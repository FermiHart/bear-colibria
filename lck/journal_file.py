"""One-writer raw WAL with selectable checkpoints. Checksums are not authentication.

NXA (default) preserves manifest v1 and its native root. Raw/Zstd use v2 with a
backend field and root = SHA256(encoded file); base_hash always hashes raw bytes.
Open requires the matching backend. Zstd uses one sized, checksummed frame without
a dictionary. Raw/Zstd bound encoded reads as well as decoded snapshot size.

Directory/CURRENT and storage are trusted. Complete frames fail closed on invalid
sequence/header/checksum; an incomplete final frame may be discarded. A damaged
length within max_record can look incomplete. min_sequence lets a host reject
rollback below its last known receipt. No external-action replay or background I/O.

create requires an existing, durably established parent; it never creates ancestors.
create/open sync the immediate parent, not its ancestors. The caller owns their
durability and must not concurrently rename or replace these trusted directories.
Directory and library paths are anchored to the initialization cwd.

A writer belongs to its creating PID and Thread object, not a reusable thread ID.
After fork an inherited instance cannot do I/O or publish, even if the parent has
closed it. The child may only close its inherited descriptors, without LOCK_UN:
the parent's flock remains owned. There is no automatic at-fork cleanup; a child
retaining descriptors can keep the flock alive after the parent closes/exits.
Close inherited instances before an independent create/open in the child, which
still requires the parent (and all other inheritors) to release the writer lock.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import threading
import uuid

from lck.nxa_segment import NXASegment

HEADER = struct.Struct("<8s16sQ32s")
RECORD = struct.Struct("<QI32s")
MAGIC = b"BJWAL001"


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("short journal write")
        view = view[count:]


class JournalFile:
    def __copy__(self):
        raise TypeError("JournalFile owns a writer lock and cannot be copied")

    def __deepcopy__(self, memo):
        return self.__copy__()

    def __init__(self, directory, core_id, nxa_library=None, *, backend="nxa", max_record=1024**2,
                 max_log=8 * 1024**2, max_records=4096, max_snapshot=32 * 1024**2):
        if not isinstance(core_id, str) or not re.fullmatch(r"[0-9a-f]{64}", core_id):
            raise ValueError("invalid core identity")
        if any(type(n) is not int or n <= 0 for n in (max_record, max_log, max_records, max_snapshot)):
            raise ValueError("positive journal limits required")
        if max_log < HEADER.size:
            raise ValueError("journal limit cannot hold its header")
        if backend not in ("nxa", "raw", "zstd"):
            raise ValueError("unsupported checkpoint backend")
        if backend == "nxa" and nxa_library is None:
            raise ValueError("NXA library required")
        cwd = Path.cwd()
        self.directory, self.core_id = cwd / directory, core_id
        self.backend, self._suffix = backend, "zst" if backend == "zstd" else backend
        self.nxa = NXASegment(str(cwd / nxa_library)) if backend == "nxa" else None
        self.max_record, self.max_log, self.max_records, self.max_snapshot = max_record, max_log, max_records, max_snapshot
        self.fd = self.lock = None
        self.poisoned = False
        self._busy = False
        self._pid, self._owner = os.getpid(), threading.current_thread()
        self.hook = None
        self.meta = None
        self.sequence = 0
        self.index = []
        self.lock = os.open(self.directory / "LOCK", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.lock)
            self.lock = None
            raise

    @classmethod
    def create(cls, directory, core_id, base, nxa_library=None, *, backend="nxa", **limits):
        """Return only after syncing contents and the new entry in the parent.

        Failure releases the writer lock but may leave a visible journal. Use open
        to reconcile a complete generation, rather than overwriting it with create.
        """
        directory = Path(directory).absolute()
        directory.mkdir(mode=0o700, exist_ok=False)
        journal = cls(directory, core_id, nxa_library, backend=backend, **limits)
        try:
            journal.checkpoint(base)
            journal._sync_dir(journal.directory.parent)
            return journal
        except BaseException:
            journal.close()
            raise

    @classmethod
    def open(cls, directory, core_id, nxa_library=None, *, backend="nxa", min_sequence=0, **limits):
        journal = cls(directory, core_id, nxa_library, backend=backend, **limits)
        try:
            if type(min_sequence) is not int or min_sequence < 0:
                raise ValueError("invalid minimum sequence")
            journal.meta = journal._manifest()
            journal.base()
            journal.fd = os.open(journal.directory / journal.meta["wal"], os.O_RDWR)
            expected = journal._header(journal.meta)
            if os.pread(journal.fd, HEADER.size, 0) != expected:
                raise ValueError("journal/checkpoint header mismatch")
            size = os.fstat(journal.fd).st_size
            if not HEADER.size <= size <= journal.max_log:
                raise ValueError("journal file size outside limit")
            cursor = HEADER.size
            sequence = journal.meta["sequence"]
            previous = hashlib.sha256(expected).digest()
            while cursor < size:
                prefix = os.pread(journal.fd, RECORD.size, cursor)
                if len(prefix) < RECORD.size:
                    break
                seq, length, prev = RECORD.unpack(prefix)
                if seq != sequence + 1 or seq >= 2**63 or prev != previous or not 0 < length <= journal.max_record:
                    raise ValueError("invalid journal sequence/header")
                remaining = size - cursor - RECORD.size
                if remaining < length + 32:
                    break
                payload = os.pread(journal.fd, length, cursor + RECORD.size)
                digest = os.pread(journal.fd, 32, cursor + RECORD.size + length)
                if len(payload) != length or hashlib.sha256(prefix + payload).digest() != digest:
                    raise ValueError("journal checksum mismatch")
                if len(journal.index) >= journal.max_records:
                    raise ValueError("journal record limit exceeded")
                journal.index.append((seq, cursor, length, previous, digest))
                sequence, previous = seq, digest
                cursor += RECORD.size + length + 32
            if sequence < min_sequence:
                raise ValueError("journal older than host receipt")
            if cursor != size:
                os.ftruncate(journal.fd, cursor)
            # Reconcile visible writes/manifest renames whose prior sync may have failed.
            os.fsync(journal.fd)
            journal._sync_dir()
            journal._sync_dir(journal.directory.parent)
            journal.sequence, journal.previous, journal.end = sequence, previous, cursor
            return journal
        except BaseException:
            journal.close()
            raise

    def _manifest(self):
        with (self.directory / "CURRENT").open("rb") as source:
            raw = source.read(8193)
        if len(raw) > 8192:
            raise ValueError("manifest too large")
        meta = json.loads(raw)
        fields = {"version", "core", "id", "sequence", "base_hash", "root", "checkpoint", "wal"}
        if self.backend != "nxa":
            fields.add("backend")
        if type(meta) is not dict or set(meta) != fields:
            raise ValueError("invalid manifest fields")
        if (type(meta["version"]) is not int or meta["version"] != (1 if self.backend == "nxa" else 2)
                or meta["core"] != self.core_id or meta.get("backend", "nxa") != self.backend):
            raise ValueError("checkpoint version/core/backend mismatch")
        for key, length in (("id", 32), ("base_hash", 64), ("root", 64)):
            if not isinstance(meta[key], str) or not re.fullmatch("[0-9a-f]{" + str(length) + "}", meta[key]):
                raise ValueError("invalid manifest identity")
        if type(meta["sequence"]) is not int or not 0 <= meta["sequence"] < 2**63:
            raise ValueError("invalid checkpoint sequence")
        for key, pattern in (("checkpoint", r"checkpoint-[0-9a-f]{32}\." + self._suffix), ("wal", r"journal-[0-9a-f]{32}\.wal")):
            if not isinstance(meta[key], str) or not re.fullmatch(pattern, meta[key]):
                raise ValueError("invalid manifest filename")
        return meta

    @staticmethod
    def _header(meta):
        return HEADER.pack(MAGIC, bytes.fromhex(meta["id"]), meta["sequence"], bytes.fromhex(meta["base_hash"]))

    def _sync_dir(self, directory=None):
        fd = os.open(self.directory if directory is None else directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _hook(self, phase):
        if self.hook:
            self.hook(phase)

    def _ready(self):
        if (os.getpid() != self._pid or threading.current_thread() is not self._owner
                or self.lock is None or self.poisoned or self._busy):
            raise RuntimeError("closed/uncertain journal; reopen before further work")

    def base(self):
        self._ready()
        blob = self._read_checkpoint(self.meta)
        if hashlib.sha256(blob).hexdigest() != self.meta["base_hash"]:
            raise ValueError("base digest mismatch")
        return blob

    def _write_checkpoint(self, meta, base):
        path = self.directory / meta["checkpoint"]
        if self.backend == "nxa":
            return self.nxa.write(path, {"state.bin": base})["root"]
        encoded = base
        if self.backend == "zstd":
            import zstandard
            encoded = zstandard.ZstdCompressor(level=3, write_checksum=True).compress(base)
        with path.open("xb") as output:
            output.write(encoded)
        return hashlib.sha256(encoded).hexdigest()

    def _read_checkpoint(self, meta):
        path = self.directory / meta["checkpoint"]
        if self.backend == "nxa":
            return self.nxa.read(path, meta["root"], "state.bin", self.max_snapshot)
        limit = self.max_snapshot if self.backend == "raw" else self.max_snapshot * 2 + 65536
        with path.open("rb") as source:
            size = os.fstat(source.fileno()).st_size
            if not 0 < size <= limit:
                raise ValueError("encoded checkpoint size outside limit")
            encoded = source.read(size)
            if len(encoded) != size or source.read(1):
                raise ValueError("checkpoint length changed")
        if hashlib.sha256(encoded).hexdigest() != meta["root"]:
            raise ValueError("checkpoint root mismatch")
        if self.backend == "raw":
            return encoded
        import zstandard
        try:
            frame = zstandard.get_frame_parameters(encoded)
            # max_output_size alone does not cap frames with an advertised size.
            if (not 0 < frame.content_size <= self.max_snapshot or frame.window_size > self.max_snapshot
                    or frame.dict_id or not frame.has_checksum):
                raise ValueError("invalid zstd checkpoint frame/size")
            blob = zstandard.ZstdDecompressor().decompress(
                encoded, max_output_size=self.max_snapshot, allow_extra_data=False)
        except zstandard.ZstdError as error:
            raise ValueError("invalid zstd checkpoint") from error
        if len(blob) != frame.content_size:
            raise ValueError("checkpoint decoded length mismatch")
        return blob

    def records(self):
        self._ready()
        for seq, _, _, _, _ in self.index:
            yield seq, self.read_record(seq)

    def read_record(self, seq):
        self._ready()
        offset = seq - self.meta["sequence"] - 1
        if not 0 <= offset < len(self.index):
            raise KeyError(seq)
        stored_seq, position, length, previous, digest = self.index[offset]
        prefix = os.pread(self.fd, RECORD.size, position)
        payload = os.pread(self.fd, length, position + RECORD.size)
        trailer = os.pread(self.fd, 32, position + RECORD.size + length)
        if (stored_seq != seq or prefix != RECORD.pack(seq, length, previous)
                or len(payload) != length or trailer != digest or hashlib.sha256(prefix + payload).digest() != digest):
            raise ValueError("journal record changed")
        return payload

    def append(self, payload, expected_sequence):
        self._ready()
        if type(expected_sequence) is not int or expected_sequence != self.sequence or self.sequence >= 2**63 - 1:
            raise ValueError("stale/exhausted journal sequence")
        if not isinstance(payload, bytes) or not 0 < len(payload) <= self.max_record:
            raise ValueError("record size outside limit")
        prefix = RECORD.pack(self.sequence + 1, len(payload), self.previous)
        digest = hashlib.sha256(prefix + payload).digest()
        frame = prefix + payload + digest
        if len(self.index) >= self.max_records or self.end + len(frame) > self.max_log:
            raise ValueError("journal full; checkpoint required")
        self._busy = True
        try:
            if os.fstat(self.fd).st_size != self.end:
                raise ValueError("unexpected journal length")
            self._hook("before_append")
            os.lseek(self.fd, self.end, os.SEEK_SET)
            _write_all(self.fd, frame)
            self._hook("after_write")
            self._hook("before_sync")
            os.fsync(self.fd)
            self._hook("after_sync")
            self.index.append((self.sequence + 1, self.end, len(payload), self.previous, digest))
            self.sequence += 1
            self.previous = digest
            self.end += len(frame)
            return self.sequence
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self._busy = False

    def checkpoint(self, base):
        self._ready()
        if not isinstance(base, bytes) or not 0 < len(base) <= self.max_snapshot:
            raise ValueError("checkpoint size outside limit")
        uid = uuid.uuid4().hex
        meta = {"version": 1 if self.backend == "nxa" else 2, "core": self.core_id, "id": self.meta["id"] if self.meta else uuid.uuid4().hex,
                "sequence": self.sequence, "base_hash": hashlib.sha256(base).hexdigest(),
                "checkpoint": f"checkpoint-{uid}.{self._suffix}", "wal": f"journal-{uid}.wal"}
        if self.backend != "nxa":
            meta["backend"] = self.backend
        new_fd = None
        self._busy = True
        try:
            meta["root"] = self._write_checkpoint(meta, base)
            if self._read_checkpoint(meta) != base:
                raise ValueError("new checkpoint failed verification")
            with (self.directory / meta["checkpoint"]).open("rb") as checkpoint:
                os.fsync(checkpoint.fileno())
            new_fd = os.open(self.directory / meta["wal"], os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            header = self._header(meta)
            _write_all(new_fd, header)
            os.fsync(new_fd)
            self._sync_dir()
            temporary = self.directory / ("manifest-" + uid + ".tmp")
            with temporary.open("xb") as output:
                output.write(json.dumps(meta, sort_keys=True).encode())
                output.flush()
                os.fsync(output.fileno())
            self._hook("before_manifest")
            os.replace(temporary, self.directory / "CURRENT")
            self._sync_dir()
            self._hook("after_manifest")
            if self.fd is not None:
                os.close(self.fd)
            self.fd, new_fd = new_fd, None
            self.meta, self.index = meta, []
            self.previous, self.end = hashlib.sha256(header).digest(), len(header)
        except BaseException:
            self.poisoned = True
            raise
        finally:
            self._busy = False
            if new_fd is not None:
                os.close(new_fd)

    def prune(self):
        self._ready()
        if self._manifest() != self.meta:
            raise ValueError("manifest changed")
        # The open descriptor may outlive the WAL named by CURRENT.
        opened = os.fstat(self.fd)
        if (not os.path.samestat(opened, (self.directory / self.meta["wal"]).stat())
                or opened.st_size != self.end
                or os.pread(self.fd, HEADER.size, 0) != self._header(self.meta)):
            raise ValueError("journal file changed")
        self.base()  # Verify the checkpoint before removing any prior generation.
        for _ in self.records():
            pass
        keep = {self.meta["checkpoint"], self.meta["wal"]}
        removed = 0
        for path in self.directory.iterdir():
            if path.name not in keep and re.fullmatch(r"(?:checkpoint-[0-9a-f]{32}\.(?:nxa|raw|zst)|journal-[0-9a-f]{32}\.wal)", path.name):
                path.unlink()
                removed += 1
        self._sync_dir()
        return removed

    def close(self):
        # In a child only drop inherited references; never unlock the shared flock.
        # In the owner process a hook/other thread must not release an active writer.
        if os.getpid() == self._pid and (self._busy or threading.current_thread() is not self._owner):
            raise RuntimeError("busy journal or wrong owner; reopen before further work")
        for field in ("fd", "lock"):
            value = getattr(self, field)
            if value is not None:
                os.close(value)
                setattr(self, field, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
