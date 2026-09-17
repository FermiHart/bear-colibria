"""Explicitly loaded ctypes adapter for immutable BearNXA v3 segments.

Use NXASegment(library_path), or NXASegment(os.environ["NXA_LIBRARY"]). Importing
this module never loads native code. Writes stage byte buffers as temporary files
and use the explicit nxa_create_v3 API. Write names must be unique safe UTF-8
basenames, not paths; BearNXA also validates them. Timing the entire write call
includes staging, native archive creation, publication, and staging cleanup.

Roots must be retained through an independently trusted channel. They commit
archive contents, not producer identity. V3 has no signing or append support,
and there is no cross-archive deduplication promise. Reads materialize one object
(the native v3 limit is 1 GiB); max_bytes bounds its size, not total native working
memory, which also includes metadata and decoded blocks. Writes replace a whole
archive; callers must coordinate concurrent writers. Publication is atomic but
this adapter does not promise crash durability of its final rename.
"""

import ctypes as C
import os
from pathlib import Path
from tempfile import TemporaryDirectory


_V3_FLAG = 1 << 9


class _CreateOpts(C.Structure):
    # Native, ABI-frozen nxa_create_opts_t, NOT the packed archive wire header.
    _fields_ = [
        ("compression", C.c_int),
        ("hash", C.c_int),
        ("chunk_size", C.c_uint32),
        ("flags", C.c_uint32),
        ("compression_level", C.c_int),
        ("deduplicate", C.c_bool),
        ("reproducible", C.c_bool),
    ]


class _ObjectInfo(C.Structure):
    _fields_ = [("size", C.c_uint64), ("chunk_count", C.c_uint64)]


class NXAError(RuntimeError):
    """Native failure, retaining the nxa_error_t value in code."""

    def __init__(self, operation: str, code: int, message: str):
        self.code = code
        super().__init__(f"{operation}: {message} (NXA error {code})")


class NXASegment:
    """Reusable library binding; every operation owns and closes its handle."""

    def __init__(self, lib_path: str):
        if not isinstance(lib_path, str):
            raise TypeError("lib_path must be a string")
        if not lib_path or "\0" in lib_path:
            raise ValueError("lib_path must be nonempty and contain no NUL")
        self._lib = C.CDLL(lib_path)
        signatures = {
            "nxa_default_opts": ([C.POINTER(_CreateOpts)], None),
            "nxa_create_v3": (
                [C.c_char_p, C.POINTER(C.c_char_p), C.c_size_t, C.POINTER(_CreateOpts)],
                C.c_int,
            ),
            "nxa_open": ([C.c_char_p, C.POINTER(C.c_void_p)], C.c_int),
            "nxa_open_pinned": (
                [C.c_char_p, C.POINTER(C.c_uint8), C.POINTER(C.c_void_p)], C.c_int,
            ),
            "nxa_get_manifest_root": ([C.c_void_p, C.POINTER(C.c_uint8)], C.c_int),
            "nxa_object_stat": (
                [C.c_void_p, C.c_char_p, C.POINTER(_ObjectInfo)], C.c_int,
            ),
            "nxa_read_object": (
                [C.c_void_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
                C.c_int,
            ),
            "nxa_free": ([C.c_void_p], None),
            "nxa_close": ([C.c_void_p], None),
            "nxa_strerror": ([C.c_int], C.c_char_p),
        }
        for name, (argtypes, restype) in signatures.items():
            try:
                function = getattr(self._lib, name)
            except AttributeError as exc:
                raise RuntimeError(f"BearNXA library lacks required function {name}") from exc
            function.argtypes = argtypes
            function.restype = restype

    def _check(self, code: int, operation: str) -> None:
        if code != 0:
            message = self._lib.nxa_strerror(code)
            text = message.decode("utf-8", errors="replace") if message else "Unknown error"
            raise NXAError(operation, code, text)

    @staticmethod
    def _name(name: str) -> bytes:
        if not isinstance(name, str):
            raise TypeError("object names must be strings")
        if "\0" in name:
            raise ValueError("object names must contain no NUL")
        # Reject C-string truncation here; leave logical-path policy to BearNXA.
        return name.encode("utf-8")

    def write(
        self, path: Path, objects: dict[str, bytes], block_size: int = 65536,
    ) -> dict[str, object]:
        """Create v3 from bytes and return root (64 hex) and actual stored_bytes.

        block_size is 1..1048576, the current v3 native limit. Names must be
        nonempty basenames of at most 255 UTF-8 bytes, with no slashes, control
        characters, or '.'/'..' components. Staging and cleanup happen inside
        this call and must be included in write timing. Failures before
        publication leave an existing destination unchanged. Python input errors
        raise TypeError or ValueError; filesystem and loader errors raise OSError.
        """
        if not isinstance(block_size, int) or isinstance(block_size, bool):
            raise TypeError("block_size must be an integer")
        if not 1 <= block_size <= 1048576:
            raise ValueError("block_size must be between 1 and 1048576")
        if not isinstance(objects, dict):
            raise TypeError("objects must be a dict of names to bytes")
        path = Path(path)
        if b"\0" in os.fsencode(path):
            raise ValueError("path must contain no NUL")

        items = list(objects.items())
        for name, data in items:
            if not isinstance(data, bytes):
                raise TypeError("object data must be bytes")
            encoded_name = self._name(name)
            if not name or name in (".", "..") or "/" in name or "\\" in name:
                raise ValueError("object names must be simple basenames, without slashes or '.'/'..'")
            if len(encoded_name) > 255 or any(ch < 32 or ch == 127 for ch in encoded_name):
                raise ValueError("object names must be safe basenames of at most 255 UTF-8 bytes")

        opts = _CreateOpts()
        self._lib.nxa_default_opts(C.byref(opts))
        opts.flags |= _V3_FLAG
        opts.chunk_size = block_size
        with TemporaryDirectory(prefix=".nxa-", dir=path.parent) as directory:
            inputs = Path(directory) / "objects"
            inputs.mkdir()
            files = (C.c_char_p * len(items))()
            for i, (name, data) in enumerate(items):
                staged = inputs / name
                # Exclusive creation also rejects aliases on case-insensitive filesystems.
                with staged.open("xb") as output:
                    output.write(data)
                files[i] = os.fsencode(staged)
            temporary = Path(directory) / "segment.nxa"
            encoded_path = os.fsencode(temporary)
            self._check(
                self._lib.nxa_create_v3(encoded_path, files, len(items), C.byref(opts)),
                "nxa_create_v3",
            )
            handle = C.c_void_p()
            root = (C.c_uint8 * 32)()
            try:
                self._check(self._lib.nxa_open(encoded_path, C.byref(handle)), "nxa_open")
                if not handle:
                    raise RuntimeError("nxa_open returned a NULL handle")
                self._check(
                    self._lib.nxa_get_manifest_root(handle, root),
                    "nxa_get_manifest_root",
                )
            finally:
                if handle:
                    self._lib.nxa_close(handle)
            result = {"root": bytes(root).hex(), "stored_bytes": temporary.stat().st_size}
            os.replace(temporary, path)
            return result

    def read(self, path: Path, root: str, name: str, max_bytes: int) -> bytes:
        """Read a named object using a trusted root and a pre-allocation size cap.

        The pinned metadata size check and payload read use the SAME handle.
        A too-large object raises ValueError before nxa_read_object is called.
        Native integrity, missing-object, and I/O failures raise NXAError.
        """
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool):
            raise TypeError("max_bytes must be an integer")
        if max_bytes < 0:
            raise ValueError("max_bytes must be nonnegative")
        if not isinstance(root, str):
            raise TypeError("root must be a string")
        if len(root) != 64:
            raise ValueError("root must contain exactly 64 hexadecimal characters")
        raw_root = bytes.fromhex(root)
        if len(raw_root) != 32:
            raise ValueError("root must contain exactly 64 hexadecimal characters")
        expected_root = (C.c_uint8 * 32).from_buffer_copy(raw_root)
        encoded_name = self._name(name)
        encoded_path = os.fsencode(path)
        if b"\0" in encoded_path:
            raise ValueError("path must contain no NUL")
        handle = C.c_void_p()
        data = C.c_void_p()
        size = C.c_size_t()
        info = _ObjectInfo()
        try:
            self._check(
                self._lib.nxa_open_pinned(encoded_path, expected_root, C.byref(handle)),
                "nxa_open_pinned",
            )
            if not handle:
                raise RuntimeError("nxa_open_pinned returned a NULL handle")
            self._check(self._lib.nxa_object_stat(handle, encoded_name, C.byref(info)), "nxa_object_stat")
            if info.size > max_bytes:
                raise ValueError(f"object size {info.size} exceeds max_bytes {max_bytes}")
            self._check(
                self._lib.nxa_read_object(handle, encoded_name, C.byref(data), C.byref(size)),
                "nxa_read_object",
            )
            if size.value != info.size or size.value > max_bytes:
                raise RuntimeError("nxa_read_object returned an unexpected size")
            if size.value and not data:
                raise RuntimeError("nxa_read_object returned NULL data for a nonempty object")
            return C.string_at(data, size.value) if size.value else b""
        finally:
            if data:
                self._lib.nxa_free(data)
            if handle:
                self._lib.nxa_close(handle)
