"""Local fsynced receipt collector with one-event crash reconciliation.

Use only with an exclusively owned field journal. Complete corrupt log entries
fail closed; only an incomplete final line is truncated. A journal one event
ahead can be reconciled using its retained last receipt. Larger gaps fail closed.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path

from lck.electricity_field import ElectricityField
from lck.journal_file import _write_all
from lck.online_reference import _json

MAX_LOG = 64*1024**2
MAX_LINE = 32768


class FieldCapture:
    def __init__(self, field, fd, last):
        self.field, self.fd, self.last = field, fd, last
        self.uncertain = False

    def __copy__(self):
        raise TypeError("collector owns writer descriptors")

    def __deepcopy__(self, memo):
        return self.__copy__()

    @classmethod
    def open(cls, directory, archive, meter_id, unit, *, min_events=0, config_sha256=None):
        archive = Path(archive).absolute()
        fd = os.open(archive, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        field = None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            size = os.fstat(fd).st_size
            if size > MAX_LOG:
                raise ValueError("collector exceeds64MiB")
            last, cursor = None, 0
            with os.fdopen(os.dup(fd), "rb") as reader:
                reader.seek(0)
                while cursor < size:
                    line = reader.readline(MAX_LINE+1)
                    if len(line) > MAX_LINE:
                        raise ValueError("collector line exceeds limit")
                    if not line.endswith(b"\n"):
                        break
                    row = json.loads(line)
                    if set(row) != {"previous", "receipt", "kind", "sha256"}:
                        raise ValueError("invalid collector schema")
                    body = {key: row[key] for key in ("previous", "receipt", "kind")}
                    if (row["sha256"] != hashlib.sha256(_json(body)).hexdigest()
                            or row["previous"] != (last["sha256"] if last else None)
                            or _json(row)+b"\n" != line):
                        raise ValueError("collector checksum or chain differs")
                    receipt = row["receipt"]
                    if (receipt["durable"] is not True or receipt["events"] != (last["receipt"]["events"]+1 if last else 0)
                            or row["kind"] not in (("accepted", "reconciled") if last else ("prepared",))):
                        raise ValueError("collector receipt sequence differs")
                    if last and (receipt["config_sha256"] != last["receipt"]["config_sha256"]
                                 or receipt["last_match"]["forecast_state_sha256"] != last["receipt"]["state_sha256"]):
                        raise ValueError("collector forecast linkage differs")
                    cursor += len(line)
                    last = row
            if cursor != size:
                os.ftruncate(fd, cursor)
            os.fsync(fd)
            parent = os.open(archive.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
            field = ElectricityField.open(directory, meter_id, unit, min_events=min_events,
                                           expected_config_hash=config_sha256)
            capture = cls(field, fd, last)
            status = field.status()
            if last is None:
                if status["events"] != 0:
                    raise ValueError("new collector requires a prepared, unconsumed deployment")
                capture._append(status, "prepared")
            else:
                if last["receipt"]["config_sha256"] != status["config_sha256"]:
                    raise ValueError("collector belongs to another deployment")
                gap = status["events"]-last["receipt"]["events"]
                if gap == 0:
                    if status["state_sha256"] != last["receipt"]["state_sha256"]:
                        raise ValueError("collector and field state differ")
                elif gap == 1:
                    capture._append(status, "reconciled")
                else:
                    raise ValueError("collector gap cannot be reconciled automatically")
            return capture
        except BaseException:
            if field is not None:
                field.close()
            os.close(fd)
            raise

    def _append(self, receipt, kind):
        if self.last is not None and receipt["last_match"]["forecast_state_sha256"] != self.last["receipt"]["state_sha256"]:
            raise ValueError("receipt does not match archived forecast")
        body = {"previous": self.last["sha256"] if self.last else None, "receipt": receipt, "kind": kind}
        row = {**body, "sha256": hashlib.sha256(_json(body)).hexdigest()}
        data = _json(row)+b"\n"
        if len(data) > MAX_LINE or os.fstat(self.fd).st_size+len(data) > MAX_LOG:
            raise ValueError("collector full; stop before continuing")
        os.lseek(self.fd, 0, os.SEEK_END)
        _write_all(self.fd, data)
        os.fsync(self.fd)
        self.last = row

    def ingest(self, event):
        if self.uncertain:
            raise RuntimeError("uncertain collector; reopen")
        self.uncertain = True
        receipt = self.field.ingest(event)
        self._append(receipt, "accepted")
        self.uncertain = False
        return {**receipt, "collector_sha256": self.last["sha256"], "collector_durable": True}

    def close(self):
        self.field.close()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
