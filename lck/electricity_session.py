"""One-writer durable electricity service, one fsynced receipt per reading."""
import hashlib
from pathlib import Path
import platform

from lck.electricity_forecast import ElectricityForecast, CAP
from lck.journal_file import JournalFile
from lck.online_reference import _json

LIMITS = {"max_snapshot": CAP, "max_record": CAP+32, "max_log": 32*1024**2, "max_records": 128}
SOURCE_FILES = ("electricity_session.py", "electricity_forecast.py", "robust_reference.py",
                "seasonal_reference.py", "empirical_reference.py", "change_reference.py",
                "online_reference.py", "journal_file.py", "nxa_segment.py")


def identity():
    root = Path(__file__).resolve().parent
    return hashlib.sha256(_json({"kind": "electricity-session-1", "python": platform.python_version(),
        "implementation": platform.python_implementation(), "machine": platform.machine(),
        "sources": {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in SOURCE_FILES}})).hexdigest()


class ElectricitySession:
    def __init__(self, file, model, raw):
        self._file, self._model, self._raw = file, model, raw
        self._busy = False

    def __copy__(self):
        raise TypeError("electricity session owns a writer lock")

    def __deepcopy__(self, memo):
        return self.__copy__()

    @classmethod
    def create(cls, directory, mean, std, *, policy="selector"):
        model = ElectricityForecast(mean, std, policy=policy)
        raw = model.dump()
        file = JournalFile.create(directory, identity(), raw, backend="raw", **LIMITS)
        return cls(file, model, raw)

    @classmethod
    def open(cls, directory, *, min_index=-1):
        if type(min_index) is not int or min_index < -1:
            raise ValueError("invalid receipt index")
        file = JournalFile.open(directory, identity(), backend="raw", min_sequence=min_index+1, **LIMITS)
        try:
            raw = file.base()
            model = ElectricityForecast.load(raw)
            if model.index+1 != file.meta["sequence"]:
                raise ValueError("electricity checkpoint clock differs")
            for sequence, payload in file.records():
                if len(payload) <= 32 or payload[:32] != hashlib.sha256(raw).digest():
                    raise ValueError("electricity state chain differs")
                candidate = ElectricityForecast.load(payload[32:])
                if candidate.index != model.index+1 or candidate.index+1 != sequence or candidate.config != model.config:
                    raise ValueError("electricity committed transition differs")
                raw, model = payload[32:], candidate
            return cls(file, model, raw)
        except BaseException:
            file.close()
            raise

    def _ready(self):
        self._file._ready()
        if self._busy or self._model.index+1 != self._file.sequence or self._model.dump() != self._raw:
            raise RuntimeError("uncertain electricity owner; reopen")

    def status(self):
        self._ready()
        return {"index": self._model.index, "forecast": self._model.forecast,
                "state_sha256": hashlib.sha256(self._raw).hexdigest(), "durable": True}

    def ingest(self, index, value):
        self._ready()
        self._busy = True
        try:
            return self._ingest(index, value)
        finally:
            self._busy = False

    def _ingest(self, index, value):
        candidate = ElectricityForecast.load(self._raw)
        result = candidate.ingest(index, value)
        raw = candidate.dump()
        ElectricityForecast.load(raw)
        # Compact before the next append, always from the durable prefix.
        if len(self._file.index) >= 64:
            self._file.checkpoint(self._raw)
            self._file.prune()
        self._file.append(hashlib.sha256(self._raw).digest()+raw, self._file.sequence)
        self._model, self._raw = candidate, raw
        return {**result, "state_sha256": hashlib.sha256(raw).hexdigest(), "durable": True}

    def close(self):
        if self._busy:
            raise RuntimeError("busy electricity session")
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
