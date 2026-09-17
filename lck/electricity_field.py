"""Timestamped one-meter pilot, atomic forecasting and prequential score state.

The qualified forecasting core is unchanged. This separately versioned connector
adds meter identity, units, frozen training provenance, UTC cadence and bounded
aggregate scores. Local timestamps are not a trusted external time attestation.
"""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time

from lck.electricity_forecast import ElectricityForecast, POLICIES
from lck.electricity_session import identity as core_identity
from lck.journal_file import JournalFile
from lck.online_reference import _integer, _scalar, _json

MAGIC = b"BEAREF01"
WARMUP = 720
LIMIT = 512*1024
LIMITS = {"max_snapshot": LIMIT, "max_record": LIMIT+32, "max_log": 32*1024**2, "max_records": 64}
FIELDS = {"timestamp", "meter_id", "unit", "value"}


def timestamp(value):
    if type(value) is not str:
        raise ValueError("timestamp must be timezone-aware ISO8601")
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid ISO8601 timestamp") from exc
    if dt.tzinfo is None or dt.utcoffset() is None or dt.microsecond:
        raise ValueError("timestamp needs timezone and whole seconds")
    seconds = dt.timestamp()
    if not 0 <= seconds <= 4102444800:
        raise ValueError("timestamp outside supported range")
    return int(seconds)


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def label(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value):
        raise ValueError("meter_id and unit need 1..64 ASCII identifier characters")
    return value


def reading(event, meter_id, unit):
    if type(event) is not dict or set(event) != FIELDS:
        raise ValueError("reading requires timestamp,meter_id,unit,value")
    if event["meter_id"] != meter_id or event["unit"] != unit:
        raise ValueError("meter identity or unit differs")
    seconds = timestamp(event["timestamp"])
    value = _scalar(event["value"], 1e12)
    if value < 0:
        raise ValueError("negative electricity reading")
    return seconds, value


def training(path, meter_id, unit):
    """Read a bounded historical CSV; fit normalization before creating a journal."""
    label(meter_id)
    label(unit)
    with Path(path).open("rb") as f:
        raw = f.read(16*1024**2+1)
    if len(raw) > 16*1024**2:
        raise ValueError("training CSV exceeds16MiB")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if reader.fieldnames is None or len(reader.fieldnames) != 4 or set(reader.fieldnames) != FIELDS:
        raise ValueError("training CSV requires unique timestamp,meter_id,unit,value columns")
    from collections import deque
    tail = deque(maxlen=WARMUP)
    count, mean, m2, first, previous = 0, 0., 0., None, None
    for row in reader:
        row["value"] = float(row["value"])
        seconds, value = reading(row, meter_id, unit)
        if previous is not None and seconds != previous+3600:
            raise ValueError("training must be contiguous hourly readings")
        if first is None:
            first = seconds
        previous = seconds
        count += 1
        if count > 100000:
            raise ValueError("training row limit exceeded")
        delta = value-mean
        mean += delta/count
        m2 += delta*(value-mean)
        tail.append((seconds, value))
    if count < WARMUP:
        raise ValueError("at least720 historical training readings required")
    std = math.sqrt(max(0., m2/count))
    ElectricityForecast(mean, std)
    return {"meter_id": meter_id, "unit": unit, "train_sha256": hashlib.sha256(raw).hexdigest(),
            "train_rows": count, "train_first": first, "train_last": previous,
            "origin": tail[0][0], "mean": mean, "std": std}, tuple(tail)


def identity():
    return hashlib.sha256(_json({"core": core_identity(), "connector": hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()})).hexdigest()


def empty_metrics():
    return {scope: {p: {"n": 0, "absolute": 0., "square": 0., "covered": 0,
                       "intervals": 0, "width": 0.} for p in POLICIES}
            for scope in ("all", "prospective")}


def encode(state):
    prefix = MAGIC+_json(state)
    if len(prefix)+32 > LIMIT:
        raise ValueError("field snapshot exceeds limit")
    return prefix+hashlib.sha256(prefix).digest()


def decode(raw, sequence):
    if (type(raw) is not bytes or not 40 < len(raw) <= LIMIT or raw[:8] != MAGIC
            or hashlib.sha256(raw[:-32]).digest() != raw[-32:]):
        raise ValueError("invalid field snapshot")
    try:
        state = json.loads(raw[8:-32])
        if set(state) != {"config", "events", "model", "issued", "last_observation", "last_match", "metrics"}:
            raise ValueError("invalid field schema")
        config = state["config"]
        expected_config = {"meter_id", "unit", "train_sha256", "train_rows", "train_first", "train_last",
                           "origin", "mean", "std", "source_kind", "created_at", "fixed_baseline"}
        if set(config) != expected_config or config["source_kind"] not in ("synthetic", "historical", "live"):
            raise ValueError("invalid field configuration")
        label(config["meter_id"])
        label(config["unit"])
        _integer(config["train_rows"], WARMUP, 100000)
        for key in ("train_first", "train_last", "origin", "created_at"):
            _integer(config[key], 0, 4102444800)
        if (not re.fullmatch(r"[0-9a-f]{64}", config["train_sha256"])
                or config["train_last"] != config["train_first"]+(config["train_rows"]-1)*3600
                or config["origin"] != config["train_last"]-(WARMUP-1)*3600
                or config["created_at"] < config["train_last"]
                or config["fixed_baseline"] not in POLICIES[:3]):
            raise ValueError("invalid training provenance")
        _integer(state["events"], sequence, sequence)
        model = ElectricityForecast.load(bytes.fromhex(state["model"]))
        if model.index != WARMUP-1+sequence or model.config != (config["mean"], config["std"], "selector"):
            raise ValueError("field/core clocks or configuration differ")
        issued = state["issued"]
        if set(issued) != {"at", "target_timestamp", "forecast"}:
            raise ValueError("invalid issued forecast schema")
        _integer(issued["at"], config["created_at"], 4102444800)
        if (issued["target_timestamp"] != iso(config["origin"]+(model.index+1)*3600)
                or issued["forecast"] != model.forecast):
            raise ValueError("issued forecast differs from model")
        last = state["last_observation"]
        match = state["last_match"]
        if (last is None) != (sequence == 0) or (match is None) != (sequence == 0):
            raise ValueError("invalid last observation")
        if last is not None:
            seconds, value = reading(last, config["meter_id"], config["unit"])
            if seconds != config["origin"]+model.index*3600 or seconds > issued["at"]:
                raise ValueError("last observation clock differs")
            if (value-config["mean"])/config["std"] != model._state[1][-1]:
                raise ValueError("last observation differs from model history")
            if set(match) != {"forecast_state_sha256", "available_at", "observed_at", "prospective"}:
                raise ValueError("invalid forecast match")
            _integer(match["available_at"], config["created_at"], issued["at"])
            _integer(match["observed_at"], max(seconds, match["available_at"]), issued["at"])
            if (not re.fullmatch(r"[0-9a-f]{64}", match["forecast_state_sha256"])
                    or type(match["prospective"]) is not bool
                    or match["prospective"] != (config["source_kind"] == "live" and match["available_at"] < seconds)):
                raise ValueError("invalid forecast availability witness")
        metrics = state["metrics"]
        if set(metrics) != {"all", "prospective"}:
            raise ValueError("invalid metric scopes")
        for scope, rows in metrics.items():
            if set(rows) != set(POLICIES):
                raise ValueError("invalid metric policies")
            counts = []
            for row in rows.values():
                if set(row) != {"n", "absolute", "square", "covered", "intervals", "width"}:
                    raise ValueError("invalid metric fields")
                _integer(row["n"], sequence if scope == "all" else 0, sequence)
                _integer(row["intervals"], 0, row["n"])
                _integer(row["covered"], 0, row["intervals"])
                for key in ("absolute", "square", "width"):
                    if _scalar(row[key], 1e30) < 0:
                        raise ValueError("negative metric")
                counts.append(row["n"])
            if len(set(counts)) != 1 or (scope == "prospective" and config["source_kind"] != "live" and any(counts)):
                raise ValueError("invalid prospective metric count")
        if encode(state) != raw:
            raise ValueError("noncanonical field state")
        return state, model
    except (TypeError, KeyError, IndexError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid field state") from exc


class ElectricityField:
    def __init__(self, journal, state, model, raw, clock):
        self._journal, self._state, self._model, self._raw, self._clock = journal, state, model, raw, clock
        self._busy = False
        # Observed AFTER create/open fsync. Reopen conservatively reissues at now;
        # it never assumes the previous process delivered an earlier receipt.
        self._available_at = int(clock())
        if self._available_at < state["issued"]["at"]:
            raise ValueError("local clock rollback")

    def __copy__(self):
        raise TypeError("field connector owns a writer")

    def __deepcopy__(self, memo):
        return self.__copy__()

    @classmethod
    def create(cls, directory, training_csv, meter_id, unit, *, source_kind, clock=time.time):
        if source_kind not in ("synthetic", "historical", "live"):
            raise ValueError("explicit source_kind required")
        config, tail = training(training_csv, meter_id, unit)
        now = int(clock())
        if config["train_last"] > now:
            raise ValueError("training contains future readings")
        model = ElectricityForecast(config["mean"], config["std"], policy="selector")
        errors = {p: 0. for p in POLICIES[:3]}
        previous = None
        for index, (_, value) in enumerate(tail):
            if index >= WARMUP-144:
                for p in errors:
                    errors[p] += abs(previous["predictions"][p]-value)
            previous = model.ingest(index, value)["forecast"]
        config.update(source_kind=source_kind, created_at=now, fixed_baseline=min(errors, key=lambda p: (errors[p], p)))
        state = {"config": config, "events": 0, "model": model.dump().hex(), "last_observation": None, "last_match": None,
                 "metrics": empty_metrics(), "issued": {"at": now,
                 "target_timestamp": iso(config["train_last"]+3600), "forecast": previous}}
        raw = encode(state)
        decode(raw, 0)
        journal = JournalFile.create(directory, identity(), raw, backend="raw", **LIMITS)
        try:
            return cls(journal, state, model, raw, clock)
        except BaseException:
            journal.close()
            raise

    @classmethod
    def open(cls, directory, meter_id, unit, *, min_events=0, expected_config_hash=None, clock=time.time):
        journal = JournalFile.open(directory, identity(), backend="raw", min_sequence=min_events, **LIMITS)
        try:
            raw = journal.base()
            state, model = decode(raw, journal.meta["sequence"])
            for sequence, payload in journal.records():
                if len(payload) <= 32 or payload[:32] != hashlib.sha256(raw).digest():
                    raise ValueError("field state chain differs")
                next_state, next_model = decode(payload[32:], sequence)
                if next_state["config"] != state["config"] or next_state["issued"]["at"] < state["issued"]["at"]:
                    raise ValueError("field identity or issue clock changed")
                state, model, raw = next_state, next_model, payload[32:]
            if state["config"]["meter_id"] != meter_id or state["config"]["unit"] != unit:
                raise ValueError("wrong meter or unit for deployment")
            digest = hashlib.sha256(_json(state["config"])).hexdigest()
            if expected_config_hash is not None and expected_config_hash != digest:
                raise ValueError("deployment configuration differs from host receipt")
            return cls(journal, state, model, raw, clock)
        except BaseException:
            journal.close()
            raise

    def _ready(self):
        self._journal._ready()
        if (self._busy or type(self._available_at) is not int or self._available_at < self._state["issued"]["at"]
                or self._state["events"] != self._journal.sequence or encode(self._state) != self._raw):
            raise RuntimeError("uncertain field state; reopen")

    def status(self):
        self._ready()
        return {"events": self._state["events"], "config": json.loads(_json(self._state["config"])),
                "config_sha256": hashlib.sha256(_json(self._state["config"])).hexdigest(),
                "issued": json.loads(_json(self._state["issued"])),
                "last_observation": json.loads(_json(self._state["last_observation"])),
                "last_match": json.loads(_json(self._state["last_match"])),
                "forecast_available_at": self._available_at,
                "state_sha256": hashlib.sha256(self._raw).hexdigest(), "durable": True}

    def ingest(self, event):
        self._ready()
        config, issued = self._state["config"], self._state["issued"]
        seconds, value = reading(event, config["meter_id"], config["unit"])
        if seconds != timestamp(issued["target_timestamp"]):
            raise ValueError("expected next hourly timestamp; gaps and duplicates require reconciliation")
        now = int(self._clock())
        if now < seconds or now < issued["at"] or now < self._available_at:
            raise ValueError("future reading or local clock rollback")
        self._busy = True
        try:
            model = ElectricityForecast.load(bytes.fromhex(self._state["model"]))
            result = model.ingest(model.index+1, value)
            state = json.loads(_json(self._state))
            prospective = config["source_kind"] == "live" and self._available_at < seconds
            for scope in ("all", "prospective") if prospective else ("all",):
                for policy in POLICIES:
                    row = state["metrics"][scope][policy]
                    error = (issued["forecast"]["predictions"][policy]-value)/config["std"]
                    row["n"] += 1
                    row["absolute"] += abs(error)
                    row["square"] += error*error
                    interval = issued["forecast"]["intervals"][policy]
                    if interval is not None:
                        lo, hi = interval
                        row["intervals"] += 1
                        row["covered"] += int(lo <= value <= hi)
                        row["width"] += (hi-lo)/config["std"]
            state.update(events=state["events"]+1, model=model.dump().hex(),
                         last_match={"forecast_state_sha256": hashlib.sha256(self._raw).hexdigest(),
                                     "available_at": self._available_at, "observed_at": now, "prospective": prospective},
                         last_observation={"timestamp": iso(seconds), "meter_id": config["meter_id"],
                                           "unit": config["unit"], "value": value},
                         issued={"at": now, "target_timestamp": iso(seconds+3600), "forecast": result["forecast"]})
            raw = encode(state)
            decode(raw, state["events"])
            if len(self._journal.index) >= 32:
                self._journal.checkpoint(self._raw)
                self._journal.prune()
            self._journal.append(hashlib.sha256(self._raw).digest()+raw, self._journal.sequence)
            self._state, self._model, self._raw = state, model, raw
            self._available_at = None
            self._available_at = int(self._clock())
            if self._available_at < now:
                raise RuntimeError("clock rollback after append; reopen")
        finally:
            self._busy = False
        return self.status()

    def report(self):
        status = self.status()
        summary = {}
        for scope, rows in self._state["metrics"].items():
            summary[scope] = {p: {"n": r["n"], "nmae": r["absolute"]/r["n"] if r["n"] else None,
                "nmse": r["square"]/r["n"] if r["n"] else None,
                "coverage90": r["covered"]/r["intervals"] if r["intervals"] else None,
                "mean_width_train_std": r["width"]/r["intervals"] if r["intervals"] else None}
                for p, r in rows.items()}
        comparisons = {}
        for scope, rows in summary.items():
            candidate = rows["selector"]["nmae"]
            comparisons[scope] = {control: None if candidate is None or not rows[control]["nmae"] else
                                  1-candidate/rows[control]["nmae"]
                                  for control in (self._state["config"]["fixed_baseline"], "baseline_selector")}
        return {**status, "metrics": summary, "relative_nmae_gain": comparisons, "external_acceptance": "not_recorded",
                "field_decision": "PENDING_EXTERNAL_EVALUATION",
                "scope": "aggregate errors of forecasts stored before ingest; prospective needs live mode and local post-fsync availability before target timestamp; no external delivery/time attestation"}

    def close(self):
        if self._busy:
            raise RuntimeError("busy field connector")
        self._journal.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
