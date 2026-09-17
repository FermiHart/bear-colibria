"""Durable pre-period forecasts using SQLite WAL/FULL and a single writer.

Preparation commits the forecast first. Confirmation samples the local clock
AFTER that commit and commits the availability witness separately. An uncertain
preparation never receives retroactive credit after the lead-time deadline.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import threading
import time

from lck.direct_electricity import DirectElectricity, POLICIES, HORIZONS
from lck.electricity_session import identity as core_identity
from lck.online_reference import _scalar, _integer, _json


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity():
    root = Path(__file__).resolve().parent
    return digest(_json({"kind": "future-electricity-1", "core": core_identity(),
                        "sources": {p: digest((root/p).read_bytes()) for p in ("future_electricity.py", "direct_electricity.py")},
                        "sqlite": sqlite3.sqlite_version, "python": platform.python_version()}))


def future_target(now, origin_end, lead=300, max_age=14400):
    _integer(now, 0, 4102444800)
    _integer(origin_end, 0, 4102444800)
    if origin_end % 3600 or not 0 <= now-origin_end <= max_age:
        raise ValueError("source too old/future or not an hourly boundary")
    start = ((now+lead+3599)//3600)*3600
    end = start+3600
    h = (end-origin_end)//3600
    if h not in HORIZONS:
        raise ValueError("required direct horizon is unsupported")
    return end, h


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class FutureElectricity:
    def __init__(self, root, *, clock=time.time):
        self.root, self.clock = Path(root), clock
        self.pid, self.owner = os.getpid(), threading.current_thread()
        self.busy, self.uncertain, self.hook = False, False, None
        self.lock = os.open(self.root/"LOCK", os.O_CREAT | os.O_RDWR, 0o600)
        self.db = None
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.db = sqlite3.connect(self.root/"future.sqlite", isolation_level=None, timeout=0)
            if self.db.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise ValueError("WAL mode required")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA wal_autocheckpoint=128")
        except BaseException:
            if self.db is not None:
                self.db.close()
            os.close(self.lock)
            raise

    def __copy__(self):
        raise TypeError("future service owns a writer")

    def __deepcopy__(self, memo):
        return self.__copy__()

    @classmethod
    def create(cls, root, records, mean, std, policy, *, source_config_hash, protocol_hash, lead=300, clock=time.time):
        if policy not in POLICIES or len(records) < 480:
            raise ValueError("qualified policy and480+ contiguous warmup readings required")
        _integer(lead, 300, 3600)
        model = DirectElectricity(mean, std)
        origin = records[0][0]
        for index, (stamp, value) in enumerate(records):
            if type(stamp) is not int or stamp != origin+3600*index or stamp > int(clock()):
                raise ValueError("seed records must be contiguous past UTC hours")
            model.ingest(index, value)
        config = {"policy": policy, "source_config_hash": source_config_hash, "protocol_hash": protocol_hash,
                  "origin": origin, "mean": mean, "std": std, "lead": lead, "seed_count": len(records)}
        root = Path(root)
        root.mkdir(mode=0o700)
        sync_dir(root.parent)
        service = cls(root, clock=clock)
        try:
            service.db.execute("BEGIN IMMEDIATE")
            service.db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            service.db.execute("CREATE TABLE model (slot INTEGER PRIMARY KEY CHECK(slot=0), raw BLOB NOT NULL)")
            service.db.execute("CREATE TABLE observations (stamp INTEGER PRIMARY KEY, value REAL NOT NULL, seen INTEGER NOT NULL, seed INTEGER NOT NULL, checksum TEXT NOT NULL)")
            service.db.execute("CREATE TABLE forecasts (target INTEGER PRIMARY KEY, payload BLOB NOT NULL, checksum TEXT NOT NULL, available INTEGER, confirmation TEXT)")
            service.db.executemany("INSERT INTO meta VALUES (?,?)", (("identity", identity()), ("config", _json(config).decode()),
                ("config_sha256", digest(_json(config)))))
            service.db.execute("INSERT INTO model VALUES (0,?)", (model.dump(),))
            seen = int(clock())
            for stamp, value in records:
                val = float(value)
                service.db.execute("INSERT INTO observations VALUES (?,?,?,?,?)", (stamp, val, seen, 1,
                    digest(_json([stamp, val, seen, 1]))))
            service.db.execute("COMMIT")
            service.config = config
            sync_dir(root)
            sync_dir(root.parent)
            return service
        except BaseException:
            service.close()
            raise

    @classmethod
    def open(cls, root, *, clock=time.time, source_config_hash=None, min_source_end=None):
        root = Path(root)
        if not (root/"future.sqlite").is_file():
            raise ValueError("existing future database required")
        service = cls(root, clock=clock)
        try:
            if service.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("future database integrity check failed")
            metadata = dict(service.db.execute("SELECT key,value FROM meta"))
            if set(metadata) != {"identity", "config", "config_sha256"} or metadata["identity"] != identity():
                raise ValueError("future runtime identity changed")
            if digest(metadata["config"].encode()) != metadata["config_sha256"]:
                raise ValueError("future configuration checksum differs")
            service.config = json.loads(metadata["config"])
            if source_config_hash is not None and service.config["source_config_hash"] != source_config_hash:
                raise ValueError("future source identity differs")
            model = service._model()
            end = service.config["origin"]+model.index*3600
            if min_source_end is not None and end < min_source_end:
                raise ValueError("future service older than host receipt")
            for stamp, value, seen, seed, checksum in service.db.execute("SELECT * FROM observations"):
                if digest(_json([stamp, value, seen, seed])) != checksum:
                    raise ValueError("observation checksum differs")
            service._forecasts()
            return service
        except BaseException:
            service.close()
            raise

    def _ready(self):
        if (self.db is None or os.getpid() != self.pid or threading.current_thread() is not self.owner
                or self.busy or self.uncertain):
            raise RuntimeError("closed/busy/uncertain future service; reopen")

    def _model(self):
        raw = self.db.execute("SELECT raw FROM model WHERE slot=0").fetchone()[0]
        model = DirectElectricity.load(raw)
        if (model._state["mean"], model._state["std"]) != (self.config["mean"], self.config["std"]):
            raise ValueError("future normalizer differs")
        count, first, last = self.db.execute("SELECT count(*),min(stamp),max(stamp) FROM observations").fetchone()
        if count != model.index+1 or first != self.config["origin"] or last != first+model.index*3600:
            raise ValueError("future source/model clock mismatch")
        return model

    def _transaction(self, statements, phase):
        self.busy = True
        try:
            self.db.execute("BEGIN IMMEDIATE")
            for sql, params in statements:
                self.db.execute(sql, params)
            self.db.execute("COMMIT")
            if self.hook:
                self.hook(phase)
        except BaseException:
            self.uncertain = True
            if self.db.in_transaction:
                self.db.execute("ROLLBACK")
            raise
        finally:
            self.busy = False

    def ingest(self, records):
        self._ready()
        model = self._model()
        latest = self.config["origin"]+model.index*3600
        seen = int(self.clock())
        statements, consumed = [], 0
        for stamp, value in records:
            value = _scalar(value, 1e12)
            if type(stamp) is not int or stamp > seen:
                raise ValueError("future-dated or invalid source label")
            if stamp <= latest:
                existing = self.db.execute("SELECT value FROM observations WHERE stamp=?", (stamp,)).fetchone()
                if existing is None or existing[0] != value:
                    raise ValueError("source history revision must not overwrite accepted vintage")
                continue
            if stamp != latest+3600 or consumed >= 48:
                raise ValueError("source gap or catch-up limit")
            model.ingest(model.index+1, value)
            statements.append(("INSERT INTO observations VALUES (?,?,?,?,?)", (stamp, value, seen, 0,
                               digest(_json([stamp, value, seen, 0])))))
            latest, consumed = stamp, consumed+1
        if consumed:
            statements.append(("UPDATE model SET raw=? WHERE slot=0", (model.dump(),)))
            self._transaction(statements, "after_observation_commit")
        return {"source_end": latest, "consumed": consumed, "durable": True}

    def prepare(self):
        self._ready()
        model = self._model()
        now = int(self.clock())
        origin_end = self.config["origin"]+model.index*3600
        end, h = future_target(now, origin_end, self.config["lead"])
        existing = self.db.execute("SELECT 1 FROM forecasts WHERE target=?", (end,)).fetchone()
        if existing:
            return end
        if self.db.execute("SELECT count(*) FROM forecasts").fetchone()[0] >= 10000:
            raise ValueError("future forecast archive limit")
        forecast = model.latest[str(h)]
        if any(interval is None for interval in forecast["intervals"].values()):
            raise ValueError("direct intervals not calibrated yet")
        payload = {"target_end": end, "target_start": end-3600, "origin_end": origin_end, "horizon": h,
                   "prepared_at": now, "deadline": end-3600-self.config["lead"],
                   "model_sha256": digest(model.dump()), "policy": self.config["policy"],
                   "predictions": forecast["predictions"], "intervals": forecast["intervals"]}
        raw = _json(payload)
        self._transaction([("INSERT INTO forecasts VALUES (?,?,?,?,?)", (end, raw, digest(raw), None, None))], "after_prepare_commit")
        return end

    def confirm(self, target):
        self._ready()
        row = self.db.execute("SELECT payload,checksum,available FROM forecasts WHERE target=?", (target,)).fetchone()
        if row is None:
            raise ValueError("prepared forecast missing")
        payload = json.loads(row[0])
        if digest(row[0]) != row[1]:
            raise ValueError("prepared forecast checksum differs")
        now = int(self.clock())  # First preparation is already committed and durable.
        if row[2] is None:
            if not payload["prepared_at"] <= now <= payload["deadline"]:
                return {"target_end": target, "status": "unconfirmed_deadline", "eligible": False}
            self._transaction([("UPDATE forecasts SET available=?,confirmation=? WHERE target=? AND available IS NULL",
                               (now, digest(_json([row[1], now])), target))], "after_confirmation_commit")
        return next(row for row in self._forecasts() if row["target_end"] == target)

    def _forecasts(self):
        rows = []
        for target, raw, checksum, available, confirmation in self.db.execute("SELECT * FROM forecasts ORDER BY target"):
            if digest(raw) != checksum:
                raise ValueError("future forecast checksum differs")
            payload = json.loads(raw)
            if (payload["target_end"] != target or payload["target_start"] != target-3600
                    or payload["origin_end"]+payload["horizon"]*3600 != target
                    or payload["deadline"] != target-3600-self.config["lead"]
                    or payload["policy"] != self.config["policy"] or _json(payload) != raw):
                raise ValueError("future forecast alignment differs")
            if available is not None and (confirmation != digest(_json([checksum, available]))
                    or not payload["prepared_at"] <= available <= payload["deadline"]):
                raise ValueError("invalid post-commit availability witness")
            rows.append({**payload, "available_at": available, "eligible": available is not None, "forecast_sha256": checksum})
        return rows

    def report(self):
        self._ready()
        model = self._model()
        forecasts = self._forecasts()
        points = []
        for row in forecasts:
            observation = self.db.execute("SELECT value,seen FROM observations WHERE stamp=?", (row["target_end"],)).fetchone()
            if row["eligible"] and observation is not None:
                points.append({**row, "value": observation[0], "observed_at": observation[1]})
        metrics = {}
        for p in POLICIES:
            errors = [(r["predictions"][p]-r["value"])/self.config["std"] for r in points]
            metrics[p] = {"n": len(points), "nmae": math.fsum(abs(e) for e in errors)/len(errors) if errors else None,
                          "coverage90": sum(r["intervals"][p][0] <= r["value"] <= r["intervals"][p][1] for r in points)/len(points) if points else None}
        first = min((r["available_at"] for r in forecasts if r["eligible"]), default=None)
        return {"source_end": self.config["origin"]+model.index*3600, "model_snapshot_bytes": len(model.dump()),
                "policy": self.config["policy"], "prepared": len(forecasts), "confirmed": sum(r["eligible"] for r in forecasts),
                "scored_future_hours": len(points), "metrics": metrics, "recent_forecasts": forecasts[-12:],
                "observation_started_at": first,
                "elapsed_observation_days": max(0., self.clock()-first)/86400 if first is not None else 0.,
                "decision": "RESEARCH_OBSERVATION_HOLD", "config": self.config}

    def close(self):
        if self.busy or os.getpid() != self.pid or threading.current_thread() is not self.owner:
            raise RuntimeError("busy future service or wrong owner")
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
