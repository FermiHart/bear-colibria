"""Separately identified risk-model service using the frozen future protocol.

Reuses the tested one-writer SQLite, two-commit availability protocol and delayed
observation handling. Existing v1 databases and identities are never migrated.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from lck.future_electricity import FutureElectricity, identity as parent_identity, digest, sync_dir, future_target
from lck.risk_electricity import RiskElectricity, POINTS, CALIBRATIONS
from lck.online_reference import _json, _integer


def identity():
    root = Path(__file__).resolve().parent
    return digest(_json({"kind": "risk-future-electricity-1", "parent": parent_identity(),
                        "sources": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                    for name in ("risk_electricity.py", "risk_future_electricity.py")}}))


class RiskFutureElectricity(FutureElectricity):
    @classmethod
    def create(cls, root, records, mean, std, policy, *, calibration, source_config_hash, protocol_hash,
               lead=300, clock=time.time):
        if policy not in POINTS or calibration not in CALIBRATIONS or len(records) < 480:
            raise ValueError("selected risk policy/calibration and480+ seed readings required")
        _integer(lead, 300, 3600)
        origin = records[0][0]
        if type(origin) is not int or origin % 3600:
            raise ValueError("UTC hourly seed required")
        model = RiskElectricity(mean, std, origin_hour=origin//3600 % 24, calibration=calibration)
        for index, (stamp, value) in enumerate(records):
            if type(stamp) is not int or stamp != origin+3600*index or stamp > int(clock()):
                raise ValueError("seed must be contiguous and in the past")
            model.ingest(index, value)
        config = {"policy": policy, "calibration": calibration, "source_config_hash": source_config_hash,
                  "protocol_hash": protocol_hash, "origin": origin, "mean": mean, "std": std,
                  "lead": lead, "seed_count": len(records)}
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
                value = float(value)
                service.db.execute("INSERT INTO observations VALUES (?,?,?,?,?)", (stamp, value, seen, 1,
                                   digest(_json([stamp, value, seen, 1]))))
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
            raise ValueError("existing risk future database required")
        service = cls(root, clock=clock)
        try:
            if service.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("risk database integrity failure")
            meta = dict(service.db.execute("SELECT key,value FROM meta"))
            if set(meta) != {"identity", "config", "config_sha256"} or meta["identity"] != identity():
                raise ValueError("risk future runtime identity differs")
            if digest(meta["config"].encode()) != meta["config_sha256"]:
                raise ValueError("risk configuration checksum differs")
            service.config = json.loads(meta["config"])
            if service.config["policy"] not in POINTS or service.config["calibration"] not in CALIBRATIONS:
                raise ValueError("invalid risk deployment choice")
            if source_config_hash is not None and source_config_hash != service.config["source_config_hash"]:
                raise ValueError("risk source configuration differs")
            model = service._model()
            if min_source_end is not None and service.config["origin"]+model.index*3600 < min_source_end:
                raise ValueError("risk service older than host receipt")
            for stamp, value, seen, seed, checksum in service.db.execute("SELECT * FROM observations"):
                if digest(_json([stamp, value, seen, seed])) != checksum:
                    raise ValueError("risk observation checksum differs")
            service._forecasts()
            return service
        except BaseException:
            service.close()
            raise

    def _model(self):
        raw = self.db.execute("SELECT raw FROM model WHERE slot=0").fetchone()[0]
        model = RiskElectricity.load(raw)
        if ((model.mean, model.std) != (self.config["mean"], self.config["std"])
                or model.aux["origin_hour"] != self.config["origin"]//3600 % 24
                or model.aux["calibration"] != self.config["calibration"]):
            raise ValueError("risk model configuration differs")
        count, first, last = self.db.execute("SELECT count(*),min(stamp),max(stamp) FROM observations").fetchone()
        if count != model.index+1 or first != self.config["origin"] or last != first+model.index*3600:
            raise ValueError("risk observation/model clocks differ")
        return model

    def prepare(self):
        self._ready()
        model = self._model()
        now = int(self.clock())
        origin_end = self.config["origin"]+model.index*3600
        end, h = future_target(now, origin_end, self.config["lead"])
        if self.db.execute("SELECT 1 FROM forecasts WHERE target=?", (end,)).fetchone():
            return end
        if self.db.execute("SELECT count(*) FROM forecasts").fetchone()[0] >= 10000:
            raise ValueError("risk forecast archive limit")
        forecast = model.latest[str(h)]
        if any(b is None for b in forecast["intervals"].values()):
            raise ValueError("risk intervals not calibrated")
        payload = {"target_end": end, "target_start": end-3600, "origin_end": origin_end, "horizon": h,
                   "prepared_at": now, "deadline": end-3600-self.config["lead"], "model_sha256": digest(model.dump()),
                   "policy": self.config["policy"], "calibration": self.config["calibration"],
                   "predictions": forecast["predictions"], "intervals": forecast["intervals"],
                   "calibration_intervals": forecast["calibration_intervals"]}
        raw = _json(payload)
        self._transaction([("INSERT INTO forecasts VALUES (?,?,?,?,?)", (end, raw, digest(raw), None, None))], "after_prepare_commit")
        return end

    def _forecasts(self):
        rows = super()._forecasts()
        for row in rows:
            if (row["calibration"] != self.config["calibration"]
                    or set(row["calibration_intervals"]) != {p+":"+c for p in POINTS for c in CALIBRATIONS}
                    or any(row["intervals"][p] != row["calibration_intervals"][p+":"+self.config["calibration"]] for p in POINTS)):
                raise ValueError("risk forecast calibration differs")
        return rows

    def report(self):
        report = super().report()
        points = []
        for forecast in self._forecasts():
            observed = self.db.execute("SELECT value FROM observations WHERE stamp=?", (forecast["target_end"],)).fetchone()
            if forecast["eligible"] and observed is not None:
                points.append((forecast, observed[0]))
        for p in POINTS:
            errors = [abs(f["predictions"][p]-y)/self.config["std"] for f, y in points]
            report["metrics"][p] = {"n": len(points), "nmae": math.fsum(errors)/len(errors) if errors else None,
                "coverage90": sum(f["intervals"][p][0] <= y <= f["intervals"][p][1] for f, y in points)/len(points) if points else None}
        report["calibration"] = self.config["calibration"]
        report["decision"] = "RISK_RESEARCH_OBSERVATION_HOLD"
        return report
