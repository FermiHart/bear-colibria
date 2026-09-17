"""Causal risk-controlled blends and horizon/time-conditioned calibration."""
import copy
import hashlib
import json
import math

from lck.direct_electricity import DirectElectricity, HORIZONS, POLICIES as BASE_POLICIES
from lck.electricity_forecast import radius
from lck.online_reference import _integer, _scalar, _json

MAGIC = b"BEARRK01"
CAP = 3*1024**2
POINTS = ("blend25", "blend50", "blend75", "guard")
CALIBRATIONS = ("rolling", "scaled", "utc_bucket")
POLICIES = BASE_POLICIES+POINTS
WEIGHTS = {"weekly": 0., "blend25": .25, "blend50": .5, "blend75": .75, "weekly_ridge": 1.}
FLOOR = 1e-4


def pack(base, aux):
    prefix = MAGIC+_json({"base": base.dump().hex(), "aux": aux})
    if len(prefix)+32 > CAP:
        raise ValueError("risk snapshot exceeds limit")
    return prefix+hashlib.sha256(prefix).digest()


def guard_weight(histories):
    """Select with completed past24-point blocks; unknown future losses never enter."""
    baseline = histories["weekly"][-168:]
    eligible = []
    for name, weight in WEIGHTS.items():
        errors = histories[name][-168:]
        if len(errors) != len(baseline):
            raise ValueError("risk histories have different clocks")
        good = True
        for offset in range(0, len(errors)-23, 24):
            stop = len(errors)-offset
            a, b = errors[stop-24:stop], baseline[stop-24:stop]
            if math.fsum(a) > 1.1*math.fsum(b):
                good = False
                break
        if good:
            eligible.append((math.fsum(errors)/len(errors) if errors else 0., weight))
    return min(eligible)[1]


class RiskElectricity:
    def __init__(self, mean, std, *, origin_hour=0, calibration="rolling"):
        _integer(origin_hour, 0, 23)
        if calibration not in CALIBRATIONS:
            raise ValueError("unknown calibration")
        self.base = DirectElectricity(mean, std)
        self.aux = {"origin_hour": origin_hour, "calibration": calibration,
                    "errors": {str(h): {p: [] for p in POINTS} for h in HORIZONS},
                    "scaled": {str(h): {p: [] for p in POINTS} for h in HORIZONS},
                    "scales": {str(h): {p: 1. for p in POINTS} for h in HORIZONS},
                    "buckets": {str(h): {p: {str(b): [] for b in range(4)} for p in POINTS} for h in HORIZONS},
                    "pending": {}, "latest": {}}

    @property
    def index(self):
        return self.base.index

    @property
    def mean(self):
        return self.base._state["mean"]

    @property
    def std(self):
        return self.base._state["std"]

    @property
    def latest(self):
        return copy.deepcopy(self.aux["latest"])

    @staticmethod
    def _forecast(base, aux, h):
        key = str(h)
        original = base.latest[key]
        predictions = dict(original["predictions"])
        weekly, ridge = predictions["weekly"], predictions["weekly_ridge"]
        for p in POINTS[:3]:
            predictions[p] = (1-WEIGHTS[p])*weekly+WEIGHTS[p]*ridge
        histories = {p: base._state["errors"][key][p] if p in ("weekly", "weekly_ridge") else aux["errors"][key][p]
                     for p in WEIGHTS}
        weight = guard_weight(histories)
        predictions["guard"] = (1-weight)*weekly+weight*ridge
        bucket = str(((aux["origin_hour"]+original["target"]) % 24)//6)
        intervals, all_intervals = dict(original["intervals"]), {}
        scales = dict(aux["scales"][key])
        for p in POINTS:
            rolling = radius(aux["errors"][key][p])
            q = radius(aux["scaled"][key][p])
            scaled = None if q is None else scales[p]*q
            local = radius(aux["buckets"][key][p][bucket])
            if local is None:
                local = rolling
            for cal, r in (("rolling", rolling), ("scaled", scaled), ("utc_bucket", local)):
                bounds = None if r is None else [max(0., predictions[p]-base._state["std"]*r),
                                                predictions[p]+base._state["std"]*r]
                all_intervals[p+":"+cal] = bounds
                if cal == aux["calibration"]:
                    intervals[p] = bounds
        return {"origin": original["origin"], "target": original["target"], "horizon": h,
                "predictions": predictions, "intervals": intervals, "calibration_intervals": all_intervals,
                "scales": scales, "bucket": bucket, "guard_weight": weight}

    def ingest(self, index, value):
        # DirectElectricity publishes immutable replacement state; shallow copy isolates it.
        base, aux = copy.copy(self.base), copy.deepcopy(self.aux)
        value = _scalar(value, 1e12)
        base.ingest(index, value)
        matured = aux["pending"].pop(str(index), [])
        for old in matured:
            h = str(old["horizon"])
            for p in POINTS:
                error = abs(old["predictions"][p]-value)/self.std
                aux["errors"][h][p] = (aux["errors"][h][p]+[error])[-256:]
                aux["scaled"][h][p] = (aux["scaled"][h][p]+[error/old["scales"][p]])[-256:]
                bucket = old["bucket"]
                aux["buckets"][h][p][bucket] = (aux["buckets"][h][p][bucket]+[error])[-128:]
                aux["scales"][h][p] = max(FLOOR, (31*aux["scales"][h][p]+error)/32)
        if index >= 167:
            aux["latest"] = {str(h): self._forecast(base, aux, h) for h in HORIZONS}
            for forecast in aux["latest"].values():
                aux["pending"].setdefault(str(forecast["target"]), []).append(forecast)
        pack(base, aux)
        self.base, self.aux = base, aux
        return matured

    def dump(self):
        return pack(self.base, self.aux)

    @classmethod
    def load(cls, raw):
        if (type(raw) is not bytes or not 40 < len(raw) <= CAP or raw[:8] != MAGIC
                or hashlib.sha256(raw[:-32]).digest() != raw[-32:]):
            raise ValueError("invalid risk snapshot")
        try:
            state = json.loads(raw[8:-32])
            if set(state) != {"base", "aux"}:
                raise ValueError("invalid risk envelope")
            base, aux = DirectElectricity.load(bytes.fromhex(state["base"])), state["aux"]
            if set(aux) != {"origin_hour", "calibration", "errors", "scaled", "scales", "buckets", "pending", "latest"}:
                raise ValueError("invalid risk schema")
            model = cls(base._state["mean"], base._state["std"], origin_hour=aux["origin_hour"], calibration=aux["calibration"])
            keys = {str(h) for h in HORIZONS}
            for name in ("errors", "scaled", "scales", "buckets"):
                if set(aux[name]) != keys or any(set(v) != set(POINTS) for v in aux[name].values()):
                    raise ValueError("invalid risk horizon/policy set")
            for h in HORIZONS:
                count = max(0, base.index-166-h)
                for p in POINTS:
                    for name in ("errors", "scaled"):
                        row = aux[name][str(h)][p]
                        if len(row) != min(256, count) or any(_scalar(v, 1e20) < 0 for v in row):
                            raise ValueError("invalid risk error queue")
                    if _scalar(aux["scales"][str(h)][p], 1e12) < FLOOR:
                        raise ValueError("invalid risk scale")
                    buckets = aux["buckets"][str(h)][p]
                    if set(buckets) != {"0", "1", "2", "3"}:
                        raise ValueError("invalid calibration buckets")
                    for b, row in buckets.items():
                        # Complete24-target cycles contribute6 labels per bucket.
                        cycles, extra = divmod(count, 24)
                        first_target = 167+h
                        expected = cycles*6+sum(((aux["origin_hour"]+first_target+i) % 24)//6 == int(b) for i in range(extra))
                        if len(row) != min(128, expected) or any(_scalar(v, 1e20) < 0 for v in row):
                            raise ValueError("invalid target-hour calibration clock")
            original = {(f["origin"], f["horizon"]): f for rows in base._state["pending"].values() for f in rows}
            found = []
            for key, rows in aux["pending"].items():
                if not rows:
                    raise ValueError("empty risk pending target")
                for f in rows:
                    pair = (f["origin"], f["horizon"])
                    if pair not in original or key != str(f["target"]) or f["target"] != f["origin"]+f["horizon"]:
                        raise ValueError("risk/base pending clocks differ")
                    found.append(pair)
                    if set(f) != {"origin", "target", "horizon", "predictions", "intervals", "calibration_intervals", "scales", "bucket", "guard_weight"}:
                        raise ValueError("invalid risk forecast schema")
                    if set(f["predictions"]) != set(POLICIES) or set(f["intervals"]) != set(POLICIES):
                        raise ValueError("invalid risk forecasts")
                    if any(f["predictions"][p] != original[pair]["predictions"][p] for p in BASE_POLICIES):
                        raise ValueError("risk primitive forecast changed")
                    if any(f["intervals"][p] != original[pair]["intervals"][p] for p in BASE_POLICIES):
                        raise ValueError("risk primitive interval changed")
                    if f["bucket"] != str(((aux["origin_hour"]+f["target"]) % 24)//6) or set(f["scales"]) != set(POINTS):
                        raise ValueError("risk forecast bucket/scale differs")
                    weight = _scalar(f["guard_weight"], 1.)
                    if weight not in WEIGHTS.values():
                        raise ValueError("invalid convex guard weight")
                    for p in POINTS:
                        w = weight if p == "guard" else WEIGHTS[p]
                        expected = (1-w)*f["predictions"]["weekly"]+w*f["predictions"]["weekly_ridge"]
                        if f["predictions"][p] != expected or _scalar(f["scales"][p], 1e12) < FLOOR:
                            raise ValueError("risk blend or forecast-time scale differs")
                    if set(f["calibration_intervals"]) != {p+":"+c for p in POINTS for c in CALIBRATIONS}:
                        raise ValueError("invalid calibration alternatives")
                    if any(f["intervals"][p] != f["calibration_intervals"][p+":"+aux["calibration"]] for p in POINTS):
                        raise ValueError("selected calibration differs")
                    for bounds in list(f["intervals"].values())+list(f["calibration_intervals"].values()):
                        if bounds is not None:
                            if len(bounds) != 2 or not 0 <= _scalar(bounds[0], 1e15) <= _scalar(bounds[1], 1e15):
                                raise ValueError("invalid risk interval")
            if sorted(found) != sorted(original):
                raise ValueError("risk pending membership differs")
            if base.index >= 167:
                latest = {str(h): cls._forecast(base, aux, h) for h in HORIZONS}
                if _json(latest) != _json(aux["latest"]):
                    raise ValueError("latest risk forecast differs")
                for h in HORIZONS:
                    matching = [f for f in aux["pending"][str(base.index+h)] if f["origin"] == base.index and f["horizon"] == h]
                    if len(matching) != 1 or _json(matching[0]) != _json(latest[str(h)]):
                        raise ValueError("latest risk forecast missing from pending")
            elif aux["latest"]:
                raise ValueError("premature risk forecast")
            model.base, model.aux = base, aux
            if model.dump() != raw:
                raise ValueError("noncanonical risk state")
            return model
        except (TypeError, KeyError, IndexError, UnicodeError, RecursionError) as exc:
            raise ValueError("invalid risk state") from exc
