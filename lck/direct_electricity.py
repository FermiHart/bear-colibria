"""Direct H1..6 electricity fits with delayed labels and causal calibration.

Established regression/online-selection methods, implemented with stdlib. Every
fit sees only matured targets; no model recursively consumes its own forecast.
"""
import copy
import hashlib
import json
import math

from lck.electricity_forecast import PRIOR, fit_update, ridge_predict, radius
from lck.online_reference import _scalar, _integer, _json
from lck.seasonal_reference import _solve

MAGIC = b"BEARDE01"
HORIZONS = tuple(range(1, 7))
EXPERTS = ("persistence", "daily", "weekly", "adjusted", "ar1", "daily_ridge", "weekly_ridge")
POLICIES = EXPERTS+("baseline_selector", "selector")
CAP = 1024**2


def pack(state):
    prefix = MAGIC+_json(state)
    if len(prefix)+32 > CAP:
        raise ValueError("direct snapshot exceeds limit")
    return prefix+hashlib.sha256(prefix).digest()


class DirectElectricity:
    def __init__(self, mean, std):
        mean, std = _scalar(mean, 1e12), _scalar(std, 1e12, positive=True)
        if mean < 0 or std <= 1e-6:
            raise ValueError("invalid direct normalizer")
        self._state = {"mean": mean, "std": std, "index": -1, "history": [], "pending": {}, "latest": {},
            "fits": {str(h): {p: [list(PRIOR), [0., 0., 0.]] for p in ("ar1", "daily_ridge", "weekly_ridge")} for h in HORIZONS},
            "errors": {str(h): {p: [] for p in POLICIES} for h in HORIZONS}}

    @property
    def index(self):
        return self._state["index"]

    @property
    def latest(self):
        return copy.deepcopy(self._state["latest"])

    @staticmethod
    def _forecast(state, h):
        history, index = state["history"], state["index"]
        origin, daily, weekly = history[-1], history[h-25], history[h-169]
        z = {"ar1": (origin, 0., 1.), "daily_ridge": (origin, daily, 1.), "weekly_ridge": (origin, weekly, 1.)}
        raw = {"persistence": origin, "daily": daily, "weekly": weekly,
               "adjusted": daily+.5*(origin-history[-25])}
        raw.update({p: ridge_predict(*state["fits"][str(h)][p], features) for p, features in z.items()})
        errors = state["errors"][str(h)]
        costs = {p: math.fsum(errors[p][-168:])/len(errors[p][-168:]) if errors[p] else 0. for p in EXPERTS}
        control = min(EXPERTS[:5], key=lambda p: (costs[p], EXPERTS.index(p)))
        chosen = min(EXPERTS, key=lambda p: (costs[p], EXPERTS.index(p)))
        raw.update(baseline_selector=raw[control], selector=raw[chosen])
        predictions = {p: max(0., state["mean"]+state["std"]*v) for p, v in raw.items()}
        intervals = {}
        for p in POLICIES:
            r = radius(errors[p])
            intervals[p] = None if r is None else [max(0., predictions[p]-state["std"]*r), predictions[p]+state["std"]*r]
        return {"origin": index, "target": index+h, "horizon": h, "features": z,
                "predictions": predictions, "intervals": intervals, "selected_expert": chosen}

    def ingest(self, index, value):
        if type(index) is not int or index != self.index+1 or index >= 2**53:
            raise ValueError("expected next direct-model index")
        value = _scalar(value, 1e12)
        if value < 0:
            raise ValueError("negative reading")
        y = _scalar((value-self._state["mean"])/self._state["std"], 1000.)
        state = copy.deepcopy(self._state)
        matured = state["pending"].pop(str(index), [])
        for forecast in matured:
            key = str(forecast["horizon"])
            for p in POLICIES:
                error = abs(forecast["predictions"][p]-value)/state["std"]
                state["errors"][key][p] = (state["errors"][key][p]+[error])[-256:]
            for p, features in forecast["features"].items():
                a, b = fit_update(*state["fits"][key][p], features, y, 1. if p == "ar1" else .995)
                state["fits"][key][p] = [list(a), list(b)]
        state["history"] = (state["history"]+[y])[-168:]
        state["index"] = index
        if index >= 167:
            state["latest"] = {str(h): self._forecast(state, h) for h in HORIZONS}
            for forecast in state["latest"].values():
                state["pending"].setdefault(str(forecast["target"]), []).append(forecast)
        raw = pack(state)  # Validate serializability and bounds before publication.
        self._state = state
        return matured

    def dump(self):
        return pack(self._state)

    @classmethod
    def load(cls, raw):
        if (type(raw) is not bytes or not 40 < len(raw) <= CAP or raw[:8] != MAGIC
                or hashlib.sha256(raw[:-32]).digest() != raw[-32:]):
            raise ValueError("invalid direct snapshot")
        try:
            state = json.loads(raw[8:-32])
            if set(state) != {"mean", "std", "index", "history", "pending", "latest", "fits", "errors"}:
                raise ValueError("invalid direct schema")
            model = cls(state["mean"], state["std"])
            index = state["index"]
            _integer(index, -1, 2**53-1)
            if len(state["history"]) != min(index+1, 168):
                raise ValueError("invalid direct history")
            for v in state["history"]:
                _scalar(v, 1000.)
            keys = {str(h) for h in HORIZONS}
            if set(state["fits"]) != keys or set(state["errors"]) != keys:
                raise ValueError("invalid horizon banks")
            for h in HORIZONS:
                fits, errors = state["fits"][str(h)], state["errors"][str(h)]
                if set(fits) != {"ar1", "daily_ridge", "weekly_ridge"} or set(errors) != set(POLICIES):
                    raise ValueError("invalid direct experts")
                for a, b in fits.values():
                    if len(a) != 6 or len(b) != 3:
                        raise ValueError("invalid direct regression dimensions")
                    for v in a+b:
                        _scalar(v, 1e12)
                    _solve(tuple(a), tuple(b))
                n = min(256, max(0, index-166-h))
                for row in errors.values():
                    if len(row) != n or any(_scalar(v, 1e20) < 0 for v in row):
                        raise ValueError("invalid matured error count")
            pairs = []
            for target, forecasts in state["pending"].items():
                if not forecasts:
                    raise ValueError("empty pending target")
                for forecast in forecasts:
                    if set(forecast) != {"origin", "target", "horizon", "features", "predictions", "intervals", "selected_expert"}:
                        raise ValueError("invalid direct forecast schema")
                    h, origin = forecast["horizon"], forecast["origin"]
                    _integer(h, 1, 6)
                    _integer(origin, 167, index)
                    if str(origin+h) != target or forecast["target"] != origin+h or origin+h <= index:
                        raise ValueError("invalid pending alignment")
                    pairs.append((origin, h))
                    if set(forecast["predictions"]) != set(POLICIES) or set(forecast["intervals"]) != set(POLICIES):
                        raise ValueError("invalid pending policies")
                    for p in POLICIES:
                        if _scalar(forecast["predictions"][p], 1e15) < 0:
                            raise ValueError("negative forecast")
                        interval = forecast["intervals"][p]
                        if interval is not None:
                            if len(interval) != 2:
                                raise ValueError("invalid interval")
                            lo, hi = (_scalar(v, 1e15) for v in interval)
                            if not 0 <= lo <= hi:
                                raise ValueError("invalid interval")
                    history = state["history"]
                    x = history[origin-index-1]
                    daily, weekly = history[origin+h-24-index-1], history[origin+h-168-index-1]
                    expected = {"ar1": [x, 0., 1.], "daily_ridge": [x, daily, 1.], "weekly_ridge": [x, weekly, 1.]}
                    if forecast["features"] != expected:
                        raise ValueError("pending features differ from past history")
            expected_pairs = [(origin, h) for h in HORIZONS for origin in range(max(167, index-h+1), index+1)]
            if sorted(pairs) != sorted(expected_pairs):
                raise ValueError("pending forecast membership differs")
            if index >= 167:
                rebuilt = {str(h): cls._forecast(state, h) for h in HORIZONS}
                if _json(rebuilt) != _json(state["latest"]):
                    raise ValueError("latest direct forecast differs")
                for h in HORIZONS:
                    last = [p for p in state["pending"][str(index+h)] if p["origin"] == index and p["horizon"] == h]
                    if len(last) != 1 or _json(last[0]) != _json(rebuilt[str(h)]):
                        raise ValueError("latest forecast absent from pending bank")
            elif state["latest"]:
                raise ValueError("premature direct forecast")
            model._state = state
            if model.dump() != raw:
                raise ValueError("noncanonical direct snapshot")
            return model
        except (TypeError, KeyError, IndexError, UnicodeError, RecursionError) as exc:
            raise ValueError("invalid direct state") from exc
