"""Causal hourly electricity forecaster with complete, bounded RAM snapshots."""
from copy import copy
import hashlib
import json
import math

from lck.online_reference import _integer, _scalar, _json
from lck.robust_reference import RobustSeasonal
from lck.seasonal_reference import _solve

MAGIC = b"BEAREL01"
EXPERTS = ("persistence", "seasonal_naive", "ar1", "seasonal_ridge", "robust")
POLICIES = EXPERTS + ("selector", "baseline_selector")
PRIOR = (1., 0., 0., 1., 0., 1.)
CAP = 262144


def fit_update(precision, rhs, z, y, decay):
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    a = tuple(decay*v+(1-decay)*prior+z[i]*z[j]
              for v, prior, (i, j) in zip(precision, PRIOR, pairs))
    b = tuple(decay*v+x*y for v, x in zip(rhs, z))
    return a, b


def ridge_predict(precision, rhs, z):
    a, b, _ = _solve(precision, rhs)
    scale = max(1., (abs(a)+abs(b))/.99)
    a, b = a/scale, b/scale
    bias = (rhs[2]-precision[2]*a-precision[4]*b)/precision[5]
    return a*z[0]+b*z[1]+bias


def radius(errors):
    if len(errors) < 64:
        return None
    rank = min(len(errors), math.ceil(.9*(len(errors)+1)))
    return sorted(errors)[rank-1]


class ElectricityForecast:
    def __init__(self, mean, std, *, policy="selector"):
        mean, std = _scalar(mean, 1e12), _scalar(std, 1e12, positive=True)
        if mean < 0 or std <= 1e-6 or policy not in POLICIES:
            raise ValueError("invalid electricity configuration")
        self.config = (mean, std, policy)
        # index, history, seasonal stats, AR1 stats, robust, expert losses, residuals, pending
        self._state = (-1, (), (PRIOR, (0., 0., 0.)), (PRIOR, (0., 0., 0.)),
                       RobustSeasonal(24, recall=True), tuple(() for _ in EXPERTS),
                       tuple(() for _ in POLICIES), None)

    @property
    def index(self):
        return self._state[0]

    @property
    def forecast(self):
        pending = self._state[-1]
        return None if pending is None else json.loads(_json(pending))

    def _predict(self, index, history, seasonal, ar1, robust, losses, residuals):
        mean, std, policy = self.config
        z = (history[-1], history[0], 1.)
        p = robust.predict(list(history)+[0.])
        raw = (history[-1], history[0], ridge_predict(*ar1, (z[0], 0., 1.)),
               ridge_predict(*seasonal, z), p.value)
        normalized = {name: max(-mean/std, value) for name, value in zip(EXPERTS, raw)}
        costs = [math.fsum(v)/len(v) if v else 0. for v in losses]
        choice = min(range(5), key=lambda i: (costs[i], i))
        baseline = min(range(3), key=lambda i: (costs[i], i))
        normalized["selector"] = normalized[EXPERTS[choice]]
        normalized["baseline_selector"] = normalized[EXPERTS[baseline]]
        predictions = {name: max(0., mean+std*v) for name, v in normalized.items()}
        intervals = {}
        for name, errors in zip(POLICIES, residuals):
            width = radius(errors)
            intervals[name] = None if width is None else [max(0., predictions[name]-std*width),
                                                         predictions[name]+std*width]
        return {"target_index": index+1, "prediction": predictions[policy], "interval90": intervals[policy],
                "policy": policy, "expert": EXPERTS[choice] if policy == "selector" else policy,
                "predictions": predictions, "intervals": intervals}

    def ingest(self, index, value):
        if type(index) is not int or index != self.index+1 or index >= 2**53:
            raise ValueError("expected the next contiguous index")
        value = _scalar(value, 1e12)
        mean, std, policy = self.config
        if value < 0:
            raise ValueError("negative electricity reading")
        y = _scalar((value-mean)/std, 1000.)
        _, history, seasonal, ar1, robust, losses, residuals, pending = self._state
        robust = copy(robust)
        if pending is not None:
            losses = tuple((old+(((pending["predictions"][name]-value)/std)**2,))[-168:]
                           for name, old in zip(EXPERTS, losses))
            residuals = tuple((old+(abs(pending["predictions"][name]-value)/std,))[-256:]
                              for name, old in zip(POLICIES, residuals))
            z = (history[-1], history[0], 1.)
            seasonal = fit_update(*seasonal, z, y, .995)
            ar1 = fit_update(*ar1, (z[0], 0., 1.), y, 1.)
            robust.observe(y, expected_sequence=robust.sequence)
        history = (history+(y,))[-24:]
        next_forecast = None if len(history) < 24 else self._predict(
            index, history, seasonal, ar1, robust, losses, residuals)
        state = index, history, seasonal, ar1, robust, losses, residuals, next_forecast
        # Construct output before publication; callers never receive live state.
        result = {"index": index, "forecast": None if next_forecast is None else json.loads(_json(next_forecast)),
                  "durable": False}
        self._state = state
        return result

    def dump(self):
        index, history, seasonal, ar1, robust, losses, residuals, pending = self._state
        body = {"config": self.config, "index": index, "history": history, "seasonal": seasonal,
                "ar1": ar1, "robust": robust.dump().hex(), "losses": losses,
                "residuals": residuals, "pending": pending}
        prefix = MAGIC+_json(body)
        if len(prefix)+32 > CAP:
            raise ValueError("electricity snapshot exceeds limit")
        return prefix+hashlib.sha256(prefix).digest()

    @classmethod
    def load(cls, data):
        if (type(data) is not bytes or not 40 < len(data) <= CAP or data[:8] != MAGIC
                or hashlib.sha256(data[:-32]).digest() != data[-32:]):
            raise ValueError("invalid electricity snapshot")
        try:
            s = json.loads(data[8:-32])
            if set(s) != {"config", "index", "history", "seasonal", "ar1", "robust", "losses", "residuals", "pending"}:
                raise ValueError("invalid electricity schema")
            mean, std, policy = s["config"]
            model = cls(mean, std, policy=policy)
            index = _integer(s["index"], -1, 2**53-1)
            # _integer validates but returns no value.
            index = s["index"]
            history = tuple(_scalar(v, 1000.) for v in s["history"])
            if len(history) != min(index+1, 24):
                raise ValueError("invalid electricity history")
            def pair(raw):
                a, b = (tuple(_scalar(v, 1e12) for v in part) for part in raw)
                if len(a) != 6 or len(b) != 3:
                    raise ValueError("invalid electricity statistics")
                _solve(a, b)
                return a, b
            seasonal, ar1 = pair(s["seasonal"]), pair(s["ar1"])
            robust = RobustSeasonal.load(bytes.fromhex(s["robust"]), season=24)
            count = max(0, index-23)
            if robust.sequence != count or not robust._recall:
                raise ValueError("electricity model clocks differ")
            def queues(raw, names, cap):
                if len(raw) != len(names):
                    raise ValueError("invalid electricity queue count")
                result = tuple(tuple(_scalar(v, 1e20) for v in row) for row in raw)
                if any(len(row) != min(count, cap) or any(v < 0 for v in row) for row in result):
                    raise ValueError("invalid electricity queue")
                return result
            losses, residuals = queues(s["losses"], EXPERTS, 168), queues(s["residuals"], POLICIES, 256)
            pending = s["pending"]
            if (pending is not None) != (index >= 23) or (robust._state[-1] is not None) != (index >= 23):
                raise ValueError("electricity pending mismatch")
            if pending is not None:
                candidate = copy(robust)
                candidate._state = candidate._state[:-1]+(None,)
                rebuilt = model._predict(index, history, seasonal, ar1, candidate, losses, residuals)
                if rebuilt != pending or candidate.dump() != robust.dump():
                    raise ValueError("electricity forecast mismatch")
            model._state = index, history, seasonal, ar1, robust, losses, residuals, pending
            if model.dump() != data:
                raise ValueError("noncanonical electricity snapshot")
            return model
        except (TypeError, KeyError, IndexError, UnicodeError, RecursionError) as exc:
            raise ValueError("invalid electricity state") from exc
