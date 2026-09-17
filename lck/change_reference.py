"""Bounded Gaussian regression mixture with fresh/retained model probability.

Own stdlib implementation of known Bayesian mechanisms, not exact unpruned BOCPD.
Predict before labels; one owner; immutable candidate state and RAM snapshots.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math

from lck.online_reference import _scalar, _integer, _json, MAX_SEQUENCE

MAGIC = b"BEARCP01"
NOISE = .03**2
PRIOR = (NOISE, 0., NOISE/.2**2, 0., 0.)  # precision00/01/11, rhs0/1
HAZARD = 1/64
CAPACITY = 32


def regression(state, x):
    a, c, d, b0, b1 = state
    if not all(math.isfinite(v) and abs(v) <= 1e12 for v in state) or a <= 0:
        raise ValueError("invalid regression information")
    l0 = math.sqrt(a)
    cross = c/l0
    pivot = d-cross*cross
    if pivot <= 0 or not math.isfinite(pivot):
        raise ValueError("nonpositive regression pivot")
    l1 = math.sqrt(pivot)
    f0 = b0/l0
    w1 = (b1-cross*f0)/l1/l1
    w0 = (f0-cross*w1)/l0
    if max(abs(w0), abs(w1)) > 1e6:
        raise ValueError("regression weights outside limit")
    u0, u1 = x/l0, (1-cross*(x/l0))/l1
    mean, variance = math.fsum((w0*x, w1)), NOISE*(1+u0*u0+u1*u1)
    if not math.isfinite(mean) or not 0 < variance <= 1e12:
        raise ValueError("invalid predictive distribution")
    return mean, variance


def update(state, x, y):
    a, c, d, b0, b1 = state
    result = a+x*x, c+x, d+1, b0+x*y, b1+y
    regression(result, 0.)
    return result


def logsum(values):
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(v-peak) for v in values))


def quantile(means, variances, probabilities, probability):
    sigmas = tuple(math.sqrt(v) for v in variances)
    scales = tuple(1/(s*math.sqrt(2)) for s in sigmas)
    low = min(m-12*s for m,s in zip(means, sigmas))
    high = max(m+12*s for m,s in zip(means, sigmas))
    for _ in range(36):
        middle = (low+high)/2
        cdf = math.fsum(p*.5*math.erfc((m-middle)*scale)
                        for m,scale,p in zip(means, scales, probabilities))
        if cdf < probability:
            low = middle
        else:
            high = middle
    return (low+high)/2


@dataclass(frozen=True)
class Forecast:
    sequence: int
    value: float
    lower: float
    upper: float
    reset_mass: float
    recall_mass: float


class ChangeReference:
    def __init__(self, *, recall=False):
        if type(recall) is not bool:
            raise ValueError("recall must be bool")
        self._recall = recall
        self._state = 0, (), PRIOR if recall else None, None, None

    @property
    def sequence(self):
        return self._state[0]

    def _components(self, x):
        sequence, active, anchor, memory, _ = self._state
        if not active:
            return (("fresh", sequence, PRIOR, 0., *regression(PRIOR, x)),)
        components = [("active", start, state, mass+math.log1p(-HAZARD), *regression(state, x))
                      for start,state,mass in active]
        share = .5 if self._recall and sequence >= 32 else 0.
        components.append(("fresh", sequence, PRIOR, math.log(HAZARD*(1-share)), *regression(PRIOR, x)))
        if share:
            log_prior = math.log(HAZARD*share)
            if memory is not None:
                log_prior = logsum((log_prior, memory+math.log1p(-HAZARD)))
            components.append(("memory", 0, anchor, log_prior, *regression(anchor, x)))
        return tuple(components)

    def predict(self, features):
        sequence, active, anchor, memory, pending = self._state
        if pending is not None:
            raise RuntimeError("observe pending forecast")
        if sequence == MAX_SEQUENCE:
            raise ValueError("sequence exhausted")
        if type(features) not in (list, tuple) or len(features) != 4:
            raise ValueError("four prior observations required")
        x = tuple(_scalar(v, 1000.) for v in features)[-1]
        components = self._components(x)
        norm = logsum(tuple(c[3] for c in components))
        probabilities = tuple(math.exp(c[3]-norm) for c in components)
        total = math.fsum(probabilities)
        probabilities = tuple(p/total for p in probabilities)
        means, variances = tuple(c[4] for c in components), tuple(c[5] for c in components)
        value = math.fsum(p*m for p,m in zip(probabilities, means))
        lower = quantile(means, variances, probabilities, .05)
        upper = quantile(means, variances, probabilities, .95)
        reset = math.fsum(p for p,c in zip(probabilities, components) if c[0] == "fresh")
        recall = math.fsum(p for p,c in zip(probabilities, components) if c[0] == "memory")
        if not all(math.isfinite(v) for v in (value, lower, upper)) or lower > upper:
            raise ValueError("invalid mixture prediction")
        result = Forecast(sequence, value, lower, upper, reset, recall)
        self._state = sequence, active, anchor, memory, (x, result)
        return result

    def observe(self, target, *, expected_sequence):
        sequence, active, anchor, memory, pending = self._state
        if type(expected_sequence) is not int or expected_sequence != sequence:
            raise ValueError("stale sequence")
        if pending is None:
            raise RuntimeError("predict before label")
        target = _scalar(target, 1000.)
        x, prediction = pending
        candidates, memory_score = [], None
        for kind,start,state,prior,mean,variance in self._components(x):
            score = prior-.5*(math.log(2*math.pi*variance)+(target-mean)**2/variance)
            if not math.isfinite(score):
                raise ValueError("nonfinite posterior likelihood")
            if kind == "memory":
                memory_score = score
            else:
                candidates.append((start, update(state, x, target), score))
        candidates.sort(key=lambda c: (-c[2], c[0]))
        candidates = candidates[:CAPACITY]
        norm = logsum(tuple(c[2] for c in candidates) + (() if memory_score is None else (memory_score,)))
        active = tuple((start, state, score-norm) for start,state,score in candidates)
        memory = None if memory_score is None else memory_score-norm
        if self._recall and sequence < 32:
            anchor = update(anchor, x, target)
        receipt = {"sequence": sequence+1, "prediction": prediction.value,
                   "error": target-prediction.value, "durable": False}
        self._state = sequence+1, active, anchor, memory, None
        return receipt

    def dump(self):
        sequence, active, anchor, memory, pending = self._state
        prefix = MAGIC + _json({"recall": self._recall, "sequence": sequence, "active": active,
            "anchor": anchor, "memory": memory,
            "pending": None if pending is None else {"x": pending[0], "forecast": asdict(pending[1])}})
        if len(prefix)+32 > 65536:
            raise ValueError("change snapshot exceeds limit")
        return prefix+hashlib.sha256(prefix).digest()

    @classmethod
    def load(cls, data):
        if (type(data) is not bytes or not 40 < len(data) <= 65536 or data[:8] != MAGIC
                or hashlib.sha256(data[:-32]).digest() != data[-32:]):
            raise ValueError("invalid change snapshot")
        try:
            s = json.loads(data[8:-32])
            if _json(s) != data[8:-32] or set(s) != {"recall", "sequence", "active", "anchor", "memory", "pending"}:
                raise ValueError("invalid change schema")
            model = cls(recall=s["recall"])
            seq = s["sequence"]
            _integer(seq, 0, MAX_SEQUENCE)
            active = []
            for start, values, mass in s["active"]:
                _integer(start, 0, seq-1)
                state = tuple(_scalar(v, 1e12) for v in values)
                regression(state, 0.)
                mass = _scalar(mass, 1e20)
                if mass > 1e-12:
                    raise ValueError("invalid posterior mass")
                active.append((start, state, mass))
            if len(active) != min(seq, CAPACITY) or len({a[0] for a in active}) != len(active):
                raise ValueError("invalid active bank size")
            anchor = None if s["anchor"] is None else tuple(_scalar(v, 1e12) for v in s["anchor"])
            memory = None if s["memory"] is None else _scalar(s["memory"], 1e20)
            if (model._recall != (anchor is not None)
                    or (memory is not None) != (model._recall and seq > 32)):
                raise ValueError("invalid recall state")
            if anchor is not None:
                regression(anchor, 0.)
            if seq == 0 and anchor != (PRIOR if model._recall else None):
                raise ValueError("invalid initial anchor")
            if active:
                norm = logsum(tuple(a[2] for a in active)+(() if memory is None else (memory,)))
                if abs(norm) > 1e-10:
                    raise ValueError("unnormalized posterior")
            model._state = seq, tuple(active), anchor, memory, None
            if s["pending"] is not None:
                x = s["pending"]["x"]
                model.predict([0., 0., 0., x])
            if model.dump() != data:
                raise ValueError("noncanonical or inconsistent forecast")
            return model
        except (TypeError, KeyError, UnicodeError, RecursionError) as exc:
            raise ValueError("invalid change state") from exc
