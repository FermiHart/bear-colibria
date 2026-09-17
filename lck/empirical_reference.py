"""Changepoint/recall mixture with empirically-frozen observation noise, stdlib only.

B2's mechanism and constants (hazard 1/64, recall share 1/2, slope/bias prior
covariance diag(1, 0.04)) unchanged; the single difference is the likelihood
noise: nominal 0.03**2 during warmup, then a frozen empirical estimate from
warmup OLS residuals (32 pairs, ML, floored/capped). Fixes B2's overconfident
likelihoods on real scales while staying exactly causal and resumable.
Warmup sufficient sums are versioned in the snapshot. The prior rescaling at
the switch (nominal- vs empirical-scaled stored prior) is a disclosed
approximation; with 32 accumulated observations its relative weight is ~1e-4.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math

from lck.change_reference import PRIOR, CAPACITY, HAZARD, regression as _regression, update, logsum, quantile
from lck.online_reference import _scalar, _integer, _json, MAX_SEQUENCE

MAGIC = b"BEAREM01"
NOMINAL = .03**2
PRIOR_PREC = (1., 25.)
NOISE_FLOOR = 1e-6
NOISE_CAP = 1e4


def warmup_noise_2d(n, sx, sy, sxx, sxy, syy):
    """ML residual variance of the warmup OLS fit, or the marginal target
    variance when the warmup design is degenerate (flat segments are valid
    real data, not corruption). Shared by observe-freeze and load-verify so
    both compute bit-identical values from the same stored sums."""
    det = n*sxx-sx*sx
    if det > 0:
        slope = (n*sxy-sx*sy)/det
        bias = (sy-slope*sx)/n
        var = (syy-slope*sxy-bias*sy)/n
    elif n > 0:
        var = (syy-sy*sy/n)/n
    else:
        var = NOISE_FLOOR
    return min(max(var, NOISE_FLOOR), NOISE_CAP)


def warmup_noise_3d(n, s1, sS, sy, s11, s1S, sSS, sy1, syS, syy):
    det = (s11*(sSS*n-sS*sS)-s1S*(s1S*n-sS*s1)+s1*(s1S*sS-sSS*s1))
    if det > 0:
        inv = [[sSS*n-sS*sS, sS*s1-s1S*n, s1S*sS-sSS*s1],
               [sS*s1-s1S*n, s11*n-s1*s1, s1*s1S-s11*sS],
               [s1S*sS-sSS*s1, s1*s1S-s11*sS, s11*sSS-s1S*s1S]]
        weights = tuple(sum(inv[i][j]*rhs for j, rhs in enumerate((sy1, syS, sy)))/det for i in range(3))
        var = (syy-weights[0]*sy1-weights[1]*syS-weights[2]*sy)/n
    elif n > 0:
        var = (syy-sy*sy/n)/n
    else:
        var = NOISE_FLOOR
    return min(max(var, NOISE_FLOOR), NOISE_CAP)


def regression(state, x, noise):
    a, c, d, b0, b1 = state
    if not all(math.isfinite(v) and abs(v) <= 1e12 for v in state) or a <= 0:
        raise ValueError("invalid regression information")
    # Scale-free floor: stored sums are raw and prior prefixes (nominal or
    # empirical, whichever built each part) never decrease. Either scale is
    # legitimate, so the floor is the smaller of the two.
    scale = noise/NOMINAL
    if a < min(PRIOR[0], PRIOR[0]*scale) or d < min(PRIOR[2], PRIOR[2]*scale):
        raise ValueError("information below prior")
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
    mean, variance = math.fsum((w0*x, w1)), noise*(1+u0*u0+u1*u1)
    if not math.isfinite(mean) or not 0 < variance <= 1e12:
        raise ValueError("invalid predictive distribution")
    return mean, variance


@dataclass(frozen=True)
class Forecast:
    sequence: int
    value: float
    lower: float
    upper: float
    reset_mass: float
    recall_mass: float


class EmpiricalReference:
    def __init__(self, *, recall=False):
        if type(recall) is not bool:
            raise ValueError("recall must be bool")
        self._recall = recall
        self._state = (0, (), PRIOR if recall else None, None, None, None, None)

    @property
    def sequence(self):
        return self._state[0]

    @property
    def noise(self):
        frozen = self._state[4]
        return NOMINAL if frozen is None else frozen

    def _prior(self):
        scale = self.noise/NOMINAL
        return (PRIOR[0]*scale, 0., PRIOR[2]*scale, 0., 0.)

    def _components(self, x):
        sequence, active, anchor, memory, noise_hat, _, _ = self._state
        noise = NOMINAL if noise_hat is None else noise_hat
        prior = self._prior()
        if not active:
            return (("fresh", sequence, prior, 0., *regression(prior, x, noise)),)
        components = [("active", start, state, mass+math.log1p(-HAZARD), *regression(state, x, noise))
                      for start, state, mass in active]
        share = .5 if self._recall and sequence >= 32 else 0.
        components.append(("fresh", sequence, prior, math.log(HAZARD*(1-share)), *regression(prior, x, noise)))
        if share:
            log_prior = math.log(HAZARD*share)
            if memory is not None:
                log_prior = logsum((log_prior, memory+math.log1p(-HAZARD)))
            components.append(("memory", 0, anchor, log_prior, *regression(anchor, x, noise)))
        return tuple(components)

    def predict(self, features):
        sequence, active, anchor, memory, noise_hat, sums, pending = self._state
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
        value = math.fsum(p*m for p, m in zip(probabilities, means))
        lower = quantile(means, variances, probabilities, .05)
        upper = quantile(means, variances, probabilities, .95)
        reset = math.fsum(p for p, c in zip(probabilities, components) if c[0] == "fresh")
        recall = math.fsum(p for p, c in zip(probabilities, components) if c[0] == "memory")
        if not all(math.isfinite(v) for v in (value, lower, upper)) or lower > upper:
            raise ValueError("invalid mixture prediction")
        result = Forecast(sequence, value, lower, upper, reset, recall)
        self._state = sequence, active, anchor, memory, noise_hat, sums, (x, result)
        return result

    def observe(self, target, *, expected_sequence):
        sequence, active, anchor, memory, noise_hat, sums, pending = self._state
        if type(expected_sequence) is not int or expected_sequence != sequence:
            raise ValueError("stale sequence")
        if pending is None:
            raise RuntimeError("predict before label")
        target = _scalar(target, 1000.)
        x, prediction = pending
        candidates, memory_score = [], None
        for kind, start, state, prior, mean, variance in self._components(x):
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
        active = tuple((start, state, score-norm) for start, state, score in candidates)
        memory = None if memory_score is None else memory_score-norm
        if self._recall and sequence < 32:
            anchor = update(anchor, x, target)
        if sequence < 32:
            n, sx, sy, sxx, sxy, syy = sums if sums is not None else (0, 0., 0., 0., 0., 0.)
            sums = n+1, sx+x, sy+target, sxx+x*x, sxy+x*target, syy+target*target
        if sequence == 31:
            n, sx, sy, sxx, sxy, syy = sums
            noise_hat = _scalar(warmup_noise_2d(n, sx, sy, sxx, sxy, syy), NOISE_CAP, positive=True)
        receipt = {"sequence": sequence+1, "prediction": prediction.value,
                   "error": target-prediction.value, "durable": False}
        self._state = sequence+1, active, anchor, memory, noise_hat, sums, None
        return receipt

    def dump(self):
        sequence, active, anchor, memory, noise_hat, sums, pending = self._state
        prefix = MAGIC + _json({"recall": self._recall, "sequence": sequence, "active": active,
            "anchor": anchor, "memory": memory, "noise_hat": noise_hat, "sums": sums,
            "pending": None if pending is None else {"x": pending[0], "forecast": asdict(pending[1])}})
        if len(prefix)+32 > 65536:
            raise ValueError("empirical snapshot exceeds limit")
        return prefix+hashlib.sha256(prefix).digest()

    @classmethod
    def load(cls, data):
        if (type(data) is not bytes or not 40 < len(data) <= 65536 or data[:8] != MAGIC
                or hashlib.sha256(data[:-32]).digest() != data[-32:]):
            raise ValueError("invalid empirical snapshot")
        try:
            s = json.loads(data[8:-32])
            if (_json(s) != data[8:-32] or set(s) != {"recall", "sequence", "active", "anchor", "memory",
                    "noise_hat", "sums", "pending"}):
                raise ValueError("invalid empirical schema")
            model = cls(recall=s["recall"])
            seq = s["sequence"]
            _integer(seq, 0, MAX_SEQUENCE)
            noise_hat = None if s["noise_hat"] is None else _scalar(s["noise_hat"], NOISE_CAP, positive=True)
            if (noise_hat is None) != (seq <= 31):
                raise ValueError("noise estimate inconsistent with warmup")
            sums = None
            if s["sums"] is not None:
                n, *rest = s["sums"]
                _integer(n, 0, 32)
                sums = (n, *tuple(_scalar(v, 1e12) for v in rest))
                if len(sums) != 6 or n != min(seq, 32):
                    raise ValueError("warmup sums inconsistent")
            elif seq != 0:
                raise ValueError("missing warmup sums")
            if noise_hat is not None:
                # The estimate is re-derived from the stored warmup sums with the
                # exact freeze-time operations: bit-exact equality required.
                n, sx, sy, sxx, sxy, syy = sums
                if warmup_noise_2d(n, sx, sy, sxx, sxy, syy) != noise_hat:
                    raise ValueError("noise estimate inconsistent with warmup sums")
            active = []
            eff = NOMINAL if noise_hat is None else noise_hat
            for start, values, mass in s["active"]:
                _integer(start, 0, seq-1)
                state = tuple(_scalar(v, 1e12) for v in values)
                regression(state, 0., eff)
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
                regression(anchor, 0., eff)
            if seq == 0 and anchor != (PRIOR if model._recall else None):
                raise ValueError("invalid initial anchor")
            if active:
                norm = logsum(tuple(a[2] for a in active)+(() if memory is None else (memory,)))
                if abs(norm) > 1e-10:
                    raise ValueError("unnormalized posterior")
            model._state = seq, tuple(active), anchor, memory, noise_hat, sums, None
            if s["pending"] is not None:
                # The replay below revalidates under the recorded empirical noise
                # before any byte is accepted.
                model.predict([0., 0., 0., s["pending"]["x"]])
            if model.dump() != data:
                raise ValueError("noncanonical or inconsistent forecast")
            return model
        except (TypeError, KeyError, UnicodeError, RecursionError) as exc:
            raise ValueError("invalid empirical state") from exc
