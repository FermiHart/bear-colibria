"""Seasonal joint changepoint/recall mixture with empirical noise, stdlib only.

AR(3-dim) hypotheses on z=(x[t-1], x[t-S], 1): one joint posterior per segment
start, so seasonal structure is learned per-regime instead of assumed global.
Otherwise B2's exact machinery: hazard 1/64, recall share 1/2, frozen warmup
anchor + frozen warmup noise estimate, keep-32 pruning, mixture quantiles.
Season S is a constructor parameter (1..168), versioned in the snapshot.
Own implementation of established ideas; 3x3 Cholesky written by hand.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math

from lck.change_reference import CAPACITY, HAZARD, logsum, quantile
from lck.empirical_reference import NOMINAL, NOISE_FLOOR, NOISE_CAP
from lck.online_reference import _scalar, _integer, _json, MAX_SEQUENCE

MAGIC = b"BEARSM01"


def _cholesky(precision):
    a00, a01, a02, a11, a12, a22 = precision
    if not all(math.isfinite(v) and abs(v) <= 1e12 for v in precision) or a00 <= 0:
        raise ValueError("invalid seasonal information")
    l00 = math.sqrt(a00)
    l10, l20 = a01/l00, a02/l00
    p11 = a11-l10*l10
    if p11 <= 0 or not math.isfinite(p11):
        raise ValueError("nonpositive seasonal pivot")
    l11 = math.sqrt(p11)
    l21 = (a12-l20*l10)/l11
    p22 = a22-l20*l20-l21*l21
    if p22 <= 0 or not math.isfinite(p22):
        raise ValueError("nonpositive seasonal pivot")
    return (l00, l10, l20, l11, l21, math.sqrt(p22))


def _solve(precision, rhs):
    l00, l10, l20, l11, l21, l22 = _cholesky(precision)
    f0 = rhs[0]/l00
    f1 = (rhs[1]-l10*f0)/l11
    f2 = (rhs[2]-l20*f0-l21*f1)/l22
    w2 = f2/l22
    w1 = (f1-l21*w2)/l11
    w0 = (f0-l10*w1-l20*w2)/l00
    return (w0, w1, w2)


def regression(state, z, noise):
    precision, rhs = state
    if len(precision) != 6 or len(rhs) != 3 or len(z) != 3:
        raise ValueError("invalid seasonal dimensions")
    if not all(math.isfinite(v) and abs(v) <= 1e12 for v in precision+rhs):
        raise ValueError("invalid seasonal information")
    # Either the nominal or the empirical scale may have built each part;
    # the floor is the smaller of the two so legitimate priors always pass.
    scale = noise/NOMINAL
    if (precision[0] < min(NOMINAL, NOMINAL*scale) or precision[3] < min(NOMINAL, NOMINAL*scale)
            or precision[5] < min(NOMINAL/.04, NOMINAL/.04*scale)):
        raise ValueError("seasonal information below prior")
    weights = _solve(precision, rhs)
    if max(abs(w) for w in weights) > 1e6:
        raise ValueError("seasonal weights outside limit")
    lever = _solve(precision, z)
    variance = noise*(1+math.fsum(a*b for a, b in zip(z, lever)))
    mean = math.fsum(w*v for w, v in zip(weights, z))
    if not math.isfinite(mean) or not 0 < variance <= 1e12:
        raise ValueError("invalid seasonal distribution")
    return mean, variance


def update(state, z, y, noise):
    precision, rhs = state
    a00, a01, a02, a11, a12, a22 = precision
    b0, b1, b2 = rhs
    result = ((a00+z[0]*z[0], a01+z[0]*z[1], a02+z[0]*z[2], a11+z[1]*z[1], a12+z[1]*z[2],
               a22+z[2]*z[2]), (b0+z[0]*y, b1+z[1]*y, b2+z[2]*y))
    regression(result, (0., 0., 0.), noise)
    return result


def fresh_prior(noise):
    scale = noise/NOMINAL
    return ((NOMINAL*scale, 0., 0., NOMINAL*scale, 0., NOMINAL/.04*scale), (0., 0., 0.))


@dataclass(frozen=True)
class Forecast:
    sequence: int
    value: float
    lower: float
    upper: float
    reset_mass: float
    recall_mass: float


class SeasonalReference:
    # state: (sequence, active, anchor, memory, noise_hat, sums, season, recall, pending)
    # sums: (n, s1, sS, sy, s11, s1S, sSS, sy1, syS, syy, 0.) or None once frozen.
    def __init__(self, season, *, recall=False):
        _integer(season, 2, 168)
        if type(recall) is not bool:
            raise ValueError("recall must be bool")
        self._season_param = season
        self._recall = recall
        self._state = (0, (), fresh_prior(NOMINAL) if recall else None,
                       None, None, None, season, recall, None)

    @property
    def sequence(self):
        return self._state[0]

    @property
    def noise(self):
        frozen = self._state[4]
        return NOMINAL if frozen is None else frozen

    @property
    def season(self):
        return self._state[6]

    def _components(self, z):
        sequence, active, anchor, memory, noise_hat, _, season, recall, _ = self._state
        noise = NOMINAL if noise_hat is None else noise_hat
        prior = fresh_prior(noise)
        if not active:
            return (("fresh", sequence, prior, 0., *regression(prior, z, noise)),)
        components = [("active", start, state, mass+math.log1p(-HAZARD), *regression(state, z, noise))
                      for start, state, mass in active]
        share = .5 if recall and sequence >= 32 else 0.
        components.append(("fresh", sequence, prior, math.log(HAZARD*(1-share)), *regression(prior, z, noise)))
        if share:
            log_prior = math.log(HAZARD*share)
            if memory is not None:
                log_prior = logsum((log_prior, memory+math.log1p(-HAZARD)))
            components.append(("memory", 0, anchor, log_prior, *regression(anchor, z, noise)))
        return tuple(components)

    def predict(self, features):
        if type(features) is not list or len(features) < self._season_param+1:
            raise ValueError("seasonal window too short")
        sequence, active, anchor, memory, noise_hat, sums, season, recall, pending = self._state
        if season != self._season_param or recall != self._recall:
            raise ValueError("seasonal configuration changed")
        if pending is not None:
            raise RuntimeError("observe pending forecast")
        if sequence == MAX_SEQUENCE:
            raise ValueError("sequence exhausted")
        window = tuple(_scalar(v, 1000.) for v in features)
        z = (window[-2], window[-1-season], 1.)
        components = self._components(z)
        norm = logsum(tuple(c[3] for c in components))
        probabilities = tuple(math.exp(c[3]-norm) for c in components)
        total = math.fsum(probabilities)
        probabilities = tuple(p/total for p in probabilities)
        means, variances = tuple(c[4] for c in components), tuple(c[5] for c in components)
        value = math.fsum(p*m for p, m in zip(probabilities, means))
        lower = quantile(means, variances, probabilities, .05)
        upper = quantile(means, variances, probabilities, .95)
        reset = math.fsum(p for p, c in zip(probabilities, components) if c[0] == "fresh")
        recall_mass = math.fsum(p for p, c in zip(probabilities, components) if c[0] == "memory")
        if not all(math.isfinite(v) for v in (value, lower, upper)) or lower > upper:
            raise ValueError("invalid seasonal forecast")
        result = Forecast(sequence, value, lower, upper, reset, recall_mass)
        self._state = sequence, active, anchor, memory, noise_hat, sums, season, recall, (z, result)
        return result

    def observe(self, target, *, expected_sequence):
        sequence, active, anchor, memory, noise_hat, sums, season, recall, pending = self._state
        if type(expected_sequence) is not int or expected_sequence != sequence:
            raise ValueError("stale sequence")
        if pending is None:
            raise RuntimeError("predict before label")
        target = _scalar(target, 1000.)
        z, prediction = pending
        noise = NOMINAL if noise_hat is None else noise_hat
        candidates, memory_score = [], None
        for kind, start, state, prior, mean, variance in self._components(z):
            score = prior-.5*(math.log(2*math.pi*variance)+(target-mean)**2/variance)
            if not math.isfinite(score):
                raise ValueError("nonfinite seasonal posterior")
            if kind == "memory":
                memory_score = score
            else:
                candidates.append((start, update(state, z, target, noise), score))
        candidates.sort(key=lambda c: (-c[2], c[0]))
        candidates = candidates[:CAPACITY]
        norm = logsum(tuple(c[2] for c in candidates) + (() if memory_score is None else (memory_score,)))
        active = tuple((start, state, score-norm) for start, state, score in candidates)
        memory = None if memory_score is None else memory_score-norm
        if recall and sequence < 32:
            anchor = update(anchor, z, target, noise)
        if sequence < 32:
            if sums is None:
                sums = (0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.)
            n, s1, sS, sy, s11, s1S, sSS, sy1, syS, syy, _ = sums
            sums = (n+1, s1+z[0], sS+z[1], sy+target, s11+z[0]*z[0], s1S+z[0]*z[1],
                    sSS+z[1]*z[1], sy1+z[0]*target, syS+z[1]*target, syy+target*target, 0.)
        if sequence == 31:
            from lck.empirical_reference import warmup_noise_3d
            n, s1, sS, sy, s11, s1S, sSS, sy1, syS, syy, _ = sums
            noise_hat = _scalar(warmup_noise_3d(n, s1, sS, sy, s11, s1S, sSS, sy1, syS, syy),
                                NOISE_CAP, positive=True)
        receipt = {"sequence": sequence+1, "prediction": prediction.value,
                   "error": target-prediction.value, "durable": False}
        self._state = sequence+1, active, anchor, memory, noise_hat, sums, season, recall, None
        return receipt

    def dump(self):
        sequence, active, anchor, memory, noise_hat, sums, season, recall, pending = self._state
        prefix = MAGIC + _json({"recall": recall, "season": season, "sequence": sequence, "active": active,
            "anchor": anchor, "memory": memory, "noise_hat": noise_hat, "sums": sums,
            "pending": None if pending is None else {"z": pending[0], "forecast": asdict(pending[1])}})
        if len(prefix)+32 > 65536:
            raise ValueError("seasonal snapshot exceeds limit")
        return prefix+hashlib.sha256(prefix).digest()

    @classmethod
    def load(cls, data, season=None):
        if (type(data) is not bytes or not 40 < len(data) <= 65536 or data[:8] != MAGIC
                or hashlib.sha256(data[:-32]).digest() != data[-32:]):
            raise ValueError("invalid seasonal snapshot")
        try:
            s = json.loads(data[8:-32])
            if (_json(s) != data[8:-32] or set(s) != {"recall", "season", "sequence", "active", "anchor",
                    "memory", "noise_hat", "sums", "pending"}):
                raise ValueError("invalid seasonal schema")
            _integer(s["season"], 2, 168)
            if season is not None and s["season"] != season:
                raise ValueError("seasonal stream mismatch")
            model = cls(s["season"], recall=s["recall"])
            seq = s["sequence"]
            _integer(seq, 0, MAX_SEQUENCE)
            noise_hat = None if s["noise_hat"] is None else _scalar(s["noise_hat"], NOISE_CAP, positive=True)
            if (noise_hat is None) != (seq <= 31):
                raise ValueError("noise estimate inconsistent with warmup")
            sums = None
            if s["sums"] is not None:
                raw = s["sums"]
                if len(raw) != 11:
                    raise ValueError("seasonal sums dimension mismatch")
                _integer(raw[0], 0, 32)
                sums = (raw[0], *tuple(_scalar(v, 1e12) for v in raw[1:]))
                if sums[0] != min(seq, 32):
                    raise ValueError("seasonal sums inconsistent")
            elif seq != 0:
                raise ValueError("missing seasonal sums")
            if noise_hat is not None:
                from lck.empirical_reference import warmup_noise_3d
                n, s1, sS, sy, s11, s1S, sSS, sy1, syS, syy, _ = sums
                if warmup_noise_3d(n, s1, sS, sy, s11, s1S, sSS, sy1, syS, syy) != noise_hat:
                    raise ValueError("noise estimate inconsistent with seasonal sums")
            eff = NOMINAL if noise_hat is None else noise_hat
            active = []
            for start, pair, mass in s["active"]:
                _integer(start, 0, seq-1)
                if len(pair) != 2:
                    raise ValueError("seasonal state pair mismatch")
                prec, rhs = pair
                if len(prec) != 6 or len(rhs) != 3:
                    raise ValueError("seasonal state dimension mismatch")
                state = (tuple(_scalar(v, 1e12) for v in prec), tuple(_scalar(v, 1e12) for v in rhs))
                regression(state, (0., 0., 0.), eff)
                mass = _scalar(mass, 1e20)
                if mass > 1e-12:
                    raise ValueError("invalid posterior mass")
                active.append((start, state, mass))
            if len(active) != min(seq, CAPACITY) or len({a[0] for a in active}) != len(active):
                raise ValueError("invalid active bank size")
            anchor = None
            if s["anchor"] is not None:
                prec, rhs = s["anchor"]
                anchor = (tuple(_scalar(v, 1e12) for v in prec), tuple(_scalar(v, 1e12) for v in rhs))
                regression(anchor, (0., 0., 0.), eff)
            memory = None if s["memory"] is None else _scalar(s["memory"], 1e20)
            if (model._recall != (anchor is not None)
                    or (memory is not None) != (model._recall and seq > 32)):
                raise ValueError("invalid seasonal recall state")
            if seq == 0 and anchor != (fresh_prior(NOMINAL) if model._recall else None):
                raise ValueError("invalid initial seasonal anchor")
            if active:
                norm = logsum(tuple(a[2] for a in active)+(() if memory is None else (memory,)))
                if abs(norm) > 1e-10:
                    raise ValueError("unnormalized seasonal posterior")
            model._state = seq, tuple(active), anchor, memory, noise_hat, sums, s["season"], s["recall"], None
            if s["pending"] is not None:
                z = tuple(s["pending"]["z"])
                model.predict([z[1]]+[0.]*(s["season"]-2)+[z[0], 0.])
            if model.dump() != data:
                raise ValueError("noncanonical or inconsistent seasonal forecast")
            return model
        except (TypeError, KeyError, UnicodeError, RecursionError) as exc:
            raise ValueError("invalid seasonal state") from exc
