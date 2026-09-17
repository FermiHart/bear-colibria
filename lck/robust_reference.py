"""Huberized seasonal changepoint/recall mixture, own stdlib implementation.

Identical to SeasonalReference (same 3-dim joint posteriors, hazard, recall,
empirical noise, quantiles) except the posterior scoring: standardized
residuals beyond HUBER_K contribute linearly instead of quadratically, so
isolated spikes cannot annihilate long-run hypotheses. Predictive
distributions are unchanged; only posterior mass allocation is robustified.
Own snapshot envelope with the Huber constant versioned.
"""
from dataclasses import asdict
import hashlib
import json
import math

from lck.change_reference import CAPACITY, HAZARD, logsum, quantile
from lck.seasonal_reference import SeasonalReference, update
from lck.online_reference import _scalar, _integer, _json, MAX_SEQUENCE

MAGIC = b"BEARSR01"
HUBER_K = 3.0


def huber_score(prior, mean, variance, target):
    residual = (target-mean)/math.sqrt(variance)
    if not math.isfinite(residual):
        raise ValueError("nonfinite robust residual")
    loss = residual*residual/2 if abs(residual) <= HUBER_K else HUBER_K*abs(residual)-HUBER_K*HUBER_K/2
    return prior-.5*math.log(2*math.pi*variance)-loss


class RobustSeasonal(SeasonalReference):
    def __init__(self, season, *, recall=False):
        super().__init__(season, recall=recall)

    def observe(self, target, *, expected_sequence):
        sequence, active, anchor, memory, noise_hat, sums, season, recall, pending = self._state
        if type(expected_sequence) is not int or expected_sequence != sequence:
            raise ValueError("stale sequence")
        if pending is None:
            raise RuntimeError("predict before label")
        if (season, recall) != (self._season_param, self._recall):
            raise ValueError("robust configuration changed")
        target = _scalar(target, 1000.)
        z, prediction = pending
        noise = self.noise
        candidates, memory_score = [], None
        for kind, start, state, prior, mean, variance in self._components(z):
            score = huber_score(prior, mean, variance, target)
            if kind == "memory":
                memory_score = score
            else:
                candidates.append((start, update(state, z, target, noise), score))
        candidates.sort(key=lambda c: (-c[2], c[0]))
        candidates = candidates[:CAPACITY]
        mem_scores = tuple(s for s in (memory_score,) if s is not None)
        norm = logsum(tuple(c[2] for c in candidates)+mem_scores)
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
            from lck.empirical_reference import warmup_noise_3d, NOISE_CAP
            n, s1, sS, sy, s11, s1S, sSS, sy1, syS, syy, _ = sums
            noise_hat = _scalar(warmup_noise_3d(n, s1, sS, sy, s11, s1S, sSS, sy1, syS, syy),
                                NOISE_CAP, positive=True)
        receipt = {"sequence": sequence+1, "prediction": prediction.value,
                   "error": target-prediction.value, "durable": False}
        self._state = sequence+1, active, anchor, memory, noise_hat, sums, season, recall, None
        return receipt

    def dump(self):
        sequence, active, anchor, memory, noise_hat, sums, season, recall, pending = self._state
        prefix = MAGIC + _json({"huber": HUBER_K, "recall": recall, "season": season, "sequence": sequence,
            "active": active, "anchor": anchor, "memory": memory, "noise_hat": noise_hat, "sums": sums,
            "pending": None if pending is None else {"z": pending[0], "forecast": asdict(pending[1])}})
        if len(prefix)+32 > 65536:
            raise ValueError("robust snapshot exceeds limit")
        return prefix+hashlib.sha256(prefix).digest()

    @classmethod
    def load(cls, data, season=None):
        from lck.seasonal_reference import SeasonalReference as Base, MAGIC as BASE_MAGIC
        if (type(data) is not bytes or not 40 < len(data) <= 65536 or data[:8] != MAGIC
                or hashlib.sha256(data[:-32]).digest() != data[-32:]):
            raise ValueError("invalid robust snapshot")
        try:
            s = json.loads(data[8:-32])
        except (UnicodeError, RecursionError) as exc:
            raise ValueError("invalid robust JSON") from exc
        if (_json(s) != data[8:-32] or set(s) != {"huber", "recall", "season", "sequence", "active",
                "anchor", "memory", "noise_hat", "sums", "pending"} or s["huber"] != HUBER_K):
            raise ValueError("invalid robust schema")
        inner = {k: s[k] for k in ("recall", "season", "sequence", "active", "anchor",
                                   "memory", "noise_hat", "sums", "pending")}
        body = _json(inner)
        base = Base.load(BASE_MAGIC+body+hashlib.sha256(BASE_MAGIC+body).digest(), season=season)
        model = cls(s["season"], recall=s["recall"])
        model._state = base._state
        if model.dump() != data:
            raise ValueError("noncanonical robust snapshot")
        return model
