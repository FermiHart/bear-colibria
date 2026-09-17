"""Authored affine NLMS reference using only Python's standard library.

NLMS is an established algorithm, not claimed as novel. One pending prediction
per instance, label afterwards; immutable tuples publish a completed update.
Snapshots restore pending predictions as well as completed weights. No durable
I/O, thread synchronization, hidden input history or arbitrary model execution.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import struct

MAGIC = b"BEARNLM1"
MAX_SNAPSHOT = 32768
MAX_SEQUENCE = 2**63 - 1


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("integer outside reference bounds")


def _scalar(value, limit, *, positive=False):
    if type(value) not in (int, float) or not -limit <= value <= limit or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError("finite scalar outside reference bounds")
    return float(value)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


@dataclass(frozen=True)
class Prediction:
    sequence: int
    value: float


class OnlineReference:
    def __init__(self, dimension=4, *, rate=.1, epsilon=.01, input_limit=1000., parameter_limit=1e6):
        _integer(dimension, 1, 128)
        self._config = (dimension, _scalar(rate, 1., positive=True), _scalar(epsilon, 1., positive=True),
                        _scalar(input_limit, 1000., positive=True), _scalar(parameter_limit, 1e6, positive=True))
        # Publish sequence, weights and pending together, including on observe.
        self._state = (0, (0.,) * (dimension + 1), None)

    @property
    def sequence(self):
        return self._state[0]

    @property
    def weights(self):
        return self._state[1]

    @property
    def pending(self):
        value = self._state[2]
        return None if value is None else Prediction(self.sequence, value[1])

    def predict(self, features):
        sequence, weights, pending = self._state
        if pending is not None:
            raise RuntimeError("observe the pending prediction before another input")
        if sequence == MAX_SEQUENCE:
            raise ValueError("reference sequence exhausted")
        dimension, _, _, limit, _ = self._config
        if type(features) not in (tuple, list) or len(features) != dimension:
            raise ValueError("reference features must match dimension")
        x = tuple(_scalar(v, limit) for v in features)
        z = x + (1.,)
        prediction = math.fsum(w * v for w, v in zip(weights, z))
        if not math.isfinite(prediction):
            raise ValueError("nonfinite prediction")
        result = Prediction(sequence, prediction)
        self._state = sequence, weights, (x, prediction)
        return result

    def observe(self, target, *, expected_sequence):
        sequence, weights, pending = self._state
        if type(expected_sequence) is not int or expected_sequence != sequence:
            raise ValueError("stale reference sequence")
        if pending is None:
            raise RuntimeError("predict before observing a label")
        _, rate, epsilon, limit, weight_limit = self._config
        target = _scalar(target, limit)
        features, prediction = pending
        z = features + (1.,)
        denominator = epsilon + math.fsum(v * v for v in z)
        error = target - prediction
        updated = tuple(w + rate * error * v / denominator for w, v in zip(weights, z))
        if any(not math.isfinite(w) or abs(w) > weight_limit for w in updated):
            raise ValueError("updated parameter outside bounds; pending prediction retained")
        self._state = sequence + 1, updated, None
        return {"sequence": sequence + 1, "prediction": prediction, "error": error, "durable": False}

    def dump(self):
        sequence, weights, pending = self._state
        body = _json({"version": 1, "algorithm": "affine-nlms-fsum-v1", "config": list(self._config),
                      "sequence": sequence, "weights": list(weights),
                      "pending": None if pending is None else {"features": list(pending[0]), "prediction": pending[1]}})
        if len(body) + 44 > MAX_SNAPSHOT:
            raise ValueError("reference snapshot exceeds limit")
        prefix = MAGIC + struct.pack("<I", len(body)) + body
        return prefix + hashlib.sha256(prefix).digest()

    @classmethod
    def load(cls, data):
        if type(data) is not bytes or not 44 <= len(data) <= MAX_SNAPSHOT or data[:8] != MAGIC:
            raise ValueError("invalid reference envelope")
        length, = struct.unpack_from("<I", data, 8)
        if length + 44 != len(data) or hashlib.sha256(data[:-32]).digest() != data[-32:]:
            raise ValueError("reference length/digest mismatch")

        def unique(pairs):
            result = {}
            for k, v in pairs:
                if k in result:
                    raise ValueError("duplicate reference field")
                result[k] = v
            return result

        try:
            state = json.loads(data[12:-32], object_pairs_hook=unique)
            if _json(state) != data[12:-32]:
                raise ValueError("noncanonical reference snapshot")
        except (UnicodeError, RecursionError) as exc:
            raise ValueError("invalid reference JSON") from exc
        if (type(state) is not dict or set(state) != {"version", "algorithm", "config", "sequence", "weights", "pending"}
                or type(state["version"]) is not int or state["version"] != 1 or state["algorithm"] != "affine-nlms-fsum-v1"
                or type(state["config"]) is not list or len(state["config"]) != 5):
            raise ValueError("invalid reference schema")
        dimension, rate, epsilon, limit, cap = state["config"]
        model = cls(dimension, rate=rate, epsilon=epsilon, input_limit=limit, parameter_limit=cap)
        _integer(state["sequence"], 0, MAX_SEQUENCE)
        if type(state["weights"]) is not list or len(state["weights"]) != dimension + 1:
            raise ValueError("invalid weight count")
        weights = tuple(_scalar(w, cap) for w in state["weights"])
        model._state = state["sequence"], weights, None
        pending = state["pending"]
        if pending is not None:
            if type(pending) is not dict or set(pending) != {"features", "prediction"}:
                raise ValueError("invalid pending schema")
            predicted = model.predict(pending["features"])
            if type(pending["prediction"]) not in (int, float) or _json(predicted.value) != _json(pending["prediction"]):
                raise ValueError("pending prediction inconsistent with weights/features")
        if model.dump() != data:
            raise ValueError("reference state not in canonical runtime form")
        return model
