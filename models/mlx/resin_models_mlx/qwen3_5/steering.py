"""Request-scoped down-projection input steering in residual-norm dose units."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SteeringCalibration:
    model: str
    token_count: int
    residual_medians: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "residual_medians", tuple(self.residual_medians))
        if (
            self.token_count < 2
            or not self.residual_medians
            or any(
                not math.isfinite(value) or value <= 0
                for value in self.residual_medians
            )
        ):
            raise ValueError(
                "calibration requires at least two tokens and positive finite residual norms"
            )

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class Qwen35Steering:
    """MediaInput built by ``runner.steer(tokens, [(layer, neuron, dose)], calibration)``.

    The first eight identity words carry 248 digest bits outside the model vocabulary;
    remaining tokens stay raw so generated-prefix continuations can hit the same trie.
    Prompts shorter than eight tokens carry only 31 bits per token. Like every finite
    digest, this is probabilistic collision resistance, not a proof of uniqueness.
    """

    token_ids: tuple[int, ...]
    neurons: tuple[tuple[int, int, float], ...]
    calibration: SteeringCalibration
    additions: tuple[tuple[int, int, float], ...]
    cache_key: tuple[int, ...] = field(init=False)
    token_spans: tuple = field(default=(), init=False)

    def __post_init__(self):
        tokens = tuple(self.token_ids)
        object.__setattr__(self, "token_ids", tokens)
        identity = json.dumps(
            (self.neurons, asdict(self.calibration), self.additions),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        keys = list(tokens)
        for index, token in enumerate(tokens[:8]):
            digest = hashlib.sha256(
                identity + struct.pack("<II", index, token)
            ).digest()
            keys[index] = int.from_bytes(digest[:4], "little") | 0x80000000
        object.__setattr__(self, "cache_key", tuple(keys))
