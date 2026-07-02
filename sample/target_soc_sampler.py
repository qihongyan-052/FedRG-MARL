from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional
import random


ParkType = Literal["residential", "office", "commercial"]


@dataclass(frozen=True)
class TruncatedNormalSpec:
    low: float
    high: float
    std: float

    @property
    def mean(self) -> float:
        return 0.5 * (self.low + self.high)


TARGET_SOC_SPECS: Dict[ParkType, TruncatedNormalSpec] = {
    "residential": TruncatedNormalSpec(low=0.85, high=0.99, std=0.05),
    "office": TruncatedNormalSpec(low=0.80, high=0.97, std=0.06),
    "commercial": TruncatedNormalSpec(low=0.75, high=0.90, std=0.07),
}


def sample_truncated_normal(
    rng: random.Random,
    mean: float,
    std: float,
    low: float,
    high: float,
    max_trials: int = 1000,
) -> float:
    for _ in range(max_trials):
        value = rng.gauss(mean, std)
        if low <= value <= high:
            return value
    return min(max(rng.gauss(mean, std), low), high)


class TargetSoCSampler:
    def __init__(self, seed: Optional[int] = None, round_digits: int = 4, default_min_delta: float = 0.05) -> None:
        self.rng = random.Random(seed)
        self.round_digits = round_digits
        self.default_min_delta = default_min_delta

    def sample_one(self, park_type: ParkType, arrival_soc: Optional[float] = None, min_delta: Optional[float] = None) -> float:
        spec = TARGET_SOC_SPECS[park_type]
        delta = self.default_min_delta if min_delta is None else min_delta
        low = spec.low
        if arrival_soc is not None:
            low = max(low, arrival_soc + delta)
        if low >= spec.high:
            return round(spec.high, self.round_digits)
        return round(
            sample_truncated_normal(self.rng, mean=spec.mean, std=spec.std, low=low, high=spec.high),
            self.round_digits,
        )

    def sample_many(self, park_type: ParkType, n: int, arrival_socs: Optional[List[float]] = None) -> List[float]:
        if arrival_socs is None:
            return [self.sample_one(park_type) for _ in range(n)]
        return [self.sample_one(park_type, arrival_soc=arrival_socs[idx]) for idx in range(n)]


if __name__ == "__main__":
    sampler = TargetSoCSampler(seed=42)
    print(sampler.sample_many("residential", 5, [0.2, 0.3, 0.4, 0.5, 0.6]))
