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


ARRIVAL_SOC_SPECS: Dict[ParkType, TruncatedNormalSpec] = {
    "residential": TruncatedNormalSpec(low=0.12, high=0.42, std=0.10),
    "office": TruncatedNormalSpec(low=0.24, high=0.50, std=0.09),
    "commercial": TruncatedNormalSpec(low=0.14, high=0.70, std=0.11),
}


def sample_truncated_normal(rng: random.Random, spec: TruncatedNormalSpec, max_trials: int = 1000) -> float:
    for _ in range(max_trials):
        value = rng.gauss(spec.mean, spec.std)
        if spec.low <= value <= spec.high:
            return value
    return min(max(rng.gauss(spec.mean, spec.std), spec.low), spec.high)


class ArrivalSoCSampler:
    def __init__(self, seed: Optional[int] = None, round_digits: int = 4) -> None:
        self.rng = random.Random(seed)
        self.round_digits = round_digits

    def sample_one(self, park_type: ParkType) -> float:
        return round(sample_truncated_normal(self.rng, ARRIVAL_SOC_SPECS[park_type]), self.round_digits)

    def sample_many(self, park_type: ParkType, n: int) -> List[float]:
        return [self.sample_one(park_type) for _ in range(n)]


if __name__ == "__main__":
    sampler = ArrivalSoCSampler(seed=42)
    for park in ("residential", "office", "commercial"):
        print(park, sampler.sample_many(park, 5))
