from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Literal, Optional
import math
import random


ParkType = Literal["residential", "office", "commercial"]
EPISODE_START_HOUR = 8.0
OVERNIGHT_DEPARTURE_WINDOW_START_HOUR_ABS = 30.0
OVERNIGHT_DEPARTURE_WINDOW_END_HOUR_ABS = 32.0
STAY_DURATION_EXTENSION_MINUTES = 45

PARK_OVERNIGHT_THRESHOLD_HOUR_ABS: Dict[ParkType, float] = {
    "residential": 20.0,
    "office": 22.0,
    "commercial": 22.0,
}


class StayTier(str, Enum):
    L4_LONGEST = "L4_longest"
    L3_MEDIUM_LONG = "L3_medium_long"
    L2_MEDIUM_SHORT = "L2_medium_short"
    L1_SHORTEST = "L1_shortest"


@dataclass(frozen=True)
class TruncatedNormalMinutesSpec:
    low_min: int
    high_min: int
    mean_min: float
    std_min: float


@dataclass(frozen=True)
class ParkSocReference:
    arrival_soc_mean: float
    target_soc_mean: float


@dataclass
class StayDurationSample:
    park_type: ParkType
    stay_tier: StayTier
    stay_minutes: int
    spec_used: TruncatedNormalMinutesSpec
    metadata: Dict[str, float | int | str | bool]


PARK_SOC_REFERENCES: Dict[ParkType, ParkSocReference] = {
    "residential": ParkSocReference(arrival_soc_mean=0.27, target_soc_mean=0.92),
    "office": ParkSocReference(arrival_soc_mean=0.37, target_soc_mean=0.885),
    "commercial": ParkSocReference(arrival_soc_mean=0.42, target_soc_mean=0.825),
}


def _build_spec(low_min: int, high_min: int, std_ratio: float = 0.20) -> TruncatedNormalMinutesSpec:
    low_min += STAY_DURATION_EXTENSION_MINUTES
    high_min += STAY_DURATION_EXTENSION_MINUTES
    return TruncatedNormalMinutesSpec(
        low_min=low_min,
        high_min=high_min,
        mean_min=0.5 * (low_min + high_min),
        std_min=max(1e-6, std_ratio * (high_min - low_min)),
    )


PARK_STAY_DURATION_SPECS: Dict[ParkType, Dict[StayTier, TruncatedNormalMinutesSpec]] = {
    "residential": {
        StayTier.L4_LONGEST: _build_spec(450, 660, 0.18),
        StayTier.L3_MEDIUM_LONG: _build_spec(345, 510, 0.19),
        StayTier.L2_MEDIUM_SHORT: _build_spec(240, 405, 0.19),
        StayTier.L1_SHORTEST: _build_spec(150, 270, 0.20),
    },
    "office": {
        StayTier.L4_LONGEST: _build_spec(285, 450, 0.18),
        StayTier.L3_MEDIUM_LONG: _build_spec(210, 345, 0.19),
        StayTier.L2_MEDIUM_SHORT: _build_spec(135, 255, 0.20),
        StayTier.L1_SHORTEST: _build_spec(75, 165, 0.20),
    },
    "commercial": {
        StayTier.L4_LONGEST: _build_spec(120, 240, 0.18),
        StayTier.L3_MEDIUM_LONG: _build_spec(90, 175, 0.18),
        StayTier.L2_MEDIUM_SHORT: _build_spec(60, 120, 0.19),
        StayTier.L1_SHORTEST: _build_spec(35, 80, 0.20),
    },
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


class StayDurationSampler:
    """根据园区、到达 SoC、目标 SoC 以及可选上下文，生成停留时长。"""

    def __init__(self, seed: Optional[int] = None, step_minutes: int = 15) -> None:
        self.rng = random.Random(seed)
        self.step_minutes = step_minutes

    def classify_stay_tier(self, park_type: ParkType, arrival_soc: float, target_soc: float) -> StayTier:
        ref = PARK_SOC_REFERENCES[park_type]
        arrival_low = arrival_soc < ref.arrival_soc_mean
        target_high = target_soc >= ref.target_soc_mean
        if arrival_low and target_high:
            return StayTier.L4_LONGEST
        if (not arrival_low) and target_high:
            return StayTier.L3_MEDIUM_LONG
        if arrival_low and (not target_high):
            return StayTier.L2_MEDIUM_SHORT
        return StayTier.L1_SHORTEST

    def sample_one(
        self,
        park_type: ParkType,
        arrival_soc: float,
        target_soc: float,
        arrival_step: Optional[int] = None,
        vehicle_info: Optional[dict] = None,
        v2g_willing: Optional[bool] = None,
    ) -> StayDurationSample:
        tier = self.classify_stay_tier(park_type, arrival_soc, target_soc)
        spec = PARK_STAY_DURATION_SPECS[park_type][tier]
        low, high, mean, std = spec.low_min, spec.high_min, spec.mean_min, spec.std_min

        stay_minutes = int(
            round(
                sample_truncated_normal(
                    self.rng,
                    mean=mean,
                    std=std,
                    low=low,
                    high=high,
                )
            )
        )
        return StayDurationSample(
            park_type=park_type,
            stay_tier=tier,
            stay_minutes=stay_minutes,
            spec_used=spec,
            metadata={
                "arrival_soc": arrival_soc,
                "target_soc": target_soc,
                "arrival_step": -1 if arrival_step is None else arrival_step,
                "v2g_willing": bool(v2g_willing),
            },
        )

    def sample_many(self, park_type: ParkType, arrival_socs: List[float], target_socs: List[float], arrival_steps: Optional[List[int]] = None) -> List[StayDurationSample]:
        results = []
        for idx, arrival_soc in enumerate(arrival_socs):
            results.append(
                self.sample_one(
                    park_type=park_type,
                    arrival_soc=arrival_soc,
                    target_soc=target_socs[idx],
                    arrival_step=None if arrival_steps is None else arrival_steps[idx],
                )
            )
        return results

    def derive_departure_step(self, arrival_step: int, stay_minutes: int, episode_total_steps: int = 96) -> int:
        stay_steps = math.ceil(stay_minutes / self.step_minutes)
        return min(episode_total_steps - 1, arrival_step + stay_steps)

    def derive_departure_hour_abs(self, arrival_hour_abs: float, stay_minutes: int, episode_total_steps: int = 96) -> float:
        departure_hour_abs = arrival_hour_abs + stay_minutes / 60.0
        episode_end_hour_abs = EPISODE_START_HOUR + episode_total_steps * self.step_minutes / 60.0
        return min(departure_hour_abs, episode_end_hour_abs)

    @staticmethod
    def is_overnight(park_type: ParkType, departure_hour_abs: float) -> bool:
        return departure_hour_abs > PARK_OVERNIGHT_THRESHOLD_HOUR_ABS[park_type]

    def sample_overnight_departure_hour_abs(self, arrival_hour_abs: float) -> float:
        low = max(OVERNIGHT_DEPARTURE_WINDOW_START_HOUR_ABS, arrival_hour_abs)
        high = OVERNIGHT_DEPARTURE_WINDOW_END_HOUR_ABS
        if low >= high:
            return high
        return sample_truncated_normal(
            self.rng,
            mean=0.5 * (low + high),
            std=max(1e-6, 0.2 * (high - low)),
            low=low,
            high=high,
        )


if __name__ == "__main__":
    sampler = StayDurationSampler(seed=42)
    print(sampler.sample_one("office", 0.5, 0.85, arrival_step=8))
