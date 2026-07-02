from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal
import math
import random


ParkType = Literal["residential", "office", "commercial"]

STEP_MINUTES = 15
EPISODE_STEPS = 96
EPISODE_START_HOUR = 8.0


@dataclass(frozen=True)
class CountRange:
    low: int
    high: int

    @property
    def mean(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def std(self) -> float:
        return max(1e-6, 0.18 * (self.high - self.low))


@dataclass(frozen=True)
class TimeSegment:
    name: str
    start_hour: float
    end_hour: float
    ratio_low: float
    ratio_high: float
    ratio_jitter_pct: float
    time_std_ratio: float

    @property
    def center_hour(self) -> float:
        return 0.5 * (self.start_hour + self.end_hour)

    @property
    def duration_hours(self) -> float:
        return self.end_hour - self.start_hour

    @property
    def ratio_mean(self) -> float:
        return 0.5 * (self.ratio_low + self.ratio_high)


PARK_ARRIVAL_COUNT_RANGES: Dict[ParkType, CountRange] = {
    "residential": CountRange(28, 40),
    "office": CountRange(36, 50),
    "commercial": CountRange(34, 46),
}

PARK_ARRIVAL_SEGMENTS: Dict[ParkType, List[TimeSegment]] = {
    "residential": [
        TimeSegment("08:00-10:00", 8, 10, 0.005, 0.015, 0.30, 0.25),
        TimeSegment("10:00-12:00", 10, 12, 0.010, 0.030, 0.30, 0.25),
        TimeSegment("12:00-14:00", 12, 14, 0.020, 0.040, 0.25, 0.22),
        TimeSegment("14:00-16:00", 14, 16, 0.040, 0.060, 0.22, 0.20),
        TimeSegment("16:00-18:00", 16, 18, 0.140, 0.180, 0.18, 0.18),
        TimeSegment("18:00-20:00", 18, 20, 0.290, 0.330, 0.15, 0.16),
        TimeSegment("20:00-22:00", 20, 22, 0.240, 0.280, 0.15, 0.16),
        TimeSegment("22:00-00:00", 22, 24, 0.100, 0.140, 0.20, 0.18),
        TimeSegment("00:00-06:00", 24, 30, 0.020, 0.060, 0.35, 0.25),
        TimeSegment("06:00-08:00", 30, 32, 0.000, 0.000, 0.00, 0.25),
    ],
    "office": [
        TimeSegment("08:00-09:00", 8, 9, 0.260, 0.300, 0.15, 0.16),
        TimeSegment("09:00-10:00", 9, 10, 0.220, 0.260, 0.15, 0.16),
        TimeSegment("10:00-12:00", 10, 12, 0.100, 0.140, 0.20, 0.18),
        TimeSegment("12:00-14:00", 12, 14, 0.160, 0.200, 0.18, 0.17),
        TimeSegment("14:00-16:00", 14, 16, 0.070, 0.110, 0.22, 0.20),
        TimeSegment("16:00-18:00", 16, 18, 0.040, 0.060, 0.22, 0.20),
        TimeSegment("18:00-20:00", 18, 20, 0.010, 0.030, 0.30, 0.22),
        TimeSegment("20:00-22:00", 20, 22, 0.005, 0.015, 0.35, 0.25),
        TimeSegment("22:00-00:00", 22, 24, 0.005, 0.015, 0.35, 0.25),
        TimeSegment("00:00-08:00", 24, 32, 0.000, 0.000, 0.00, 0.25),
    ],
    "commercial": [
        TimeSegment("08:00-10:00", 8, 10, 0.060, 0.100, 0.22, 0.20),
        TimeSegment("10:00-12:00", 10, 12, 0.160, 0.200, 0.18, 0.18),
        TimeSegment("12:00-14:00", 12, 14, 0.220, 0.260, 0.15, 0.16),
        TimeSegment("14:00-16:00", 14, 16, 0.090, 0.130, 0.20, 0.18),
        TimeSegment("16:00-18:00", 16, 18, 0.140, 0.180, 0.18, 0.17),
        TimeSegment("18:00-20:00", 18, 20, 0.160, 0.200, 0.18, 0.17),
        TimeSegment("20:00-22:00", 20, 22, 0.040, 0.060, 0.25, 0.20),
        TimeSegment("22:00-08:00", 22, 32, 0.000, 0.000, 0.00, 0.25),
    ],
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


def normalize_weights(weights: List[float]) -> List[float]:
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("weights sum must be positive")
    return [weight / total for weight in weights]


def allocate_integer_counts(total: int, weights: List[float]) -> List[int]:
    raw = [total * weight for weight in weights]
    base = [math.floor(value) for value in raw]
    remain = total - sum(base)
    frac = sorted(((idx, raw[idx] - base[idx]) for idx in range(len(raw))), key=lambda item: item[1], reverse=True)
    for idx, _ in frac[:remain]:
        base[idx] += 1
    return base


def hour_to_step(hour_abs: float) -> int:
    step = int((hour_abs - EPISODE_START_HOUR) * 60 // STEP_MINUTES)
    return max(0, min(EPISODE_STEPS - 1, step))


class ArrivalTimeSampler:
    """三园区 EV 到达时刻采样器。"""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def sample_daily_arrival_count(self, park_type: ParkType) -> int:
        cfg = PARK_ARRIVAL_COUNT_RANGES[park_type]
        return int(
            round(
                sample_truncated_normal(
                    self.rng,
                    mean=cfg.mean,
                    std=cfg.std,
                    low=cfg.low,
                    high=cfg.high,
                )
            )
        )

    def sample_arrivals_for_park(self, park_type: ParkType) -> Dict[str, object]:
        segments = PARK_ARRIVAL_SEGMENTS[park_type]
        total_count = self.sample_daily_arrival_count(park_type)
        weights = []
        for segment in segments:
            if segment.ratio_mean <= 0.0:
                weights.append(0.0)
                continue
            weights.append(
                sample_truncated_normal(
                    self.rng,
                    mean=segment.ratio_mean,
                    std=max(1e-6, segment.ratio_mean * 0.08),
                    low=segment.ratio_low,
                    high=segment.ratio_high,
                )
            )
        seg_counts = allocate_integer_counts(total_count, normalize_weights(weights))

        arrivals: List[Dict[str, object]] = []
        for segment, count in zip(segments, seg_counts):
            if count <= 0 or segment.ratio_mean <= 0.0:
                continue
            for _ in range(count):
                hour_abs = sample_truncated_normal(
                    self.rng,
                    mean=segment.center_hour,
                    std=max(1e-6, segment.duration_hours * segment.time_std_ratio),
                    low=segment.start_hour,
                    high=segment.end_hour,
                )
                arrivals.append(
                    {
                        "park_type": park_type,
                        "segment_name": segment.name,
                        "arrival_hour_abs": round(hour_abs, 4),
                        "arrival_step": hour_to_step(hour_abs),
                        "arrival_hour_label": self.format_hour_label(hour_abs),
                    }
                )
        arrivals.sort(key=lambda item: (item["arrival_step"], item["arrival_hour_abs"]))
        return {
            "park_type": park_type,
            "daily_total": total_count,
            "segment_counts": [{"segment_name": segment.name, "count": count} for segment, count in zip(segments, seg_counts)],
            "arrivals": arrivals,
        }

    def sample_arrivals_for_all_parks(self) -> Dict[ParkType, Dict[str, object]]:
        return {park_type: self.sample_arrivals_for_park(park_type) for park_type in ("residential", "office", "commercial")}

    @staticmethod
    def format_hour_label(hour_abs: float) -> str:
        if hour_abs < 24.0:
            prefix = ""
            local_hour = hour_abs
        else:
            prefix = "next_day "
            local_hour = hour_abs - 24.0
        hh = int(local_hour)
        mm = int(round((local_hour - hh) * 60))
        if mm == 60:
            hh += 1
            mm = 0
        return f"{prefix}{hh % 24:02d}:{mm:02d}"


if __name__ == "__main__":
    sampler = ArrivalTimeSampler(seed=42)
    print(sampler.sample_arrivals_for_all_parks())
