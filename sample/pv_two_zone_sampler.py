from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import csv
import random


@dataclass
class PVSamplerConfig:
    csv_path: str
    residential_scale: float = 0.0
    office_scale: float = 0.82
    commercial_scale: float = 0.62
    weather_probs: Dict[str, float] = field(
        default_factory=lambda: {"sunny": 0.40, "cloudy": 0.30, "overcast": 0.20, "rainy": 0.10}
    )
    std_ratio: float = 0.05
    trunc_ratio: float = 0.10


def _resolve_csv_path(csv_path: str) -> Path:
    path = Path(csv_path)
    if path.exists():
        return path
    local_path = Path(__file__).resolve().parents[1] / "config_files" / csv_path
    if local_path.exists():
        return local_path
    raise FileNotFoundError(csv_path)


def _load_pv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(_resolve_csv_path(csv_path), "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _sample_truncated_gaussian(rng: random.Random, mean: float, std_ratio: float, trunc_ratio: float) -> float:
    if mean <= 0.0:
        return 0.0
    low = max(0.0, mean * (1.0 - trunc_ratio))
    high = mean * (1.0 + trunc_ratio)
    std = max(1e-6, mean * std_ratio)
    for _ in range(1000):
        value = rng.gauss(mean, std)
        if low <= value <= high:
            return value
    return min(max(rng.gauss(mean, std), low), high)


class ThreeParkPVSampler:
    """工作日单日 PV 采样器。住宅区无 PV，办公区最大，商业区中等。"""

    def __init__(self, config: PVSamplerConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self.rng = random.Random(seed)
        self.rows = _load_pv_rows(config.csv_path)

    def sample_day(self, weather: Optional[str] = None) -> Dict[str, object]:
        chosen_weather = weather or self.rng.choices(
            population=list(self.config.weather_probs.keys()),
            weights=list(self.config.weather_probs.values()),
            k=1,
        )[0]
        office_kw = []
        commercial_kw = []
        residential_kw = []
        for row in self.rows:
            residential_base = float(row[chosen_weather]) * self.config.residential_scale
            commercial_base = float(row[chosen_weather]) * self.config.commercial_scale
            office_base = float(row[chosen_weather]) * self.config.office_scale
            residential_kw.append(_sample_truncated_gaussian(self.rng, residential_base, self.config.std_ratio, self.config.trunc_ratio))
            commercial_kw.append(_sample_truncated_gaussian(self.rng, commercial_base, self.config.std_ratio, self.config.trunc_ratio))
            office_kw.append(_sample_truncated_gaussian(self.rng, office_base, self.config.std_ratio, self.config.trunc_ratio))
        return {
            "weather": chosen_weather,
            "time": [row["time"] for row in self.rows],
            "park_pv_kw": {
                "residential": residential_kw,
                "office": office_kw,
                "commercial": commercial_kw,
            },
        }


if __name__ == "__main__":
    sampler = ThreeParkPVSampler(PVSamplerConfig(csv_path="pv_4weather.csv"), seed=42)
    print(sampler.sample_day())
