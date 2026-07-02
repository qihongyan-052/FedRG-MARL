from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import json
import random


ParkType = Literal["residential", "office", "commercial"]


@dataclass(frozen=True)
class EVModel:
    id: str
    size_type: str
    battery_capacity_kwh: float
    p_ch_max_kw: float
    soc_knee: float
    soc_tail_start: float
    tail_power_kw: float
    eta_ch: float
    eta_dis: float
    p_dis_max_kw: float
    soc_min: float
    v2g_capable: bool


@dataclass(frozen=True)
class VehicleSample:
    park_type: ParkType
    model_id: str
    size_type: str
    battery_capacity_kwh: float
    p_ch_max_kw: float
    p_dis_max_kw: float
    soc_knee: float
    soc_tail_start: float
    tail_power_kw: float
    eta_ch: float
    eta_dis: float
    soc_min: float
    v2g_capable: bool
    v2g_willing_user: bool
    v2g_enabled: bool
    metadata: Dict[str, Any]


PARK_SIZE_WEIGHTS: Dict[ParkType, Dict[str, float]] = {
    "residential": {"small": 0.58, "medium": 0.32, "large": 0.10},
    "office": {"small": 0.24, "medium": 0.51, "large": 0.25},
    "commercial": {"small": 0.16, "medium": 0.39, "large": 0.45},
}

PARK_POWER_SCALE: Dict[ParkType, float] = {
    "residential": 0.95,
    "office": 1.00,
    "commercial": 1.05,
}

PARK_V2G_WILLING_PROB: Dict[ParkType, float] = {
    "residential": 0.80,
    "office": 0.80,
    "commercial": 0.50,
}


class VehicleAndV2GSampler:
    """基于车型库和园区车型比例，采样 EV 品牌车型与 V2G 意愿。"""

    def __init__(
        self,
        json_path: str | Path,
        seed: Optional[int] = None,
    ) -> None:
        self.json_path = Path(json_path)
        self.rng = random.Random(seed)
        self.models = self._load_models()
        self.models_by_size = self._group_models_by_size(self.models)

    def sample_one(self, park_type: ParkType, custom_context: Optional[Dict[str, Any]] = None) -> VehicleSample:
        self._validate_park_type(park_type)
        size_type = self._weighted_choice(PARK_SIZE_WEIGHTS[park_type])
        model = self.rng.choice(self.models_by_size[size_type])
        v2g_willing_user = self.rng.random() < PARK_V2G_WILLING_PROB[park_type]
        power_scale = PARK_POWER_SCALE[park_type]
        return VehicleSample(
            park_type=park_type,
            model_id=model.id,
            size_type=model.size_type,
            battery_capacity_kwh=model.battery_capacity_kwh,
            p_ch_max_kw=round(model.p_ch_max_kw * power_scale, 4),
            p_dis_max_kw=round(model.p_dis_max_kw * power_scale, 4),
            soc_knee=model.soc_knee,
            soc_tail_start=model.soc_tail_start,
            tail_power_kw=round(model.tail_power_kw * power_scale, 4),
            eta_ch=model.eta_ch,
            eta_dis=model.eta_dis,
            soc_min=model.soc_min,
            v2g_capable=model.v2g_capable,
            v2g_willing_user=v2g_willing_user,
            v2g_enabled=bool(model.v2g_capable and v2g_willing_user),
            metadata={"custom_context": custom_context or {}},
        )

    def sample_many(self, park_type: ParkType, n: int) -> List[VehicleSample]:
        return [self.sample_one(park_type=park_type) for _ in range(n)]

    def summarize_model_pool(self) -> Dict[str, int]:
        return {size_type: len(pool) for size_type, pool in self.models_by_size.items()}

    def _load_models(self) -> List[EVModel]:
        with open(self.json_path, "r", encoding="utf-8") as file:
            raw = json.load(file)["ev_models"]
        return [
            EVModel(
                id=item["id"],
                size_type=item["size_type"],
                battery_capacity_kwh=float(item["battery_capacity_kwh"]),
                p_ch_max_kw=float(item["p_ch_max_kw"]),
                soc_knee=float(item["soc_knee"]),
                soc_tail_start=float(item["soc_tail_start"]),
                tail_power_kw=float(item["tail_power_kw"]),
                eta_ch=float(item["eta_ch"]),
                eta_dis=float(item["eta_dis"]),
                p_dis_max_kw=float(item["p_dis_max_kw"]),
                soc_min=float(item["soc_min"]),
                v2g_capable=bool(item["v2g_capable"]),
            )
            for item in raw
        ]

    @staticmethod
    def _group_models_by_size(models: List[EVModel]) -> Dict[str, List[EVModel]]:
        grouped: Dict[str, List[EVModel]] = {"small": [], "medium": [], "large": []}
        for model in models:
            grouped.setdefault(model.size_type, []).append(model)
        return grouped

    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        items = list(weights.keys())
        probs = list(weights.values())
        return self.rng.choices(items, weights=probs, k=1)[0]

    @staticmethod
    def _validate_park_type(park_type: str) -> None:
        if park_type not in PARK_SIZE_WEIGHTS:
            raise ValueError(f"Unknown park_type: {park_type}")


if __name__ == "__main__":
    sampler = VehicleAndV2GSampler(
        json_path=Path(__file__).resolve().parents[1] / "config_files" / "ev_20_brand_models.json",
        seed=42,
    )
    for park in ("residential", "office", "commercial"):
        print(park, sampler.sample_one(park))
