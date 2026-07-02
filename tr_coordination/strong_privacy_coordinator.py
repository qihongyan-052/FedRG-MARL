from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Mapping


EPS = 1e-9
DEFAULT_SECURE_AGGREGATION_SCALE = 1_000_000
DEFAULT_SECURE_MASK_BOUND = 10**9


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class ParkPrivacyActionProfile:
    park_id: str
    raw_net_kwh: float
    same_direction_controllable_kwh: float
    ev_charge_kwh: float
    ev_discharge_kwh: float
    bes_charge_kwh: float
    bes_discharge_kwh: float
    pv_export_kwh: float


@dataclass(frozen=True)
class LocalPrivacyMetrics:
    park_id: str
    same_direction_controllable_kwh: float
    score_raw: float
    score_probe_20: float
    score_probe_40: float
    probe_curtailment_20_kwh: float
    probe_curtailment_40_kwh: float
    unit_cmdp_loss: float
    local_curvature: float
    composite_curtailment_cost: float
    curtailment_preference: float
    preference_capacity_kwh: float
    final_mixing_weight: float


@dataclass(frozen=True)
class ParkPrivacyAllocation:
    park_id: str
    same_direction_controllable_kwh: float
    final_mixing_weight: float
    curtailment_kwh: float
    pv_curtailment_kwh: float
    shrink_ratio: float


@dataclass(frozen=True)
class ParkPrivacyProjectionResult:
    park_id: str
    ev_energy_grid_side_by_id: Dict[str, float]
    bes_energy_grid_side_kwh: float
    pv_effective_energy_kwh: float
    pv_curtailment_kwh: float
    scaling_factor: float
    projected_park_net_kwh: float
    reduction_kwh: float
    triggered: bool


@dataclass(frozen=True)
class TRCoordinationBroadcast:
    overload_direction: str
    triggered: bool
    total_raw_net_kwh: float
    limit_kwh: float
    overload_kwh: float
    total_capacity_kwh: float
    total_preference_capacity_kwh: float
    safety_base_ratio: float
    blended_capacity_kwh: float
    scaling_coefficient: float
    total_responsibility: float
    tr_penalty_coefficient: float
    infeasible_residual_kwh: float
    actual_total_reduction_kwh: float


@dataclass(frozen=True)
class TRCoordinationControlSignal:
    triggered: bool
    overload_direction: str
    safety_base_ratio: float
    scaling_coefficient: float
    infeasible_residual_kwh: float
    tr_penalty_coefficient: float


@dataclass(frozen=True)
class TRCoordinationSummary:
    broadcast: TRCoordinationBroadcast
    local_metrics_by_park: Dict[str, LocalPrivacyMetrics]
    allocations_by_park: Dict[str, ParkPrivacyAllocation]
    park_results_by_id: Dict[str, ParkPrivacyProjectionResult]

    @property
    def triggered(self) -> bool:
        return self.broadcast.triggered

    @property
    def total_net_before_kwh(self) -> float:
        return self.broadcast.total_raw_net_kwh

    @property
    def total_net_after_kwh(self) -> float:
        if not self.broadcast.triggered:
            return self.broadcast.total_raw_net_kwh
        direction = 1.0 if self.broadcast.overload_direction == "import" else -1.0
        return self.broadcast.total_raw_net_kwh - direction * self.broadcast.actual_total_reduction_kwh

    @property
    def overload_direction(self) -> str:
        return self.broadcast.overload_direction


def compute_raw_overload(total_raw_net_kwh: float, limit_kwh: float) -> float:
    return max(0.0, abs(total_raw_net_kwh) - limit_kwh)


def compute_unit_cmdp_loss(
    score_raw: float,
    score_probe: float,
    probe_curtailment_kwh: float,
) -> float:
    if probe_curtailment_kwh <= EPS:
        return 0.0
    return max(0.0, (score_raw - score_probe) / (probe_curtailment_kwh + EPS))


def compute_local_curvature(
    loss_probe_20: float,
    loss_probe_40: float,
    probe_curtailment_20_kwh: float,
) -> float:
    if probe_curtailment_20_kwh <= EPS:
        return 0.0
    return max(0.0, (loss_probe_40 - 2.0 * loss_probe_20) / ((probe_curtailment_20_kwh + EPS) ** 2))


def compute_composite_curtailment_cost(
    unit_cmdp_loss: float,
    local_curvature: float,
    same_direction_controllable_kwh: float,
    curvature_weight: float,
) -> float:
    return max(0.0, unit_cmdp_loss + curvature_weight * local_curvature * same_direction_controllable_kwh)


def compute_curtailment_preference(
    composite_curtailment_cost: float,
) -> float:
    return 1.0 / (1.0 + max(0.0, composite_curtailment_cost))


def compute_local_shrink_ratio(
    curtailment_kwh: float,
    same_direction_controllable_kwh: float,
) -> float:
    if curtailment_kwh <= EPS or same_direction_controllable_kwh <= EPS:
        return 0.0
    return clamp(curtailment_kwh / same_direction_controllable_kwh, 0.0, 1.0)


def secure_masked_sum(
    values_by_park: Mapping[str, float],
    rng: random.Random,
    *,
    scale: int = DEFAULT_SECURE_AGGREGATION_SCALE,
    mask_bound: int = DEFAULT_SECURE_MASK_BOUND,
) -> float:
    park_ids = sorted(values_by_park.keys())
    scaled_values = {
        park_id: int(round(values_by_park[park_id] * scale))
        for park_id in park_ids
    }
    masked_messages: Dict[str, int] = dict(scaled_values)

    for idx, left_park_id in enumerate(park_ids):
        for right_park_id in park_ids[idx + 1:]:
            mask = rng.randint(-mask_bound, mask_bound)
            masked_messages[left_park_id] += mask
            masked_messages[right_park_id] -= mask

    aggregated_scaled_value = sum(masked_messages.values())
    return aggregated_scaled_value / scale


def compute_global_tr_penalty(
    penalty_weight: float,
    overload_kwh: float,
) -> float:
    if overload_kwh <= EPS:
        return 0.0
    return penalty_weight * overload_kwh * overload_kwh


def compute_local_tr_responsibility(
    overload_direction: str,
    projected_park_net_kwh: float,
) -> float:
    if overload_direction == "import":
        return max(0.0, projected_park_net_kwh)
    if overload_direction == "export":
        return max(0.0, -projected_park_net_kwh)
    return 0.0


def compute_tr_penalty_coefficient(
    penalty_weight: float,
    overload_kwh: float,
    total_responsibility: float,
) -> float:
    if overload_kwh <= EPS or total_responsibility <= EPS:
        return 0.0
    return compute_global_tr_penalty(
        penalty_weight=penalty_weight,
        overload_kwh=overload_kwh,
    ) / total_responsibility


def compute_local_tr_penalty(
    tr_penalty_coefficient: float,
    local_responsibility: float,
) -> float:
    if tr_penalty_coefficient <= EPS or local_responsibility <= EPS:
        return 0.0
    return tr_penalty_coefficient * local_responsibility
