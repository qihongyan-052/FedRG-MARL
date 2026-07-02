from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


EPS = 1e-9


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class ParkCSLimitSpec:
    park_id: str
    max_exchange_energy_kwh: float


@dataclass(frozen=True)
class ParkCSProjectionInput:
    park_id: str
    ev_energy_grid_side_by_id: Dict[str, float] = field(default_factory=dict)
    bes_energy_grid_side_kwh: float = 0.0
    pv_energy_kwh: float = 0.0


@dataclass(frozen=True)
class ParkCSProjectionResult:
    park_id: str
    raw_ev_energy_grid_side_by_id: Dict[str, float]
    raw_bes_energy_grid_side_kwh: float
    raw_pv_energy_kwh: float
    ev_energy_grid_side_by_id: Dict[str, float]
    bes_energy_grid_side_kwh: float
    scaling_factor: float
    raw_device_net_kwh: float
    raw_net_after_pv_kwh: float
    projected_device_net_kwh: float
    projected_net_after_pv_kwh: float
    pv_energy_kwh: float
    pv_curtailment_kwh: float
    triggered: bool


def _sum_device_net(ev_energy_grid_side_by_id: Dict[str, float], bes_energy_grid_side_kwh: float) -> float:
    return sum(ev_energy_grid_side_by_id.values()) + bes_energy_grid_side_kwh


def _scale_directional_devices(
    ev_energy_grid_side_by_id: Dict[str, float],
    bes_energy_grid_side_kwh: float,
    scaling_factor: float,
    overload_direction: str,
) -> tuple[Dict[str, float], float]:
    def scale_value(value: float) -> float:
        if overload_direction == "import" and value > 0.0:
            return value * scaling_factor
        if overload_direction == "export" and value < 0.0:
            return value * scaling_factor
        return value

    return (
        {ev_id: scale_value(energy) for ev_id, energy in ev_energy_grid_side_by_id.items()},
        scale_value(bes_energy_grid_side_kwh),
    )


def project_single_park_cs(park_input: ParkCSProjectionInput, cs_limit: ParkCSLimitSpec) -> ParkCSProjectionResult:
    raw_device_net = _sum_device_net(park_input.ev_energy_grid_side_by_id, park_input.bes_energy_grid_side_kwh)
    raw_net_after_pv = raw_device_net - park_input.pv_energy_kwh
    limit = cs_limit.max_exchange_energy_kwh
    triggered = abs(raw_net_after_pv) > limit + EPS
    overload_direction = "none"
    effective_pv_energy_kwh = park_input.pv_energy_kwh
    pv_curtailment_kwh = 0.0

    if raw_net_after_pv > limit + EPS:
        overload_direction = "import"
        positive_device_net = sum(max(energy, 0.0) for energy in park_input.ev_energy_grid_side_by_id.values()) + max(park_input.bes_energy_grid_side_kwh, 0.0)
        negative_device_net = sum(min(energy, 0.0) for energy in park_input.ev_energy_grid_side_by_id.values()) + min(park_input.bes_energy_grid_side_kwh, 0.0)
        target_device_net = limit + park_input.pv_energy_kwh
        retained_positive_net = target_device_net - negative_device_net
        scaling_factor = clamp(retained_positive_net / max(positive_device_net, EPS), 0.0, 1.0)
    elif raw_net_after_pv < -limit - EPS:
        overload_direction = "export"
        positive_device_net = sum(max(energy, 0.0) for energy in park_input.ev_energy_grid_side_by_id.values()) + max(park_input.bes_energy_grid_side_kwh, 0.0)
        negative_magnitude = sum(max(-energy, 0.0) for energy in park_input.ev_energy_grid_side_by_id.values()) + max(-park_input.bes_energy_grid_side_kwh, 0.0)
        pv_export_kwh = max(0.0, park_input.pv_energy_kwh - positive_device_net)
        export_pressure_kwh = negative_magnitude + pv_export_kwh
        target_device_net = -limit + park_input.pv_energy_kwh
        retained_negative_magnitude = positive_device_net - target_device_net
        retained_export_pressure_kwh = clamp(retained_negative_magnitude, 0.0, export_pressure_kwh)
        scaling_factor = clamp(retained_export_pressure_kwh / max(export_pressure_kwh, EPS), 0.0, 1.0)
        effective_pv_export_kwh = pv_export_kwh * scaling_factor
        pv_curtailment_kwh = pv_export_kwh - effective_pv_export_kwh
        effective_pv_energy_kwh = park_input.pv_energy_kwh - pv_curtailment_kwh
    elif abs(raw_device_net) <= EPS:
        scaling_factor = 1.0
    else:
        scaling_factor = 1.0

    ev_after, bes_after = _scale_directional_devices(
        park_input.ev_energy_grid_side_by_id,
        park_input.bes_energy_grid_side_kwh,
        scaling_factor,
        overload_direction,
    )
    projected_device_net = _sum_device_net(ev_after, bes_after)
    projected_net_after_pv = projected_device_net - effective_pv_energy_kwh
    return ParkCSProjectionResult(
        park_id=park_input.park_id,
        raw_ev_energy_grid_side_by_id=dict(park_input.ev_energy_grid_side_by_id),
        raw_bes_energy_grid_side_kwh=park_input.bes_energy_grid_side_kwh,
        raw_pv_energy_kwh=park_input.pv_energy_kwh,
        ev_energy_grid_side_by_id=ev_after,
        bes_energy_grid_side_kwh=bes_after,
        scaling_factor=scaling_factor,
        raw_device_net_kwh=raw_device_net,
        raw_net_after_pv_kwh=raw_net_after_pv,
        projected_device_net_kwh=projected_device_net,
        projected_net_after_pv_kwh=projected_net_after_pv,
        pv_energy_kwh=effective_pv_energy_kwh,
        pv_curtailment_kwh=pv_curtailment_kwh,
        triggered=triggered,
    )


def project_three_park_cs(
    park_inputs_by_id: Dict[str, ParkCSProjectionInput],
    cs_limits_by_id: Dict[str, ParkCSLimitSpec],
) -> Dict[str, ParkCSProjectionResult]:
    return {
        park_id: project_single_park_cs(park_inputs_by_id[park_id], cs_limits_by_id[park_id])
        for park_id in park_inputs_by_id.keys()
    }


if __name__ == "__main__":
    demo = ParkCSProjectionInput("office", {"ev1": 10.0, "ev2": -2.0}, 4.0, 6.0)
    print(project_single_park_cs(demo, ParkCSLimitSpec("office", 12.0)))
