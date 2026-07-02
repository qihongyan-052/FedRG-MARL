from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


EPS = 1e-9


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class BESModelSpec:
    park_id: str
    energy_capacity_kwh: float
    soc_min: float
    soc_max: float
    max_charge_power_kw: float
    max_discharge_power_kw: float
    eta_ch: float
    eta_dis: float
    initial_soc: float


@dataclass(frozen=True)
class BESStepState:
    park_id: str
    soc: float
    available: bool = True


@dataclass(frozen=True)
class BESStepEnergyBound:
    park_id: str
    lower_bound_kwh: float
    upper_bound_kwh: float
    max_charge_energy_port_kwh: float
    max_discharge_energy_port_kwh: float


def compute_bes_step_energy_bound(bes_state: BESStepState, bes_spec: BESModelSpec, step_hours: float) -> BESStepEnergyBound:
    if not bes_state.available:
        return BESStepEnergyBound(bes_state.park_id, 0.0, 0.0, 0.0, 0.0)
    soc = clamp(bes_state.soc, bes_spec.soc_min, bes_spec.soc_max)
    charge_power_kwh = bes_spec.max_charge_power_kw * step_hours
    discharge_power_kwh = bes_spec.max_discharge_power_kw * step_hours
    charge_room_battery_kwh = max(0.0, (bes_spec.soc_max - soc) * bes_spec.energy_capacity_kwh)
    discharge_room_battery_kwh = max(0.0, (soc - bes_spec.soc_min) * bes_spec.energy_capacity_kwh)
    charge_room_port_kwh = charge_room_battery_kwh / max(bes_spec.eta_ch, EPS)
    discharge_room_port_kwh = discharge_room_battery_kwh * bes_spec.eta_dis
    max_charge_energy = min(charge_power_kwh, charge_room_port_kwh)
    max_discharge_energy = min(discharge_power_kwh, discharge_room_port_kwh)
    return BESStepEnergyBound(
        park_id=bes_state.park_id,
        lower_bound_kwh=-max(0.0, max_discharge_energy),
        upper_bound_kwh=max(0.0, max_charge_energy),
        max_charge_energy_port_kwh=max(0.0, max_charge_energy),
        max_discharge_energy_port_kwh=max(0.0, max_discharge_energy),
    )


def map_raw_action_to_bes_energy(raw_action: float, bound: BESStepEnergyBound) -> float:
    raw_action = clamp(raw_action, -1.0, 1.0)
    if raw_action >= 0.0:
        return raw_action * bound.upper_bound_kwh
    return (-raw_action) * bound.lower_bound_kwh


def compute_three_park_bes_bounds(
    bes_states_by_park: Dict[str, BESStepState],
    bes_specs_by_park: Dict[str, BESModelSpec],
    step_hours: float,
) -> Dict[str, BESStepEnergyBound]:
    return {
        park_id: compute_bes_step_energy_bound(bes_states_by_park[park_id], bes_specs_by_park[park_id], step_hours)
        for park_id in bes_states_by_park.keys()
    }


if __name__ == "__main__":
    spec = BESModelSpec("residential", 280.0, 0.2, 0.9, 90.0, 90.0, 0.95, 0.95, 0.2)
    state = BESStepState("residential", 0.2)
    print(compute_bes_step_energy_bound(state, spec, 0.25))
