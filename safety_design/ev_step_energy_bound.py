from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


EPS = 1e-9


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _tail_start_soc(ev_spec: "EVModelSpec", target_soc: float) -> float:
    return min(max(ev_spec.soc_tail_start, ev_spec.soc_knee), target_soc)


@dataclass(frozen=True)
class EVModelSpec:
    model_name: str
    battery_capacity_kwh: float
    p_ch_max_kw: float
    p_dis_max_kw: float
    soc_knee: float
    soc_tail_start: float
    tail_power_kw: float
    soc_min: float
    eta_ch: float
    eta_dis: float


@dataclass(frozen=True)
class EVStepState:
    ev_id: str
    soc: float
    target_soc: float
    connected: bool
    v2g_enabled: bool
    discharge_soc_floor: float = 0.0


@dataclass(frozen=True)
class CPSpec:
    cp_id: str
    max_charge_power_kw: float
    max_discharge_power_kw: float


@dataclass(frozen=True)
class EVStepEnergyBound:
    ev_id: str
    lower_bound_kwh: float
    upper_bound_kwh: float
    max_charge_energy_port_kwh: float
    max_discharge_energy_port_kwh: float
    charge_power_limit_kw: float
    discharge_power_limit_kw: float


def _charge_power_on_curve(ev_spec: EVModelSpec, soc: float, target_soc: float) -> float:
    soc = clamp(soc, 0.0, target_soc)
    tail_start_soc = _tail_start_soc(ev_spec, target_soc)
    if soc <= ev_spec.soc_knee:
        return ev_spec.p_ch_max_kw
    if soc >= tail_start_soc:
        return ev_spec.tail_power_kw
    ratio = (soc - ev_spec.soc_knee) / max(tail_start_soc - ev_spec.soc_knee, EPS)
    # 中后段采用非线性下降，直到涓流功率
    return ev_spec.tail_power_kw + (ev_spec.p_ch_max_kw - ev_spec.tail_power_kw) * (1.0 - ratio**2)


def _discharge_power_on_curve(ev_spec: EVModelSpec, soc: float) -> float:
    if soc <= ev_spec.soc_min:
        return 0.0
    return ev_spec.p_dis_max_kw


def _estimate_taper_charge_battery_energy(
    ev_spec: EVModelSpec,
    cp_spec: CPSpec,
    start_soc: float,
    target_soc: float,
    duration_hours: float,
    segments: int = 3,
) -> float:
    if duration_hours <= EPS or start_soc >= target_soc - EPS:
        return 0.0

    soc = start_soc
    total_battery_energy = 0.0
    dt = duration_hours / max(segments, 1)
    for _ in range(max(segments, 1)):
        if soc >= target_soc - EPS:
            break
        start_power_kw = min(_charge_power_on_curve(ev_spec, soc, target_soc), cp_spec.max_charge_power_kw)
        mid_soc = min(
            target_soc,
            soc + (start_power_kw * ev_spec.eta_ch * 0.5 * dt) / max(ev_spec.battery_capacity_kwh, EPS),
        )
        mid_power_kw = min(_charge_power_on_curve(ev_spec, mid_soc, target_soc), cp_spec.max_charge_power_kw)
        battery_increment = mid_power_kw * ev_spec.eta_ch * dt
        battery_room = max(0.0, (target_soc - soc) * ev_spec.battery_capacity_kwh)
        battery_increment = min(battery_increment, battery_room)
        total_battery_energy += battery_increment
        soc += battery_increment / max(ev_spec.battery_capacity_kwh, EPS)
    return total_battery_energy


def _estimate_charge_energy_port(
    ev_spec: EVModelSpec,
    cp_spec: CPSpec,
    soc: float,
    target_soc: float,
    step_hours: float,
) -> float:
    if step_hours <= EPS or soc >= target_soc - EPS:
        return 0.0

    current_power_limit_kw = min(_charge_power_on_curve(ev_spec, soc, target_soc), cp_spec.max_charge_power_kw)
    if current_power_limit_kw <= EPS:
        return 0.0

    remaining_hours = step_hours
    battery_energy = 0.0
    cc_end_soc = min(ev_spec.soc_knee, target_soc)

    if soc < cc_end_soc - EPS:
        cc_battery_power_kw = min(ev_spec.p_ch_max_kw, cp_spec.max_charge_power_kw) * ev_spec.eta_ch
        if cc_battery_power_kw > EPS:
            battery_to_knee = max(0.0, (cc_end_soc - soc) * ev_spec.battery_capacity_kwh)
            time_to_knee = battery_to_knee / cc_battery_power_kw
            if time_to_knee >= remaining_hours:
                battery_energy += cc_battery_power_kw * remaining_hours
                return battery_energy / max(ev_spec.eta_ch, EPS)
            battery_energy += battery_to_knee
            remaining_hours -= time_to_knee
            soc = cc_end_soc

    if remaining_hours > EPS and soc < target_soc - EPS:
        battery_energy += _estimate_taper_charge_battery_energy(
            ev_spec=ev_spec,
            cp_spec=cp_spec,
            start_soc=soc,
            target_soc=target_soc,
            duration_hours=remaining_hours,
        )

    return battery_energy / max(ev_spec.eta_ch, EPS)


def compute_ev_step_energy_bound(
    ev_state: EVStepState,
    ev_spec: EVModelSpec,
    cp_spec: CPSpec,
    step_hours: float,
) -> EVStepEnergyBound:
    if not ev_state.connected:
        return EVStepEnergyBound(ev_state.ev_id, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    soc = clamp(ev_state.soc, 0.0, 1.0)
    target_soc = clamp(max(ev_state.target_soc, ev_spec.soc_min), ev_spec.soc_min, 1.0)

    charge_curve_kw = _charge_power_on_curve(ev_spec, soc, target_soc)
    charge_power_limit_kw = min(charge_curve_kw, cp_spec.max_charge_power_kw)
    charge_from_power = _estimate_charge_energy_port(ev_spec, cp_spec, soc, target_soc, step_hours)
    charge_room_battery_kwh = max(0.0, (target_soc - soc) * ev_spec.battery_capacity_kwh)
    charge_room_port_kwh = charge_room_battery_kwh / max(ev_spec.eta_ch, EPS)
    max_charge_energy = min(charge_from_power, charge_room_port_kwh)

    if ev_state.v2g_enabled:
        discharge_soc_floor = clamp(max(ev_spec.soc_min, ev_state.discharge_soc_floor), ev_spec.soc_min, 1.0)
        discharge_curve_kw = _discharge_power_on_curve(ev_spec, soc)
        discharge_power_limit_kw = min(discharge_curve_kw, cp_spec.max_discharge_power_kw)
        discharge_from_power = max(0.0, discharge_power_limit_kw) * step_hours
        discharge_room_battery_kwh = max(0.0, (soc - discharge_soc_floor) * ev_spec.battery_capacity_kwh)
        discharge_room_port_kwh = discharge_room_battery_kwh * ev_spec.eta_dis
        max_discharge_energy = min(discharge_from_power, discharge_room_port_kwh)
    else:
        discharge_power_limit_kw = 0.0
        max_discharge_energy = 0.0

    return EVStepEnergyBound(
        ev_id=ev_state.ev_id,
        lower_bound_kwh=-max(0.0, max_discharge_energy),
        upper_bound_kwh=max(0.0, max_charge_energy),
        max_charge_energy_port_kwh=max(0.0, max_charge_energy),
        max_discharge_energy_port_kwh=max(0.0, max_discharge_energy),
        charge_power_limit_kw=max(0.0, charge_power_limit_kw),
        discharge_power_limit_kw=max(0.0, discharge_power_limit_kw),
    )


def map_raw_action_to_ev_energy(raw_action: float, bound: EVStepEnergyBound) -> float:
    raw_action = clamp(raw_action, -1.0, 1.0)
    if raw_action >= 0.0:
        return raw_action * bound.upper_bound_kwh
    return (-raw_action) * bound.lower_bound_kwh


def batch_compute_ev_bounds(
    ev_states: Sequence[EVStepState],
    ev_specs: Sequence[EVModelSpec],
    cp_specs: Sequence[CPSpec],
    step_hours: float,
) -> List[EVStepEnergyBound]:
    if not (len(ev_states) == len(ev_specs) == len(cp_specs)):
        raise ValueError("batch length mismatch")
    return [
        compute_ev_step_energy_bound(state, spec, cp_spec, step_hours)
        for state, spec, cp_spec in zip(ev_states, ev_specs, cp_specs)
    ]


if __name__ == "__main__":
    spec = EVModelSpec("demo", 60.0, 22.0, 11.0, 0.85, 0.95, 2.0, 0.1, 0.95, 0.95)
    state = EVStepState("ev_1", 0.5, 0.9, True, True)
    cp = CPSpec("cp_1", 11.0, 11.0)
    print(compute_ev_step_energy_bound(state, spec, cp, 0.25))
