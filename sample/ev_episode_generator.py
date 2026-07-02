from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional
import json
import random

from safety_design.ev_step_energy_bound import CPSpec, EVModelSpec, EVStepState, compute_ev_step_energy_bound
from sample.arrival_soc_sampler import ArrivalSoCSampler
from sample.arrival_time_sampler import ArrivalTimeSampler
from sample.stay_time_sampler import StayDurationSampler
from sample.target_soc_sampler import TargetSoCSampler
from sample.vehicle_and_v2g_sampler import VehicleAndV2GSampler


ParkType = Literal["residential", "office", "commercial"]
PARK_TYPES: tuple[ParkType, ...] = ("residential", "office", "commercial")
EPISODE_STEPS = 96
STEP_HOURS = 0.25
TARGET_SOC_REPAIR_MARGIN = 0.005


@dataclass
class EVSession:
    ev_id: str
    park_type: ParkType
    cp_id: str | None
    admitted: bool
    arrival_step: int
    arrival_hour_abs: float
    arrival_hour_label: str
    departure_step: int
    departure_hour_abs: float
    departure_hour_label: str
    is_overnight: bool
    arrival_soc: float
    target_soc: float
    target_soc_original: float
    stay_minutes: int
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


class EpisodeEVGenerator:
    """生成单个 episode 中三园区全部 EV 完整属性，并按 CP 空位直接筛选纳入。"""

    def __init__(
        self,
        config_dir: str | Path,
        sample_dir: str | Path | None = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.sample_dir = Path(sample_dir) if sample_dir is not None else Path(__file__).resolve().parent
        self.seed = seed
        self.rng = random.Random(seed)
        self.topology = self._load_topology()
        self.arrival_sampler = ArrivalTimeSampler(seed=self._next_seed())
        self.arrival_soc_sampler = ArrivalSoCSampler(seed=self._next_seed())
        self.target_soc_sampler = TargetSoCSampler(seed=self._next_seed())
        self.stay_sampler = StayDurationSampler(seed=self._next_seed())
        self.vehicle_sampler = VehicleAndV2GSampler(
            json_path=self.config_dir / "ev_20_brand_models.json",
            seed=self._next_seed(),
        )

    def generate_episode(self) -> Dict[str, object]:
        raw_arrivals = self.arrival_sampler.sample_arrivals_for_all_parks()
        sessions_by_park: Dict[ParkType, List[EVSession]] = {park: [] for park in PARK_TYPES}
        admitted_by_step: Dict[int, Dict[str, List[Dict[str, object]]]] = {
            step: {park: [] for park in PARK_TYPES} for step in range(EPISODE_STEPS)
        }
        active_until_by_park: Dict[ParkType, Dict[str, int]] = {park: {} for park in PARK_TYPES}
        idle_cp_by_park: Dict[ParkType, List[str]] = {}
        for park in self.topology["parks"]:
            park_type = park["id"]
            idle_cp_by_park[park_type] = [f"{park_type}_cp_{idx:02d}" for idx in range(int(park["cp"]["count"]))]

        for park_type in PARK_TYPES:
            arrivals = raw_arrivals[park_type]["arrivals"]
            arrivals = sorted(arrivals, key=lambda item: (item["arrival_step"], item["arrival_hour_abs"]))
            for idx, arrival in enumerate(arrivals):
                self._release_departed_cps(
                    active_until=active_until_by_park[park_type],
                    idle_cp_ids=idle_cp_by_park[park_type],
                    current_step=int(arrival["arrival_step"]),
                )
                vehicle = self.vehicle_sampler.sample_one(park_type)
                arrival_soc = self.arrival_soc_sampler.sample_one(park_type)
                target_soc = self.target_soc_sampler.sample_one(park_type, arrival_soc=arrival_soc)
                target_soc_original = target_soc
                stay = self.stay_sampler.sample_one(
                    park_type=park_type,
                    arrival_soc=arrival_soc,
                    target_soc=target_soc_original,
                    arrival_step=int(arrival["arrival_step"]),
                    vehicle_info=asdict(vehicle),
                    v2g_willing=vehicle.v2g_enabled,
                )
                departure_step = self.stay_sampler.derive_departure_step(
                    arrival_step=int(arrival["arrival_step"]),
                    stay_minutes=stay.stay_minutes,
                    episode_total_steps=EPISODE_STEPS,
                )
                departure_hour_abs = self.stay_sampler.derive_departure_hour_abs(
                    arrival_hour_abs=float(arrival["arrival_hour_abs"]),
                    stay_minutes=stay.stay_minutes,
                    episode_total_steps=EPISODE_STEPS,
                )
                is_overnight = self.stay_sampler.is_overnight(park_type, departure_hour_abs)
                stay_minutes = stay.stay_minutes
                if is_overnight:
                    departure_hour_abs = self.stay_sampler.sample_overnight_departure_hour_abs(
                        arrival_hour_abs=float(arrival["arrival_hour_abs"]),
                    )
                    departure_step = self.stay_sampler.derive_departure_step(
                        arrival_step=int(arrival["arrival_step"]),
                        stay_minutes=int(round((departure_hour_abs - float(arrival["arrival_hour_abs"])) * 60.0)),
                        episode_total_steps=EPISODE_STEPS,
                    )
                    stay_minutes = int(round((departure_hour_abs - float(arrival["arrival_hour_abs"])) * 60.0))
                cp_id = idle_cp_by_park[park_type].pop(0) if idle_cp_by_park[park_type] else None
                admitted = cp_id is not None
                if admitted:
                    active_until_by_park[park_type][cp_id] = departure_step
                    target_soc = self._repair_target_soc_if_needed(
                        park_type=park_type,
                        cp_id=cp_id,
                        arrival_step=int(arrival["arrival_step"]),
                        departure_step=departure_step,
                        arrival_soc=arrival_soc,
                        target_soc=target_soc,
                        vehicle=vehicle,
                    )
                session = EVSession(
                    ev_id=f"{park_type[:1]}_ev_{idx:03d}",
                    park_type=park_type,
                    cp_id=cp_id,
                    admitted=admitted,
                    arrival_step=int(arrival["arrival_step"]),
                    arrival_hour_abs=float(arrival["arrival_hour_abs"]),
                    arrival_hour_label=str(arrival["arrival_hour_label"]),
                    departure_step=departure_step,
                    departure_hour_abs=departure_hour_abs,
                    departure_hour_label=self.arrival_sampler.format_hour_label(departure_hour_abs),
                    is_overnight=is_overnight,
                    arrival_soc=arrival_soc,
                    target_soc=target_soc,
                    target_soc_original=target_soc_original,
                    stay_minutes=stay_minutes,
                    model_id=vehicle.model_id,
                    size_type=vehicle.size_type,
                    battery_capacity_kwh=vehicle.battery_capacity_kwh,
                    p_ch_max_kw=vehicle.p_ch_max_kw,
                    p_dis_max_kw=vehicle.p_dis_max_kw,
                    soc_knee=vehicle.soc_knee,
                    soc_tail_start=vehicle.soc_tail_start,
                    tail_power_kw=vehicle.tail_power_kw,
                    eta_ch=vehicle.eta_ch,
                    eta_dis=vehicle.eta_dis,
                    soc_min=vehicle.soc_min,
                    v2g_capable=vehicle.v2g_capable,
                    v2g_willing_user=vehicle.v2g_willing_user,
                    v2g_enabled=vehicle.v2g_enabled,
                )
                sessions_by_park[park_type].append(session)
                if admitted:
                    admitted_by_step[session.arrival_step][park_type].append(asdict(session))

        return {
            "seed": self.seed,
            "sessions_by_park": {park: [asdict(session) for session in sessions] for park, sessions in sessions_by_park.items()},
            "admitted_by_step": admitted_by_step,
        }

    def _load_topology(self) -> Dict[str, object]:
        with open(self.config_dir / "three_parks_topology_config.json", "r", encoding="utf-8") as file:
            return json.load(file)

    def _next_seed(self) -> int:
        return self.rng.randint(0, 2**31 - 1)

    @staticmethod
    def _release_departed_cps(active_until: Dict[str, int], idle_cp_ids: List[str], current_step: int) -> None:
        released = [cp_id for cp_id, departure_step in active_until.items() if departure_step <= current_step]
        for cp_id in released:
            idle_cp_ids.append(cp_id)
            del active_until[cp_id]
        idle_cp_ids.sort()

    def _repair_target_soc_if_needed(
        self,
        park_type: ParkType,
        cp_id: str,
        arrival_step: int,
        departure_step: int,
        arrival_soc: float,
        target_soc: float,
        vehicle: object,
    ) -> float:
        reachable_soc = self._simulate_max_reachable_soc(
            park_type=park_type,
            cp_id=cp_id,
            arrival_step=arrival_step,
            departure_step=departure_step,
            arrival_soc=arrival_soc,
            target_soc=target_soc,
            vehicle=vehicle,
        )
        if reachable_soc + 1e-9 >= target_soc:
            return target_soc
        repaired_target = max(arrival_soc + 0.02, reachable_soc - TARGET_SOC_REPAIR_MARGIN)
        return round(min(target_soc, repaired_target), 4)

    def _simulate_max_reachable_soc(
        self,
        park_type: ParkType,
        cp_id: str,
        arrival_step: int,
        departure_step: int,
        arrival_soc: float,
        target_soc: float,
        vehicle: object,
    ) -> float:
        park_cfg = next(park for park in self.topology["parks"] if park["id"] == park_type)
        cp_spec = CPSpec(
            cp_id=cp_id,
            max_charge_power_kw=min(float(vehicle.p_ch_max_kw), float(park_cfg["cp"]["p_ch_max_kw"])),
            max_discharge_power_kw=min(float(vehicle.p_dis_max_kw), float(park_cfg["cp"]["p_dis_max_kw"])),
        )
        ev_spec = EVModelSpec(
            model_name=vehicle.model_id,
            battery_capacity_kwh=float(vehicle.battery_capacity_kwh),
            p_ch_max_kw=float(vehicle.p_ch_max_kw),
            p_dis_max_kw=float(vehicle.p_dis_max_kw),
            soc_knee=float(vehicle.soc_knee),
            soc_tail_start=float(vehicle.soc_tail_start),
            tail_power_kw=float(vehicle.tail_power_kw),
            soc_min=float(vehicle.soc_min),
            eta_ch=float(vehicle.eta_ch),
            eta_dis=float(vehicle.eta_dis),
        )
        soc = float(arrival_soc)
        for _ in range(max(0, departure_step - arrival_step)):
            bound = compute_ev_step_energy_bound(
                EVStepState(
                    ev_id=cp_id,
                    soc=soc,
                    target_soc=target_soc,
                    connected=True,
                    v2g_enabled=False,
                ),
                ev_spec,
                cp_spec,
                STEP_HOURS,
            )
            battery_increment_kwh = bound.max_charge_energy_port_kwh * ev_spec.eta_ch
            soc = min(target_soc, soc + battery_increment_kwh / max(ev_spec.battery_capacity_kwh, 1e-9))
        return soc


if __name__ == "__main__":
    generator = EpisodeEVGenerator(config_dir=Path(__file__).resolve().parents[1] / "config_files", seed=42)
    episode = generator.generate_episode()
    print(episode["seed"])
    print(len(episode["sessions_by_park"]["residential"]))
