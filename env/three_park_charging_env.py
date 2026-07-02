from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import csv
import json
import random

import torch

from agent.reward import EVDepartureRecord, RewardBuilder, StepRewardInput
from agent.state import StateBuilder, normalize_privacy_mode
from sample.ev_episode_generator import EpisodeEVGenerator
from sample.pv_two_zone_sampler import PVSamplerConfig, ThreeParkPVSampler
from tr_coordination.strong_privacy_coordinator import (
    EPS,
    LocalPrivacyMetrics,
    ParkPrivacyActionProfile,
    ParkPrivacyAllocation,
    ParkPrivacyProjectionResult,
    TRCoordinationBroadcast,
    TRCoordinationControlSignal,
    TRCoordinationSummary,
    compute_composite_curtailment_cost,
    compute_curtailment_preference,
    compute_global_tr_penalty,
    compute_local_curvature,
    compute_local_shrink_ratio,
    compute_local_tr_penalty,
    compute_local_tr_responsibility,
    compute_raw_overload,
    compute_tr_penalty_coefficient,
    compute_unit_cmdp_loss,
    secure_masked_sum,
)
from safety_design.bes_step_energy_bound import BESModelSpec, BESStepState, compute_three_park_bes_bounds, map_raw_action_to_bes_energy
from safety_design.cs_projection import ParkCSLimitSpec, ParkCSProjectionInput, project_three_park_cs
from safety_design.ev_step_energy_bound import CPSpec, EVModelSpec, EVStepState, compute_ev_step_energy_bound, map_raw_action_to_ev_energy
from utilities.v2g_debt import MultiParkDebtManager


ParkType = Literal["residential", "office", "commercial"]
PARK_TYPES: tuple[ParkType, ...] = ("residential", "office", "commercial")
STEP_MINUTES = 15
STEP_HOURS = STEP_MINUTES / 60.0
EPISODE_STEPS = 96


@dataclass
class ConnectedEVRuntime:
    session: Dict[str, Any]
    cp_id: str


@dataclass
class ParkRuntimeState:
    park_type: ParkType
    cp_count: int
    cp_charge_power_kw: float
    cp_discharge_power_kw: float
    cs_limit_kwh: float
    bes_spec: BESModelSpec
    bes_soc: float
    pv_kw: List[float]
    connected_evs: Dict[str, ConnectedEVRuntime]


@dataclass
class ExecutedParkFlow:
    ev_grid_energy_by_id: Dict[str, float]
    bes_grid_energy_kwh: float
    pv_energy_kwh: float
    park_grid_exchange_kwh: float
    internal_balance_residual_kwh: float


@dataclass(frozen=True)
class TRLimitSpec:
    max_exchange_energy_kwh: float


class ThreeParkChargingEnv:
    def __init__(self, root_dir: str | Path | None = None, seed: Optional[int] = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[1]
        self.config_dir = self.root_dir / "config_files"
        self.sample_dir = self.root_dir / "sample"
        self.seed = seed
        self.current_step = 0
        self.done = False

        self.topology = self._load_json(self.config_dir / "three_parks_topology_config.json")
        self.sell_price_table = self._load_park_sell_price_table()
        self.grid_price_table = self._load_grid_price_table()

        self.runtime_states: Dict[ParkType, ParkRuntimeState] = {}
        self.arrivals_by_step: Dict[int, List[Dict[str, Any]]] = {}
        self.departures_by_step: Dict[int, List[Dict[str, Any]]] = {}

        self.prev_cs_projection_stats: Dict[str, Dict[str, float]] = {}
        self.prev_tr_projection_stats: Dict[str, Dict[str, float]] = {}
        self.energy_balance_tolerance_kwh: float = 1e-6
        self.reward_consistency_tolerance: float = 1e-6

        self.privacy_mode: str = "strong"
        self.state_builder = StateBuilder(config_dir=self.config_dir, privacy_mode=self.privacy_mode)
        self.reward_builder = RewardBuilder()
        self.local_agents: Optional[Dict[str, Any]] = None
        self.bes_only_mode: bool = False
        self.transformer_overload_penalty_weight: float = 1.0
        self.tr_probe_ratio_1: float = 0.2
        self.tr_probe_ratio_2: float = 0.4
        self.tr_curvature_weight: float = 0.5
        self.use_central_tr_hgt_agent: bool = False
        self.use_strong_tr_projection_for_nonprivacy: bool = False
        self.privacy_mask_rng = random.Random(self.seed if self.seed is not None else 0)

    def attach_local_agents(self, local_agents: Optional[Dict[str, Any]]) -> None:
        self.local_agents = local_agents
        if local_agents is not None:
            for local_agent in local_agents.values():
                bind_environment = getattr(local_agent, "bind_environment", None)
                if bind_environment is not None:
                    bind_environment(self)

    def reset(self, seed: Optional[int] = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.seed = seed
        self.current_step = 0
        self.done = False

        episode_seed = self.seed if self.seed is not None else 0
        self.privacy_mask_rng = random.Random(episode_seed + 20260427)
        pv_sampler = ThreeParkPVSampler(PVSamplerConfig(csv_path="pv_4weather.csv"), seed=episode_seed + 1)
        self.daily_pv = pv_sampler.sample_day()
        episode_generator = EpisodeEVGenerator(config_dir=self.config_dir, sample_dir=self.sample_dir, seed=episode_seed + 2)
        self.daily_episode = episode_generator.generate_episode()

        self.arrivals_by_step = {step: [] for step in range(EPISODE_STEPS)}
        self.departures_by_step = {step: [] for step in range(EPISODE_STEPS)}
        for park_type in PARK_TYPES:
            for session in self.daily_episode["sessions_by_park"][park_type]:
                if self.bes_only_mode:
                    continue
                if not session["admitted"]:
                    continue
                self.arrivals_by_step[session["arrival_step"]].append(session)
                self.departures_by_step[session["departure_step"]].append(session)

        self.debt_manager = MultiParkDebtManager(self.sell_price_table)
        self.runtime_states = self._build_runtime_states()
        self._reset_projection_memory()

        event_info = self._apply_step_events(self.current_step)
        obs = self._build_state()
        return obs, {
            "weather": self.daily_pv["weather"],
            "event_info": event_info,
            "bes_only_mode": self.bes_only_mode,
        }

    def step(
        self,
        action: Optional[Dict[str, Any]] = None,
        raw_node_actions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        action = action or {}
        ev_bounds, bes_bounds = self._compute_action_bounds()
        requested_ev_grid_energy, requested_bes_grid_energy = self._decode_actions(action, ev_bounds, bes_bounds)
        cs_results = self._run_cs_projection(requested_ev_grid_energy, requested_bes_grid_energy)
        tr_summary = self._run_tr_projection(
            cs_results=cs_results,
            ev_bounds=ev_bounds,
            bes_bounds=bes_bounds,
            raw_node_actions=raw_node_actions,
        )
        transition_info = self._apply_projected_actions(tr_summary)
        self.current_step += 1
        terminated = self.current_step >= EPISODE_STEPS
        truncated = False

        if terminated:
            terminal_departure_records = self._settle_remaining_departures()
            self.done = True
            next_event_info = {
                "arrivals": [],
                "departures": [],
                "rejected_arrivals": [],
                "departure_records": terminal_departure_records,
            }
        else:
            next_event_info = self._apply_step_events(self.current_step)

        self._update_projection_memory(cs_results, tr_summary)

        park_reward_breakdown = self._build_park_reward_breakdown(
            transition_info=transition_info,
            cs_results=cs_results,
            tr_summary=tr_summary,
            departure_records=next_event_info["departure_records"],
            is_terminal_step=terminated,
        )
        aggregated_reward_breakdown = self._aggregate_park_reward_breakdown(park_reward_breakdown)
        reward = aggregated_reward_breakdown["total_reward"]
        reward_sum_consistency_error = reward - sum(
            breakdown["total_reward"] for breakdown in park_reward_breakdown.values()
        )
        if abs(reward_sum_consistency_error) > self.reward_consistency_tolerance:
            raise RuntimeError("system reward is not equal to the sum of park rewards")
        obs = self._build_state()
        info = {
            "privacy_mode": self.privacy_mode,
            "bounds": self._serialize_bounds(ev_bounds, bes_bounds),
            "cs_projection": {
                park: {
                    "triggered": result.triggered,
                    "scaling_factor": result.scaling_factor,
                    "projected_net_after_pv_kwh": result.projected_net_after_pv_kwh,
                }
                for park, result in cs_results.items()
            },
            "tr_projection": {
                "triggered": tr_summary.triggered,
                "overload_direction": tr_summary.overload_direction,
                "total_net_before_kwh": tr_summary.total_net_before_kwh,
                "total_net_after_kwh": tr_summary.total_net_after_kwh,
            },
            "privacy_coordination": {
                "broadcast": {
                    "overload_direction": tr_summary.broadcast.overload_direction,
                    "triggered": tr_summary.broadcast.triggered,
                    "total_raw_net_kwh": tr_summary.broadcast.total_raw_net_kwh,
                    "limit_kwh": tr_summary.broadcast.limit_kwh,
                    "overload_kwh": tr_summary.broadcast.overload_kwh,
                    "total_capacity_kwh": tr_summary.broadcast.total_capacity_kwh,
                    "total_preference_capacity_kwh": tr_summary.broadcast.total_preference_capacity_kwh,
                    "safety_base_ratio": tr_summary.broadcast.safety_base_ratio,
                    "blended_capacity_kwh": tr_summary.broadcast.blended_capacity_kwh,
                    "scaling_coefficient": tr_summary.broadcast.scaling_coefficient,
                    "total_responsibility": tr_summary.broadcast.total_responsibility,
                    "tr_penalty_coefficient": tr_summary.broadcast.tr_penalty_coefficient,
                    "infeasible_residual_kwh": tr_summary.broadcast.infeasible_residual_kwh,
                    "actual_total_reduction_kwh": tr_summary.broadcast.actual_total_reduction_kwh,
                },
            },
            "transition": transition_info,
            "energy_balance": transition_info["energy_balance"],
            "next_event_info": next_event_info,
            "park_reward_breakdown": park_reward_breakdown,
            "reward_breakdown": {
                **aggregated_reward_breakdown,
                "reward_sum_consistency_error": reward_sum_consistency_error,
            },
        }
        info["energy_log"] = self._build_energy_log_row(
            transition_info,
            cs_results,
            tr_summary,
            next_event_info["departure_records"],
            raw_node_actions=raw_node_actions,
            requested_bes_grid_energy=requested_bes_grid_energy,
        )
        info["reward_log"] = self._build_reward_log_row(
            transition_info,
            cs_results,
            tr_summary,
            next_event_info["departure_records"],
            park_reward_breakdown,
            terminated,
        )
        info["training_reward_log"] = self._build_training_reward_log_row(park_reward_breakdown)
        info["projection_trace"] = self._build_projection_trace_row(cs_results, tr_summary)
        return obs, reward, terminated, truncated, info

    def _build_runtime_states(self) -> Dict[ParkType, ParkRuntimeState]:
        runtime_states: Dict[ParkType, ParkRuntimeState] = {}
        pv_by_park = self.daily_pv["park_pv_kw"]
        tr_limit_kwh = float(self.topology["tr"]["p_limit_kw"]) * STEP_HOURS
        self.tr_limit = TRLimitSpec(max_exchange_energy_kwh=tr_limit_kwh)

        for park_cfg in self.topology["parks"]:
            park_type = park_cfg["id"]
            bes_cfg = park_cfg["bes"]
            bes_spec = BESModelSpec(
                park_id=park_type,
                energy_capacity_kwh=float(bes_cfg["cap_kwh"]),
                soc_min=float(bes_cfg["soc_min"]),
                soc_max=float(bes_cfg["soc_max"]),
                max_charge_power_kw=float(bes_cfg["p_ch_kw"]),
                max_discharge_power_kw=float(bes_cfg["p_dis_kw"]),
                eta_ch=float(bes_cfg["eta_ch"]),
                eta_dis=float(bes_cfg["eta_dis"]),
                initial_soc=float(bes_cfg.get("soc0", bes_cfg["soc_min"])),
            )
            runtime_states[park_type] = ParkRuntimeState(
                park_type=park_type,
                cp_count=int(park_cfg["cp"]["count"]),
                cp_charge_power_kw=float(park_cfg["cp"]["p_ch_max_kw"]),
                cp_discharge_power_kw=float(park_cfg["cp"]["p_dis_max_kw"]),
                cs_limit_kwh=float(park_cfg["cs"]["p_limit_kw"]) * STEP_HOURS,
                bes_spec=bes_spec,
                bes_soc=bes_spec.initial_soc,
                pv_kw=pv_by_park[park_type],
                connected_evs={},
            )
        return runtime_states

    def _apply_step_events(self, step: int) -> Dict[str, Any]:
        event_info = {"arrivals": [], "departures": [], "rejected_arrivals": [], "departure_records": []}

        for session in self.departures_by_step.get(step, []):
            park_state = self.runtime_states[session["park_type"]]
            if session["ev_id"] not in park_state.connected_evs:
                continue
            settlement = self.debt_manager.settle_departure(session["ev_id"], step)
            event_info["departures"].append(
                {
                    "ev_id": session["ev_id"],
                    "park_type": session["park_type"],
                    "total_penalty": settlement.total_penalty,
                }
            )
            event_info["departure_records"].append(
                {
                    "ev_id": session["ev_id"],
                    "park_type": session["park_type"],
                    "soc_at_departure": settlement.soc_at_departure,
                    "target_soc": settlement.target_departure_soc,
                    "debt_remaining_kwh": settlement.debt_remaining_kwh,
                    "soc_shortfall_kwh": settlement.soc_shortfall_kwh,
                    "debt_penalty": settlement.debt_penalty,
                    "soc_shortfall_penalty": settlement.soc_shortfall_penalty,
                }
            )
            del park_state.connected_evs[session["ev_id"]]
            self.debt_manager.remove_ev(session["ev_id"])

        for session in self.arrivals_by_step.get(step, []):
            park_state = self.runtime_states[session["park_type"]]
            if len(park_state.connected_evs) >= park_state.cp_count:
                event_info["rejected_arrivals"].append({"ev_id": session["ev_id"], "park_type": session["park_type"]})
                continue
            park_state.connected_evs[session["ev_id"]] = ConnectedEVRuntime(session=session, cp_id=session["cp_id"])
            self.debt_manager.register_ev(
                ev_id=session["ev_id"],
                park_type=session["park_type"],
                battery_capacity_kwh=session["battery_capacity_kwh"],
                current_soc=session["arrival_soc"],
                target_departure_soc=session["target_soc"],
                min_soc=session["soc_min"],
                departure_step=session["departure_step"],
            )
            event_info["arrivals"].append(
                {
                    "ev_id": session["ev_id"],
                    "park_type": session["park_type"],
                    "cp_id": session["cp_id"],
                }
            )
        return event_info

    def _compute_action_bounds(self) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        ev_bounds: Dict[str, Dict[str, Any]] = {park: {} for park in PARK_TYPES}
        bes_states: Dict[str, BESStepState] = {}
        bes_specs: Dict[str, BESModelSpec] = {}

        for park_type, park_state in self.runtime_states.items():
            bes_states[park_type] = BESStepState(park_id=park_type, soc=park_state.bes_soc, available=True)
            bes_specs[park_type] = park_state.bes_spec
            for ev_id, runtime in park_state.connected_evs.items():
                session = runtime.session
                account = self.debt_manager.get_account(ev_id)
                debt_cfg = self.debt_manager.park_configs[park_type]
                v2g_enabled = session["v2g_enabled"]
                ev_spec = EVModelSpec(
                    model_name=session["model_id"],
                    battery_capacity_kwh=session["battery_capacity_kwh"],
                    p_ch_max_kw=session["p_ch_max_kw"],
                    p_dis_max_kw=session["p_dis_max_kw"],
                    soc_knee=session["soc_knee"],
                    soc_tail_start=session["soc_tail_start"],
                    tail_power_kw=session["tail_power_kw"],
                    soc_min=session["soc_min"],
                    eta_ch=session["eta_ch"],
                    eta_dis=session["eta_dis"],
                )
                ev_state = EVStepState(
                    ev_id=ev_id,
                    soc=account.current_soc,
                    target_soc=session["target_soc"],
                    connected=True,
                    v2g_enabled=v2g_enabled,
                    discharge_soc_floor=min(1.0, session["soc_min"] + debt_cfg.min_soc_margin_for_v2g),
                )
                cp_spec = CPSpec(
                    cp_id=runtime.cp_id,
                    max_charge_power_kw=min(session["p_ch_max_kw"], park_state.cp_charge_power_kw),
                    max_discharge_power_kw=min(session["p_dis_max_kw"], park_state.cp_discharge_power_kw),
                )
                ev_bounds[park_type][ev_id] = compute_ev_step_energy_bound(ev_state, ev_spec, cp_spec, STEP_HOURS)

        bes_bounds = compute_three_park_bes_bounds(bes_states, bes_specs, STEP_HOURS)
        return ev_bounds, bes_bounds

    def _decode_actions(
        self,
        action: Dict[str, Any],
        ev_bounds: Dict[str, Dict[str, Any]],
        bes_bounds: Dict[str, Any],
    ) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        requested_ev_grid_energy: Dict[str, Dict[str, float]] = {park: {} for park in PARK_TYPES}
        requested_bes_grid_energy: Dict[str, float] = {park: 0.0 for park in PARK_TYPES}
        park_actions = action.get("parks", {})

        for park_type in PARK_TYPES:
            park_action = park_actions.get(park_type, {})
            ev_action = park_action.get("ev", {})
            for ev_id, bound in ev_bounds[park_type].items():
                requested_ev_grid_energy[park_type][ev_id] = map_raw_action_to_ev_energy(float(ev_action.get(ev_id, 0.0)), bound)
            requested_bes_grid_energy[park_type] = map_raw_action_to_bes_energy(float(park_action.get("bes", 0.0)), bes_bounds[park_type])

        return requested_ev_grid_energy, requested_bes_grid_energy

    def _run_cs_projection(
        self,
        requested_ev_grid_energy: Dict[str, Dict[str, float]],
        requested_bes_grid_energy: Dict[str, float],
    ) -> Dict[str, Any]:
        cs_inputs = {}
        cs_limits = {}
        for park_type, park_state in self.runtime_states.items():
            cs_inputs[park_type] = ParkCSProjectionInput(
                park_id=park_type,
                ev_energy_grid_side_by_id=requested_ev_grid_energy[park_type],
                bes_energy_grid_side_kwh=requested_bes_grid_energy[park_type],
                pv_energy_kwh=park_state.pv_kw[self.current_step] * STEP_HOURS,
            )
            cs_limits[park_type] = ParkCSLimitSpec(park_id=park_type, max_exchange_energy_kwh=park_state.cs_limit_kwh)
        return project_three_park_cs(cs_inputs, cs_limits)

    def _run_tr_projection(
        self,
        cs_results: Dict[str, Any],
        ev_bounds: Dict[str, Dict[str, Any]],
        bes_bounds: Dict[str, Any],
        raw_node_actions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> TRCoordinationSummary:
        if self.use_central_tr_hgt_agent:
            return self._run_central_global_scaling_tr_projection(
                cs_results=cs_results,
            )
        if (
            normalize_privacy_mode(self.privacy_mode) == "none"
            and not self.use_strong_tr_projection_for_nonprivacy
        ):
            return self._run_nonprivacy_tr_projection(
                cs_results=cs_results,
                ev_bounds=ev_bounds,
                bes_bounds=bes_bounds,
                raw_node_actions=raw_node_actions,
            )
        return self._run_strong_privacy_tr_projection(
            cs_results=cs_results,
            ev_bounds=ev_bounds,
            bes_bounds=bes_bounds,
            raw_node_actions=raw_node_actions,
        )

    def _run_strong_privacy_tr_projection(
        self,
        cs_results: Dict[str, Any],
        ev_bounds: Dict[str, Dict[str, Any]],
        bes_bounds: Dict[str, Any],
        raw_node_actions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> TRCoordinationSummary:
        raw_total_net_kwh = self._secure_aggregate_by_park(
            {
                park_type: result.projected_net_after_pv_kwh
                for park_type, result in cs_results.items()
            }
        )
        overload_kwh = compute_raw_overload(raw_total_net_kwh, self.tr_limit.max_exchange_energy_kwh)
        triggered = overload_kwh > self.reward_consistency_tolerance
        overload_direction = "import" if raw_total_net_kwh > 0.0 else "export"
        profiles_by_park = {
            park_type: self._build_privacy_action_profile(
                park_type=park_type,
                cs_result=cs_results[park_type],
                overload_direction=overload_direction,
            )
            for park_type in PARK_TYPES
        }

        if not triggered:
            return self._build_non_triggered_privacy_summary(cs_results, profiles_by_park, raw_total_net_kwh)

        responsibility_by_park = self._compute_local_tr_responsibility_by_park(
            cs_results=cs_results,
            overload_direction=overload_direction,
        )
        local_metrics = self._compute_local_preliminary_privacy_metrics(
            profiles_by_park=profiles_by_park,
            cs_results=cs_results,
            ev_bounds=ev_bounds,
            bes_bounds=bes_bounds,
            overload_direction=overload_direction,
            raw_node_actions=raw_node_actions,
        )
        total_capacity_kwh = self._secure_aggregate_by_park(
            {
                park_type: metric.same_direction_controllable_kwh
                for park_type, metric in local_metrics.items()
            }
        )
        total_preference_capacity_kwh = self._secure_aggregate_by_park(
            {
                park_type: metric.preference_capacity_kwh
                for park_type, metric in local_metrics.items()
            }
        )
        responsibility_by_park = self._compute_local_tr_responsibility_by_park(
            cs_results=cs_results,
            overload_direction=overload_direction,
        )
        total_responsibility = self._secure_aggregate_by_park(responsibility_by_park)
        broadcast = self._compute_operator_broadcast(
            raw_total_net_kwh=raw_total_net_kwh,
            overload_direction=overload_direction,
            overload_kwh=overload_kwh,
            total_capacity_kwh=total_capacity_kwh,
            total_preference_capacity_kwh=total_preference_capacity_kwh,
            total_responsibility=total_responsibility,
        )
        control_signal = self._build_control_signal(broadcast)
        allocations, park_results = self._compute_local_allocations_and_results(
            profiles_by_park=profiles_by_park,
            local_metrics=local_metrics,
            cs_results=cs_results,
            control_signal=control_signal,
            overload_kwh=overload_kwh,
        )
        return TRCoordinationSummary(
            broadcast=broadcast,
            local_metrics_by_park=local_metrics,
            allocations_by_park=allocations,
            park_results_by_id=park_results,
        )

    def _run_central_global_scaling_tr_projection(
        self,
        cs_results: Dict[str, Any],
    ) -> TRCoordinationSummary:
        raw_total_net_kwh = sum(
            result.projected_net_after_pv_kwh
            for result in cs_results.values()
        )
        overload_kwh = compute_raw_overload(raw_total_net_kwh, self.tr_limit.max_exchange_energy_kwh)
        triggered = overload_kwh > self.reward_consistency_tolerance
        overload_direction = "import" if raw_total_net_kwh > 0.0 else "export"
        profiles_by_park = {
            park_type: self._build_privacy_action_profile(
                park_type=park_type,
                cs_result=cs_results[park_type],
                overload_direction=overload_direction,
            )
            for park_type in PARK_TYPES
        }

        if not triggered:
            return self._build_non_triggered_privacy_summary(cs_results, profiles_by_park, raw_total_net_kwh)

        responsibility_by_park = self._compute_local_tr_responsibility_by_park(
            cs_results=cs_results,
            overload_direction=overload_direction,
        )
        total_capacity_kwh = sum(
            profile.same_direction_controllable_kwh
            for profile in profiles_by_park.values()
        )
        total_responsibility = sum(max(0.0, responsibility_by_park[park_type]) for park_type in PARK_TYPES)
        broadcast = self._compute_operator_broadcast(
            raw_total_net_kwh=raw_total_net_kwh,
            overload_direction=overload_direction,
            overload_kwh=overload_kwh,
            total_capacity_kwh=total_capacity_kwh,
            total_preference_capacity_kwh=total_capacity_kwh,
            total_responsibility=total_responsibility,
        )
        shrink_ratio = compute_local_shrink_ratio(
            curtailment_kwh=overload_kwh,
            same_direction_controllable_kwh=total_capacity_kwh,
        )
        control_signal = self._build_control_signal(broadcast)
        local_metrics: Dict[str, LocalPrivacyMetrics] = {}
        allocations: Dict[str, ParkPrivacyAllocation] = {}
        park_results: Dict[str, ParkPrivacyProjectionResult] = {}
        for park_type in PARK_TYPES:
            same_direction_controllable_kwh = profiles_by_park[park_type].same_direction_controllable_kwh
            curtailment_kwh = min(
                overload_kwh * (same_direction_controllable_kwh / max(total_capacity_kwh, EPS)),
                same_direction_controllable_kwh,
            )
            capacity_share = (
                same_direction_controllable_kwh / max(total_capacity_kwh, EPS)
                if total_capacity_kwh > self.reward_consistency_tolerance
                else 0.0
            )
            local_metrics[park_type] = LocalPrivacyMetrics(
                park_id=park_type,
                same_direction_controllable_kwh=same_direction_controllable_kwh,
                score_raw=0.0,
                score_probe_20=0.0,
                score_probe_40=0.0,
                probe_curtailment_20_kwh=0.0,
                probe_curtailment_40_kwh=0.0,
                unit_cmdp_loss=0.0,
                local_curvature=0.0,
                composite_curtailment_cost=0.0,
                curtailment_preference=capacity_share,
                preference_capacity_kwh=same_direction_controllable_kwh,
                final_mixing_weight=capacity_share,
            )
            allocations[park_type] = ParkPrivacyAllocation(
                park_id=park_type,
                same_direction_controllable_kwh=same_direction_controllable_kwh,
                final_mixing_weight=capacity_share,
                curtailment_kwh=curtailment_kwh,
                pv_curtailment_kwh=0.0,
                shrink_ratio=shrink_ratio,
            )
            park_results[park_type] = self._project_local_privacy_result(
                park_type=park_type,
                cs_result=cs_results[park_type],
                allocation=allocations[park_type],
                control_signal=control_signal,
            )
        return TRCoordinationSummary(
            broadcast=broadcast,
            local_metrics_by_park=local_metrics,
            allocations_by_park=allocations,
            park_results_by_id=park_results,
        )

    def _run_nonprivacy_tr_projection(
        self,
        cs_results: Dict[str, Any],
        ev_bounds: Dict[str, Dict[str, Any]],
        bes_bounds: Dict[str, Any],
        raw_node_actions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> TRCoordinationSummary:
        raw_total_net_kwh = sum(
            result.projected_net_after_pv_kwh
            for result in cs_results.values()
        )
        overload_kwh = compute_raw_overload(raw_total_net_kwh, self.tr_limit.max_exchange_energy_kwh)
        triggered = overload_kwh > self.reward_consistency_tolerance
        overload_direction = "import" if raw_total_net_kwh > 0.0 else "export"
        profiles_by_park = {
            park_type: self._build_privacy_action_profile(
                park_type=park_type,
                cs_result=cs_results[park_type],
                overload_direction=overload_direction,
            )
            for park_type in PARK_TYPES
        }

        if not triggered:
            return self._build_non_triggered_privacy_summary(cs_results, profiles_by_park, raw_total_net_kwh)

        responsibility_by_park = self._compute_local_tr_responsibility_by_park(
            cs_results=cs_results,
            overload_direction=overload_direction,
        )
        local_metrics = self._build_nonprivacy_local_metrics(
            profiles_by_park=profiles_by_park,
            responsibility_by_park=responsibility_by_park,
        )
        total_capacity_kwh = sum(
            metric.same_direction_controllable_kwh
            for metric in local_metrics.values()
        )
        total_preference_capacity_kwh = total_capacity_kwh
        total_responsibility = sum(responsibility_by_park.values())
        broadcast = self._compute_operator_broadcast(
            raw_total_net_kwh=raw_total_net_kwh,
            overload_direction=overload_direction,
            overload_kwh=overload_kwh,
            total_capacity_kwh=total_capacity_kwh,
            total_preference_capacity_kwh=total_preference_capacity_kwh,
            total_responsibility=total_responsibility,
        )
        allocations = self._compute_nonprivacy_allocations(
            profiles_by_park=profiles_by_park,
            responsibility_by_park=responsibility_by_park,
            overload_kwh=overload_kwh,
        )
        control_signal = self._build_control_signal(broadcast)
        park_results = {
            park_type: self._project_local_privacy_result(
                park_type=park_type,
                cs_result=cs_results[park_type],
                allocation=allocations[park_type],
                control_signal=control_signal,
            )
            for park_type in PARK_TYPES
        }
        return TRCoordinationSummary(
            broadcast=broadcast,
            local_metrics_by_park=local_metrics,
            allocations_by_park=allocations,
            park_results_by_id=park_results,
        )

    def _build_nonprivacy_local_metrics(
        self,
        profiles_by_park: Dict[str, ParkPrivacyActionProfile],
        responsibility_by_park: Dict[str, float],
    ) -> Dict[str, LocalPrivacyMetrics]:
        total_responsibility = sum(responsibility_by_park.values())
        metrics: Dict[str, LocalPrivacyMetrics] = {}
        for park_type in PARK_TYPES:
            same_direction_controllable_kwh = profiles_by_park[park_type].same_direction_controllable_kwh
            responsibility_share = (
                responsibility_by_park[park_type] / max(total_responsibility, EPS)
                if total_responsibility > self.reward_consistency_tolerance
                else 0.0
            )
            metrics[park_type] = LocalPrivacyMetrics(
                park_id=park_type,
                same_direction_controllable_kwh=same_direction_controllable_kwh,
                score_raw=0.0,
                score_probe_20=0.0,
                score_probe_40=0.0,
                probe_curtailment_20_kwh=0.0,
                probe_curtailment_40_kwh=0.0,
                unit_cmdp_loss=0.0,
                local_curvature=0.0,
                composite_curtailment_cost=0.0,
                curtailment_preference=responsibility_share,
                preference_capacity_kwh=same_direction_controllable_kwh,
                final_mixing_weight=responsibility_share,
            )
        return metrics

    def _build_non_triggered_privacy_summary(
        self,
        cs_results: Dict[str, Any],
        profiles_by_park: Dict[str, ParkPrivacyActionProfile],
        raw_total_net_kwh: float,
    ) -> TRCoordinationSummary:
        broadcast = TRCoordinationBroadcast(
            overload_direction="none",
            triggered=False,
            total_raw_net_kwh=raw_total_net_kwh,
            limit_kwh=self.tr_limit.max_exchange_energy_kwh,
            overload_kwh=0.0,
            total_capacity_kwh=0.0,
            total_preference_capacity_kwh=0.0,
            safety_base_ratio=0.0,
            blended_capacity_kwh=0.0,
            scaling_coefficient=0.0,
            total_responsibility=0.0,
            tr_penalty_coefficient=0.0,
            infeasible_residual_kwh=0.0,
            actual_total_reduction_kwh=0.0,
        )
        local_metrics = {
            park_type: LocalPrivacyMetrics(
                park_id=park_type,
                same_direction_controllable_kwh=profiles_by_park[park_type].same_direction_controllable_kwh,
                score_raw=0.0,
                score_probe_20=0.0,
                score_probe_40=0.0,
                probe_curtailment_20_kwh=0.0,
                probe_curtailment_40_kwh=0.0,
                unit_cmdp_loss=0.0,
                local_curvature=0.0,
                composite_curtailment_cost=0.0,
                curtailment_preference=0.0,
                preference_capacity_kwh=0.0,
                final_mixing_weight=0.0,
            )
            for park_type in PARK_TYPES
        }
        allocations = {
            park_type: ParkPrivacyAllocation(
                park_id=park_type,
                same_direction_controllable_kwh=profiles_by_park[park_type].same_direction_controllable_kwh,
                final_mixing_weight=0.0,
                curtailment_kwh=0.0,
                pv_curtailment_kwh=0.0,
                shrink_ratio=0.0,
            )
            for park_type in PARK_TYPES
        }
        park_results = {
            park_type: ParkPrivacyProjectionResult(
                park_id=park_type,
                ev_energy_grid_side_by_id=dict(cs_results[park_type].ev_energy_grid_side_by_id),
                bes_energy_grid_side_kwh=cs_results[park_type].bes_energy_grid_side_kwh,
                pv_effective_energy_kwh=cs_results[park_type].pv_energy_kwh,
                pv_curtailment_kwh=0.0,
                scaling_factor=1.0,
                projected_park_net_kwh=cs_results[park_type].projected_net_after_pv_kwh,
                reduction_kwh=0.0,
                triggered=False,
            )
            for park_type in PARK_TYPES
        }
        return TRCoordinationSummary(
            broadcast=broadcast,
            local_metrics_by_park=local_metrics,
            allocations_by_park=allocations,
            park_results_by_id=park_results,
        )

    def _compute_local_preliminary_privacy_metrics(
        self,
        profiles_by_park: Dict[str, ParkPrivacyActionProfile],
        cs_results: Dict[str, Any],
        ev_bounds: Dict[str, Dict[str, Any]],
        bes_bounds: Dict[str, Any],
        overload_direction: str,
        raw_node_actions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, LocalPrivacyMetrics]:
        local_pre_metrics: Dict[str, LocalPrivacyMetrics] = {}
        for park_type in PARK_TYPES:
            profile = profiles_by_park[park_type]
            (
                score_raw,
                score_probe_20,
                score_probe_40,
                probe_curtailment_20_kwh,
                probe_curtailment_40_kwh,
            ) = self._evaluate_privacy_guidance(
                park_type=park_type,
                profile=profile,
                cs_result=cs_results[park_type],
                ev_bounds=ev_bounds,
                bes_bounds=bes_bounds,
                overload_direction=overload_direction,
                base_node_action=None if raw_node_actions is None else raw_node_actions.get(park_type),
            )
            unit_cmdp_loss = compute_unit_cmdp_loss(
                score_raw=score_raw,
                score_probe=score_probe_20,
                probe_curtailment_kwh=probe_curtailment_20_kwh,
            )
            loss_probe_20 = max(0.0, score_raw - score_probe_20)
            loss_probe_40 = max(0.0, score_raw - score_probe_40)
            local_curvature = compute_local_curvature(
                loss_probe_20=loss_probe_20,
                loss_probe_40=loss_probe_40,
                probe_curtailment_20_kwh=probe_curtailment_20_kwh,
            )
            composite_curtailment_cost = compute_composite_curtailment_cost(
                unit_cmdp_loss=unit_cmdp_loss,
                local_curvature=local_curvature,
                same_direction_controllable_kwh=profile.same_direction_controllable_kwh,
                curvature_weight=self.tr_curvature_weight,
            )
            curtailment_preference = compute_curtailment_preference(
                composite_curtailment_cost=composite_curtailment_cost,
            )
            local_pre_metrics[park_type] = LocalPrivacyMetrics(
                park_id=park_type,
                same_direction_controllable_kwh=profile.same_direction_controllable_kwh,
                score_raw=score_raw,
                score_probe_20=score_probe_20,
                score_probe_40=score_probe_40,
                probe_curtailment_20_kwh=probe_curtailment_20_kwh,
                probe_curtailment_40_kwh=probe_curtailment_40_kwh,
                unit_cmdp_loss=unit_cmdp_loss,
                local_curvature=local_curvature,
                composite_curtailment_cost=composite_curtailment_cost,
                curtailment_preference=curtailment_preference,
                preference_capacity_kwh=profile.same_direction_controllable_kwh * curtailment_preference,
                final_mixing_weight=curtailment_preference,
            )
        return local_pre_metrics

    def _compute_operator_broadcast(
        self,
        raw_total_net_kwh: float,
        overload_direction: str,
        overload_kwh: float,
        total_capacity_kwh: float,
        total_preference_capacity_kwh: float,
        total_responsibility: float,
    ) -> TRCoordinationBroadcast:
        infeasible = total_capacity_kwh + self.reward_consistency_tolerance < overload_kwh
        infeasible_residual_kwh = max(0.0, overload_kwh - total_capacity_kwh)
        if infeasible or total_capacity_kwh <= self.reward_consistency_tolerance:
            safety_base_ratio = 1.0 if total_capacity_kwh > self.reward_consistency_tolerance else 0.0
            blended_capacity_kwh = total_capacity_kwh
            scaling_coefficient = 1.0 if total_capacity_kwh > self.reward_consistency_tolerance else 0.0
            actual_total_reduction_kwh = min(overload_kwh, total_capacity_kwh)
        else:
            safety_base_ratio = overload_kwh / max(total_capacity_kwh, EPS)
            blended_capacity_kwh = (
                safety_base_ratio * total_capacity_kwh
                + (1.0 - safety_base_ratio) * total_preference_capacity_kwh
            )
            scaling_coefficient = overload_kwh / max(blended_capacity_kwh, EPS)
            actual_total_reduction_kwh = overload_kwh
        return TRCoordinationBroadcast(
            overload_direction=overload_direction,
            triggered=True,
            total_raw_net_kwh=raw_total_net_kwh,
            limit_kwh=self.tr_limit.max_exchange_energy_kwh,
            overload_kwh=overload_kwh,
            total_capacity_kwh=total_capacity_kwh,
            total_preference_capacity_kwh=total_preference_capacity_kwh,
            safety_base_ratio=safety_base_ratio,
            blended_capacity_kwh=blended_capacity_kwh,
            scaling_coefficient=scaling_coefficient,
            total_responsibility=total_responsibility,
            tr_penalty_coefficient=compute_tr_penalty_coefficient(
                penalty_weight=self.transformer_overload_penalty_weight,
                overload_kwh=overload_kwh,
                total_responsibility=total_responsibility,
            ),
            infeasible_residual_kwh=infeasible_residual_kwh,
            actual_total_reduction_kwh=actual_total_reduction_kwh,
        )

    @staticmethod
    def _build_control_signal(broadcast: TRCoordinationBroadcast) -> TRCoordinationControlSignal:
        return TRCoordinationControlSignal(
            triggered=broadcast.triggered,
            overload_direction=broadcast.overload_direction,
            safety_base_ratio=broadcast.safety_base_ratio,
            scaling_coefficient=broadcast.scaling_coefficient,
            infeasible_residual_kwh=broadcast.infeasible_residual_kwh,
            tr_penalty_coefficient=broadcast.tr_penalty_coefficient,
        )

    def _compute_local_allocations_and_results(
        self,
        profiles_by_park: Dict[str, ParkPrivacyActionProfile],
        local_metrics: Dict[str, LocalPrivacyMetrics],
        cs_results: Dict[str, Any],
        control_signal: TRCoordinationControlSignal,
        overload_kwh: float,
    ) -> tuple[Dict[str, ParkPrivacyAllocation], Dict[str, ParkPrivacyProjectionResult]]:
        allocations: Dict[str, ParkPrivacyAllocation] = {}
        for park_type in PARK_TYPES:
            profile = profiles_by_park[park_type]
            metric = local_metrics[park_type]
            allocation = self._compute_local_privacy_allocation(
                park_type=park_type,
                profile=profile,
                metric=metric,
                control_signal=control_signal,
            )
            allocations[park_type] = allocation

        expected_total_curtailment_kwh = max(
            0.0,
            overload_kwh - control_signal.infeasible_residual_kwh,
        )
        allocated_total_curtailment_kwh = sum(
            allocation.curtailment_kwh
            for allocation in allocations.values()
        )
        self._validate_privacy_curtailment_total(
            stage="allocation",
            actual_total_curtailment_kwh=allocated_total_curtailment_kwh,
            expected_total_curtailment_kwh=expected_total_curtailment_kwh,
        )

        park_results: Dict[str, ParkPrivacyProjectionResult] = {}
        for park_type in PARK_TYPES:
            allocation = allocations[park_type]
            park_results[park_type] = self._project_local_privacy_result(
                park_type=park_type,
                cs_result=cs_results[park_type],
                allocation=allocation,
                control_signal=control_signal,
            )
        projected_total_curtailment_kwh = sum(
            abs(
                profiles_by_park[park_type].raw_net_kwh
                - park_results[park_type].projected_park_net_kwh
            )
            for park_type in PARK_TYPES
        )
        self._validate_privacy_curtailment_total(
            stage="projection",
            actual_total_curtailment_kwh=projected_total_curtailment_kwh,
            expected_total_curtailment_kwh=expected_total_curtailment_kwh,
        )
        return allocations, park_results

    def _validate_privacy_curtailment_total(
        self,
        stage: str,
        actual_total_curtailment_kwh: float,
        expected_total_curtailment_kwh: float,
    ) -> None:
        # Secure aggregation rounds to 1e-6 kWh before summation.
        tolerance_kwh = max(self.reward_consistency_tolerance, 1e-5)
        residual_kwh = actual_total_curtailment_kwh - expected_total_curtailment_kwh
        if abs(residual_kwh) > tolerance_kwh:
            raise RuntimeError(
                "strong-privacy TR curtailment total mismatch: "
                f"stage={stage}, actual={actual_total_curtailment_kwh}, "
                f"expected={expected_total_curtailment_kwh}, residual={residual_kwh}"
            )

    def _compute_local_privacy_allocation(
        self,
        park_type: ParkType,
        profile: ParkPrivacyActionProfile,
        metric: LocalPrivacyMetrics,
        control_signal: TRCoordinationControlSignal,
    ) -> ParkPrivacyAllocation:
        if profile.same_direction_controllable_kwh <= self.reward_consistency_tolerance:
            final_mixing_weight = 0.0
            curtailment_kwh = 0.0
        elif control_signal.infeasible_residual_kwh > self.reward_consistency_tolerance:
            final_mixing_weight = 1.0
            curtailment_kwh = profile.same_direction_controllable_kwh
        else:
            final_mixing_weight = (
                control_signal.safety_base_ratio
                + (1.0 - control_signal.safety_base_ratio) * metric.curtailment_preference
            )
            curtailment_kwh = min(
                control_signal.scaling_coefficient
                * profile.same_direction_controllable_kwh
                * final_mixing_weight,
                profile.same_direction_controllable_kwh,
            )
        shrink_ratio = compute_local_shrink_ratio(
            curtailment_kwh=curtailment_kwh,
            same_direction_controllable_kwh=profile.same_direction_controllable_kwh,
        )
        return ParkPrivacyAllocation(
            park_id=park_type,
            same_direction_controllable_kwh=profile.same_direction_controllable_kwh,
            final_mixing_weight=final_mixing_weight,
            curtailment_kwh=curtailment_kwh,
            pv_curtailment_kwh=0.0,
            shrink_ratio=shrink_ratio,
        )

    def _project_local_privacy_result(
        self,
        park_type: ParkType,
        cs_result: Any,
        allocation: ParkPrivacyAllocation,
        control_signal: TRCoordinationControlSignal,
    ) -> ParkPrivacyProjectionResult:
        return self._apply_privacy_curtailment_to_park(
            park_type=park_type,
            cs_result=cs_result,
            allocation=allocation,
            overload_direction=control_signal.overload_direction,
            triggered=control_signal.triggered,
        )

    def _build_privacy_action_profile(
        self,
        park_type: ParkType,
        cs_result: Any,
        overload_direction: str,
    ) -> ParkPrivacyActionProfile:
        ev_charge_kwh = sum(max(energy, 0.0) for energy in cs_result.ev_energy_grid_side_by_id.values())
        ev_discharge_kwh = sum(max(-energy, 0.0) for energy in cs_result.ev_energy_grid_side_by_id.values())
        bes_charge_kwh = max(cs_result.bes_energy_grid_side_kwh, 0.0)
        bes_discharge_kwh = max(-cs_result.bes_energy_grid_side_kwh, 0.0)
        positive_device_net = ev_charge_kwh + bes_charge_kwh
        pv_export_kwh = max(0.0, cs_result.pv_energy_kwh - positive_device_net)
        if overload_direction == "import":
            same_direction_controllable_kwh = ev_charge_kwh + bes_charge_kwh
        else:
            same_direction_controllable_kwh = ev_discharge_kwh + bes_discharge_kwh + pv_export_kwh
        return ParkPrivacyActionProfile(
            park_id=park_type,
            raw_net_kwh=cs_result.projected_net_after_pv_kwh,
            same_direction_controllable_kwh=same_direction_controllable_kwh,
            ev_charge_kwh=ev_charge_kwh,
            ev_discharge_kwh=ev_discharge_kwh,
            bes_charge_kwh=bes_charge_kwh,
            bes_discharge_kwh=bes_discharge_kwh,
            pv_export_kwh=pv_export_kwh,
        )

    def _evaluate_privacy_guidance(
        self,
        park_type: ParkType,
        profile: ParkPrivacyActionProfile,
        cs_result: Any,
        ev_bounds: Dict[str, Dict[str, Any]],
        bes_bounds: Dict[str, Any],
        overload_direction: str,
        base_node_action: Optional[torch.Tensor] = None,
    ) -> tuple[float, float, float, float, float]:
        if profile.same_direction_controllable_kwh <= self.reward_consistency_tolerance:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        if self.local_agents is None or park_type not in self.local_agents:
            probe_20 = self.tr_probe_ratio_1 * profile.same_direction_controllable_kwh
            probe_40 = self.tr_probe_ratio_2 * profile.same_direction_controllable_kwh
            return 0.0, 0.0, 0.0, probe_20, probe_40

        local_agent = self.local_agents[park_type]
        park_graph = self._build_state()["park_graphs"][park_type]
        raw_env_action = self._physical_actions_to_env_action(
            park_type=park_type,
            ev_grid_energy_by_id=cs_result.ev_energy_grid_side_by_id,
            bes_grid_energy_kwh=cs_result.bes_energy_grid_side_kwh,
            ev_bounds=ev_bounds,
            bes_bounds=bes_bounds,
        )
        raw_node_action = self._compose_node_action_for_q(
            park_graph=park_graph,
            env_action=raw_env_action,
            base_node_action=base_node_action,
        )

        score_raw = local_agent.evaluate_cmdp_score(park_graph, raw_node_action)
        score_probes: List[float] = []
        probe_curtailments: List[float] = []
        for probe_ratio in (self.tr_probe_ratio_1, self.tr_probe_ratio_2):
            test_ev_energy = dict(cs_result.ev_energy_grid_side_by_id)
            test_bes_energy = cs_result.bes_energy_grid_side_kwh
            test_pv_energy_kwh = cs_result.pv_energy_kwh
            requested_curtailment_kwh = probe_ratio * profile.same_direction_controllable_kwh
            if overload_direction == "import":
                test_ev_energy, test_bes_energy = self._apply_import_curtailment_proportional(
                    ev_energy_by_id=test_ev_energy,
                    bes_energy_kwh=test_bes_energy,
                    curtailment_kwh=requested_curtailment_kwh,
                )
            else:
                test_ev_energy, test_bes_energy, test_pv_energy_kwh = self._apply_export_curtailment_proportional(
                    ev_energy_by_id=test_ev_energy,
                    bes_energy_kwh=test_bes_energy,
                    pv_energy_kwh=test_pv_energy_kwh,
                    curtailment_kwh=requested_curtailment_kwh,
                )

            test_env_action = self._physical_actions_to_env_action(
                park_type=park_type,
                ev_grid_energy_by_id=test_ev_energy,
                bes_grid_energy_kwh=test_bes_energy,
                ev_bounds=ev_bounds,
                bes_bounds=bes_bounds,
            )
            test_node_action = self._compose_node_action_for_q(
                park_graph=park_graph,
                env_action=test_env_action,
                base_node_action=base_node_action,
            )
            score_probes.append(local_agent.evaluate_cmdp_score(park_graph, test_node_action))
            test_raw_net_kwh = sum(test_ev_energy.values()) + test_bes_energy - test_pv_energy_kwh
            probe_curtailments.append(abs(profile.raw_net_kwh - test_raw_net_kwh))

        return (
            score_raw,
            score_probes[0],
            score_probes[1],
            probe_curtailments[0],
            probe_curtailments[1],
        )

    @staticmethod
    def _compose_node_action_for_q(
        park_graph: Dict[str, Any],
        env_action: Dict[str, Any],
        base_node_action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if base_node_action is None or int(base_node_action.numel()) != len(park_graph["node_types"]):
            node_action = torch.zeros(len(park_graph["node_types"]), dtype=torch.float32)
        else:
            node_action = base_node_action.detach().clone().to(dtype=torch.float32).reshape(-1)

        bes_node_idx = None
        for node_idx, fixed_idx in zip(park_graph["action_node_indices"], park_graph["action_mapper"]):
            if fixed_idx == park_graph["bes_action_index"]:
                bes_node_idx = node_idx
                break
        if bes_node_idx is not None:
            node_action[bes_node_idx] = float(env_action.get("bes", 0.0))

        ev_action = env_action.get("ev", {})
        for ev_idx in park_graph["ev_indexes"]:
            ev_id = park_graph["node_names"][ev_idx]
            node_action[ev_idx] = float(ev_action.get(ev_id, 0.0))
        return node_action

    def _physical_actions_to_env_action(
        self,
        park_type: ParkType,
        ev_grid_energy_by_id: Dict[str, float],
        bes_grid_energy_kwh: float,
        ev_bounds: Dict[str, Dict[str, Any]],
        bes_bounds: Dict[str, Any],
    ) -> Dict[str, Any]:
        ev_action: Dict[str, float] = {}
        for ev_id, bound in ev_bounds[park_type].items():
            energy = ev_grid_energy_by_id.get(ev_id, 0.0)
            if energy >= 0.0:
                raw_value = energy / max(bound.upper_bound_kwh, EPS)
            else:
                raw_value = energy / max(bound.max_discharge_energy_port_kwh, EPS)
            ev_action[ev_id] = max(-1.0, min(1.0, raw_value))

        bes_bound = bes_bounds[park_type]
        if bes_grid_energy_kwh >= 0.0:
            bes_raw = bes_grid_energy_kwh / max(bes_bound.upper_bound_kwh, EPS)
        else:
            bes_raw = bes_grid_energy_kwh / max(bes_bound.max_discharge_energy_port_kwh, EPS)
        return {"bes": max(-1.0, min(1.0, bes_raw)), "ev": ev_action}

    def _secure_aggregate_by_park(self, values_by_park: Dict[str, float]) -> float:
        return secure_masked_sum(values_by_park, self.privacy_mask_rng)

    def _compute_nonprivacy_allocations(
        self,
        profiles_by_park: Dict[str, ParkPrivacyActionProfile],
        responsibility_by_park: Dict[str, float],
        overload_kwh: float,
    ) -> Dict[str, ParkPrivacyAllocation]:
        capacity_by_park = {
            park_type: profiles_by_park[park_type].same_direction_controllable_kwh
            for park_type in PARK_TYPES
        }
        weight_by_park = {
            park_type: max(responsibility_by_park[park_type], 0.0)
            for park_type in PARK_TYPES
        }
        curtailment_by_park = self._allocate_nonprivacy_curtailment(
            capacity_by_park=capacity_by_park,
            weight_by_park=weight_by_park,
            target_total_curtailment=overload_kwh,
        )
        total_weight = sum(weight_by_park.values())
        allocations: Dict[str, ParkPrivacyAllocation] = {}
        for park_type in PARK_TYPES:
            same_direction_controllable_kwh = capacity_by_park[park_type]
            curtailment_kwh = curtailment_by_park[park_type]
            shrink_ratio = compute_local_shrink_ratio(
                curtailment_kwh=curtailment_kwh,
                same_direction_controllable_kwh=same_direction_controllable_kwh,
            )
            final_mixing_weight = (
                weight_by_park[park_type] / max(total_weight, EPS)
                if total_weight > self.reward_consistency_tolerance
                else 0.0
            )
            allocations[park_type] = ParkPrivacyAllocation(
                park_id=park_type,
                same_direction_controllable_kwh=same_direction_controllable_kwh,
                final_mixing_weight=final_mixing_weight,
                curtailment_kwh=curtailment_kwh,
                pv_curtailment_kwh=0.0,
                shrink_ratio=shrink_ratio,
            )
        return allocations

    def _allocate_nonprivacy_curtailment(
        self,
        capacity_by_park: Dict[str, float],
        weight_by_park: Dict[str, float],
        target_total_curtailment: float,
    ) -> Dict[str, float]:
        allocations = {park_type: 0.0 for park_type in PARK_TYPES}
        remaining_target = max(0.0, target_total_curtailment)
        remaining_capacity = {
            park_type: max(0.0, capacity)
            for park_type, capacity in capacity_by_park.items()
        }
        active_parks = {
            park_type
            for park_type, capacity in remaining_capacity.items()
            if capacity > self.reward_consistency_tolerance
        }

        while remaining_target > self.reward_consistency_tolerance and active_parks:
            total_weight = sum(max(weight_by_park[park_type], 0.0) for park_type in active_parks)
            if total_weight <= self.reward_consistency_tolerance:
                capacity_sum = sum(remaining_capacity[park_type] for park_type in active_parks)
                desired_share = {
                    park_type: remaining_capacity[park_type] / max(capacity_sum, EPS)
                    for park_type in active_parks
                }
            else:
                desired_share = {
                    park_type: max(weight_by_park[park_type], 0.0) / total_weight
                    for park_type in active_parks
                }

            progress = 0.0
            saturated_parks = set()
            for park_type in list(active_parks):
                requested = remaining_target * desired_share[park_type]
                granted = min(requested, remaining_capacity[park_type])
                allocations[park_type] += granted
                remaining_capacity[park_type] -= granted
                progress += granted
                if remaining_capacity[park_type] <= self.reward_consistency_tolerance:
                    saturated_parks.add(park_type)

            remaining_target = max(0.0, target_total_curtailment - sum(allocations.values()))
            active_parks -= saturated_parks
            if progress <= self.reward_consistency_tolerance:
                break

        if remaining_target > self.reward_consistency_tolerance:
            for park_type in sorted(
                remaining_capacity.keys(),
                key=lambda item: remaining_capacity[item],
                reverse=True,
            ):
                if remaining_target <= self.reward_consistency_tolerance:
                    break
                granted = min(remaining_target, remaining_capacity[park_type])
                allocations[park_type] += granted
                remaining_target -= granted

        return allocations

    def _apply_privacy_curtailment_to_park(
        self,
        park_type: ParkType,
        cs_result: Any,
        allocation: ParkPrivacyAllocation,
        overload_direction: str,
        triggered: bool,
    ) -> ParkPrivacyProjectionResult:
        factor = 1.0 - allocation.shrink_ratio
        ev_after = dict(cs_result.ev_energy_grid_side_by_id)
        bes_after = cs_result.bes_energy_grid_side_kwh
        pv_effective_energy_kwh = cs_result.pv_energy_kwh
        pv_curtailment_kwh = 0.0

        if overload_direction == "import":
            ev_after, bes_after = self._apply_import_curtailment_proportional(
                ev_energy_by_id=ev_after,
                bes_energy_kwh=bes_after,
                curtailment_kwh=allocation.curtailment_kwh,
            )
        else:
            ev_after, bes_after, pv_effective_energy_kwh = self._apply_export_curtailment_proportional(
                ev_energy_by_id=ev_after,
                bes_energy_kwh=bes_after,
                pv_energy_kwh=pv_effective_energy_kwh,
                curtailment_kwh=allocation.curtailment_kwh,
            )
            pv_curtailment_kwh = max(0.0, cs_result.pv_energy_kwh - pv_effective_energy_kwh)

        projected_park_net_kwh = sum(ev_after.values()) + bes_after - pv_effective_energy_kwh
        return ParkPrivacyProjectionResult(
            park_id=park_type,
            ev_energy_grid_side_by_id=ev_after,
            bes_energy_grid_side_kwh=bes_after,
            pv_effective_energy_kwh=pv_effective_energy_kwh,
            pv_curtailment_kwh=pv_curtailment_kwh,
            scaling_factor=factor,
            projected_park_net_kwh=projected_park_net_kwh,
            reduction_kwh=allocation.curtailment_kwh,
            triggered=triggered,
        )

    @staticmethod
    def _apply_import_curtailment_proportional(
        ev_energy_by_id: Dict[str, float],
        bes_energy_kwh: float,
        curtailment_kwh: float,
    ) -> tuple[Dict[str, float], float]:
        ev_after = dict(ev_energy_by_id)
        bes_after = bes_energy_kwh
        positive_total_kwh = (
            sum(max(energy, 0.0) for energy in ev_after.values())
            + max(bes_after, 0.0)
        )
        reduced_charge_kwh = min(max(0.0, curtailment_kwh), positive_total_kwh)
        if reduced_charge_kwh <= EPS or positive_total_kwh <= EPS:
            return ev_after, bes_after

        retained_ratio = max(0.0, (positive_total_kwh - reduced_charge_kwh) / positive_total_kwh)
        ev_after = {
            ev_id: (energy * retained_ratio if energy > 0.0 else energy)
            for ev_id, energy in ev_after.items()
        }
        if bes_after > 0.0:
            bes_after *= retained_ratio
        return ev_after, bes_after

    @staticmethod
    def _apply_export_curtailment_proportional(
        ev_energy_by_id: Dict[str, float],
        bes_energy_kwh: float,
        pv_energy_kwh: float,
        curtailment_kwh: float,
    ) -> tuple[Dict[str, float], float, float]:
        ev_after = dict(ev_energy_by_id)
        bes_after = bes_energy_kwh
        positive_device_net = (
            sum(max(energy, 0.0) for energy in ev_after.values())
            + max(bes_after, 0.0)
        )
        total_export_kwh = (
            sum(max(-energy, 0.0) for energy in ev_after.values())
            + max(-bes_after, 0.0)
            + max(0.0, pv_energy_kwh - positive_device_net)
        )
        reduced_export_kwh = min(max(0.0, curtailment_kwh), total_export_kwh)
        if reduced_export_kwh <= EPS or total_export_kwh <= EPS:
            return ev_after, bes_after, pv_energy_kwh

        retained_ratio = max(0.0, (total_export_kwh - reduced_export_kwh) / total_export_kwh)
        ev_after = {
            ev_id: (energy * retained_ratio if energy < 0.0 else energy)
            for ev_id, energy in ev_after.items()
        }
        if bes_after < 0.0:
            bes_after *= retained_ratio
        positive_device_net = (
            sum(max(energy, 0.0) for energy in ev_after.values())
            + max(bes_after, 0.0)
        )
        raw_pv_export_kwh = max(0.0, pv_energy_kwh - positive_device_net)
        pv_export_after = raw_pv_export_kwh * retained_ratio
        pv_effective_energy_kwh = pv_energy_kwh - (raw_pv_export_kwh - pv_export_after)
        return ev_after, bes_after, pv_effective_energy_kwh

    def _apply_projected_actions(self, tr_summary: Any) -> Dict[str, Any]:
        ev_charge_revenue = 0.0
        grid_sale_revenue = 0.0
        grid_purchase_cost = 0.0
        v2g_compensation_cost = 0.0
        park_financials = {
            park_type: {
                "ev_charge_revenue": 0.0,
                "grid_sale_revenue": 0.0,
                "grid_purchase_cost": 0.0,
                "v2g_compensation_cost": 0.0,
            }
            for park_type in PARK_TYPES
        }
        executed_flows: Dict[str, ExecutedParkFlow] = {}

        for park_type, park_result in tr_summary.park_results_by_id.items():
            park_state = self.runtime_states[park_type]
            executed_ev_grid_energy_by_id: Dict[str, float] = {}
            for ev_id, grid_energy_kwh in park_result.ev_energy_grid_side_by_id.items():
                runtime = park_state.connected_evs[ev_id]
                session = runtime.session
                if grid_energy_kwh > 0.0:
                    battery_charge_kwh = grid_energy_kwh * session["eta_ch"]
                    charge_result = self.debt_manager.process_charge(ev_id, self.current_step, battery_charge_kwh)
                    actual_grid_energy_kwh = charge_result.executed_battery_kwh / max(session["eta_ch"], 1e-9)
                    executed_ev_grid_energy_by_id[ev_id] = actual_grid_energy_kwh
                    ev_charge_revenue += charge_result.charge_revenue
                    park_financials[park_type]["ev_charge_revenue"] += charge_result.charge_revenue
                elif grid_energy_kwh < 0.0:
                    grid_discharge_kwh = -grid_energy_kwh
                    battery_discharge_kwh = grid_discharge_kwh / max(session["eta_dis"], 1e-9)
                    discharge_result = self.debt_manager.process_v2g_discharge(
                        ev_id=ev_id,
                        step=self.current_step,
                        battery_discharge_kwh=battery_discharge_kwh,
                        grid_discharge_kwh=grid_discharge_kwh,
                    )
                    v2g_compensation_cost += discharge_result.cash_compensation
                    park_financials[park_type]["v2g_compensation_cost"] += discharge_result.cash_compensation
                    executed_ev_grid_energy_by_id[ev_id] = -discharge_result.executed_grid_kwh
                else:
                    executed_ev_grid_energy_by_id[ev_id] = 0.0

            bes_grid_kwh = park_result.bes_energy_grid_side_kwh
            if bes_grid_kwh >= 0.0:
                delta_battery_kwh = bes_grid_kwh * park_state.bes_spec.eta_ch
            else:
                delta_battery_kwh = bes_grid_kwh / max(park_state.bes_spec.eta_dis, 1e-9)
            park_state.bes_soc = min(
                park_state.bes_spec.soc_max,
                max(
                    park_state.bes_spec.soc_min,
                    park_state.bes_soc + delta_battery_kwh / park_state.bes_spec.energy_capacity_kwh,
                ),
            )
            pv_energy_kwh = park_result.pv_effective_energy_kwh
            park_grid_exchange_kwh = sum(executed_ev_grid_energy_by_id.values()) + bes_grid_kwh - pv_energy_kwh
            internal_balance_residual_kwh = (
                sum(executed_ev_grid_energy_by_id.values()) + bes_grid_kwh - pv_energy_kwh - park_grid_exchange_kwh
            )
            executed_flows[park_type] = ExecutedParkFlow(
                ev_grid_energy_by_id=executed_ev_grid_energy_by_id,
                bes_grid_energy_kwh=bes_grid_kwh,
                pv_energy_kwh=pv_energy_kwh,
                park_grid_exchange_kwh=park_grid_exchange_kwh,
                internal_balance_residual_kwh=internal_balance_residual_kwh,
            )

        actual_total_grid_exchange_kwh = sum(flow.park_grid_exchange_kwh for flow in executed_flows.values())
        charge_price = self.grid_price_table["charge_price"][self.current_step]
        discharge_price = self.grid_price_table["discharge_price"][self.current_step]
        (
            settlement_direction,
            settlement_price,
            system_purchase_cost,
            system_sale_revenue,
            aggregated_grid_purchase_cost,
            aggregated_grid_sale_revenue,
        ) = self._allocate_fused_grid_settlement(
            executed_flows=executed_flows,
            charge_price=charge_price,
            discharge_price=discharge_price,
            park_financials=park_financials,
        )
        grid_purchase_cost += system_purchase_cost
        grid_sale_revenue += system_sale_revenue
        actual_net_settlement = grid_sale_revenue - grid_purchase_cost
        allocated_net_settlement = aggregated_grid_sale_revenue - aggregated_grid_purchase_cost
        if abs(actual_net_settlement - allocated_net_settlement) > self.reward_consistency_tolerance:
            raise RuntimeError("shared grid settlement decomposition violated")

        energy_balance = self._build_energy_balance_summary(executed_flows, actual_total_grid_exchange_kwh)

        return {
            "ev_charge_revenue": ev_charge_revenue,
            "grid_sale_revenue": grid_sale_revenue,
            "grid_purchase_cost": grid_purchase_cost,
            "v2g_compensation_cost": v2g_compensation_cost,
            "parks": park_financials,
            "executed_flows": {
                park_type: {
                    "ev_grid_energy_by_id": flow.ev_grid_energy_by_id,
                    "bes_grid_energy_kwh": flow.bes_grid_energy_kwh,
                    "pv_energy_kwh": flow.pv_energy_kwh,
                    "park_grid_exchange_kwh": flow.park_grid_exchange_kwh,
                }
                for park_type, flow in executed_flows.items()
            },
            "grid_settlement": {
                "direction": settlement_direction,
                "price": settlement_price,
                "actual_total_grid_exchange_kwh": actual_total_grid_exchange_kwh,
                "system_purchase_cost": system_purchase_cost,
                "system_sale_revenue": system_sale_revenue,
                "aggregated_purchase_cost": aggregated_grid_purchase_cost,
                "aggregated_sale_revenue": aggregated_grid_sale_revenue,
                "system_net_settlement": actual_net_settlement,
                "aggregated_park_net_settlement": allocated_net_settlement,
                "consistency_error": allocated_net_settlement - actual_net_settlement,
            },
            "energy_balance": energy_balance,
        }

    def _build_state(self) -> Dict[str, Any]:
        ev_bounds, bes_bounds = self._compute_action_bounds()
        return self.state_builder.build_state(env=self, ev_bounds=ev_bounds, bes_bounds=bes_bounds)

    def _settle_remaining_departures(self) -> List[Dict[str, Any]]:
        departure_records: List[Dict[str, Any]] = []
        for park_state in self.runtime_states.values():
            for ev_id in list(park_state.connected_evs.keys()):
                settlement = self.debt_manager.settle_departure(ev_id, EPISODE_STEPS - 1)
                departure_records.append(
                    {
                        "ev_id": ev_id,
                        "park_type": park_state.park_type,
                        "soc_at_departure": settlement.soc_at_departure,
                        "target_soc": settlement.target_departure_soc,
                        "debt_remaining_kwh": settlement.debt_remaining_kwh,
                        "soc_shortfall_kwh": settlement.soc_shortfall_kwh,
                        "debt_penalty": settlement.debt_penalty,
                        "soc_shortfall_penalty": settlement.soc_shortfall_penalty,
                    }
                )
                self.debt_manager.remove_ev(ev_id)
                del park_state.connected_evs[ev_id]
        return departure_records

    def _compute_bes_terminal_energy_penalty_abs(self) -> float:
        total_penalty_abs = 0.0
        for park_state in self.runtime_states.values():
            total_penalty_abs += abs(park_state.bes_soc - park_state.bes_spec.soc_min) * park_state.bes_spec.energy_capacity_kwh
        return total_penalty_abs

    def _compute_local_tr_responsibility_by_park(
        self,
        cs_results: Dict[str, Any],
        overload_direction: str,
    ) -> Dict[str, float]:
        return {
            park_type: compute_local_tr_responsibility(
                overload_direction=overload_direction,
                projected_park_net_kwh=cs_results[park_type].projected_net_after_pv_kwh,
            )
            for park_type in PARK_TYPES
        }

    def _compute_local_tr_penalty_by_park(self, cs_results: Dict[str, Any], tr_summary: Any) -> Dict[str, float]:
        if not tr_summary.triggered:
            return {park_type: 0.0 for park_type in PARK_TYPES}
        control_signal = self._build_control_signal(tr_summary.broadcast)
        responsibility_by_park = self._compute_local_tr_responsibility_by_park(
            cs_results=cs_results,
            overload_direction=control_signal.overload_direction,
        )
        penalty_by_park = {
            park_type: compute_local_tr_penalty(
                tr_penalty_coefficient=control_signal.tr_penalty_coefficient,
                local_responsibility=responsibility_by_park[park_type],
            )
            for park_type in PARK_TYPES
        }
        total_penalty = sum(penalty_by_park.values())
        expected_total_penalty = compute_global_tr_penalty(
            penalty_weight=self.transformer_overload_penalty_weight,
            overload_kwh=tr_summary.broadcast.overload_kwh,
        )
        consistency_residual = expected_total_penalty - total_penalty
        if abs(consistency_residual) > self.reward_consistency_tolerance:
            anchor_park = max(responsibility_by_park, key=responsibility_by_park.get)
            if responsibility_by_park[anchor_park] > self.reward_consistency_tolerance:
                penalty_by_park[anchor_park] = max(0.0, penalty_by_park[anchor_park] + consistency_residual)
                total_penalty = sum(penalty_by_park.values())
        if abs(total_penalty - expected_total_penalty) > self.reward_consistency_tolerance:
            raise RuntimeError("shared transformer penalty decomposition violated")
        return penalty_by_park

    def _build_park_reward_breakdown(
        self,
        transition_info: Dict[str, Any],
        cs_results: Dict[str, Any],
        tr_summary: Any,
        departure_records: List[Dict[str, Any]],
        is_terminal_step: bool,
    ) -> Dict[str, Dict[str, float]]:
        breakdown_by_park: Dict[str, Dict[str, float]] = {}
        local_tr_penalty_by_park = self._compute_local_tr_penalty_by_park(cs_results, tr_summary)
        for park_type in PARK_TYPES:
            park_departures = [
                self._to_departure_record(record)
                for record in departure_records
                if record["park_type"] == park_type
            ]
            park_reward = self.reward_builder.compute(
                StepRewardInput(
                    ev_charge_revenue=transition_info["parks"][park_type]["ev_charge_revenue"],
                    grid_sale_revenue=transition_info["parks"][park_type]["grid_sale_revenue"],
                    grid_purchase_cost=transition_info["parks"][park_type]["grid_purchase_cost"],
                    v2g_compensation_cost=transition_info["parks"][park_type]["v2g_compensation_cost"],
                    cs_projection_penalty_abs=abs(
                        cs_results[park_type].raw_net_after_pv_kwh - cs_results[park_type].projected_net_after_pv_kwh
                    ),
                    tr_projection_penalty_abs=local_tr_penalty_by_park[park_type],
                    departure_records=park_departures,
                    is_terminal_step=is_terminal_step,
                    bes_terminal_energy_penalty_abs=(
                        abs(self.runtime_states[park_type].bes_soc - self.runtime_states[park_type].bes_spec.soc_min)
                        * self.runtime_states[park_type].bes_spec.energy_capacity_kwh
                        if is_terminal_step
                        else 0.0
                    ),
                )
            )
            breakdown_by_park[park_type] = {
                "profit_term": park_reward.profit_term,
                "user_satisfaction_penalty": park_reward.user_satisfaction_penalty,
                "cs_projection_penalty": park_reward.cs_projection_penalty,
                "tr_projection_penalty": park_reward.tr_projection_penalty,
                "bes_terminal_penalty": park_reward.bes_terminal_penalty,
                "debt_penalty": park_reward.debt_penalty,
                "profit_reward": park_reward.training_profit_reward,
                "constraint_cost": park_reward.training_constraint_cost,
                "local_constraint_cost": max(
                    0.0,
                    park_reward.training_constraint_cost - park_reward.training_tr_projection_penalty,
                ),
                "regional_constraint_cost": park_reward.training_tr_projection_penalty,
                "training_user_satisfaction_penalty": park_reward.training_user_satisfaction_penalty,
                "training_cs_projection_penalty": park_reward.training_cs_projection_penalty,
                "training_tr_projection_penalty": park_reward.training_tr_projection_penalty,
                "training_bes_terminal_penalty": park_reward.training_bes_terminal_penalty,
                "training_debt_penalty": park_reward.training_debt_penalty,
                "training_total_reward": park_reward.training_total_reward,
                "logging_profit_reward": park_reward.logging_profit_reward,
                "logging_constraint_cost": park_reward.logging_constraint_cost,
                "total_reward": park_reward.total_reward,
            }
        return breakdown_by_park

    @staticmethod
    def _aggregate_park_reward_breakdown(park_reward_breakdown: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        totals = {
            "profit_term": 0.0,
            "user_satisfaction_penalty": 0.0,
            "cs_projection_penalty": 0.0,
            "tr_projection_penalty": 0.0,
            "bes_terminal_penalty": 0.0,
            "debt_penalty": 0.0,
            "profit_reward": 0.0,
            "constraint_cost": 0.0,
            "training_user_satisfaction_penalty": 0.0,
            "training_cs_projection_penalty": 0.0,
            "training_tr_projection_penalty": 0.0,
            "training_bes_terminal_penalty": 0.0,
            "training_debt_penalty": 0.0,
            "training_total_reward": 0.0,
            "logging_profit_reward": 0.0,
            "logging_constraint_cost": 0.0,
            "total_reward": 0.0,
        }
        for breakdown in park_reward_breakdown.values():
            for key in totals.keys():
                totals[key] += breakdown[key]
        return totals

    def _log_step_index(self) -> int:
        return max(self.current_step - 1, 0)

    def _log_time_label(self) -> Any:
        log_step = self._log_step_index()
        return self.grid_price_table["time"][min(log_step, len(self.grid_price_table["time"]) - 1)]

    def _build_energy_log_row(
        self,
        transition_info: Dict[str, Any],
        cs_results: Dict[str, Any],
        tr_summary: Any,
        departure_records: List[Dict[str, Any]],
        raw_node_actions: Optional[Dict[str, torch.Tensor]] = None,
        requested_bes_grid_energy: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "step": self._log_step_index(),
            "time": self._log_time_label(),
            "weather": self.daily_pv["weather"],
        }
        totals = {
            "pv_energy_kwh": 0.0,
            "grid_purchase_energy_kwh": 0.0,
            "grid_sale_energy_kwh": 0.0,
            "bes_charge_grid_energy_kwh": 0.0,
            "bes_discharge_grid_energy_kwh": 0.0,
            "ev_charge_grid_energy_kwh": 0.0,
            "ev_discharge_grid_energy_kwh": 0.0,
            "cs_trunc_charge_kwh": 0.0,
            "cs_trunc_discharge_kwh": 0.0,
            "tr_trunc_charge_kwh": 0.0,
            "tr_trunc_discharge_kwh": 0.0,
            "departure_debt_energy_kwh": 0.0,
            "departure_soc_shortfall_energy_kwh": 0.0,
        }

        departures_by_park = {park_type: [] for park_type in PARK_TYPES}
        for record in departure_records:
            departures_by_park[record["park_type"]].append(record)

        for park_type in PARK_TYPES:
            executed = transition_info["executed_flows"][park_type]
            active_ev_count = len(self.runtime_states[park_type].connected_evs)
            raw_bes_action = 0.0
            if raw_node_actions is not None:
                park_graph = self._build_state()["park_graphs"][park_type]
                bes_node_index = next(
                    node_idx
                    for node_idx, fixed_idx in zip(park_graph["action_node_indices"], park_graph["action_mapper"])
                    if fixed_idx == park_graph["bes_action_index"]
                )
                raw_bes_action = float(raw_node_actions[park_type][bes_node_index].detach().cpu().item())
            requested_bes_energy = 0.0 if requested_bes_grid_energy is None else float(requested_bes_grid_energy[park_type])
            cs_projected_bes_energy = float(cs_results[park_type].bes_energy_grid_side_kwh)
            tr_projected_bes_energy = float(tr_summary.park_results_by_id[park_type].bes_energy_grid_side_kwh)
            ev_charge_grid_energy = sum(max(energy, 0.0) for energy in executed["ev_grid_energy_by_id"].values())
            ev_discharge_grid_energy = sum(max(-energy, 0.0) for energy in executed["ev_grid_energy_by_id"].values())
            bes_charge_grid_energy = max(executed["bes_grid_energy_kwh"], 0.0)
            bes_discharge_grid_energy = max(-executed["bes_grid_energy_kwh"], 0.0)
            park_grid_purchase_energy = max(executed["park_grid_exchange_kwh"], 0.0)
            park_grid_sale_energy = max(-executed["park_grid_exchange_kwh"], 0.0)
            cs_charge_trunc, cs_discharge_trunc = self._directional_projection_delta(
                cs_results[park_type].raw_ev_energy_grid_side_by_id,
                cs_results[park_type].raw_bes_energy_grid_side_kwh,
                cs_results[park_type].ev_energy_grid_side_by_id,
                cs_results[park_type].bes_energy_grid_side_kwh,
            )
            cs_discharge_trunc += cs_results[park_type].pv_curtailment_kwh
            tr_charge_trunc, tr_discharge_trunc = self._directional_projection_delta(
                cs_results[park_type].ev_energy_grid_side_by_id,
                cs_results[park_type].bes_energy_grid_side_kwh,
                tr_summary.park_results_by_id[park_type].ev_energy_grid_side_by_id,
                tr_summary.park_results_by_id[park_type].bes_energy_grid_side_kwh,
            )
            tr_discharge_trunc += tr_summary.park_results_by_id[park_type].pv_curtailment_kwh
            departure_debt_energy = sum(record["debt_remaining_kwh"] for record in departures_by_park[park_type])
            departure_soc_shortfall_energy = sum(record["soc_shortfall_kwh"] for record in departures_by_park[park_type])

            row[f"{park_type}_active_ev_count"] = active_ev_count
            row[f"{park_type}_pv_energy_kwh"] = executed["pv_energy_kwh"]
            row[f"{park_type}_raw_bes_action"] = raw_bes_action
            row[f"{park_type}_requested_bes_grid_energy_kwh"] = requested_bes_energy
            row[f"{park_type}_cs_projected_bes_grid_energy_kwh"] = cs_projected_bes_energy
            row[f"{park_type}_tr_projected_bes_grid_energy_kwh"] = tr_projected_bes_energy
            row[f"{park_type}_grid_purchase_energy_kwh"] = park_grid_purchase_energy
            row[f"{park_type}_grid_sale_energy_kwh"] = park_grid_sale_energy
            row[f"{park_type}_bes_charge_grid_energy_kwh"] = bes_charge_grid_energy
            row[f"{park_type}_bes_discharge_grid_energy_kwh"] = bes_discharge_grid_energy
            row[f"{park_type}_ev_charge_grid_energy_kwh"] = ev_charge_grid_energy
            row[f"{park_type}_ev_discharge_grid_energy_kwh"] = ev_discharge_grid_energy
            row[f"{park_type}_cs_trunc_charge_kwh"] = cs_charge_trunc
            row[f"{park_type}_cs_trunc_discharge_kwh"] = cs_discharge_trunc
            row[f"{park_type}_tr_trunc_charge_kwh"] = tr_charge_trunc
            row[f"{park_type}_tr_trunc_discharge_kwh"] = tr_discharge_trunc
            row[f"{park_type}_departure_debt_energy_kwh"] = departure_debt_energy
            row[f"{park_type}_departure_soc_shortfall_energy_kwh"] = departure_soc_shortfall_energy
            row[f"{park_type}_bes_soc"] = self.runtime_states[park_type].bes_soc

            totals["pv_energy_kwh"] += executed["pv_energy_kwh"]
            totals["grid_purchase_energy_kwh"] += park_grid_purchase_energy
            totals["grid_sale_energy_kwh"] += park_grid_sale_energy
            totals["bes_charge_grid_energy_kwh"] += bes_charge_grid_energy
            totals["bes_discharge_grid_energy_kwh"] += bes_discharge_grid_energy
            totals["ev_charge_grid_energy_kwh"] += ev_charge_grid_energy
            totals["ev_discharge_grid_energy_kwh"] += ev_discharge_grid_energy
            totals["cs_trunc_charge_kwh"] += cs_charge_trunc
            totals["cs_trunc_discharge_kwh"] += cs_discharge_trunc
            totals["tr_trunc_charge_kwh"] += tr_charge_trunc
            totals["tr_trunc_discharge_kwh"] += tr_discharge_trunc
            totals["departure_debt_energy_kwh"] += departure_debt_energy
            totals["departure_soc_shortfall_energy_kwh"] += departure_soc_shortfall_energy

        for key, value in totals.items():
            row[f"total_{key}"] = value
        return row

    def _build_projection_trace_row(
        self,
        cs_results: Dict[str, Any],
        tr_summary: Any,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "step": self._log_step_index(),
            "time": self._log_time_label(),
            "weather": self.daily_pv["weather"],
        }
        for park_type in PARK_TYPES:
            row[f"{park_type}_cs_projected_demand_kwh"] = cs_results[park_type].projected_net_after_pv_kwh
            row[f"{park_type}_cs_limit_kwh"] = self.runtime_states[park_type].cs_limit_kwh
        row["tr_net_demand_kwh"] = tr_summary.total_net_before_kwh
        row["tr_limit_kwh"] = self.tr_limit.max_exchange_energy_kwh
        row["tr_triggered"] = 1 if tr_summary.broadcast.triggered else 0
        row["tr_overload_direction"] = tr_summary.broadcast.overload_direction
        row["tr_overload_kwh"] = tr_summary.broadcast.overload_kwh
        row["tr_total_capacity_kwh"] = tr_summary.broadcast.total_capacity_kwh
        row["tr_total_preference_capacity_kwh"] = tr_summary.broadcast.total_preference_capacity_kwh
        row["tr_total_responsibility"] = tr_summary.broadcast.total_responsibility
        row["tr_safety_base_ratio"] = tr_summary.broadcast.safety_base_ratio
        row["tr_blended_capacity_kwh"] = tr_summary.broadcast.blended_capacity_kwh
        row["tr_scaling_coefficient"] = tr_summary.broadcast.scaling_coefficient
        row["tr_penalty_coefficient"] = tr_summary.broadcast.tr_penalty_coefficient
        row["tr_infeasible_residual_kwh"] = tr_summary.broadcast.infeasible_residual_kwh
        return row

    def _build_reward_log_row(
        self,
        transition_info: Dict[str, Any],
        cs_results: Dict[str, Any],
        tr_summary: Any,
        departure_records: List[Dict[str, Any]],
        park_reward_breakdown: Dict[str, Dict[str, float]],
        is_terminal_step: bool,
    ) -> Dict[str, Any]:
        aggregated = self._aggregate_park_reward_breakdown(park_reward_breakdown)
        profit_weight = self.reward_builder.training_weights.profit_weight
        row: Dict[str, Any] = {
            "step": self._log_step_index(),
            "time": self._log_time_label(),
            "weather": self.daily_pv["weather"],
            "total_profit_reward": aggregated["logging_profit_reward"],
            "total_constraint_cost": aggregated["logging_constraint_cost"],
            "total_immediate_reward": aggregated["total_reward"],
            "total_grid_purchase_cost": profit_weight * transition_info["grid_purchase_cost"],
            "total_grid_sale_revenue": profit_weight * transition_info["grid_sale_revenue"],
            "total_v2g_compensation_cost": profit_weight * transition_info["v2g_compensation_cost"],
            "total_ev_charge_revenue": profit_weight * transition_info["ev_charge_revenue"],
            "total_cs_projection_penalty": aggregated["training_cs_projection_penalty"],
            "total_tr_projection_penalty": aggregated["training_tr_projection_penalty"],
            "total_debt_penalty": aggregated["training_debt_penalty"],
            "total_soc_shortfall_penalty": aggregated["training_user_satisfaction_penalty"],
            "total_bes_terminal_penalty": aggregated["training_bes_terminal_penalty"],
            "grid_settlement_consistency_error": transition_info["grid_settlement"]["consistency_error"],
            "tr_penalty_consistency_error": aggregated["tr_projection_penalty"] - compute_global_tr_penalty(
                penalty_weight=self.transformer_overload_penalty_weight,
                overload_kwh=tr_summary.broadcast.overload_kwh,
            ),
            "reward_sum_consistency_error": aggregated["total_reward"] - sum(
                breakdown["total_reward"] for breakdown in park_reward_breakdown.values()
            ),
        }

        for park_type in PARK_TYPES:
            row[f"{park_type}_profit_reward"] = park_reward_breakdown[park_type]["logging_profit_reward"]
            row[f"{park_type}_constraint_cost"] = park_reward_breakdown[park_type]["logging_constraint_cost"]
            row[f"{park_type}_immediate_reward"] = park_reward_breakdown[park_type]["total_reward"]
            row[f"{park_type}_grid_purchase_cost"] = profit_weight * transition_info["parks"][park_type]["grid_purchase_cost"]
            row[f"{park_type}_grid_sale_revenue"] = profit_weight * transition_info["parks"][park_type]["grid_sale_revenue"]
            row[f"{park_type}_v2g_compensation_cost"] = profit_weight * transition_info["parks"][park_type]["v2g_compensation_cost"]
            row[f"{park_type}_ev_charge_revenue"] = profit_weight * transition_info["parks"][park_type]["ev_charge_revenue"]
            row[f"{park_type}_cs_projection_penalty"] = park_reward_breakdown[park_type]["training_cs_projection_penalty"]
            row[f"{park_type}_tr_projection_penalty"] = park_reward_breakdown[park_type]["training_tr_projection_penalty"]
            row[f"{park_type}_debt_penalty"] = park_reward_breakdown[park_type]["training_debt_penalty"]
            row[f"{park_type}_soc_shortfall_penalty"] = park_reward_breakdown[park_type]["training_user_satisfaction_penalty"]
            row[f"{park_type}_bes_terminal_penalty"] = park_reward_breakdown[park_type]["training_bes_terminal_penalty"]
        return row

    def _build_training_reward_log_row(
        self,
        park_reward_breakdown: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        aggregated = self._aggregate_park_reward_breakdown(park_reward_breakdown)
        row: Dict[str, Any] = {
            "step": self._log_step_index(),
            "time": self._log_time_label(),
            "weather": self.daily_pv["weather"],
            "total_profit_reward": aggregated["profit_reward"],
            "total_constraint_cost": aggregated["constraint_cost"],
            "total_training_reward": aggregated["training_total_reward"],
            "total_user_satisfaction_penalty": aggregated["training_user_satisfaction_penalty"],
            "total_cs_projection_penalty": aggregated["training_cs_projection_penalty"],
            "total_tr_projection_penalty": aggregated["training_tr_projection_penalty"],
            "total_bes_terminal_penalty": aggregated["training_bes_terminal_penalty"],
            "total_debt_penalty": aggregated["training_debt_penalty"],
        }
        for park_type in PARK_TYPES:
            row[f"{park_type}_profit_reward"] = park_reward_breakdown[park_type]["profit_reward"]
            row[f"{park_type}_constraint_cost"] = park_reward_breakdown[park_type]["constraint_cost"]
            row[f"{park_type}_training_reward"] = park_reward_breakdown[park_type]["training_total_reward"]
            row[f"{park_type}_user_satisfaction_penalty"] = park_reward_breakdown[park_type]["training_user_satisfaction_penalty"]
            row[f"{park_type}_cs_projection_penalty"] = park_reward_breakdown[park_type]["training_cs_projection_penalty"]
            row[f"{park_type}_tr_projection_penalty"] = park_reward_breakdown[park_type]["training_tr_projection_penalty"]
            row[f"{park_type}_bes_terminal_penalty"] = park_reward_breakdown[park_type]["training_bes_terminal_penalty"]
            row[f"{park_type}_debt_penalty"] = park_reward_breakdown[park_type]["training_debt_penalty"]
        return row

    def _build_energy_balance_summary(
        self,
        executed_flows: Dict[str, ExecutedParkFlow],
        actual_total_grid_exchange_kwh: float,
    ) -> Dict[str, Any]:
        parks: Dict[str, Dict[str, float]] = {}
        for park_type, flow in executed_flows.items():
            parks[park_type] = {
                "ev_grid_total_kwh": sum(flow.ev_grid_energy_by_id.values()),
                "bes_grid_energy_kwh": flow.bes_grid_energy_kwh,
                "pv_energy_kwh": flow.pv_energy_kwh,
                "park_grid_exchange_kwh": flow.park_grid_exchange_kwh,
                "internal_balance_residual_kwh": flow.internal_balance_residual_kwh,
            }
            if abs(flow.internal_balance_residual_kwh) > self.energy_balance_tolerance_kwh:
                raise RuntimeError(f"park internal energy balance violated: {park_type}")

        upper_level_residual_kwh = sum(flow.park_grid_exchange_kwh for flow in executed_flows.values()) - actual_total_grid_exchange_kwh
        if abs(upper_level_residual_kwh) > self.energy_balance_tolerance_kwh:
            raise RuntimeError("upper-level energy balance violated")

        return {
            "parks": parks,
            "upper_level": {
                "sum_park_grid_exchange_kwh": sum(flow.park_grid_exchange_kwh for flow in executed_flows.values()),
                "tr_grid_exchange_kwh": actual_total_grid_exchange_kwh,
                "residual_kwh": upper_level_residual_kwh,
            },
        }

    def _allocate_fused_grid_settlement(
        self,
        executed_flows: Dict[str, ExecutedParkFlow],
        charge_price: float,
        discharge_price: float,
        park_financials: Dict[str, Dict[str, float]],
    ) -> tuple[str, float, float, float, float, float]:
        total_grid_exchange_kwh = sum(flow.park_grid_exchange_kwh for flow in executed_flows.values())
        if total_grid_exchange_kwh >= 0.0:
            settlement_direction = "import"
            settlement_price = charge_price
            system_purchase_cost = total_grid_exchange_kwh * settlement_price
            system_sale_revenue = 0.0
        else:
            settlement_direction = "export"
            settlement_price = discharge_price
            system_purchase_cost = 0.0
            system_sale_revenue = (-total_grid_exchange_kwh) * settlement_price

        for park_type, flow in executed_flows.items():
            park_financials[park_type]["grid_purchase_cost"] = settlement_price * max(flow.park_grid_exchange_kwh, 0.0)
            park_financials[park_type]["grid_sale_revenue"] = settlement_price * max(-flow.park_grid_exchange_kwh, 0.0)

        aggregated_grid_purchase_cost = sum(
            financials["grid_purchase_cost"]
            for financials in park_financials.values()
        )
        aggregated_grid_sale_revenue = sum(
            financials["grid_sale_revenue"]
            for financials in park_financials.values()
        )
        if settlement_direction == "import":
            if abs((aggregated_grid_purchase_cost - aggregated_grid_sale_revenue) - system_purchase_cost) > self.reward_consistency_tolerance:
                raise RuntimeError("fused import settlement decomposition violated")
        else:
            if abs((aggregated_grid_sale_revenue - aggregated_grid_purchase_cost) - system_sale_revenue) > self.reward_consistency_tolerance:
                raise RuntimeError("fused export settlement decomposition violated")

        return (
            settlement_direction,
            settlement_price,
            system_purchase_cost,
            system_sale_revenue,
            aggregated_grid_purchase_cost,
            aggregated_grid_sale_revenue,
        )

    @staticmethod
    def _to_departure_record(record: Dict[str, Any]) -> EVDepartureRecord:
        return EVDepartureRecord(
            ev_id=record["ev_id"],
            park_type=record["park_type"],
            soc_at_departure=record["soc_at_departure"],
            target_soc=record["target_soc"],
            debt_remaining_kwh=record["debt_remaining_kwh"],
            soc_shortfall_kwh=record.get("soc_shortfall_kwh", 0.0),
        )

    @staticmethod
    def _directional_projection_delta(
        before_ev_by_id: Dict[str, float],
        before_bes_kwh: float,
        after_ev_by_id: Dict[str, float],
        after_bes_kwh: float,
    ) -> tuple[float, float]:
        before_charge = sum(max(energy, 0.0) for energy in before_ev_by_id.values()) + max(before_bes_kwh, 0.0)
        before_discharge = sum(max(-energy, 0.0) for energy in before_ev_by_id.values()) + max(-before_bes_kwh, 0.0)
        after_charge = sum(max(energy, 0.0) for energy in after_ev_by_id.values()) + max(after_bes_kwh, 0.0)
        after_discharge = sum(max(-energy, 0.0) for energy in after_ev_by_id.values()) + max(-after_bes_kwh, 0.0)
        return max(0.0, before_charge - after_charge), max(0.0, before_discharge - after_discharge)
    def _reset_projection_memory(self) -> None:
        self.prev_cs_projection_stats = {
            park_type: {"triggered": 0.0, "reduction_degree": 0.0}
            for park_type in PARK_TYPES
        }
        self.prev_tr_projection_stats = {
            park_type: {
                "reduction_kwh": 0.0,
                "reduction_ratio": 0.0,
                "pv_curtailment_ratio": 0.0,
                "signed_control_signal": 0.0,
                "global_overload_ratio": 0.0,
                "local_penalty_normalized": 0.0,
            }
            for park_type in PARK_TYPES
        }

    def _update_projection_memory(self, cs_results: Dict[str, Any], tr_summary: Any) -> None:
        self.prev_cs_projection_stats = {}
        for park_type, result in cs_results.items():
            reduction_degree = 0.0 if abs(result.raw_device_net_kwh) <= 1e-9 else max(0.0, 1.0 - result.scaling_factor)
            self.prev_cs_projection_stats[park_type] = {
                "triggered": 1.0 if result.triggered else 0.0,
                "reduction_degree": reduction_degree,
            }
        local_tr_penalty_by_park = self._compute_local_tr_penalty_by_park(cs_results, tr_summary)
        normalized_penalty_scale = (
            self.transformer_overload_penalty_weight
            * self.tr_limit.max_exchange_energy_kwh
            * self.tr_limit.max_exchange_energy_kwh
        )
        global_overload_ratio = (
            tr_summary.broadcast.overload_kwh / max(self.tr_limit.max_exchange_energy_kwh, EPS)
            if tr_summary.broadcast.triggered
            else 0.0
        )
        self.prev_tr_projection_stats = {
            park_type: {
                "reduction_kwh": tr_summary.park_results_by_id[park_type].reduction_kwh,
                "reduction_ratio": tr_summary.allocations_by_park[park_type].shrink_ratio,
                "pv_curtailment_ratio": (
                    (
                        cs_results[park_type].pv_curtailment_kwh
                        + tr_summary.park_results_by_id[park_type].pv_curtailment_kwh
                    )
                    / max(cs_results[park_type].raw_pv_energy_kwh, EPS)
                    if cs_results[park_type].raw_pv_energy_kwh > self.reward_consistency_tolerance
                    else 0.0
                ),
                "signed_control_signal": self._compute_signed_tr_control_signal(tr_summary),
                "global_overload_ratio": global_overload_ratio,
                "local_penalty_normalized": (
                    local_tr_penalty_by_park[park_type] / max(normalized_penalty_scale, EPS)
                    if normalized_penalty_scale > self.reward_consistency_tolerance
                    else 0.0
                ),
            }
            for park_type in PARK_TYPES
        }

    @staticmethod
    def _compute_signed_tr_control_signal(tr_summary: Any) -> float:
        if not tr_summary.broadcast.triggered:
            return 0.0
        if tr_summary.broadcast.overload_direction == "import":
            return tr_summary.broadcast.scaling_coefficient
        if tr_summary.broadcast.overload_direction == "export":
            return -tr_summary.broadcast.scaling_coefficient
        return 0.0

    @staticmethod
    def _serialize_bounds(ev_bounds: Dict[str, Dict[str, Any]], bes_bounds: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ev": {
                park: {
                    ev_id: {"lower": bound.lower_bound_kwh, "upper": bound.upper_bound_kwh}
                    for ev_id, bound in park_bounds.items()
                }
                for park, park_bounds in ev_bounds.items()
            },
            "bes": {
                park: {"lower": bound.lower_bound_kwh, "upper": bound.upper_bound_kwh}
                for park, bound in bes_bounds.items()
            },
        }

    def _load_park_sell_price_table(self) -> Dict[ParkType, List[float]]:
        file_map = {
            "residential": self.config_dir / "residential_park_price.csv",
            "office": self.config_dir / "office_park_price.csv",
            "commercial": self.config_dir / "commercial_park_price.csv",
        }
        table: Dict[ParkType, List[float]] = {}
        for park_type, path in file_map.items():
            with open(path, "r", encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            table[park_type] = [float(row["total_sell_price"]) for row in rows]
        return table

    def _load_grid_price_table(self) -> Dict[str, List[float]]:
        with open(self.config_dir / "grid_price_qinhuangdao_2026-3.csv", "r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        return {
            "time": [row["time"] for row in rows],
            "charge_price": [float(row["charge_price"]) for row in rows],
            "discharge_price": [float(row["discharge_price"]) for row in rows],
        }

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)


if __name__ == "__main__":
    env = ThreeParkChargingEnv(seed=42)
    obs, info = env.reset()
    print(info["weather"], obs["step"])
    for _ in range(2):
        obs, reward, terminated, truncated, step_info = env.step({})
        print(obs["step"], reward, terminated, truncated)
        print(step_info["tr_projection"])
