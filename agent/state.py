from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math

NODE_TYPE_TO_ID = {
    "cs": 0,
    "bes": 1,
    "pv": 2,
    "external": 3,
    "ev": 4,
}

RELATION_TYPE_TO_ID = {
    "bes_to_cs": 0,
    "cs_to_bes": 1,
    "pv_to_cs": 2,
    "cs_to_pv": 3,
    "external_to_cs": 4,
    "cs_to_external": 5,
    "ev_to_cs": 6,
    "cs_to_ev": 7,
}

HIGH_URGENCY_THRESHOLD = 1.0

LOCAL_NODE_SIZES = {
    "cs": 7,
    "bes": 4,
    "pv": 3,
    "external": 7,
    "ev": 8,
}

GLOBAL_NODE_SIZES = {
    "cs": 11,
    "bes": 4,
    "pv": 3,
    "external": 7,
    "ev": 8,
}


def normalize_privacy_mode(privacy_mode: str = "strong") -> str:
    if privacy_mode in {"strong", "local"}:
        return "strong"
    if privacy_mode in {"none", "global"}:
        return "none"
    raise ValueError("privacy_mode must be either 'strong' or 'none'.")


def get_observation_mode(privacy_mode: str = "strong") -> str:
    normalized_mode = normalize_privacy_mode(privacy_mode)
    return "local" if normalized_mode == "strong" else "global"


def get_node_sizes(privacy_mode: str = "strong") -> Dict[str, int]:
    observation_mode = get_observation_mode(privacy_mode)
    if observation_mode == "local":
        return dict(LOCAL_NODE_SIZES)
    if observation_mode == "global":
        return dict(GLOBAL_NODE_SIZES)
    raise ValueError("observation_mode must be either 'local' or 'global'.")


@dataclass
class ParkGraphState:
    park_type: str
    node_types: List[str]
    node_type_ids: List[int]
    node_names: List[str]
    edge_index: List[Tuple[int, int]]
    edge_type_ids: List[int]
    edge_type_names: List[str]
    active_ev_ids: List[str]
    ev_indexes: List[int]
    cs_indexes: List[int]
    bes_indexes: List[int]
    pv_indexes: List[int]
    external_indexes: List[int]
    ev_features: List[List[float]]
    cs_features: List[List[float]]
    bes_features: List[List[float]]
    pv_features: List[List[float]]
    external_features: List[List[float]]
    action_node_indices: List[int]
    action_mapper: List[int]
    fixed_action_dim: int
    bes_action_index: int

class StateBuilder:
    """
    Build one heterogeneous graph per park.

    Nodes:
    - CS
    - BES
    - PV
    - External
    - active EVs

    Bidirectional edges:
    - EV <-> CS
    - BES <-> CS
    - PV <-> CS
    - External <-> CS
    """

    def __init__(self, config_dir: str | Path, privacy_mode: str = "strong") -> None:
        self.privacy_mode = normalize_privacy_mode(privacy_mode)
        self.observation_mode = get_observation_mode(self.privacy_mode)
        self.weather_intensity = {
            "sunny": 0.0,
            "cloudy": 0.33,
            "overcast": 0.67,
            "rainy": 1.0,
        }
        self.park_type_one_hot = {
            "residential": [1.0, 0.0, 0.0],
            "office": [0.0, 1.0, 0.0],
            "commercial": [0.0, 0.0, 1.0],
        }
        self.node_sizes = get_node_sizes(self.privacy_mode)

    def build_state(self, env: Any, ev_bounds: Dict[str, Dict[str, Any]], bes_bounds: Dict[str, Any]) -> Dict[str, Any]:
        global_summary = self._build_global_summary(env=env, ev_bounds=ev_bounds, bes_bounds=bes_bounds)
        coordination_context = self._build_coordination_context(
            env=env,
            all_ev_bounds=ev_bounds,
            all_bes_bounds=bes_bounds,
        )
        park_graphs = {
            park_type: self.build_park_graph(
                env=env,
                park_type=park_type,
                all_ev_bounds=ev_bounds,
                all_bes_bounds=bes_bounds,
                global_summary=global_summary,
                coordination_context=coordination_context,
            )
            for park_type in env.runtime_states.keys()
        }
        return {
            "step": env.current_step,
            "time": env.grid_price_table["time"][min(env.current_step, len(env.grid_price_table["time"]) - 1)],
            "weather": env.daily_pv["weather"],
            "privacy_mode": self.privacy_mode,
            "park_graphs": {park_type: self.to_dict(graph) for park_type, graph in park_graphs.items()},
            "parks": {
                park_type: {"active_ev_ids": graph.active_ev_ids}
                for park_type, graph in park_graphs.items()
            },
        }

    def build_park_graph(
        self,
        env: Any,
        park_type: str,
        all_ev_bounds: Dict[str, Dict[str, Any]],
        all_bes_bounds: Dict[str, Any],
        global_summary: Dict[str, float],
        coordination_context: Dict[str, Dict[str, float]],
    ) -> ParkGraphState:
        park_state = env.runtime_states[park_type]
        ev_bounds = all_ev_bounds[park_type]
        bes_bound = all_bes_bounds[park_type]
        current_idx = min(env.current_step, len(env.grid_price_table["charge_price"]) - 1)
        episode_steps = max(1, len(env.grid_price_table["charge_price"]))
        day_progress = env.current_step / episode_steps
        angle = 2.0 * math.pi * day_progress
        active_ev_ids = list(park_state.connected_evs.keys())
        max_cs_limit_kwh = max(
            runtime_state.cs_limit_kwh
            for runtime_state in env.runtime_states.values()
        )
        charge_price = env.grid_price_table["charge_price"][current_idx]
        sell_price = env.sell_price_table[park_type][current_idx]
        pv_output_kw = park_state.pv_kw[current_idx]
        pv_reference_kw = self._compute_pv_reference_kw(park_state.pv_kw)
        step_hours = float(env.topology.get("time_step_min", 15)) / 60.0
        pv_output_kwh = pv_output_kw * step_hours
        weather_intensity = self._encode_weather_intensity(env.daily_pv["weather"])
        current_local_flexibility_ratio = self._compute_local_flexibility_ratio(
            park_state=park_state,
            ev_bounds=ev_bounds,
            bes_bound=bes_bound,
        )
        current_local_flexibility_ratio = self._squash_nonnegative_ratio(current_local_flexibility_ratio)
        local_control_summary = self._compute_local_control_summary(
            env=env,
            park_type=park_type,
            active_ev_ids=active_ev_ids,
            park_state=park_state,
            ev_bounds=ev_bounds,
            bes_bound=bes_bound,
            pv_output_kwh=pv_output_kwh,
            episode_steps=episode_steps,
        )
        local_occupancy_ratio = len(active_ev_ids) / max(1, park_state.cp_count)
        local_pv_ratio = pv_output_kwh / max(park_state.cs_limit_kwh, 1e-9)
        local_pv_ratio = self._squash_nonnegative_ratio(local_pv_ratio)
        node_types: List[str] = []
        node_type_ids: List[int] = []
        node_names: List[str] = []
        edge_index: List[Tuple[int, int]] = []
        edge_type_ids: List[int] = []
        edge_type_names: List[str] = []
        ev_indexes: List[int] = []
        cs_indexes: List[int] = []
        bes_indexes: List[int] = []
        pv_indexes: List[int] = []
        external_indexes: List[int] = []
        ev_features: List[List[float]] = []
        cs_features: List[List[float]] = []
        bes_features: List[List[float]] = []
        pv_features: List[List[float]] = []
        external_features: List[List[float]] = []
        action_node_indices: List[int] = []
        action_mapper: List[int] = []

        cs_index = self._append_node(node_types, node_type_ids, node_names, "cs", f"{park_type}_cs")
        cs_indexes.append(cs_index)
        if self.observation_mode == "global":
            other_coordination = coordination_context[park_type]
            cs_features.append(
                [
                    park_state.cs_limit_kwh / max(max_cs_limit_kwh, 1e-9),
                    local_occupancy_ratio,
                    current_local_flexibility_ratio,
                    local_control_summary["local_net_demand_ratio"],
                    local_control_summary["local_same_direction_controllable_ratio"],
                    env.prev_tr_projection_stats[park_type]["reduction_ratio"],
                    env.prev_tr_projection_stats[park_type]["local_penalty_normalized"],
                    other_coordination["other_total_responsibility_ratio"],
                    other_coordination["other_total_same_direction_capacity_ratio"],
                    other_coordination["max_other_responsibility_share"],
                    other_coordination["max_other_capacity_share"],
                ]
            )
        else:
            cs_features.append(
                [
                    park_state.cs_limit_kwh / max(max_cs_limit_kwh, 1e-9),
                    local_occupancy_ratio,
                    current_local_flexibility_ratio,
                    local_control_summary["local_net_demand_ratio"],
                    local_control_summary["local_same_direction_controllable_ratio"],
                    env.prev_tr_projection_stats[park_type]["reduction_ratio"],
                    env.prev_tr_projection_stats[park_type]["local_penalty_normalized"],
                ]
            )

        bes_index = self._append_node(node_types, node_type_ids, node_names, "bes", f"{park_type}_bes")
        bes_indexes.append(bes_index)
        if self.observation_mode == "global":
            bes_features.append(
                [
                    park_state.bes_soc,
                    self._squash_nonnegative_ratio(
                        bes_bound.max_charge_energy_port_kwh / max(park_state.cs_limit_kwh, 1e-9)
                    ),
                    self._squash_nonnegative_ratio(
                        bes_bound.max_discharge_energy_port_kwh / max(park_state.cs_limit_kwh, 1e-9)
                    ),
                    self._compute_bes_charge_headroom_ratio(park_state),
                ]
            )
        else:
            bes_features.append(
                [
                    park_state.bes_soc,
                    self._squash_nonnegative_ratio(
                        bes_bound.max_charge_energy_port_kwh / max(park_state.cs_limit_kwh, 1e-9)
                    ),
                    self._squash_nonnegative_ratio(
                        bes_bound.max_discharge_energy_port_kwh / max(park_state.cs_limit_kwh, 1e-9)
                    ),
                    self._compute_bes_charge_headroom_ratio(park_state),
                ]
            )
        action_node_indices.append(bes_index)
        action_mapper.append(park_state.cp_count)
        self._connect_bi(
            edge_index,
            edge_type_ids,
            edge_type_names,
            bes_index,
            cs_index,
            "bes_to_cs",
            "cs_to_bes",
        )

        pv_index = self._append_node(node_types, node_type_ids, node_names, "pv", f"{park_type}_pv")
        pv_indexes.append(pv_index)
        if self.observation_mode == "global":
            pv_features.append(
                [
                    pv_output_kw / max(pv_reference_kw, 1e-9),
                    local_pv_ratio,
                    env.prev_tr_projection_stats[park_type]["pv_curtailment_ratio"],
                ]
            )
        else:
            pv_features.append(
                [
                    pv_output_kw / max(pv_reference_kw, 1e-9),
                    local_pv_ratio,
                    env.prev_tr_projection_stats[park_type]["pv_curtailment_ratio"],
                ]
            )
        self._connect_bi(
            edge_index,
            edge_type_ids,
            edge_type_names,
            pv_index,
            cs_index,
            "pv_to_cs",
            "cs_to_pv",
        )

        external_index = self._append_node(node_types, node_type_ids, node_names, "external", f"{park_type}_external")
        external_indexes.append(external_index)
        if self.observation_mode == "global":
            external_features.append(
                [
                    math.sin(angle),
                    math.cos(angle),
                    weather_intensity,
                    self._min_max_normalize(
                        charge_price,
                        min(env.grid_price_table["charge_price"]),
                        max(env.grid_price_table["charge_price"]),
                    ),
                    self._min_max_normalize(
                        sell_price,
                        min(min(price_series) for price_series in env.sell_price_table.values()),
                        max(max(price_series) for price_series in env.sell_price_table.values()),
                    ),
                    env.prev_tr_projection_stats[park_type]["global_overload_ratio"],
                    env.prev_tr_projection_stats[park_type]["signed_control_signal"],
                ]
            )
        else:
            external_features.append(
                [
                    math.sin(angle),
                    math.cos(angle),
                    weather_intensity,
                    self._min_max_normalize(
                        charge_price,
                        min(env.grid_price_table["charge_price"]),
                        max(env.grid_price_table["charge_price"]),
                    ),
                    self._min_max_normalize(
                        sell_price,
                        min(min(price_series) for price_series in env.sell_price_table.values()),
                        max(max(price_series) for price_series in env.sell_price_table.values()),
                    ),
                    env.prev_tr_projection_stats[park_type]["global_overload_ratio"],
                    env.prev_tr_projection_stats[park_type]["signed_control_signal"],
                ]
            )
        self._connect_bi(
            edge_index,
            edge_type_ids,
            edge_type_names,
            external_index,
            cs_index,
            "external_to_cs",
            "cs_to_external",
        )

        for ev_id in active_ev_ids:
            runtime = park_state.connected_evs[ev_id]
            session = runtime.session
            account = env.debt_manager.get_account(ev_id)
            bound = ev_bounds[ev_id]
            remaining_steps = max(0, session["departure_step"] - env.current_step)
            urgency_ratio = self._compute_ev_urgency_ratio(
                current_soc=account.current_soc,
                target_soc=session["target_soc"],
                battery_capacity_kwh=session["battery_capacity_kwh"],
                remaining_steps=remaining_steps,
                max_charge_energy_per_step_kwh=bound.max_charge_energy_port_kwh,
            )
            ev_index = self._append_node(node_types, node_type_ids, node_names, "ev", ev_id)
            ev_indexes.append(ev_index)
            if self.observation_mode == "global":
                ev_features.append(
                    [
                        account.current_soc,
                        session["target_soc"],
                        float(remaining_steps) / episode_steps,
                        self._squash_nonnegative_ratio(
                            bound.max_charge_energy_port_kwh / max(park_state.cs_limit_kwh, 1e-9)
                        ),
                        self._squash_nonnegative_ratio(
                            bound.max_discharge_energy_port_kwh / max(park_state.cs_limit_kwh, 1e-9)
                        ),
                        local_control_summary["high_urgency_ev_ratio"],
                        account.debt_kwh / max(account.battery_capacity_kwh, 1e-9),
                        1.0 if session["v2g_enabled"] else 0.0,
                    ]
                )
            else:
                ev_features.append(
                    [
                        account.current_soc,
                        session["target_soc"],
                        float(remaining_steps) / episode_steps,
                        self._squash_nonnegative_ratio(
                            bound.max_charge_energy_port_kwh / max(park_state.cs_limit_kwh, 1e-9)
                        ),
                        self._squash_nonnegative_ratio(
                            bound.max_discharge_energy_port_kwh / max(park_state.cs_limit_kwh, 1e-9)
                        ),
                        local_control_summary["high_urgency_ev_ratio"],
                        account.debt_kwh / max(account.battery_capacity_kwh, 1e-9),
                        1.0 if session["v2g_enabled"] else 0.0,
                    ]
                )
            action_node_indices.append(ev_index)
            action_mapper.append(self._cp_id_to_action_index(runtime.cp_id))
            self._connect_bi(
                edge_index,
                edge_type_ids,
                edge_type_names,
                ev_index,
                cs_index,
                "ev_to_cs",
                "cs_to_ev",
            )

        self._validate_feature_dimensions(
            cs_features=cs_features,
            bes_features=bes_features,
            pv_features=pv_features,
            external_features=external_features,
            ev_features=ev_features,
        )

        return ParkGraphState(
            park_type=park_type,
            node_types=node_types,
            node_type_ids=node_type_ids,
            node_names=node_names,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_type_names=edge_type_names,
            active_ev_ids=active_ev_ids,
            ev_indexes=ev_indexes,
            cs_indexes=cs_indexes,
            bes_indexes=bes_indexes,
            pv_indexes=pv_indexes,
            external_indexes=external_indexes,
            ev_features=ev_features,
            cs_features=cs_features,
            bes_features=bes_features,
            pv_features=pv_features,
            external_features=external_features,
            action_node_indices=action_node_indices,
            action_mapper=action_mapper,
            fixed_action_dim=park_state.cp_count + 1,
            bes_action_index=park_state.cp_count,
        )

    @staticmethod
    def to_dict(graph: ParkGraphState) -> Dict[str, Any]:
        return {
            "park_type": graph.park_type,
            "node_types": graph.node_types,
            "node_type_ids": graph.node_type_ids,
            "node_names": graph.node_names,
            "edge_index": graph.edge_index,
            "edge_type_ids": graph.edge_type_ids,
            "edge_type_names": graph.edge_type_names,
            "active_ev_ids": graph.active_ev_ids,
            "ev_indexes": graph.ev_indexes,
            "cs_indexes": graph.cs_indexes,
            "bes_indexes": graph.bes_indexes,
            "pv_indexes": graph.pv_indexes,
            "external_indexes": graph.external_indexes,
            "ev_features": graph.ev_features,
            "cs_features": graph.cs_features,
            "bes_features": graph.bes_features,
            "pv_features": graph.pv_features,
            "external_features": graph.external_features,
            "action_node_indices": graph.action_node_indices,
            "action_mapper": graph.action_mapper,
            "fixed_action_dim": graph.fixed_action_dim,
            "bes_action_index": graph.bes_action_index,
        }

    @staticmethod
    def _append_node(
        node_types: List[str],
        node_type_ids: List[int],
        node_names: List[str],
        node_type: str,
        node_name: str,
    ) -> int:
        index = len(node_types)
        node_types.append(node_type)
        node_type_ids.append(NODE_TYPE_TO_ID[node_type])
        node_names.append(node_name)
        return index

    @staticmethod
    def _connect_bi(
        edge_index: List[Tuple[int, int]],
        edge_type_ids: List[int],
        edge_type_names: List[str],
        src: int,
        dst: int,
        forward_relation: str,
        reverse_relation: str,
    ) -> None:
        edge_index.append((src, dst))
        edge_type_ids.append(RELATION_TYPE_TO_ID[forward_relation])
        edge_type_names.append(forward_relation)
        edge_index.append((dst, src))
        edge_type_ids.append(RELATION_TYPE_TO_ID[reverse_relation])
        edge_type_names.append(reverse_relation)

    @staticmethod
    def _cp_id_to_action_index(cp_id: str) -> int:
        return int(cp_id.rsplit("_", 1)[-1])

    @staticmethod
    def _compute_local_flexibility_ratio(
        park_state: Any,
        ev_bounds: Dict[str, Any],
        bes_bound: Any,
    ) -> float:
        ev_total_flexibility_kwh = sum(
            bound.max_charge_energy_port_kwh + bound.max_discharge_energy_port_kwh
            for bound in ev_bounds.values()
        )
        bes_total_flexibility_kwh = (
            bes_bound.max_charge_energy_port_kwh
            + bes_bound.max_discharge_energy_port_kwh
        )
        return (ev_total_flexibility_kwh + bes_total_flexibility_kwh) / max(park_state.cs_limit_kwh, 1e-9)

    @staticmethod
    def _compute_bes_charge_headroom_ratio(park_state: Any) -> float:
        feasible_soc_span = max(park_state.bes_spec.soc_max - park_state.bes_spec.soc_min, 1e-9)
        return max(0.0, park_state.bes_spec.soc_max - park_state.bes_soc) / feasible_soc_span

    @staticmethod
    def _compute_pv_reference_kw(pv_profile_kw: List[float]) -> float:
        return max(max(pv_profile_kw, default=0.0), 1e-9)

    @staticmethod
    def _min_max_normalize(value: float, lower: float, upper: float) -> float:
        if upper - lower <= 1e-9:
            return 0.0
        return (value - lower) / (upper - lower)

    def _encode_weather_intensity(self, weather: str) -> float:
        return self.weather_intensity.get(weather, 0.5)

    @staticmethod
    def _build_global_summary(env: Any, ev_bounds: Dict[str, Dict[str, Any]], bes_bounds: Dict[str, Any]) -> Dict[str, float]:
        total_cs_limit_kwh = sum(runtime_state.cs_limit_kwh for runtime_state in env.runtime_states.values())
        total_active_evs = sum(len(runtime_state.connected_evs) for runtime_state in env.runtime_states.values())
        total_cp_count = sum(runtime_state.cp_count for runtime_state in env.runtime_states.values())
        total_pv_output_kwh = sum(
            runtime_state.pv_kw[min(env.current_step, len(runtime_state.pv_kw) - 1)] * (float(env.topology.get("time_step_min", 15)) / 60.0)
            for runtime_state in env.runtime_states.values()
        )
        total_ev_charge_capacity_kwh = sum(
            bound.max_charge_energy_port_kwh
            for park_bounds in ev_bounds.values()
            for bound in park_bounds.values()
        )
        total_ev_discharge_capacity_kwh = sum(
            bound.max_discharge_energy_port_kwh
            for park_bounds in ev_bounds.values()
            for bound in park_bounds.values()
        )
        total_bes_charge_capacity_kwh = sum(bound.max_charge_energy_port_kwh for bound in bes_bounds.values())
        total_bes_discharge_capacity_kwh = sum(bound.max_discharge_energy_port_kwh for bound in bes_bounds.values())
        total_charge_capacity_kwh = total_ev_charge_capacity_kwh + total_bes_charge_capacity_kwh
        total_export_support_kwh = total_ev_discharge_capacity_kwh + total_bes_discharge_capacity_kwh + total_pv_output_kwh
        total_discharge_capacity_kwh = total_ev_discharge_capacity_kwh + total_bes_discharge_capacity_kwh
        net_demand_kwh = total_charge_capacity_kwh - total_export_support_kwh
        return {
            "total_cs_limit_kwh": total_cs_limit_kwh,
            "total_active_evs": float(total_active_evs),
            "total_cp_count": float(total_cp_count),
            "global_active_ev_ratio": total_active_evs / max(total_cp_count, 1),
            "total_pv_output_kwh": total_pv_output_kwh,
            "global_pv_ratio": total_pv_output_kwh / max(total_cs_limit_kwh, 1e-9),
            "global_charge_capacity_ratio": total_charge_capacity_kwh / max(total_cs_limit_kwh, 1e-9),
            "global_discharge_capacity_ratio": total_discharge_capacity_kwh / max(total_cs_limit_kwh, 1e-9),
            "global_flexibility_ratio": (total_charge_capacity_kwh + total_discharge_capacity_kwh) / max(total_cs_limit_kwh, 1e-9),
            "global_net_demand_ratio": StateBuilder._squash_signed_value(
                net_demand_kwh / max(total_cs_limit_kwh, 1e-9)
            ),
        }

    @staticmethod
    def _compute_ev_urgency_ratio(
        current_soc: float,
        target_soc: float,
        battery_capacity_kwh: float,
        remaining_steps: int,
        max_charge_energy_per_step_kwh: float,
    ) -> float:
        needed_energy_kwh = max(0.0, target_soc - current_soc) * battery_capacity_kwh
        future_charge_capacity_kwh = max(1e-9, remaining_steps * max_charge_energy_per_step_kwh)
        return needed_energy_kwh / future_charge_capacity_kwh

    @staticmethod
    def _compute_local_control_summary(
        env: Any,
        park_type: str,
        active_ev_ids: List[str],
        park_state: Any,
        ev_bounds: Dict[str, Any],
        bes_bound: Any,
        pv_output_kwh: float,
        episode_steps: int,
    ) -> Dict[str, float]:
        del park_type, episode_steps
        total_ev_charge_capacity_kwh = 0.0
        total_ev_discharge_capacity_kwh = 0.0
        high_urgency_count = 0

        for ev_id in active_ev_ids:
            runtime = park_state.connected_evs[ev_id]
            session = runtime.session
            account = env.debt_manager.get_account(ev_id)
            bound = ev_bounds[ev_id]
            total_ev_charge_capacity_kwh += float(bound.max_charge_energy_port_kwh)
            total_ev_discharge_capacity_kwh += float(bound.max_discharge_energy_port_kwh)
            remaining_steps = max(0, session["departure_step"] - env.current_step)
            urgency_ratio = StateBuilder._compute_ev_urgency_ratio(
                current_soc=account.current_soc,
                target_soc=session["target_soc"],
                battery_capacity_kwh=session["battery_capacity_kwh"],
                remaining_steps=remaining_steps,
                max_charge_energy_per_step_kwh=bound.max_charge_energy_port_kwh,
            )
            if urgency_ratio >= HIGH_URGENCY_THRESHOLD:
                high_urgency_count += 1

        total_charge_capacity_kwh = total_ev_charge_capacity_kwh + float(bes_bound.max_charge_energy_port_kwh)
        total_export_support_kwh = (
            total_ev_discharge_capacity_kwh
            + float(bes_bound.max_discharge_energy_port_kwh)
            + max(0.0, pv_output_kwh)
        )
        cs_limit_kwh = max(float(park_state.cs_limit_kwh), 1e-9)
        net_pressure_kwh = total_charge_capacity_kwh - total_export_support_kwh

        if net_pressure_kwh >= 0.0:
            same_direction_controllable_kwh = total_charge_capacity_kwh
            responsibility_kwh = max(0.0, net_pressure_kwh)
        else:
            same_direction_controllable_kwh = total_export_support_kwh
            responsibility_kwh = max(0.0, -net_pressure_kwh)

        return {
            "local_net_demand_ratio": StateBuilder._squash_signed_value(net_pressure_kwh / cs_limit_kwh),
            "local_same_direction_controllable_ratio": StateBuilder._squash_nonnegative_ratio(
                same_direction_controllable_kwh / cs_limit_kwh
            ),
            "local_tr_responsibility_ratio": StateBuilder._squash_nonnegative_ratio(
                responsibility_kwh / cs_limit_kwh
            ),
            "high_urgency_ev_ratio": high_urgency_count / max(1, len(active_ev_ids)),
        }

    @staticmethod
    def _squash_nonnegative_ratio(value: float) -> float:
        return math.tanh(max(0.0, value))

    @staticmethod
    def _squash_signed_value(value: float) -> float:
        return math.tanh(value)

    def _validate_feature_dimensions(
        self,
        cs_features: List[List[float]],
        bes_features: List[List[float]],
        pv_features: List[List[float]],
        external_features: List[List[float]],
        ev_features: List[List[float]],
    ) -> None:
        feature_groups = {
            "cs": cs_features,
            "bes": bes_features,
            "pv": pv_features,
            "external": external_features,
            "ev": ev_features,
        }
        for node_type, features in feature_groups.items():
            expected_dim = self.node_sizes[node_type]
            for feature_vector in features:
                if len(feature_vector) != expected_dim:
                    raise ValueError(
                        f"{self.observation_mode} mode {node_type} feature dim mismatch: "
                        f"expected {expected_dim}, got {len(feature_vector)}"
                    )

    @staticmethod
    def _build_other_park_context(
        env: Any,
        park_type: str,
        all_ev_bounds: Dict[str, Dict[str, Any]],
        all_bes_bounds: Dict[str, Any],
    ) -> Dict[str, List[float]]:
        ordered_parks = ("residential", "office", "commercial")
        occupancy_ratios: List[float] = []
        flex_ratios: List[float] = []
        net_demand_ratios: List[float] = []
        same_direction_controllable_ratios: List[float] = []
        pv_ratios: List[float] = []
        bes_soc: List[float] = []

        for slot_park in ordered_parks:
            if slot_park == park_type or slot_park not in env.runtime_states:
                occupancy_ratios.append(0.0)
                flex_ratios.append(0.0)
                net_demand_ratios.append(0.0)
                same_direction_controllable_ratios.append(0.0)
                pv_ratios.append(0.0)
                bes_soc.append(0.0)
                continue

            other_state = env.runtime_states[slot_park]
            other_ev_bounds = all_ev_bounds[slot_park]
            other_bes_bound = all_bes_bounds[slot_park]
            occupancy_ratios.append(len(other_state.connected_evs) / max(1, other_state.cp_count))
            flex_ratios.append(
                StateBuilder._squash_nonnegative_ratio(
                    StateBuilder._compute_local_flexibility_ratio(
                        park_state=other_state,
                        ev_bounds=other_ev_bounds,
                        bes_bound=other_bes_bound,
                    )
                )
            )
            pv_output_kwh = other_state.pv_kw[min(env.current_step, len(other_state.pv_kw) - 1)] * (
                float(env.topology.get("time_step_min", 15)) / 60.0
            )
            other_control_summary = StateBuilder._compute_local_control_summary(
                env=env,
                park_type=slot_park,
                active_ev_ids=list(other_state.connected_evs.keys()),
                park_state=other_state,
                ev_bounds=other_ev_bounds,
                bes_bound=other_bes_bound,
                pv_output_kwh=pv_output_kwh,
                episode_steps=max(1, len(env.grid_price_table["charge_price"])),
            )
            net_demand_ratios.append(other_control_summary["local_net_demand_ratio"])
            same_direction_controllable_ratios.append(
                other_control_summary["local_same_direction_controllable_ratio"]
            )
            pv_ratios.append(
                StateBuilder._squash_nonnegative_ratio(
                    pv_output_kwh / max(other_state.cs_limit_kwh, 1e-9)
                )
            )
            bes_soc.append(other_state.bes_soc)

        return {
            "park_occupancy_ratios": occupancy_ratios,
            "park_flex_ratios": flex_ratios,
            "park_net_demand_ratios": net_demand_ratios,
            "park_same_direction_controllable_ratios": same_direction_controllable_ratios,
            "park_pv_ratios": pv_ratios,
            "park_bes_soc": bes_soc,
        }

    @staticmethod
    def _build_coordination_context(
        env: Any,
        all_ev_bounds: Dict[str, Dict[str, Any]],
        all_bes_bounds: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        responsibility_ratio_by_park: Dict[str, float] = {}
        capacity_ratio_by_park: Dict[str, float] = {}

        for park_type, park_state in env.runtime_states.items():
            pv_output_kwh = park_state.pv_kw[min(env.current_step, len(park_state.pv_kw) - 1)] * (
                float(env.topology.get("time_step_min", 15)) / 60.0
            )
            control_summary = StateBuilder._compute_local_control_summary(
                env=env,
                park_type=park_type,
                active_ev_ids=list(park_state.connected_evs.keys()),
                park_state=park_state,
                ev_bounds=all_ev_bounds[park_type],
                bes_bound=all_bes_bounds[park_type],
                pv_output_kwh=pv_output_kwh,
                episode_steps=max(1, len(env.grid_price_table["charge_price"])),
            )
            responsibility_ratio_by_park[park_type] = control_summary["local_tr_responsibility_ratio"]
            capacity_ratio_by_park[park_type] = control_summary["local_same_direction_controllable_ratio"]

        total_responsibility_ratio = sum(responsibility_ratio_by_park.values())
        total_capacity_ratio = sum(capacity_ratio_by_park.values())
        context: Dict[str, Dict[str, float]] = {}
        for park_type in env.runtime_states.keys():
            other_responsibilities = [
                responsibility_ratio_by_park[other_park]
                for other_park in env.runtime_states.keys()
                if other_park != park_type
            ]
            other_capacities = [
                capacity_ratio_by_park[other_park]
                for other_park in env.runtime_states.keys()
                if other_park != park_type
            ]
            other_total_responsibility = sum(other_responsibilities)
            other_total_capacity = sum(other_capacities)
            context[park_type] = {
                "other_total_responsibility_ratio": other_total_responsibility,
                "other_total_same_direction_capacity_ratio": other_total_capacity,
                "max_other_responsibility_share": (
                    max(other_responsibilities) / max(total_responsibility_ratio, 1e-9)
                    if other_responsibilities
                    else 0.0
                ),
                "max_other_capacity_share": (
                    max(other_capacities) / max(total_capacity_ratio, 1e-9)
                    if other_capacities
                    else 0.0
                ),
            }
        return context
