from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from safety_design.ev_step_energy_bound import map_raw_action_to_ev_energy
from env.three_park_charging_env import STEP_HOURS


EPS = 1e-9


def _quantile(values: list[float], ratio: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile input must not be empty")
    position = max(0.0, min(1.0, ratio)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class TOUSOCUrgencyRuleConfig:
    park_type: str
    algorithm_variant: str = "rule_based_tou_soc"
    low_price_quantile: float = 0.30
    high_price_quantile: float = 0.70
    force_charge_urgency: float = 0.80
    allow_v2g_urgency: float = 0.40
    pv_absorption_enabled: bool = True


class LocalTOUSOCUrgencyRuleAgent:
    """Local TOU-SOC-urgency EMS baseline with no learned parameters."""

    def __init__(self, config: TOUSOCUrgencyRuleConfig) -> None:
        self.config = config
        self.env: Any | None = None
        self.low_charge_price: float = 0.0
        self.high_charge_price: float = 0.0

    def bind_environment(self, env: Any) -> None:
        self.env = env
        charge_prices = [float(value) for value in env.grid_price_table["charge_price"]]
        self.low_charge_price = _quantile(charge_prices, self.config.low_price_quantile)
        self.high_charge_price = _quantile(charge_prices, self.config.high_price_quantile)

    def act(
        self,
        park_graph: Dict[str, Any],
        deterministic: bool = False,
        return_node_action: bool = False,
    ) -> Dict[str, Any] | Tuple[Dict[str, Any], torch.Tensor]:
        del deterministic
        env_action = self._build_env_action()
        node_action = self._compose_node_action(park_graph, env_action)
        if return_node_action:
            return env_action, node_action
        return env_action

    def evaluate_cmdp_score(self, park_graph: Dict[str, Any], node_action: torch.Tensor) -> float:
        """Return a local long-horizon proxy score for strong-privacy TR probes."""
        env = self._require_env()
        park_state = env.runtime_states[self.config.park_type]
        ev_bounds, _ = env._compute_action_bounds()
        local_ev_bounds = ev_bounds[self.config.park_type]
        actions_by_node = node_action.detach().cpu().to(dtype=torch.float32).reshape(-1)
        score_loss = 0.0

        for ev_idx in park_graph["ev_indexes"]:
            ev_id = park_graph["node_names"][ev_idx]
            runtime = park_state.connected_evs[ev_id]
            account = env.debt_manager.get_account(ev_id)
            bound = local_ev_bounds[ev_id]
            raw_action = float(actions_by_node[ev_idx].item())
            executed_grid_kwh = map_raw_action_to_ev_energy(raw_action, bound)
            remaining_steps = max(0, int(runtime.session["departure_step"]) - env.current_step)
            required_kwh = max(
                0.0,
                (float(runtime.session["target_soc"]) - account.current_soc)
                * account.battery_capacity_kwh,
            )
            future_charge_capacity = max(EPS, remaining_steps * bound.max_charge_energy_port_kwh)
            urgency = required_kwh / future_charge_capacity
            charge_shortfall = max(0.0, required_kwh - max(0.0, executed_grid_kwh))
            score_loss += min(2.0, urgency) * charge_shortfall
            score_loss += 2.0 * max(0.0, account.debt_kwh - max(0.0, executed_grid_kwh))
            score_loss += 1.5 * max(0.0, -executed_grid_kwh)

        bes_idx = self._bes_node_index(park_graph)
        if bes_idx is not None:
            desired_bes_action = self._build_bes_action(ev_actions={})
            score_loss += 0.25 * abs(float(actions_by_node[bes_idx].item()) - desired_bes_action)
        return -score_loss

    def _build_env_action(self) -> Dict[str, Any]:
        env = self._require_env()
        park_state = env.runtime_states[self.config.park_type]
        ev_bounds, _ = env._compute_action_bounds()
        local_ev_bounds = ev_bounds[self.config.park_type]
        ev_actions: Dict[str, float] = {}
        for ev_id, runtime in park_state.connected_evs.items():
            account = env.debt_manager.get_account(ev_id)
            bound = local_ev_bounds[ev_id]
            remaining_steps = max(0, int(runtime.session["departure_step"]) - env.current_step)
            required_kwh = max(
                0.0,
                (float(runtime.session["target_soc"]) - account.current_soc)
                * account.battery_capacity_kwh,
            )
            future_charge_capacity = max(EPS, remaining_steps * bound.max_charge_energy_port_kwh)
            urgency = required_kwh / future_charge_capacity
            if account.debt_kwh > EPS or urgency >= self.config.force_charge_urgency:
                ev_actions[ev_id] = 1.0
            elif self._is_low_price() and required_kwh > EPS:
                ev_actions[ev_id] = 1.0
            elif (
                self._is_high_price()
                and bool(runtime.session["v2g_enabled"])
                and account.debt_kwh <= EPS
                and bound.max_discharge_energy_port_kwh > EPS
                and urgency <= self.config.allow_v2g_urgency
            ):
                ev_actions[ev_id] = -1.0
            else:
                ev_actions[ev_id] = 0.0
        return {
            "bes": self._build_bes_action(ev_actions),
            "ev": ev_actions,
        }

    def _build_bes_action(self, ev_actions: Dict[str, float]) -> float:
        env = self._require_env()
        if self._is_low_price():
            return 1.0
        if self._is_high_price():
            return -1.0
        if not self.config.pv_absorption_enabled:
            return 0.0

        park_state = env.runtime_states[self.config.park_type]
        ev_bounds, bes_bounds = env._compute_action_bounds()
        ev_charge_kwh = sum(
            max(0.0, map_raw_action_to_ev_energy(action, ev_bounds[self.config.park_type][ev_id]))
            for ev_id, action in ev_actions.items()
        )
        pv_energy_kwh = float(park_state.pv_kw[env.current_step]) * STEP_HOURS
        if pv_energy_kwh > ev_charge_kwh + EPS and bes_bounds[self.config.park_type].max_charge_energy_port_kwh > EPS:
            return 1.0
        return 0.0

    def _is_low_price(self) -> bool:
        env = self._require_env()
        return float(env.grid_price_table["charge_price"][env.current_step]) <= self.low_charge_price + EPS

    def _is_high_price(self) -> bool:
        env = self._require_env()
        return float(env.grid_price_table["charge_price"][env.current_step]) >= self.high_charge_price - EPS

    @staticmethod
    def _compose_node_action(park_graph: Dict[str, Any], env_action: Dict[str, Any]) -> torch.Tensor:
        node_action = torch.zeros(len(park_graph["node_types"]), dtype=torch.float32)
        bes_idx = LocalTOUSOCUrgencyRuleAgent._bes_node_index(park_graph)
        if bes_idx is not None:
            node_action[bes_idx] = float(env_action["bes"])
        for ev_idx in park_graph["ev_indexes"]:
            ev_id = park_graph["node_names"][ev_idx]
            node_action[ev_idx] = float(env_action["ev"].get(ev_id, 0.0))
        return node_action

    @staticmethod
    def _bes_node_index(park_graph: Dict[str, Any]) -> int | None:
        for node_idx, fixed_idx in zip(park_graph["action_node_indices"], park_graph["action_mapper"]):
            if fixed_idx == park_graph["bes_action_index"]:
                return int(node_idx)
        return None

    def _require_env(self) -> Any:
        if self.env is None:
            raise RuntimeError("rule-based agent must be bound to an environment")
        return self.env
