from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from env.three_park_charging_env import STEP_HOURS
from safety_design.ev_step_energy_bound import map_raw_action_to_ev_energy


EPS = 1e-9


@dataclass(frozen=True)
class LPMPCConfig:
    park_type: str
    algorithm_variant: str = "lp_mpc"
    horizon_steps: int = 8
    terminal_soc_weight: float = 8.0
    debt_weight: float = 10.0
    v2g_cycle_margin: float = 0.02
    bes_cycle_margin: float = 0.02
    pv_absorption_enabled: bool = True


class LocalLPMPCController:
    """Local receding-horizon linear EMS baseline with a merit-order LP solver."""

    def __init__(self, config: LPMPCConfig) -> None:
        self.config = config
        self.env: Any | None = None

    def bind_environment(self, env: Any) -> None:
        self.env = env

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
        """Evaluate TR probe actions with the same local MPC marginal-value proxy."""
        env = self._require_env()
        park_state = env.runtime_states[self.config.park_type]
        ev_bounds, _ = env._compute_action_bounds()
        local_bounds = ev_bounds[self.config.park_type]
        actions = node_action.detach().cpu().to(dtype=torch.float32).reshape(-1)
        score = 0.0
        for ev_idx in park_graph["ev_indexes"]:
            ev_id = park_graph["node_names"][ev_idx]
            account = env.debt_manager.get_account(ev_id)
            runtime = park_state.connected_evs[ev_id]
            grid_kwh = map_raw_action_to_ev_energy(float(actions[ev_idx].item()), local_bounds[ev_id])
            remaining_steps = max(1, int(runtime.session["departure_step"]) - env.current_step)
            target_gap = max(
                0.0,
                (float(runtime.session["target_soc"]) - account.current_soc)
                * account.battery_capacity_kwh,
            )
            required_battery_kwh = max(target_gap, account.debt_kwh)
            urgency = required_battery_kwh / max(
                EPS,
                remaining_steps * local_bounds[ev_id].max_charge_energy_port_kwh,
            )
            score += min(2.0, urgency) * max(0.0, grid_kwh)
            score -= self.config.debt_weight * max(0.0, account.debt_kwh - max(0.0, grid_kwh))
            score -= self.config.terminal_soc_weight * max(0.0, -grid_kwh)

        bes_idx = self._bes_node_index(park_graph)
        if bes_idx is not None:
            score -= 0.25 * abs(float(actions[bes_idx].item()) - self._build_bes_action({}))
        return score

    def _build_env_action(self) -> Dict[str, Any]:
        env = self._require_env()
        park_state = env.runtime_states[self.config.park_type]
        ev_bounds, _ = env._compute_action_bounds()
        local_bounds = ev_bounds[self.config.park_type]
        ev_actions: Dict[str, float] = {}
        for ev_id, runtime in park_state.connected_evs.items():
            account = env.debt_manager.get_account(ev_id)
            bound = local_bounds[ev_id]
            current_cap = bound.max_charge_energy_port_kwh
            remaining_steps = max(0, int(runtime.session["departure_step"]) - env.current_step)
            horizon = min(max(1, self.config.horizon_steps), max(1, remaining_steps))
            target_gap_battery = max(
                0.0,
                (float(runtime.session["target_soc"]) - account.current_soc)
                * account.battery_capacity_kwh,
            )
            required_battery = max(target_gap_battery, account.debt_kwh)
            required_grid = required_battery / max(float(runtime.session["eta_ch"]), EPS)
            slot_caps = [current_cap for _ in range(horizon)]
            charge_plan = self._allocate_required_charge(
                required_grid_kwh=required_grid,
                slot_caps_kwh=slot_caps,
                prices=self._future_charge_prices(horizon),
            )
            raw_action = charge_plan[0] / max(current_cap, EPS) if current_cap > EPS else 0.0
            if raw_action <= EPS:
                raw_action = self._optional_v2g_action(
                    runtime=runtime,
                    debt_kwh=account.debt_kwh,
                    required_grid_kwh=required_grid,
                    future_charge_capacity_kwh=sum(slot_caps[1:]),
                    max_discharge_kwh=bound.max_discharge_energy_port_kwh,
                )
            ev_actions[ev_id] = max(-1.0, min(1.0, raw_action))
        return {
            "bes": self._build_bes_action(ev_actions),
            "ev": ev_actions,
        }

    @staticmethod
    def _allocate_required_charge(
        required_grid_kwh: float,
        slot_caps_kwh: list[float],
        prices: list[float],
    ) -> list[float]:
        allocation = [0.0 for _ in slot_caps_kwh]
        remaining = max(0.0, required_grid_kwh)
        for slot in sorted(range(len(slot_caps_kwh)), key=lambda idx: (prices[idx], idx)):
            assigned = min(max(0.0, slot_caps_kwh[slot]), remaining)
            allocation[slot] = assigned
            remaining -= assigned
            if remaining <= EPS:
                break
        return allocation

    def _optional_v2g_action(
        self,
        runtime: Any,
        debt_kwh: float,
        required_grid_kwh: float,
        future_charge_capacity_kwh: float,
        max_discharge_kwh: float,
    ) -> float:
        env = self._require_env()
        if (
            not bool(runtime.session["v2g_enabled"])
            or debt_kwh > EPS
            or max_discharge_kwh <= EPS
            or future_charge_capacity_kwh <= required_grid_kwh + EPS
        ):
            return 0.0
        current_sale_price = float(env.grid_price_table["discharge_price"][env.current_step])
        compensation = env.debt_manager.get_compensation_price(self.config.park_type, env.current_step)
        horizon = min(self.config.horizon_steps, len(env.grid_price_table["charge_price"]) - env.current_step)
        cheapest_future_charge = min(self._future_charge_prices(max(1, horizon)))
        if current_sale_price - compensation <= cheapest_future_charge + self.config.v2g_cycle_margin:
            return 0.0
        slack = max(0.0, future_charge_capacity_kwh - required_grid_kwh)
        return -min(1.0, slack / max(max_discharge_kwh, EPS))

    def _build_bes_action(self, ev_actions: Dict[str, float]) -> float:
        env = self._require_env()
        park_state = env.runtime_states[self.config.park_type]
        ev_bounds, bes_bounds = env._compute_action_bounds()
        bound = bes_bounds[self.config.park_type]
        horizon = min(self.config.horizon_steps, len(env.grid_price_table["charge_price"]) - env.current_step)
        charge_prices = self._future_charge_prices(max(1, horizon))
        sale_prices = self._future_sale_prices(max(1, horizon))
        current_charge = charge_prices[0]
        current_sale = sale_prices[0]

        if bound.max_discharge_energy_port_kwh > EPS and current_sale >= max(sale_prices) - EPS:
            if current_sale > min(charge_prices) + self.config.bes_cycle_margin:
                return -1.0
        if bound.max_charge_energy_port_kwh > EPS and current_charge <= min(charge_prices) + EPS:
            if max(sale_prices) > current_charge + self.config.bes_cycle_margin:
                return 1.0
        if self.config.pv_absorption_enabled and bound.max_charge_energy_port_kwh > EPS:
            ev_charge_kwh = sum(
                max(0.0, map_raw_action_to_ev_energy(action, ev_bounds[self.config.park_type][ev_id]))
                for ev_id, action in ev_actions.items()
            )
            pv_energy_kwh = float(park_state.pv_kw[env.current_step]) * STEP_HOURS
            if pv_energy_kwh > ev_charge_kwh + EPS:
                return 1.0
        return 0.0

    def _future_charge_prices(self, horizon: int) -> list[float]:
        env = self._require_env()
        values = env.grid_price_table["charge_price"]
        return [float(values[min(env.current_step + offset, len(values) - 1)]) for offset in range(horizon)]

    def _future_sale_prices(self, horizon: int) -> list[float]:
        env = self._require_env()
        values = env.grid_price_table["discharge_price"]
        return [float(values[min(env.current_step + offset, len(values) - 1)]) for offset in range(horizon)]

    @staticmethod
    def _compose_node_action(park_graph: Dict[str, Any], env_action: Dict[str, Any]) -> torch.Tensor:
        node_action = torch.zeros(len(park_graph["node_types"]), dtype=torch.float32)
        bes_idx = LocalLPMPCController._bes_node_index(park_graph)
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
            raise RuntimeError("LP-MPC controller must be bound to an environment")
        return self.env

