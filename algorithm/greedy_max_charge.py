from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from safety_design.ev_step_energy_bound import map_raw_action_to_ev_energy


EPS = 1e-9


@dataclass(frozen=True)
class GreedyMaxChargeConfig:
    park_type: str
    algorithm_variant: str = "greedy_max_charge"


class LocalGreedyMaxChargeAgent:
    """Greedy baseline: keep BES idle and charge every connected EV at max power."""

    def __init__(self, config: GreedyMaxChargeConfig) -> None:
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
        env_action = {
            "bes": 0.0,
            "ev": {
                str(park_graph["node_names"][ev_idx]): 1.0
                for ev_idx in park_graph["ev_indexes"]
            },
        }
        node_action = self._compose_node_action(park_graph, env_action)
        if return_node_action:
            return env_action, node_action
        return env_action

    def evaluate_cmdp_score(self, park_graph: Dict[str, Any], node_action: torch.Tensor) -> float:
        """Local proxy used by strong-privacy TR probes."""
        env = self._require_env()
        park_state = env.runtime_states[self.config.park_type]
        ev_bounds, _ = env._compute_action_bounds()
        local_ev_bounds = ev_bounds[self.config.park_type]
        actions_by_node = node_action.detach().cpu().to(dtype=torch.float32).reshape(-1)
        score = 0.0

        for ev_idx in park_graph["ev_indexes"]:
            ev_id = park_graph["node_names"][ev_idx]
            runtime = park_state.connected_evs[ev_id]
            account = env.debt_manager.get_account(ev_id)
            bound = local_ev_bounds[ev_id]
            raw_action = float(actions_by_node[ev_idx].item())
            executed_grid_kwh = max(0.0, map_raw_action_to_ev_energy(raw_action, bound))
            required_kwh = max(
                0.0,
                (float(runtime.session["target_soc"]) - account.current_soc)
                * account.battery_capacity_kwh,
            )
            score += min(required_kwh, executed_grid_kwh)
            score -= 2.0 * max(0.0, account.debt_kwh - executed_grid_kwh)
            score -= max(0.0, required_kwh - executed_grid_kwh)

        bes_idx = self._bes_node_index(park_graph)
        if bes_idx is not None:
            score -= 0.1 * abs(float(actions_by_node[bes_idx].item()))
        return float(score)

    @staticmethod
    def _compose_node_action(park_graph: Dict[str, Any], env_action: Dict[str, Any]) -> torch.Tensor:
        node_action = torch.zeros(len(park_graph["node_types"]), dtype=torch.float32)
        bes_idx = LocalGreedyMaxChargeAgent._bes_node_index(park_graph)
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
            raise RuntimeError("greedy max-charge agent must be bound to an environment")
        return self.env
