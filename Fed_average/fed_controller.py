from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable
import copy

import torch


@dataclass
class FederatedConfig:
    warmup_episodes: int = 0
    early_phase_end_episode: int = 150
    mid_phase_end_episode: int = 300
    early_phase_interval: int = 5
    mid_phase_interval: int = 10
    late_phase_interval: int = 20


class FederatedAveragingCoordinator:
    """Coordinate weighted FedAvg over the shared backbones."""

    def __init__(self, config: FederatedConfig | None = None) -> None:
        self.config = config or FederatedConfig()

    def should_aggregate(self, episode_idx: int) -> bool:
        completed_episodes = episode_idx + 1
        if completed_episodes <= self.config.warmup_episodes:
            return False

        if completed_episodes <= self.config.early_phase_end_episode:
            aggregation_interval = self.config.early_phase_interval
        elif completed_episodes <= self.config.mid_phase_end_episode:
            aggregation_interval = self.config.mid_phase_interval
        else:
            aggregation_interval = self.config.late_phase_interval

        return (completed_episodes - self.config.warmup_episodes) % aggregation_interval == 0

    def aggregate(
        self,
        local_agents: Dict[str, object],
        normalized_weights: Dict[str, float] | None = None,
        block_weights: Dict[str, Dict[str, float]] | None = None,
        selected_blocks: Iterable[str] | None = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        park_ids = list(local_agents.keys())
        if not normalized_weights:
            normalized_weights = {park_id: 1.0 / len(park_ids) for park_id in park_ids}
        if block_weights is None:
            block_weights = {}

        local_states = {park_id: local_agents[park_id].get_shared_state() for park_id in park_ids}
        relation_masks = {
            park_id: (
                local_agents[park_id].get_actor_relation_fed_mask()
                if hasattr(local_agents[park_id], "get_actor_relation_fed_mask")
                else {}
            )
            for park_id in park_ids
        }
        block_names = list(selected_blocks) if selected_blocks is not None else list(local_states[park_ids[0]].keys())
        aggregated = {
            block_name: copy.deepcopy(local_states[park_ids[0]][block_name])
            for block_name in block_names
        }

        for block_name in aggregated.keys():
            current_block_weights = block_weights.get(block_name, normalized_weights)
            for param_name in aggregated[block_name].keys():
                relation_name = self._relation_name_from_state_key(param_name)
                aggregated_value = None
                weight_sum = 0.0
                for park_id in park_ids:
                    relation_mask = relation_masks.get(park_id, {})
                    if relation_name is not None and relation_mask and not relation_mask.get(relation_name, True):
                        continue
                    source_weight = current_block_weights[park_id]
                    weighted_value = local_states[park_id][block_name][param_name] * source_weight
                    aggregated_value = weighted_value if aggregated_value is None else aggregated_value + weighted_value
                    weight_sum += float(source_weight)
                if aggregated_value is not None and weight_sum > 0.0:
                    aggregated[block_name][param_name] = aggregated_value / weight_sum

        for park_id in park_ids:
            local_agents[park_id].load_shared_state(aggregated)
            local_agents[park_id].set_global_actor_reference(aggregated)
        return aggregated

    @staticmethod
    def _relation_name_from_state_key(key: str) -> str | None:
        marker = ".shared."
        if marker not in key:
            return None
        tail = key.split(marker, 1)[1]
        return tail.split(".", 1)[0]
