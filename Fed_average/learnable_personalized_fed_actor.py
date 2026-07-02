from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import math

import torch
from torch import nn

from Fed_average.fed_controller import FederatedConfig


EPS = 1e-9


@dataclass
class LearnablePersonalizedFedActorState:
    fed_logits: torch.Tensor
    rho_logits: torch.Tensor
    fed_optimizer: Dict[str, Any]
    rho_optimizer: Dict[str, Any]
    aggregate_count: int
    last_metrics: Dict[str, Any]


class LearnablePersonalizedFedActorCoordinator(nn.Module):
    """Learnable advantage-gated personalized federated actor coordinator."""

    def __init__(
        self,
        config: FederatedConfig | None,
        park_ids: List[str],
        fed_logits_lr: float,
        rho_logits_lr: float,
        fed_logits_diag_init: float,
        fed_logits_offdiag_init: float,
        rho_init: float,
        candidate_gate_margin: Dict[str, float] | float,
        eta_probe: Dict[str, float] | float,
        eta_max: Dict[str, float] | float,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.config = config or FederatedConfig()
        self.park_ids = list(park_ids)
        self.num_parks = len(self.park_ids)
        self.park_index = {park_id: idx for idx, park_id in enumerate(self.park_ids)}
        self.device = torch.device(device)

        fed_init = torch.full(
            (self.num_parks, self.num_parks),
            float(fed_logits_offdiag_init),
            dtype=torch.float32,
            device=self.device,
        )
        fed_init.fill_diagonal_(float(fed_logits_diag_init))
        self.fed_logits = nn.Parameter(fed_init)
        rho_init_clamped = min(max(float(rho_init), EPS), 1.0 - EPS)
        rho_logit_init = math.log(rho_init_clamped / (1.0 - rho_init_clamped))
        self.rho_logits = nn.Parameter(
            torch.full((self.num_parks,), rho_logit_init, dtype=torch.float32, device=self.device)
        )
        self.fed_optimizer = torch.optim.Adam([self.fed_logits], lr=fed_logits_lr)
        self.rho_optimizer = torch.optim.Adam([self.rho_logits], lr=rho_logits_lr)
        self.candidate_gate_margin = self._resolve_park_values(candidate_gate_margin)
        self.eta_probe = self._resolve_park_values(eta_probe)
        self.eta_max = self._resolve_park_values(eta_max)
        self.aggregate_count = 0
        self.last_metrics: Dict[str, Any] = {}

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

    def aggregate(self, local_agents: Dict[str, object]) -> Dict[str, Any]:
        local_states = {
            park_id: local_agents[park_id].get_shared_state()["actor_backbone"]
            for park_id in self.park_ids
        }
        relation_masks = {
            park_id: (
                local_agents[park_id].get_actor_relation_fed_mask()
                if hasattr(local_agents[park_id], "get_actor_relation_fed_mask")
                else {}
            )
            for park_id in self.park_ids
        }

        source_advantage = self.compute_source_advantage_matrix(local_agents, local_states)
        source_advantage_norm = self._rowwise_normalize_by_mean_abs(source_advantage)
        self.update_fed_logits(source_advantage_norm)

        fed_weights = torch.softmax(self.fed_logits, dim=1).detach()
        candidates = self.build_personalized_candidates(local_states, fed_weights, relation_masks)

        candidate_advantage = self.compute_candidate_advantages(local_agents, candidates)
        candidate_advantage_norm = self._normalize_candidate_advantages(candidate_advantage, source_advantage)
        self.update_rho_logits(candidate_advantage_norm)

        rho = torch.sigmoid(self.rho_logits).detach()
        eta = self.soft_load_candidates(local_agents, candidates, candidate_advantage_norm, rho)

        metrics = self._build_metrics(
            fed_weights=fed_weights,
            rho=rho,
            eta=eta,
            source_advantage=source_advantage,
            source_advantage_norm=source_advantage_norm,
            candidate_advantage=candidate_advantage,
            candidate_advantage_norm=candidate_advantage_norm,
        )
        self.aggregate_count += 1
        metrics["aggregate_count"] = self.aggregate_count
        self.last_metrics = metrics
        return metrics

    def compute_source_advantage_matrix(
        self,
        local_agents: Dict[str, object],
        local_states: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        advantage = torch.zeros((self.num_parks, self.num_parks), dtype=torch.float32, device=self.device)
        for i, target_park in enumerate(self.park_ids):
            agent = local_agents[target_park]
            for j, source_park in enumerate(self.park_ids):
                if i == j:
                    continue
                score = agent.evaluate_external_backbone_advantage(local_states[source_park])
                advantage[i, j] = float(score)
        return advantage

    def update_fed_logits(self, normalized_advantage: torch.Tensor) -> None:
        self.fed_optimizer.zero_grad()
        weights = torch.softmax(self.fed_logits, dim=1)
        loss = -(weights * normalized_advantage.detach()).sum()
        loss.backward()
        self.fed_optimizer.step()

    def build_personalized_candidates(
        self,
        local_states: Dict[str, Dict[str, torch.Tensor]],
        fed_weights: torch.Tensor,
        relation_masks: Dict[str, Dict[str, bool]] | None = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        candidates: Dict[str, Dict[str, torch.Tensor]] = {}
        state_keys = list(local_states[self.park_ids[0]].keys())
        for i, target_park in enumerate(self.park_ids):
            candidate_block: Dict[str, torch.Tensor] = {}
            for key in state_keys:
                relation_name = self._relation_name_from_state_key(key)
                target_mask = (relation_masks or {}).get(target_park, {})
                if relation_name is not None and target_mask and not target_mask.get(relation_name, True):
                    candidate_block[key] = local_states[target_park][key].detach().cpu()
                    continue
                mixed = None
                weight_sum = 0.0
                for j, source_park in enumerate(self.park_ids):
                    source_mask = (relation_masks or {}).get(source_park, {})
                    if relation_name is not None and source_mask and not source_mask.get(relation_name, True):
                        continue
                    source_value = local_states[source_park][key].detach().to(self.device)
                    source_weight = fed_weights[i, j]
                    weighted = source_value * source_weight
                    mixed = weighted if mixed is None else mixed + weighted
                    weight_sum += float(source_weight.detach().cpu().item())
                if mixed is None or weight_sum <= EPS:
                    candidate_block[key] = local_states[target_park][key].detach().cpu()
                    continue
                mixed = mixed / weight_sum
                candidate_block[key] = mixed.detach().cpu()
            candidates[target_park] = candidate_block
        return candidates

    def compute_candidate_advantages(
        self,
        local_agents: Dict[str, object],
        candidates: Dict[str, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        advantage = torch.zeros((self.num_parks,), dtype=torch.float32, device=self.device)
        for i, park_id in enumerate(self.park_ids):
            advantage[i] = float(local_agents[park_id].evaluate_candidate_backbone_advantage(candidates[park_id]))
        return advantage

    def update_rho_logits(self, normalized_candidate_advantage: torch.Tensor) -> None:
        self.rho_optimizer.zero_grad()
        rho = torch.sigmoid(self.rho_logits)
        loss = -(rho * normalized_candidate_advantage.detach()).sum()
        loss.backward()
        self.rho_optimizer.step()

    def soft_load_candidates(
        self,
        local_agents: Dict[str, object],
        candidates: Dict[str, Dict[str, torch.Tensor]],
        normalized_candidate_advantage: torch.Tensor,
        rho: torch.Tensor,
    ) -> torch.Tensor:
        eta = torch.zeros((self.num_parks,), dtype=torch.float32, device=self.device)
        for i, park_id in enumerate(self.park_ids):
            margin_i = self.candidate_gate_margin[i]
            probe_i = self.eta_probe[i]
            eta_max_i = self.eta_max[i]
            a_norm_i = float(normalized_candidate_advantage[i].detach().cpu().item())
            rho_i = float(rho[i].detach().cpu().item())
            if a_norm_i > 0.0:
                eta_i = min(rho_i * a_norm_i, eta_max_i)
            elif a_norm_i >= -margin_i:
                eta_i = probe_i
            else:
                eta_i = 0.0
            eta[i] = eta_i
            local_agents[park_id].soft_load_actor_backbone(candidates[park_id], eta_i)
            local_agents[park_id].set_global_actor_reference(
                {"actor_backbone": local_agents[park_id].get_shared_state()["actor_backbone"]}
            )
        return eta

    def export_state(self) -> LearnablePersonalizedFedActorState:
        return LearnablePersonalizedFedActorState(
            fed_logits=self.fed_logits.detach().cpu().clone(),
            rho_logits=self.rho_logits.detach().cpu().clone(),
            fed_optimizer=self.fed_optimizer.state_dict(),
            rho_optimizer=self.rho_optimizer.state_dict(),
            aggregate_count=self.aggregate_count,
            last_metrics=self.last_metrics,
        )

    def load_state(self, state: LearnablePersonalizedFedActorState | Dict[str, Any]) -> None:
        if isinstance(state, dict):
            fed_logits = state["fed_logits"]
            rho_logits = state["rho_logits"]
            if "fed_optimizer" in state and "rho_optimizer" in state:
                fed_optimizer_state = state["fed_optimizer"]
                rho_optimizer_state = state["rho_optimizer"]
            else:
                legacy_optimizer_state = state["optimizer"]
                fed_optimizer_state = legacy_optimizer_state
                rho_optimizer_state = legacy_optimizer_state
            aggregate_count = int(state.get("aggregate_count", 0))
            last_metrics = dict(state.get("last_metrics", {}))
        else:
            fed_logits = state.fed_logits
            rho_logits = state.rho_logits
            fed_optimizer_state = state.fed_optimizer
            rho_optimizer_state = state.rho_optimizer
            aggregate_count = int(state.aggregate_count)
            last_metrics = dict(state.last_metrics)
        self.fed_logits.data.copy_(fed_logits.detach().to(self.device))
        self.rho_logits.data.copy_(rho_logits.detach().to(self.device))
        self.fed_optimizer.load_state_dict(fed_optimizer_state)
        self.rho_optimizer.load_state_dict(rho_optimizer_state)
        self.aggregate_count = aggregate_count
        self.last_metrics = last_metrics

    def _resolve_park_values(self, value: Dict[str, float] | float) -> torch.Tensor:
        if isinstance(value, dict):
            resolved = [float(value[park_id]) for park_id in self.park_ids]
        else:
            resolved = [float(value) for _ in self.park_ids]
        return torch.tensor(resolved, dtype=torch.float32, device=self.device)

    @staticmethod
    def _rowwise_normalize_by_mean_abs(advantage: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(advantage)
        if advantage.numel() == 0:
            return normalized
        for row_idx in range(advantage.shape[0]):
            row = advantage[row_idx]
            scale = torch.mean(torch.abs(row))
            if float(scale.detach().cpu().item()) <= EPS:
                continue
            normalized[row_idx] = torch.clamp(row / scale, min=-1.0, max=1.0)
        return normalized

    @staticmethod
    def _normalize_candidate_advantages(candidate_advantage: torch.Tensor, source_advantage: torch.Tensor) -> torch.Tensor:
        normalized = torch.zeros_like(candidate_advantage)
        for row_idx in range(candidate_advantage.shape[0]):
            source_row = source_advantage[row_idx]
            scale = torch.mean(torch.abs(source_row))
            if float(scale.detach().cpu().item()) <= EPS:
                continue
            normalized[row_idx] = torch.clamp(candidate_advantage[row_idx] / scale, min=-1.0, max=1.0)
        return normalized

    def _build_metrics(
        self,
        fed_weights: torch.Tensor,
        rho: torch.Tensor,
        eta: torch.Tensor,
        source_advantage: torch.Tensor,
        source_advantage_norm: torch.Tensor,
        candidate_advantage: torch.Tensor,
        candidate_advantage_norm: torch.Tensor,
    ) -> Dict[str, Any]:
        return {
            "fed_weights": fed_weights.detach().cpu().tolist(),
            "rho": rho.detach().cpu().tolist(),
            "eta": eta.detach().cpu().tolist(),
            "source_advantage": source_advantage.detach().cpu().tolist(),
            "source_advantage_norm": source_advantage_norm.detach().cpu().tolist(),
            "candidate_advantage": candidate_advantage.detach().cpu().tolist(),
            "candidate_advantage_norm": candidate_advantage_norm.detach().cpu().tolist(),
            "candidate_gate_margin": self.candidate_gate_margin.detach().cpu().tolist(),
            "eta_probe": self.eta_probe.detach().cpu().tolist(),
            "eta_max": self.eta_max.detach().cpu().tolist(),
        }

    @staticmethod
    def _relation_name_from_state_key(key: str) -> str | None:
        marker = ".shared."
        if marker not in key:
            return None
        tail = key.split(marker, 1)[1]
        return tail.split(".", 1)[0]
