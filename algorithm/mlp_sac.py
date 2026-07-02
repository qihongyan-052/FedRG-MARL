from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from agent.replaybuffer import ReplayBuffer
from agent.state import get_node_sizes


LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0
EPSILON = 1e-6
MLP_HIDDEN_DIM = 256
NONFED_BACKBONE_LR = 3e-4
NONFED_HEAD_LR = 3e-4
FED_BACKBONE_LR = 3e-4
FED_HEAD_LR = 3e-4


def weights_init_(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
        torch.nn.init.constant_(module.bias, 0.0)


class MLPActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(
        self,
        state: torch.Tensor,
        valid_action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = mean if deterministic else normal.rsample()
        squashed_action = torch.tanh(raw_action)
        mask = valid_action_mask.to(dtype=squashed_action.dtype)
        masked_action = squashed_action * mask

        log_prob = normal.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - squashed_action.pow(2) + EPSILON)
        masked_log_prob = (log_prob * mask).sum(dim=-1)
        masked_mean = torch.tanh(mean) * mask
        return masked_action, masked_log_prob, masked_mean


class MLPCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        valid_action_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_action = action * valid_action_mask.to(dtype=action.dtype)
        x = torch.cat([state, masked_action], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x).squeeze(-1)


class MLPCriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = MLPCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.q2 = MLPCritic(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        valid_action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(state, action, valid_action_mask),
            self.q2(state, action, valid_action_mask),
        )


@dataclass
class LocalMLPSACConfig:
    park_type: str
    cp_count: int
    algorithm_variant: str
    enable_federation: bool
    privacy_mode: str
    alpha_lr: float
    gamma: float
    tau: float
    batch_size: int
    replay_size: int
    target_entropy: float
    actor_proximal_weight: float
    seed: int
    act_device: str
    update_device: str
    d: float
    lambda_lr: float
    federate_critic_backbone: bool = False
    actor_backbone_lr: float | None = None
    actor_head_lr: float | None = None
    critic_backbone_lr: float | None = None
    critic_head_lr: float | None = None
    mlp_hidden_dim: int = MLP_HIDDEN_DIM

    def __post_init__(self) -> None:
        if self.actor_backbone_lr is None:
            self.actor_backbone_lr = FED_BACKBONE_LR if self.enable_federation else NONFED_BACKBONE_LR
        if self.actor_head_lr is None:
            self.actor_head_lr = FED_HEAD_LR if self.enable_federation else NONFED_HEAD_LR
        if self.critic_backbone_lr is None:
            self.critic_backbone_lr = FED_BACKBONE_LR if self.enable_federation else NONFED_BACKBONE_LR
        if self.critic_head_lr is None:
            self.critic_head_lr = FED_HEAD_LR if self.enable_federation else NONFED_HEAD_LR


class LocalMLPSACAgent:
    def __init__(self, config: LocalMLPSACConfig) -> None:
        self.config = config
        self.enable_federation = config.enable_federation
        self.device = torch.device(config.update_device)
        self.act_device = torch.device(config.act_device)
        torch.manual_seed(config.seed)
        self.node_sizes = get_node_sizes(config.privacy_mode)
        self.state_dim = (
            self.node_sizes["cs"]
            + self.node_sizes["bes"]
            + self.node_sizes["pv"]
            + self.node_sizes["external"]
            + config.cp_count * self.node_sizes["ev"]
        )
        self.action_dim = config.cp_count + 1
        hidden_dim = int(config.mlp_hidden_dim)

        self.actor = MLPActor(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic = MLPCriticNetwork(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.actor_inference = self._build_inference_actor()
        self.global_actor_reference: Dict[str, torch.Tensor] = {}

        self.actor_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.net.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.mean.parameters(), "lr": config.actor_head_lr},
                {"params": self.actor.log_std.parameters(), "lr": config.actor_head_lr},
            ]
        )
        self.critic_optimizer = torch.optim.Adam(
            [
                {"params": self.critic.q1.l1.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q1.l2.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q1.l3.parameters(), "lr": config.critic_head_lr},
                {"params": self.critic.q2.l1.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q2.l2.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q2.l3.parameters(), "lr": config.critic_head_lr},
            ]
        )

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

        self.replay_buffer = ReplayBuffer(capacity=config.replay_size, seed=config.seed)
        self._sync_inference_actor()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def lambda_value(self) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float32, device=self.device)

    def act(
        self,
        park_graph: Dict[str, Any],
        deterministic: bool = False,
        return_node_action: bool = False,
    ) -> Dict[str, Any] | Tuple[Dict[str, Any], torch.Tensor]:
        with torch.inference_mode():
            state_vec, valid_mask = self._graph_to_state_and_mask(park_graph, device=self.act_device)
            fixed_action, _, fixed_mean = self.actor_inference.sample(
                state_vec.unsqueeze(0),
                valid_mask.unsqueeze(0),
                deterministic=deterministic,
            )
            chosen_fixed_action = fixed_mean[0] if deterministic else fixed_action[0]
            node_action = self._fixed_action_to_node_action(park_graph, chosen_fixed_action.detach().cpu())
            env_action = self._fixed_action_to_env_action(park_graph, chosen_fixed_action.detach().cpu())
            if return_node_action:
                return env_action, node_action
            return env_action

    def evaluate_cmdp_score(self, park_graph: Dict[str, Any], node_action: torch.Tensor) -> float:
        with torch.inference_mode():
            state_vec, valid_mask = self._graph_to_state_and_mask(park_graph, device=self.device)
            fixed_action = self._node_action_to_fixed_action(park_graph, node_action, device=self.device)
            q1, q2 = self.critic(state_vec.unsqueeze(0), fixed_action.unsqueeze(0), valid_mask.unsqueeze(0))
            q_value = torch.min(q1, q2)
            return float(q_value.detach().cpu().item())

    def store_transition(
        self,
        obs: Dict[str, Any],
        action: torch.Tensor,
        reward: float,
        cost: float,
        next_obs: Dict[str, Any],
        done: bool,
    ) -> None:
        self.replay_buffer.push(obs, action, reward, cost, next_obs, done)

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "cost_critic_loss": 0.0,
                "alpha_loss": 0.0,
                "lambda_loss": 0.0,
                "lambda_value": 0.0,
                "mean_qcf_pi": 0.0,
            }

        batch = self.replay_buffer.sample(self.config.batch_size)
        states, valid_masks = self._graphs_to_batch_tensors([transition.obs for transition in batch], device=self.device)
        next_states, next_valid_masks = self._graphs_to_batch_tensors([transition.next_obs for transition in batch], device=self.device)
        action_batch = torch.stack(
            [
                self._node_action_to_fixed_action(transition.obs, transition.action, device=self.device)
                for transition in batch
            ],
            dim=0,
        )
        reward = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device)
        done = torch.tensor([transition.done for transition in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_states, next_valid_masks)
            qf1_next_target, qf2_next_target = self.critic_target(next_states, next_action, next_valid_masks)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha.detach() * next_log_pi
            next_q_value = reward + (1.0 - done) * self.config.gamma * min_qf_next_target

        qf1, qf2 = self.critic(states, action_batch, valid_masks)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        critic_loss = qf1_loss + qf2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(states, valid_masks)
        qf1_pi, qf2_pi = self.critic(states, pi, valid_masks)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha.detach() * log_pi) - min_qf_pi).mean()
        actor_proximal_penalty = self._compute_actor_proximal_penalty()
        actor_total_loss = actor_loss + self.config.actor_proximal_weight * actor_proximal_penalty
        self.actor_optimizer.zero_grad()
        actor_total_loss.backward()
        self.actor_optimizer.step()

        target_entropy = -valid_masks.sum(dim=-1)
        alpha_loss = -(self.log_alpha * (log_pi + target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update_targets()
        self._sync_inference_actor()
        return {
            "actor_loss": float(actor_total_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "cost_critic_loss": 0.0,
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "lambda_loss": 0.0,
            "lambda_value": 0.0,
            "mean_qcf_pi": 0.0,
        }

    def soft_update_targets(self) -> None:
        for target_param, source_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau)

    def get_shared_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        shared_state = {
            "actor_backbone": {k: v.detach().cpu().clone() for k, v in self.actor.net.state_dict().items()}
        }
        if self.enable_federation and self.config.federate_critic_backbone:
            shared_state["critic_backbone"] = {
                **{f"q1.l1.{k}": v.detach().cpu().clone() for k, v in self.critic.q1.l1.state_dict().items()},
                **{f"q1.l2.{k}": v.detach().cpu().clone() for k, v in self.critic.q1.l2.state_dict().items()},
                **{f"q2.l1.{k}": v.detach().cpu().clone() for k, v in self.critic.q2.l1.state_dict().items()},
                **{f"q2.l2.{k}": v.detach().cpu().clone() for k, v in self.critic.q2.l2.state_dict().items()},
            }
        return shared_state

    def set_global_actor_reference(self, shared_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        if not self.enable_federation:
            self.global_actor_reference = {}
            return
        self.global_actor_reference = {
            key: value.detach().to(self.device).clone()
            for key, value in shared_state["actor_backbone"].items()
        }

    def load_shared_state(self, shared_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        if not self.enable_federation:
            return
        local_state = self.actor.net.state_dict()
        blended_state = {}
        for key, local_value in local_state.items():
            global_value = shared_state["actor_backbone"][key].detach().to(self.device)
            blended_state[key] = 0.7 * local_value + 0.3 * global_value
        self.actor.net.load_state_dict(blended_state)
        if self.config.federate_critic_backbone and "critic_backbone" in shared_state:
            self._blend_critic_backbone(shared_state["critic_backbone"])
        self._sync_inference_actor()

    def _blend_critic_backbone(self, shared_block: Dict[str, torch.Tensor]) -> None:
        for critic_name, critic_module, target_module in (
            ("q1", self.critic.q1, self.critic_target.q1),
            ("q2", self.critic.q2, self.critic_target.q2),
        ):
            for layer_name in ("l1", "l2"):
                self._blend_linear_layer(
                    layer=getattr(critic_module, layer_name),
                    target_layer=getattr(target_module, layer_name),
                    shared_block=shared_block,
                    prefix=f"{critic_name}.{layer_name}",
                )

    def _blend_linear_layer(
        self,
        layer: nn.Linear,
        target_layer: nn.Linear,
        shared_block: Dict[str, torch.Tensor],
        prefix: str,
    ) -> None:
        local_state = layer.state_dict()
        blended_state = {}
        for key, local_value in local_state.items():
            global_value = shared_block[f"{prefix}.{key}"].detach().to(self.device)
            blended_state[key] = 0.7 * local_value + 0.3 * global_value
        layer.load_state_dict(blended_state)
        target_layer.load_state_dict(blended_state)

    def export_checkpoint(self, park_type: str, episode: int) -> Dict[str, Any]:
        return {
            "checkpoint_format": "mlp_sac_v1",
            "park_type": park_type,
            "episode": episode,
            "agent_config": asdict(self.config),
            "state_spec": {
                "node_sizes": dict(self.node_sizes),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            "models": {
                "actor": {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()},
                "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
                "critic_target": {k: v.detach().cpu().clone() for k, v in self.critic_target.state_dict().items()},
            },
            "optimizers": {
                "actor": self.actor_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict(),
                "alpha": self.alpha_optimizer.state_dict(),
            },
            "temperature": {
                "log_alpha": float(self.log_alpha.detach().cpu().item()),
            },
            "replay_buffer": self.replay_buffer.state_dict(),
        }

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.actor.load_state_dict(checkpoint["models"]["actor"])
        self.critic.load_state_dict(checkpoint["models"]["critic"])
        self.critic_target.load_state_dict(checkpoint["models"]["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["optimizers"]["actor"])
        self.critic_optimizer.load_state_dict(checkpoint["optimizers"]["critic"])
        self.alpha_optimizer.load_state_dict(checkpoint["optimizers"]["alpha"])
        self.log_alpha.data.copy_(torch.tensor(checkpoint["temperature"]["log_alpha"], device=self.device))
        self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        self._sync_inference_actor()

    def _build_inference_actor(self) -> MLPActor:
        hidden_dim = int(self.config.mlp_hidden_dim)
        return MLPActor(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=hidden_dim).to(self.act_device)

    def _sync_inference_actor(self) -> None:
        self.actor_inference.load_state_dict(
            {key: value.detach().to(self.act_device) for key, value in self.actor.state_dict().items()}
        )
        self.actor_inference.eval()

    def _compute_actor_proximal_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        if not self.enable_federation or not self.global_actor_reference:
            return penalty
        for name, param in self.actor.net.named_parameters():
            penalty = penalty + (param - self.global_actor_reference[name]).pow(2).sum()
        return penalty

    def _graph_to_state_and_mask(
        self,
        graph: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cs = torch.tensor(graph["cs_features"][0] if graph["cs_features"] else [0.0] * self.node_sizes["cs"], dtype=torch.float32, device=device)
        bes = torch.tensor(graph["bes_features"][0] if graph["bes_features"] else [0.0] * self.node_sizes["bes"], dtype=torch.float32, device=device)
        pv = torch.tensor(graph["pv_features"][0] if graph["pv_features"] else [0.0] * self.node_sizes["pv"], dtype=torch.float32, device=device)
        external = torch.tensor(graph["external_features"][0] if graph["external_features"] else [0.0] * self.node_sizes["external"], dtype=torch.float32, device=device)

        ev_slots = torch.zeros((self.config.cp_count, self.node_sizes["ev"]), dtype=torch.float32, device=device)
        valid_action_mask = torch.zeros((self.action_dim,), dtype=torch.float32, device=device)
        valid_action_mask[self.config.cp_count] = 1.0

        action_slot_by_node = {
            int(node_idx): int(fixed_idx)
            for node_idx, fixed_idx in zip(graph["action_node_indices"], graph["action_mapper"])
        }
        for ev_idx, ev_feature in zip(graph["ev_indexes"], graph["ev_features"]):
            fixed_idx = action_slot_by_node[int(ev_idx)]
            ev_slots[fixed_idx] = torch.tensor(ev_feature, dtype=torch.float32, device=device)
            valid_action_mask[fixed_idx] = 1.0

        state_vec = torch.cat([cs, bes, pv, external, ev_slots.reshape(-1)], dim=0)
        return state_vec, valid_action_mask

    def _graphs_to_batch_tensors(
        self,
        graphs: List[Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state_list: List[torch.Tensor] = []
        mask_list: List[torch.Tensor] = []
        for graph in graphs:
            state_vec, valid_mask = self._graph_to_state_and_mask(graph, device=device)
            state_list.append(state_vec)
            mask_list.append(valid_mask)
        return torch.stack(state_list, dim=0), torch.stack(mask_list, dim=0)

    def _node_action_to_fixed_action(
        self,
        graph: Dict[str, Any],
        node_action: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        fixed_action = torch.zeros((self.action_dim,), dtype=torch.float32, device=device)
        node_action = node_action.to(device=device, dtype=torch.float32).reshape(-1)
        for node_idx, fixed_idx in zip(graph["action_node_indices"], graph["action_mapper"]):
            fixed_action[int(fixed_idx)] = node_action[int(node_idx)]
        return fixed_action

    def _fixed_action_to_node_action(self, graph: Dict[str, Any], fixed_action: torch.Tensor) -> torch.Tensor:
        node_action = torch.zeros((len(graph["node_types"]),), dtype=torch.float32)
        fixed_action = fixed_action.detach().cpu().reshape(-1)
        for node_idx, fixed_idx in zip(graph["action_node_indices"], graph["action_mapper"]):
            node_action[int(node_idx)] = fixed_action[int(fixed_idx)]
        return node_action

    def _fixed_action_to_env_action(self, graph: Dict[str, Any], fixed_action: torch.Tensor) -> Dict[str, Any]:
        fixed_action = fixed_action.detach().cpu().reshape(-1)
        env_action = {"bes": float(fixed_action[self.config.cp_count].item()), "ev": {}}
        action_slot_by_node = {
            int(node_idx): int(fixed_idx)
            for node_idx, fixed_idx in zip(graph["action_node_indices"], graph["action_mapper"])
        }
        for ev_idx in graph["ev_indexes"]:
            slot_idx = action_slot_by_node[int(ev_idx)]
            env_action["ev"][graph["node_names"][int(ev_idx)]] = float(fixed_action[slot_idx].item())
        return env_action
