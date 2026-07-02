from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import copy

import torch
import torch.nn.functional as F
from torch import nn

from agent.replaybuffer import ReplayBuffer
from agent.state import get_node_sizes


MLP_HIDDEN_DIM = 256
NONFED_BACKBONE_LR = 3e-4
NONFED_HEAD_LR = 3e-4


def weights_init_(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
        torch.nn.init.constant_(module.bias, 0.0)


class TD3Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(self.out(self.net(state)))
        return action * valid_action_mask.to(dtype=action.dtype)


class TD3Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor, valid_action_mask: torch.Tensor) -> torch.Tensor:
        masked_action = action * valid_action_mask.to(dtype=action.dtype)
        x = torch.cat([state, masked_action], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x).squeeze(-1)


class TD3CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.q1 = TD3Critic(state_dim, action_dim, hidden_dim)
        self.q2 = TD3Critic(state_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor, valid_action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state, action, valid_action_mask), self.q2(state, action, valid_action_mask)


@dataclass
class LocalMLPTD3Config:
    park_type: str
    cp_count: int
    algorithm_variant: str
    enable_federation: bool
    privacy_mode: str
    gamma: float
    tau: float
    batch_size: int
    replay_size: int
    actor_proximal_weight: float
    seed: int
    act_device: str
    update_device: str
    d: float
    lambda_lr: float
    actor_backbone_lr: float | None = None
    actor_head_lr: float | None = None
    critic_backbone_lr: float | None = None
    critic_head_lr: float | None = None
    mlp_hidden_dim: int = MLP_HIDDEN_DIM
    policy_noise: float = 0.20
    noise_clip: float = 0.50
    exploration_noise: float = 0.10
    policy_delay: int = 2

    def __post_init__(self) -> None:
        if self.enable_federation:
            raise RuntimeError("mlp_td3 does not support federation because park MLP input dimensions differ.")
        if self.actor_backbone_lr is None:
            self.actor_backbone_lr = NONFED_BACKBONE_LR
        if self.actor_head_lr is None:
            self.actor_head_lr = NONFED_HEAD_LR
        if self.critic_backbone_lr is None:
            self.critic_backbone_lr = NONFED_BACKBONE_LR
        if self.critic_head_lr is None:
            self.critic_head_lr = NONFED_HEAD_LR


class LocalMLPTD3Agent:
    def __init__(self, config: LocalMLPTD3Config) -> None:
        self.config = config
        self.enable_federation = False
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

        self.actor = TD3Actor(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = TD3CriticNetwork(self.state_dim, self.action_dim, hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.actor_inference = self._build_inference_actor()
        self.total_it = 0

        self.actor_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.net.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.out.parameters(), "lr": config.actor_head_lr},
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

        self.replay_buffer = ReplayBuffer(capacity=config.replay_size, seed=config.seed)
        self._sync_inference_actor()

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
            fixed_action = self.actor_inference(state_vec.unsqueeze(0), valid_mask.unsqueeze(0))[0]
            if not deterministic:
                noise = torch.randn_like(fixed_action) * float(self.config.exploration_noise)
                fixed_action = (fixed_action + noise * valid_mask).clamp(-1.0, 1.0) * valid_mask
            fixed_action_cpu = fixed_action.detach().cpu()
            node_action = self._fixed_action_to_node_action(park_graph, fixed_action_cpu)
            env_action = self._fixed_action_to_env_action(park_graph, fixed_action_cpu)
            if return_node_action:
                return env_action, node_action
            return env_action

    def evaluate_cmdp_score(self, park_graph: Dict[str, Any], node_action: torch.Tensor) -> float:
        with torch.inference_mode():
            state_vec, valid_mask = self._graph_to_state_and_mask(park_graph, device=self.device)
            fixed_action = self._node_action_to_fixed_action(park_graph, node_action, device=self.device)
            q1, q2 = self.critic(state_vec.unsqueeze(0), fixed_action.unsqueeze(0), valid_mask.unsqueeze(0))
            return float(torch.min(q1, q2).detach().cpu().item())

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
            return self._empty_metrics()

        self.total_it += 1
        batch = self.replay_buffer.sample(self.config.batch_size)
        states, valid_masks = self._graphs_to_batch_tensors([transition.obs for transition in batch], self.device)
        next_states, next_valid_masks = self._graphs_to_batch_tensors([transition.next_obs for transition in batch], self.device)
        action_batch = torch.stack(
            [self._node_action_to_fixed_action(transition.obs, transition.action, self.device) for transition in batch],
            dim=0,
        )
        reward = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device)
        done = torch.tensor([transition.done for transition in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            noise = torch.randn_like(action_batch) * float(self.config.policy_noise)
            noise = noise.clamp(-float(self.config.noise_clip), float(self.config.noise_clip))
            next_action = self.actor_target(next_states, next_valid_masks)
            next_action = (next_action + noise * next_valid_masks).clamp(-1.0, 1.0) * next_valid_masks
            target_q1, target_q2 = self.critic_target(next_states, next_action, next_valid_masks)
            target_q = reward + (1.0 - done) * self.config.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(states, action_batch, valid_masks)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = 0.0
        if self.total_it % max(1, int(self.config.policy_delay)) == 0:
            actor_action = self.actor(states, valid_masks)
            actor_loss = -self.critic.q1(states, actor_action, valid_masks).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu().item())
            self.soft_update_targets(update_actor=True)
            self._sync_inference_actor()
        else:
            self.soft_update_targets(update_actor=False)

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "cost_critic_loss": 0.0,
            "alpha_loss": 0.0,
            "lambda_loss": 0.0,
            "lambda_value": 0.0,
            "mean_qcf_pi": 0.0,
        }

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "cost_critic_loss": 0.0,
            "alpha_loss": 0.0,
            "lambda_loss": 0.0,
            "lambda_value": 0.0,
            "mean_qcf_pi": 0.0,
        }

    def soft_update_targets(self, update_actor: bool = True) -> None:
        if update_actor:
            for target_param, source_param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau)
        for target_param, source_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau)

    def get_shared_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {}

    def set_global_actor_reference(self, shared_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        del shared_state

    def load_shared_state(self, shared_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        del shared_state

    def export_checkpoint(self, park_type: str, episode: int) -> Dict[str, Any]:
        return {
            "checkpoint_format": "mlp_td3_v1",
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
                "actor_target": {k: v.detach().cpu().clone() for k, v in self.actor_target.state_dict().items()},
                "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
                "critic_target": {k: v.detach().cpu().clone() for k, v in self.critic_target.state_dict().items()},
            },
            "optimizers": {
                "actor": self.actor_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict(),
            },
            "counters": {"total_it": int(self.total_it)},
            "replay_buffer": self.replay_buffer.state_dict(),
        }

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.actor.load_state_dict(checkpoint["models"]["actor"])
        self.actor_target.load_state_dict(checkpoint["models"].get("actor_target", checkpoint["models"]["actor"]))
        self.critic.load_state_dict(checkpoint["models"]["critic"])
        self.critic_target.load_state_dict(checkpoint["models"].get("critic_target", checkpoint["models"]["critic"]))
        self.actor_optimizer.load_state_dict(checkpoint["optimizers"]["actor"])
        self.critic_optimizer.load_state_dict(checkpoint["optimizers"]["critic"])
        self.total_it = int(checkpoint.get("counters", {}).get("total_it", 0))
        self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        self._sync_inference_actor()

    def _build_inference_actor(self) -> TD3Actor:
        return TD3Actor(self.state_dim, self.action_dim, int(self.config.mlp_hidden_dim)).to(self.act_device)

    def _sync_inference_actor(self) -> None:
        self.actor_inference.load_state_dict(
            {key: value.detach().to(self.act_device) for key, value in self.actor.state_dict().items()}
        )
        self.actor_inference.eval()

    def _graph_to_state_and_mask(self, graph: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return torch.cat([cs, bes, pv, external, ev_slots.reshape(-1)], dim=0), valid_action_mask

    def _graphs_to_batch_tensors(self, graphs: List[Dict[str, Any]], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        states: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for graph in graphs:
            state, mask = self._graph_to_state_and_mask(graph, device)
            states.append(state)
            masks.append(mask)
        return torch.stack(states, dim=0), torch.stack(masks, dim=0)

    def _node_action_to_fixed_action(self, graph: Dict[str, Any], node_action: torch.Tensor, device: torch.device) -> torch.Tensor:
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


__all__ = ["LocalMLPTD3Config", "LocalMLPTD3Agent", "TD3Actor", "TD3Critic", "TD3CriticNetwork"]
