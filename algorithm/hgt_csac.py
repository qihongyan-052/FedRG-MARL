from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import HGTConv, global_add_pool, global_mean_pool

from agent.replaybuffer import ReplayBuffer
from agent.state import get_node_sizes


LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0
EPSILON = 1e-6
NODE_TYPES = ("cs", "bes", "pv", "external", "ev")
HGT_FEATURE_DIM = 32
HGT_ACTOR_HIDDEN_DIM_1 = 64
HGT_ACTOR_HIDDEN_DIM_2 = 128
HGT_CRITIC_HIDDEN_DIM_1 = 64
HGT_CRITIC_HIDDEN_DIM_2 = 128
HGT_NUM_HEADS = 2
NONFED_BACKBONE_LR = 3e-4
NONFED_HEAD_LR = 3e-4
FED_BACKBONE_LR = 1e-4
FED_LOCAL_BACKBONE_LR = 3e-4
FED_HEAD_LR = 3e-4
EDGE_TYPES = (
    ("ev", "to", "cs"),
    ("cs", "rev_to_ev", "ev"),
    ("bes", "to", "cs"),
    ("cs", "rev_to_bes", "bes"),
    ("pv", "to", "cs"),
    ("cs", "rev_to_pv", "pv"),
    ("external", "to", "cs"),
    ("cs", "rev_to_external", "external"),
)
METADATA = (list(NODE_TYPES), list(EDGE_TYPES))


def weights_init_(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
        torch.nn.init.constant_(module.bias, 0.0)


class HeteroTypeEmbedding(nn.Module):
    def __init__(self, node_sizes: Dict[str, int], feature_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {node_type: nn.Linear(node_sizes[node_type], feature_dim) for node_type in NODE_TYPES}
        )
        self.apply(weights_init_)

    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embedded: Dict[str, torch.Tensor] = {}
        for node_type in NODE_TYPES:
            embedded[node_type] = F.relu(self.embeddings[node_type](x_dict[node_type]))
        return embedded


class HGTActor(nn.Module):
    def __init__(self, node_sizes: Dict[str, int], feature_dim: int, decouple_actor_output_heads: bool = False) -> None:
        super().__init__()
        self.decouple_actor_output_heads = decouple_actor_output_heads
        self.node_embedding = HeteroTypeEmbedding(node_sizes=node_sizes, feature_dim=feature_dim)
        self.hgt_conv1 = HGTConv(
            in_channels={node_type: feature_dim for node_type in NODE_TYPES},
            out_channels=HGT_ACTOR_HIDDEN_DIM_1,
            metadata=METADATA,
            heads=HGT_NUM_HEADS,
        )
        self.norm1 = nn.ModuleDict({node_type: nn.LayerNorm(HGT_ACTOR_HIDDEN_DIM_1) for node_type in NODE_TYPES})
        self.hgt_conv2 = HGTConv(
            in_channels={node_type: HGT_ACTOR_HIDDEN_DIM_1 for node_type in NODE_TYPES},
            out_channels=HGT_ACTOR_HIDDEN_DIM_2,
            metadata=METADATA,
            heads=HGT_NUM_HEADS,
        )
        self.norm2 = nn.ModuleDict({node_type: nn.LayerNorm(HGT_ACTOR_HIDDEN_DIM_2) for node_type in NODE_TYPES})
        if self.decouple_actor_output_heads:
            self.mean_heads = nn.ModuleDict({"bes": nn.Linear(HGT_ACTOR_HIDDEN_DIM_2, 1), "ev": nn.Linear(HGT_ACTOR_HIDDEN_DIM_2, 1)})
            self.log_std_heads = nn.ModuleDict({"bes": nn.Linear(HGT_ACTOR_HIDDEN_DIM_2, 1), "ev": nn.Linear(HGT_ACTOR_HIDDEN_DIM_2, 1)})
        else:
            self.mean_head = nn.Linear(HGT_ACTOR_HIDDEN_DIM_2, 1)
            self.log_std_head = nn.Linear(HGT_ACTOR_HIDDEN_DIM_2, 1)
        self.apply(weights_init_)

    def _encode(self, batch: Batch) -> Dict[str, torch.Tensor]:
        x_dict = {node_type: batch[node_type].x for node_type in NODE_TYPES}
        edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in EDGE_TYPES}
        x_dict = self.node_embedding(x_dict)
        x_dict = {
            key: F.relu(self.norm1[key](value))
            for key, value in self.hgt_conv1(x_dict, edge_index_dict).items()
        }
        x_dict = {
            key: F.relu(self.norm2[key](value))
            for key, value in self.hgt_conv2(x_dict, edge_index_dict).items()
        }
        return x_dict

    def forward(self, batch: Batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        x_dict = self._encode(batch)
        mean_dict: Dict[str, torch.Tensor] = {}
        log_std_dict: Dict[str, torch.Tensor] = {}
        for node_type in ("bes", "ev"):
            if self.decouple_actor_output_heads:
                mean = self.mean_heads[node_type](x_dict[node_type]).squeeze(-1)
                log_std = self.log_std_heads[node_type](x_dict[node_type]).squeeze(-1)
            else:
                mean = self.mean_head(x_dict[node_type]).squeeze(-1)
                log_std = self.log_std_head(x_dict[node_type]).squeeze(-1)
            mean_dict[node_type] = mean
            log_std_dict[node_type] = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean_dict, log_std_dict

    def actor_head_parameters(self) -> list[nn.Parameter]:
        if self.decouple_actor_output_heads:
            return list(self.mean_heads.parameters()) + list(self.log_std_heads.parameters())
        return list(self.mean_head.parameters()) + list(self.log_std_head.parameters())

    def sample(
        self,
        batch: Batch,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        mean_dict, log_std_dict = self.forward(batch)
        action_dict: Dict[str, torch.Tensor] = {}
        mean_action_dict: Dict[str, torch.Tensor] = {}
        log_prob_by_graph: torch.Tensor | None = None
        num_graphs = batch.num_graphs
        for node_type in ("bes", "ev"):
            mean = mean_dict[node_type]
            log_std = log_std_dict[node_type]
            std = log_std.exp()
            normal = Normal(mean, std)
            raw_action = mean if deterministic else normal.rsample()
            squashed_action = torch.tanh(raw_action)
            action_dict[node_type] = squashed_action
            mean_action_dict[node_type] = torch.tanh(mean)
            log_prob = normal.log_prob(raw_action)
            log_prob = log_prob - torch.log(1 - squashed_action.pow(2) + EPSILON)
            pooled = global_add_pool(
                log_prob.unsqueeze(-1),
                batch[node_type].batch,
                size=num_graphs,
            ).squeeze(-1)
            log_prob_by_graph = pooled if log_prob_by_graph is None else log_prob_by_graph + pooled
        if log_prob_by_graph is None:
            log_prob_by_graph = torch.zeros((num_graphs,), dtype=torch.float32, device=batch["cs"].x.device)
        return action_dict, log_prob_by_graph, mean_action_dict


class HGTCritic(nn.Module):
    def __init__(self, node_sizes: Dict[str, int], feature_dim: int) -> None:
        super().__init__()
        self.node_embedding = HeteroTypeEmbedding(node_sizes=node_sizes, feature_dim=feature_dim)
        critic_input_dims = {
            "cs": feature_dim,
            "bes": feature_dim + 1,
            "pv": feature_dim,
            "external": feature_dim,
            "ev": feature_dim + 1,
        }
        self.hgt_conv1 = HGTConv(
            in_channels=critic_input_dims,
            out_channels=HGT_CRITIC_HIDDEN_DIM_1,
            metadata=METADATA,
            heads=HGT_NUM_HEADS,
        )
        self.norm1 = nn.ModuleDict({node_type: nn.LayerNorm(HGT_CRITIC_HIDDEN_DIM_1) for node_type in NODE_TYPES})
        self.hgt_conv2 = HGTConv(
            in_channels={node_type: HGT_CRITIC_HIDDEN_DIM_1 for node_type in NODE_TYPES},
            out_channels=HGT_CRITIC_HIDDEN_DIM_2,
            metadata=METADATA,
            heads=HGT_NUM_HEADS,
        )
        self.norm2 = nn.ModuleDict({node_type: nn.LayerNorm(HGT_CRITIC_HIDDEN_DIM_2) for node_type in NODE_TYPES})
        self.l1 = nn.Linear(HGT_CRITIC_HIDDEN_DIM_2, 64)
        self.l2 = nn.Linear(64, 1)
        self.apply(weights_init_)

    def forward(self, batch: Batch, action_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_dict = {node_type: batch[node_type].x for node_type in NODE_TYPES}
        edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in EDGE_TYPES}
        x_dict = self.node_embedding(x_dict)
        x_dict = {
            "cs": x_dict["cs"],
            "bes": torch.cat([x_dict["bes"], action_dict["bes"].unsqueeze(-1)], dim=-1),
            "pv": x_dict["pv"],
            "external": x_dict["external"],
            "ev": torch.cat([x_dict["ev"], action_dict["ev"].unsqueeze(-1)], dim=-1),
        }
        x_dict = {
            key: F.relu(self.norm1[key](value))
            for key, value in self.hgt_conv1(x_dict, edge_index_dict).items()
        }
        x_dict = {
            key: F.relu(self.norm2[key](value))
            for key, value in self.hgt_conv2(x_dict, edge_index_dict).items()
        }

        batch_size = batch.num_graphs
        pooled_sum = torch.zeros((batch_size, HGT_CRITIC_HIDDEN_DIM_2), dtype=torch.float32, device=batch["cs"].x.device)
        present_counts = torch.zeros((batch_size, 1), dtype=torch.float32, device=batch["cs"].x.device)
        for node_type in NODE_TYPES:
            x = x_dict[node_type]
            if x.numel() == 0:
                continue
            pooled = global_mean_pool(x, batch[node_type].batch, size=batch_size)
            pooled_sum = pooled_sum + pooled
            counts = global_add_pool(
                torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device),
                batch[node_type].batch,
                size=batch_size,
            )
            present_counts = present_counts + (counts > 0).to(dtype=torch.float32)
        pooled_graph = pooled_sum / torch.clamp(present_counts, min=1.0)
        x = F.relu(self.l1(pooled_graph))
        return self.l2(x).squeeze(-1)


class HGTCriticNetwork(nn.Module):
    def __init__(self, node_sizes: Dict[str, int], feature_dim: int) -> None:
        super().__init__()
        self.q1 = HGTCritic(node_sizes=node_sizes, feature_dim=feature_dim)
        self.q2 = HGTCritic(node_sizes=node_sizes, feature_dim=feature_dim)

    def forward(self, batch: Batch, action_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(batch, action_dict), self.q2(batch, action_dict)


@dataclass
class LocalHGTCSACConfig:
    park_type: str
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
    critic_proximal_weight: float
    seed: int
    act_device: str
    update_device: str
    d: float
    lambda_lr: float
    federate_critic_backbone: bool = False
    actor_backbone_lr: float | None = None
    actor_local_backbone_lr: float | None = None
    actor_head_lr: float | None = None
    critic_backbone_lr: float | None = None
    critic_head_lr: float | None = None
    feature_dim: int = HGT_FEATURE_DIM
    decouple_actor_output_heads: bool | None = None

    def __post_init__(self) -> None:
        if self.decouple_actor_output_heads is None:
            self.decouple_actor_output_heads = False
        if self.actor_backbone_lr is None:
            self.actor_backbone_lr = FED_BACKBONE_LR if self.enable_federation else NONFED_BACKBONE_LR
        if self.actor_local_backbone_lr is None:
            self.actor_local_backbone_lr = FED_LOCAL_BACKBONE_LR if self.enable_federation else NONFED_BACKBONE_LR
        if self.actor_head_lr is None:
            self.actor_head_lr = FED_HEAD_LR if self.enable_federation else NONFED_HEAD_LR
        if self.critic_backbone_lr is None:
            self.critic_backbone_lr = FED_BACKBONE_LR if self.enable_federation and self.federate_critic_backbone else NONFED_BACKBONE_LR
        if self.critic_head_lr is None:
            self.critic_head_lr = FED_HEAD_LR if self.enable_federation and self.federate_critic_backbone else NONFED_HEAD_LR


class LocalHGTCSACAgent:
    def __init__(self, config: LocalHGTCSACConfig) -> None:
        self.config = config
        self.enable_federation = config.enable_federation
        self.device = torch.device(config.update_device)
        self.act_device = torch.device(config.act_device)
        torch.manual_seed(config.seed)
        self.node_sizes = get_node_sizes(config.privacy_mode)

        self.actor = HGTActor(
            node_sizes=self.node_sizes,
            feature_dim=config.feature_dim,
            decouple_actor_output_heads=config.decouple_actor_output_heads,
        ).to(self.device)
        self.critic = HGTCriticNetwork(node_sizes=self.node_sizes, feature_dim=config.feature_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.cost_critic = HGTCriticNetwork(node_sizes=self.node_sizes, feature_dim=config.feature_dim).to(self.device)
        self.cost_critic_target = copy.deepcopy(self.cost_critic).to(self.device)
        self.actor_inference = self._build_inference_actor()
        self.global_actor_reference: Dict[str, torch.Tensor] = {}
        self.global_critic1_reference: Dict[str, torch.Tensor] = {}
        self.global_critic2_reference: Dict[str, torch.Tensor] = {}
        self.global_cost_critic1_reference: Dict[str, torch.Tensor] = {}
        self.global_cost_critic2_reference: Dict[str, torch.Tensor] = {}

        self.actor_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.node_embedding.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.hgt_conv1.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.norm1.parameters(), "lr": config.actor_local_backbone_lr},
                {"params": self.actor.hgt_conv2.parameters(), "lr": config.actor_local_backbone_lr},
                {"params": self.actor.norm2.parameters(), "lr": config.actor_local_backbone_lr},
                {"params": self.actor.actor_head_parameters(), "lr": config.actor_head_lr},
            ]
        )
        self.critic_optimizer = torch.optim.Adam(
            [
                {"params": self.critic.q1.node_embedding.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q1.hgt_conv1.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q1.norm1.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q1.hgt_conv2.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q1.norm2.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q1.l1.parameters(), "lr": config.critic_head_lr},
                {"params": self.critic.q1.l2.parameters(), "lr": config.critic_head_lr},
                {"params": self.critic.q2.node_embedding.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q2.hgt_conv1.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q2.norm1.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q2.hgt_conv2.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q2.norm2.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q2.l1.parameters(), "lr": config.critic_head_lr},
                {"params": self.critic.q2.l2.parameters(), "lr": config.critic_head_lr},
            ]
        )
        self.cost_critic_optimizer = torch.optim.Adam(
            [
                {"params": self.cost_critic.q1.node_embedding.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q1.hgt_conv1.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q1.norm1.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q1.hgt_conv2.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q1.norm2.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q1.l1.parameters(), "lr": config.critic_head_lr},
                {"params": self.cost_critic.q1.l2.parameters(), "lr": config.critic_head_lr},
                {"params": self.cost_critic.q2.node_embedding.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q2.hgt_conv1.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q2.norm1.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q2.hgt_conv2.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q2.norm2.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q2.l1.parameters(), "lr": config.critic_head_lr},
                {"params": self.cost_critic.q2.l2.parameters(), "lr": config.critic_head_lr},
            ]
        )
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.log_lambda = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.lambda_optimizer = torch.optim.Adam([self.log_lambda], lr=config.lambda_lr)

        self.replay_buffer = ReplayBuffer(capacity=config.replay_size, seed=config.seed)
        self._sync_inference_actor()

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def lambda_value(self) -> torch.Tensor:
        return self.log_lambda.exp()

    def act(
        self,
        park_graph: Dict[str, Any],
        deterministic: bool = False,
        return_node_action: bool = False,
    ) -> Dict[str, Any] | Tuple[Dict[str, Any], torch.Tensor]:
        with torch.inference_mode():
            batch = self._batch_graphs([park_graph], device=self.act_device)
            action_dict, _, mean_action_dict = self.actor_inference.sample(batch, deterministic=deterministic)
            chosen = mean_action_dict if deterministic else action_dict
            node_action = self._hetero_action_to_node_action(park_graph, chosen)
            env_action = self._hetero_action_to_env_action(park_graph, chosen)
            if return_node_action:
                return env_action, node_action
            return env_action

    def evaluate_cmdp_score(self, park_graph: Dict[str, Any], node_action: torch.Tensor) -> float:
        with torch.inference_mode():
            batch = self._batch_graphs([park_graph], device=self.device)
            action_dict = self._flat_node_action_to_hetero_action([park_graph], [node_action], device=self.device)
            q1, q2 = self.critic(batch, action_dict)
            qc1, qc2 = self.cost_critic(batch, action_dict)
            q_value = torch.min(q1, q2)
            qc_value = 0.5 * (qc1 + qc2)
            score = q_value - self.lambda_value.detach() * qc_value
            return float(score.detach().cpu().item())

    def make_actor_with_external_backbone(self, shared_backbone: Dict[str, torch.Tensor]) -> HGTActor:
        actor_copy = copy.deepcopy(self.actor)
        self._load_backbone_module(actor_copy, shared_backbone)
        actor_copy.to(self.device)
        actor_copy.eval()
        return actor_copy

    def evaluate_external_backbone_advantage(
        self,
        external_backbone: Dict[str, torch.Tensor],
        batch_size: int | None = None,
    ) -> float:
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        return self._evaluate_actor_backbone_advantage(external_backbone, batch_size=batch_size)

    def evaluate_candidate_backbone_advantage(
        self,
        candidate_backbone: Dict[str, torch.Tensor],
        batch_size: int | None = None,
    ) -> float:
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        return self._evaluate_actor_backbone_advantage(candidate_backbone, batch_size=batch_size)

    def soft_load_actor_backbone(self, candidate_backbone: Dict[str, torch.Tensor], eta: float) -> None:
        if eta <= 0.0:
            return
        current_backbone = self.get_shared_state()["actor_backbone"]
        mixed_backbone: Dict[str, torch.Tensor] = {}
        for key, current_value in current_backbone.items():
            local_tensor = current_value.to(self.device)
            candidate_tensor = candidate_backbone[key].to(self.device)
            mixed_backbone[key] = ((1.0 - eta) * local_tensor + eta * candidate_tensor).detach().cpu()
        self._load_backbone_module(self.actor, mixed_backbone)
        self._sync_inference_actor()

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
                "lambda_value": float(self.lambda_value.detach().cpu().item()),
                "mean_qcf_pi": 0.0,
            }

        batch = self.replay_buffer.sample(self.config.batch_size)
        obs_graphs = [transition.obs for transition in batch]
        next_obs_graphs = [transition.next_obs for transition in batch]
        obs_batch = self._batch_graphs(obs_graphs, device=self.device)
        next_batch = self._batch_graphs(next_obs_graphs, device=self.device)
        action_batch = self._flat_node_action_to_hetero_action(
            obs_graphs,
            [transition.action for transition in batch],
            device=self.device,
        )
        reward = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device)
        cost = torch.tensor([transition.cost for transition in batch], dtype=torch.float32, device=self.device)
        done = torch.tensor([transition.done for transition in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_batch, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha.detach() * next_log_pi
            next_q_value = reward + (1.0 - done) * self.config.gamma * min_qf_next_target
            qcf1_next_target, qcf2_next_target = self.cost_critic_target(next_batch, next_action)
            mean_qcf_next_target = 0.5 * (qcf1_next_target + qcf2_next_target)
            next_qc_value = cost + (1.0 - done) * self.config.gamma * mean_qcf_next_target

        qf1, qf2 = self.critic(obs_batch, action_batch)
        critic_loss = F.mse_loss(qf1, next_q_value) + F.mse_loss(qf2, next_q_value)
        critic_proximal_penalty = self._compute_critic_proximal_penalty()
        critic_loss = critic_loss + self.config.critic_proximal_weight * critic_proximal_penalty
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        qcf1, qcf2 = self.cost_critic(obs_batch, action_batch)
        cost_critic_loss = F.mse_loss(qcf1, next_qc_value) + F.mse_loss(qcf2, next_qc_value)
        cost_critic_proximal_penalty = self._compute_cost_critic_proximal_penalty()
        cost_critic_loss = cost_critic_loss + self.config.critic_proximal_weight * cost_critic_proximal_penalty
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(obs_batch)
        qf1_pi, qf2_pi = self.critic(obs_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        qcf1_pi, qcf2_pi = self.cost_critic(obs_batch, pi)
        mean_qcf_pi = 0.5 * (qcf1_pi + qcf2_pi)
        actor_loss = ((self.alpha.detach() * log_pi) - min_qf_pi + self.lambda_value.detach() * mean_qcf_pi).mean()
        actor_proximal_penalty = self._compute_actor_proximal_penalty()
        actor_total_loss = actor_loss + self.config.actor_proximal_weight * actor_proximal_penalty
        self.actor_optimizer.zero_grad()
        actor_total_loss.backward()
        self.actor_optimizer.step()

        target_entropy = self._compute_batch_target_entropy(obs_batch)
        alpha_loss = -(self.log_alpha * (log_pi + target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        lambda_loss = (self.log_lambda * (self.config.d - mean_qcf_pi.detach())).mean()
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        self.soft_update_targets()
        self._sync_inference_actor()
        return {
            "actor_loss": float(actor_total_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "cost_critic_loss": float(cost_critic_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "lambda_loss": float(lambda_loss.detach().cpu().item()),
            "lambda_value": float(self.lambda_value.detach().cpu().item()),
            "mean_qcf_pi": float(mean_qcf_pi.detach().mean().cpu().item()),
        }

    def soft_update_targets(self) -> None:
        for target_param, source_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau)
        for target_param, source_param in zip(self.cost_critic_target.parameters(), self.cost_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau)

    def get_shared_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        shared_state = {
            "actor_backbone": {
                **{f"node_embedding.{k}": v.detach().cpu().clone() for k, v in self.actor.node_embedding.state_dict().items()},
                **{f"hgt_conv1.{k}": v.detach().cpu().clone() for k, v in self.actor.hgt_conv1.state_dict().items()},
            }
        }
        if self.enable_federation and self.config.federate_critic_backbone:
            shared_state["critic_backbone"] = {
                **{f"q1.node_embedding.{k}": v.detach().cpu().clone() for k, v in self.critic.q1.node_embedding.state_dict().items()},
                **{f"q1.hgt_conv1.{k}": v.detach().cpu().clone() for k, v in self.critic.q1.hgt_conv1.state_dict().items()},
                **{f"q1.norm1.{k}": v.detach().cpu().clone() for k, v in self.critic.q1.norm1.state_dict().items()},
                **{f"q1.hgt_conv2.{k}": v.detach().cpu().clone() for k, v in self.critic.q1.hgt_conv2.state_dict().items()},
                **{f"q1.norm2.{k}": v.detach().cpu().clone() for k, v in self.critic.q1.norm2.state_dict().items()},
                **{f"q2.node_embedding.{k}": v.detach().cpu().clone() for k, v in self.critic.q2.node_embedding.state_dict().items()},
                **{f"q2.hgt_conv1.{k}": v.detach().cpu().clone() for k, v in self.critic.q2.hgt_conv1.state_dict().items()},
                **{f"q2.norm1.{k}": v.detach().cpu().clone() for k, v in self.critic.q2.norm1.state_dict().items()},
                **{f"q2.hgt_conv2.{k}": v.detach().cpu().clone() for k, v in self.critic.q2.hgt_conv2.state_dict().items()},
                **{f"q2.norm2.{k}": v.detach().cpu().clone() for k, v in self.critic.q2.norm2.state_dict().items()},
            }
            shared_state["cost_critic_backbone"] = {
                **{f"q1.node_embedding.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q1.node_embedding.state_dict().items()},
                **{f"q1.hgt_conv1.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q1.hgt_conv1.state_dict().items()},
                **{f"q1.norm1.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q1.norm1.state_dict().items()},
                **{f"q1.hgt_conv2.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q1.hgt_conv2.state_dict().items()},
                **{f"q1.norm2.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q1.norm2.state_dict().items()},
                **{f"q2.node_embedding.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q2.node_embedding.state_dict().items()},
                **{f"q2.hgt_conv1.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q2.hgt_conv1.state_dict().items()},
                **{f"q2.norm1.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q2.norm1.state_dict().items()},
                **{f"q2.hgt_conv2.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q2.hgt_conv2.state_dict().items()},
                **{f"q2.norm2.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q2.norm2.state_dict().items()},
            }
        return shared_state

    def set_global_actor_reference(self, shared_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        if not self.enable_federation:
            self.global_actor_reference = {}
            self.global_critic1_reference = {}
            self.global_critic2_reference = {}
            self.global_cost_critic1_reference = {}
            self.global_cost_critic2_reference = {}
            return
        if "actor_backbone" in shared_state:
            actor_backbone = shared_state["actor_backbone"]
            self.global_actor_reference = {
                key: value.detach().to(self.device).clone()
                for key, value in actor_backbone.items()
            }
        elif "critic_backbone" not in shared_state and "cost_critic_backbone" not in shared_state:
            self.global_actor_reference = {}
        if self.config.federate_critic_backbone and "critic_backbone" in shared_state:
            critic_backbone = shared_state["critic_backbone"]
            self.global_critic1_reference = {
                key[len("q1."):]: value.detach().to(self.device).clone()
                for key, value in critic_backbone.items()
                if key.startswith("q1.")
            }
            self.global_critic2_reference = {
                key[len("q2."):]: value.detach().to(self.device).clone()
                for key, value in critic_backbone.items()
                if key.startswith("q2.")
            }
        if self.config.federate_critic_backbone and "cost_critic_backbone" in shared_state:
            cost_critic_backbone = shared_state["cost_critic_backbone"]
            self.global_cost_critic1_reference = {
                key[len("q1."):]: value.detach().to(self.device).clone()
                for key, value in cost_critic_backbone.items()
                if key.startswith("q1.")
            }
            self.global_cost_critic2_reference = {
                key[len("q2."):]: value.detach().to(self.device).clone()
                for key, value in cost_critic_backbone.items()
                if key.startswith("q2.")
            }
        elif "actor_backbone" in shared_state:
            self.global_critic1_reference = {}
            self.global_critic2_reference = {}
            self.global_cost_critic1_reference = {}
            self.global_cost_critic2_reference = {}

    def load_shared_state(self, shared_state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        if not self.enable_federation:
            return
        if "actor_backbone" in shared_state:
            self._load_backbone_module(
                module=self.actor,
                shared_block=shared_state["actor_backbone"],
            )
        if self.config.federate_critic_backbone and "critic_backbone" in shared_state:
            for module, prefix in (
                (self.critic.q1, "q1."),
                (self.critic.q2, "q2."),
            ):
                self._load_named_backbone_module(
                    module=module,
                    shared_block=shared_state["critic_backbone"],
                    block_prefix=prefix,
                )
        if self.config.federate_critic_backbone and "cost_critic_backbone" in shared_state:
            for module, prefix in (
                (self.cost_critic.q1, "q1."),
                (self.cost_critic.q2, "q2."),
            ):
                self._load_named_backbone_module(
                    module=module,
                    shared_block=shared_state["cost_critic_backbone"],
                    block_prefix=prefix,
                )
        self._sync_inference_actor()

    def export_checkpoint(self, park_type: str, episode: int) -> Dict[str, Any]:
        return {
            "checkpoint_format": "hgt_csac_v1",
            "park_type": park_type,
            "episode": episode,
            "agent_config": asdict(self.config),
            "state_spec": {
                "node_sizes": dict(self.node_sizes),
                "metadata": {
                    "node_types": list(NODE_TYPES),
                    "edge_types": [list(edge_type) for edge_type in EDGE_TYPES],
                },
            },
            "models": {
                "actor": {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()},
                "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
                "critic_target": {k: v.detach().cpu().clone() for k, v in self.critic_target.state_dict().items()},
                "cost_critic": {k: v.detach().cpu().clone() for k, v in self.cost_critic.state_dict().items()},
                "cost_critic_target": {k: v.detach().cpu().clone() for k, v in self.cost_critic_target.state_dict().items()},
            },
            "optimizers": {
                "actor": self.actor_optimizer.state_dict(),
                "critic": self.critic_optimizer.state_dict(),
                "cost_critic": self.cost_critic_optimizer.state_dict(),
                "alpha": self.alpha_optimizer.state_dict(),
                "lambda": self.lambda_optimizer.state_dict(),
            },
            "temperature": {
                "log_alpha": float(self.log_alpha.detach().cpu().item()),
                "log_lambda": float(self.log_lambda.detach().cpu().item()),
            },
            "replay_buffer": self.replay_buffer.state_dict(),
        }

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.actor.load_state_dict(checkpoint["models"]["actor"])
        self.critic.load_state_dict(checkpoint["models"]["critic"])
        self.critic_target.load_state_dict(checkpoint["models"]["critic_target"])
        self.cost_critic.load_state_dict(checkpoint["models"]["cost_critic"])
        self.cost_critic_target.load_state_dict(checkpoint["models"]["cost_critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["optimizers"]["actor"])
        self.critic_optimizer.load_state_dict(checkpoint["optimizers"]["critic"])
        self.cost_critic_optimizer.load_state_dict(checkpoint["optimizers"]["cost_critic"])
        self.alpha_optimizer.load_state_dict(checkpoint["optimizers"]["alpha"])
        self.lambda_optimizer.load_state_dict(checkpoint["optimizers"]["lambda"])
        self.log_alpha.data.copy_(torch.tensor(checkpoint["temperature"]["log_alpha"], device=self.device))
        self.log_lambda.data.copy_(torch.tensor(checkpoint["temperature"]["log_lambda"], device=self.device))
        self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        self._sync_inference_actor()

    def _build_inference_actor(self) -> HGTActor:
        return HGTActor(
            node_sizes=self.node_sizes,
            feature_dim=self.config.feature_dim,
            decouple_actor_output_heads=self.config.decouple_actor_output_heads,
        ).to(self.act_device)

    def _sync_inference_actor(self) -> None:
        self.actor_inference.load_state_dict(
            {key: value.detach().to(self.act_device) for key, value in self.actor.state_dict().items()}
        )
        self.actor_inference.eval()

    def _load_backbone_module(
        self,
        module: nn.Module,
        shared_block: Dict[str, torch.Tensor],
    ) -> None:
        self._load_named_backbone_module(
            module=module,
            shared_block=shared_block,
            block_prefix="",
        )

    def _load_named_backbone_module(
        self,
        module: nn.Module,
        shared_block: Dict[str, torch.Tensor],
        block_prefix: str,
    ) -> None:
        for prefix, submodule in (
            ("node_embedding", module.node_embedding),
            ("hgt_conv1", module.hgt_conv1),
            ("norm1", module.norm1),
            ("hgt_conv2", module.hgt_conv2),
            ("norm2", module.norm2),
        ):
            local_state = submodule.state_dict()
            loaded_state = {}
            for key, local_value in local_state.items():
                shared_key = f"{block_prefix}{prefix}.{key}"
                if shared_key not in shared_block:
                    continue
                loaded_state[key] = shared_block[shared_key].detach().to(
                    self.device,
                    dtype=local_value.dtype,
                )
            if loaded_state:
                submodule.load_state_dict(loaded_state, strict=False)

    def _compute_actor_proximal_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        if not self.enable_federation or not self.global_actor_reference:
            return penalty
        for prefix, submodule in (
            ("node_embedding", self.actor.node_embedding),
            ("hgt_conv1", self.actor.hgt_conv1),
            ("norm1", self.actor.norm1),
            ("hgt_conv2", self.actor.hgt_conv2),
            ("norm2", self.actor.norm2),
        ):
            for name, param in submodule.named_parameters():
                reference_key = f"{prefix}.{name}"
                if reference_key not in self.global_actor_reference:
                    continue
                penalty = penalty + (param - self.global_actor_reference[reference_key]).pow(2).sum()
        return penalty

    def _compute_critic_proximal_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        if not self.enable_federation or not self.config.federate_critic_backbone:
            return penalty
        penalty = penalty + self._compute_named_critic_backbone_penalty(self.critic.q1, self.global_critic1_reference)
        penalty = penalty + self._compute_named_critic_backbone_penalty(self.critic.q2, self.global_critic2_reference)
        return penalty

    def _compute_cost_critic_proximal_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        if not self.enable_federation or not self.config.federate_critic_backbone:
            return penalty
        penalty = penalty + self._compute_named_critic_backbone_penalty(
            self.cost_critic.q1,
            self.global_cost_critic1_reference,
        )
        penalty = penalty + self._compute_named_critic_backbone_penalty(
            self.cost_critic.q2,
            self.global_cost_critic2_reference,
        )
        return penalty

    @staticmethod
    def _compute_named_critic_backbone_penalty(
        module: nn.Module,
        reference: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not reference:
            param = next(module.parameters(), None)
            device = torch.device("cpu") if param is None else param.device
            return torch.zeros((), dtype=torch.float32, device=device)

        penalty = torch.zeros((), dtype=torch.float32, device=next(module.parameters()).device)
        for prefix, submodule in (
            ("node_embedding", module.node_embedding),
            ("hgt_conv1", module.hgt_conv1),
            ("norm1", module.norm1),
            ("hgt_conv2", module.hgt_conv2),
            ("norm2", module.norm2),
        ):
            for name, param in submodule.named_parameters():
                penalty = penalty + (param - reference[f"{prefix}.{name}"]).pow(2).sum()
        return penalty

    def _evaluate_actor_backbone_advantage(
        self,
        external_backbone: Dict[str, torch.Tensor],
        batch_size: int | None = None,
    ) -> float:
        batch_size = batch_size or self.config.batch_size
        batch = self.replay_buffer.sample(batch_size)
        obs_graphs = [transition.obs for transition in batch]
        obs_batch = self._batch_graphs(obs_graphs, device=self.device)
        with torch.inference_mode():
            local_action, local_log_pi, _ = self.actor.sample(obs_batch, deterministic=True)
            score_local = self._evaluate_action_dict_score(obs_batch, local_action, local_log_pi)
            candidate_actor = self.make_actor_with_external_backbone(external_backbone)
            candidate_action, candidate_log_pi, _ = candidate_actor.sample(obs_batch, deterministic=True)
            score_candidate = self._evaluate_action_dict_score(obs_batch, candidate_action, candidate_log_pi)
            advantage = score_candidate - score_local
        return float(advantage.detach().cpu().item())

    def _evaluate_action_dict_score(
        self,
        obs_batch: Batch,
        action_dict: Dict[str, torch.Tensor],
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2 = self.critic(obs_batch, action_dict)
        qc1, qc2 = self.cost_critic(obs_batch, action_dict)
        q_value = torch.min(q1, q2)
        qc_value = 0.5 * (qc1 + qc2)
        return (q_value - self.lambda_value.detach() * qc_value - self.alpha.detach() * log_pi).mean()

    def _graph_to_heterodata(self, graph: Dict[str, Any], device: torch.device) -> HeteroData:
        data = HeteroData()
        for node_type in NODE_TYPES:
            feature_key = f"{node_type}_features"
            features = graph[feature_key]
            if features:
                data[node_type].x = torch.tensor(features, dtype=torch.float32, device=device)
            else:
                data[node_type].x = torch.zeros((0, self.node_sizes[node_type]), dtype=torch.float32, device=device)

        global_to_local = {}
        for node_type in NODE_TYPES:
            global_indexes = graph[f"{node_type}_indexes"]
            for local_idx, global_idx in enumerate(global_indexes):
                global_to_local[int(global_idx)] = (node_type, local_idx)

        relation_edges: Dict[Tuple[str, str, str], List[List[int]]] = {
            edge_type: [[], []] for edge_type in EDGE_TYPES
        }
        for src_global, dst_global in graph["edge_index"]:
            src_type, src_local = global_to_local[int(src_global)]
            dst_type, dst_local = global_to_local[int(dst_global)]
            if src_type == "ev" and dst_type == "cs":
                relation = ("ev", "to", "cs")
            elif src_type == "cs" and dst_type == "ev":
                relation = ("cs", "rev_to_ev", "ev")
            elif src_type == "bes" and dst_type == "cs":
                relation = ("bes", "to", "cs")
            elif src_type == "cs" and dst_type == "bes":
                relation = ("cs", "rev_to_bes", "bes")
            elif src_type == "pv" and dst_type == "cs":
                relation = ("pv", "to", "cs")
            elif src_type == "cs" and dst_type == "pv":
                relation = ("cs", "rev_to_pv", "pv")
            elif src_type == "external" and dst_type == "cs":
                relation = ("external", "to", "cs")
            elif src_type == "cs" and dst_type == "external":
                relation = ("cs", "rev_to_external", "external")
            else:
                continue
            relation_edges[relation][0].append(src_local)
            relation_edges[relation][1].append(dst_local)

        for edge_type in EDGE_TYPES:
            edges = relation_edges[edge_type]
            if edges[0]:
                data[edge_type].edge_index = torch.tensor(edges, dtype=torch.long, device=device)
            else:
                data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        return data

    def _batch_graphs(self, graphs: List[Dict[str, Any]], device: torch.device) -> Batch:
        hetero_graphs = [self._graph_to_heterodata(graph, device=device) for graph in graphs]
        return Batch.from_data_list(hetero_graphs)

    def _flat_node_action_to_hetero_action(
        self,
        graphs: List[Dict[str, Any]],
        node_actions: List[torch.Tensor],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        bes_actions: List[torch.Tensor] = []
        ev_actions: List[torch.Tensor] = []
        for graph, node_action in zip(graphs, node_actions):
            flat_action = node_action.to(device=device, dtype=torch.float32).reshape(-1)
            bes_index = int(graph["bes_indexes"][0])
            bes_actions.append(flat_action[bes_index].reshape(1))
            for ev_index in graph["ev_indexes"]:
                ev_actions.append(flat_action[int(ev_index)].reshape(1))
        bes_tensor = (
            torch.cat(bes_actions, dim=0)
            if bes_actions
            else torch.zeros((0,), dtype=torch.float32, device=device)
        )
        ev_tensor = (
            torch.cat(ev_actions, dim=0)
            if ev_actions
            else torch.zeros((0,), dtype=torch.float32, device=device)
        )
        return {"bes": bes_tensor, "ev": ev_tensor}

    @staticmethod
    def _compute_batch_target_entropy(batch: Batch) -> torch.Tensor:
        batch_size = batch.num_graphs
        bes_counts = global_add_pool(
            torch.ones((batch["bes"].x.shape[0], 1), dtype=torch.float32, device=batch["bes"].x.device),
            batch["bes"].batch,
            size=batch_size,
        ).squeeze(-1)
        if batch["ev"].x.shape[0] > 0:
            ev_counts = global_add_pool(
                torch.ones((batch["ev"].x.shape[0], 1), dtype=torch.float32, device=batch["ev"].x.device),
                batch["ev"].batch,
                size=batch_size,
            ).squeeze(-1)
        else:
            ev_counts = torch.zeros((batch_size,), dtype=torch.float32, device=batch["bes"].x.device)
        return -(bes_counts + ev_counts)

    @staticmethod
    def _hetero_action_to_node_action(graph: Dict[str, Any], action_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        node_action = torch.zeros((len(graph["node_types"]),), dtype=torch.float32)
        if graph["bes_indexes"]:
            node_action[int(graph["bes_indexes"][0])] = action_dict["bes"][0].detach().cpu()
        for offset, ev_index in enumerate(graph["ev_indexes"]):
            node_action[int(ev_index)] = action_dict["ev"][offset].detach().cpu()
        return node_action

    @staticmethod
    def _hetero_action_to_env_action(graph: Dict[str, Any], action_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        env_action = {"bes": 0.0, "ev": {}}
        if graph["bes_indexes"]:
            env_action["bes"] = float(action_dict["bes"][0].detach().cpu().item())
        for offset, ev_index in enumerate(graph["ev_indexes"]):
            env_action["ev"][graph["node_names"][int(ev_index)]] = float(
                action_dict["ev"][offset].detach().cpu().item()
            )
        return env_action
