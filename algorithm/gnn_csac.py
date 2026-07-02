from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

from agent.replaybuffer import ReplayBuffer
from agent.state import get_node_sizes


GNN_FEATURE_DIM = 32
GNN_HIDDEN_DIM = 128
GNN_ACTOR_NUM_LAYERS = 2
GNN_CRITIC_NUM_LAYERS = 2
NONFED_BACKBONE_LR = 3e-4
NONFED_HEAD_LR = 3e-4
FED_BACKBONE_LR = 1e-4
FED_HEAD_LR = 3e-4
LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0
EPSILON = 1e-6
NODE_TYPE_NAMES = ("cs", "bes", "pv", "external", "ev")


def weights_init_(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
        torch.nn.init.constant_(module.bias, 0.0)


class TypeSpecificEmbedding(nn.Module):
    def __init__(self, node_sizes: Dict[str, int], feature_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {name: nn.Linear(node_sizes[name], feature_dim) for name in NODE_TYPE_NAMES}
        )
        self.apply(weights_init_)

    def forward(self, graph_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_nodes = int(graph_batch["node_type_ids"].shape[0])
        embedded = torch.zeros(
            (total_nodes, next(iter(self.embeddings.values())).out_features),
            dtype=torch.float32,
            device=graph_batch["node_type_ids"].device,
        )
        for node_type_name in NODE_TYPE_NAMES:
            index_key = f"{node_type_name}_indexes"
            feature_key = f"{node_type_name}_features"
            indexes = graph_batch[index_key]
            features = graph_batch[feature_key]
            if indexes.numel() > 0:
                embedded[indexes] = self.embeddings[node_type_name](features)
        return F.relu(embedded)


class ActorActionGNN(nn.Module):
    def __init__(
        self,
        node_sizes: Dict[str, int],
        feature_dim: int,
        gnn_hidden_dim: int,
        num_gcn_layers: int,
        decouple_actor_output_heads: bool = False,
    ) -> None:
        super().__init__()
        self.decouple_actor_output_heads = decouple_actor_output_heads
        self.node_embedding = TypeSpecificEmbedding(node_sizes=node_sizes, feature_dim=feature_dim)
        if num_gcn_layers != 2:
            raise ValueError("Current actor architecture is fixed to 2 GCN layers. Set actor_num_gcn_layers=2.")
        hidden_dim_1 = gnn_hidden_dim // 2
        hidden_dim_2 = gnn_hidden_dim

        self.gcn_conv = GCNConv(feature_dim, hidden_dim_1)
        self.gcn_layers = nn.ModuleList([GCNConv(hidden_dim_1, hidden_dim_2)])
        if self.decouple_actor_output_heads:
            self.mean_heads = nn.ModuleDict(
                {
                    "bes": GCNConv(hidden_dim_2, 1),
                    "ev": GCNConv(hidden_dim_2, 1),
                }
            )
            self.log_std_heads = nn.ModuleDict(
                {
                    "bes": GCNConv(hidden_dim_2, 1),
                    "ev": GCNConv(hidden_dim_2, 1),
                }
            )
        else:
            self.mean_linear = GCNConv(hidden_dim_2, 1)
            self.log_std_linear = GCNConv(hidden_dim_2, 1)

    def _encode(
        self,
        graph_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = self.node_embedding(graph_batch)
        x = F.relu(self.gcn_conv(x, graph_batch["edge_index"]))
        for layer in self.gcn_layers:
            x = F.relu(layer(x, graph_batch["edge_index"]))
        return x

    def forward(
        self,
        graph_batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(graph_batch)
        if self.decouple_actor_output_heads:
            mean = torch.zeros((x.shape[0],), dtype=torch.float32, device=x.device)
            log_std = torch.zeros((x.shape[0],), dtype=torch.float32, device=x.device)
            for node_type_name, indexes in (
                ("bes", graph_batch["bes_indexes"]),
                ("ev", graph_batch["ev_indexes"]),
            ):
                if indexes.numel() == 0:
                    continue
                mean_by_type = self.mean_heads[node_type_name](x, graph_batch["edge_index"]).squeeze(-1)
                log_std_by_type = self.log_std_heads[node_type_name](x, graph_batch["edge_index"]).squeeze(-1)
                mean[indexes] = mean_by_type[indexes]
                log_std[indexes] = log_std_by_type[indexes]
        else:
            mean = self.mean_linear(x, graph_batch["edge_index"]).squeeze(-1)
            log_std = self.log_std_linear(x, graph_batch["edge_index"]).squeeze(-1)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(
        self,
        graph_batch: Dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(graph_batch)
        action_node_indices = graph_batch["action_node_indices"]
        action_batches = graph_batch["batch"][action_node_indices]

        action_mean = mean[action_node_indices]
        action_log_std = log_std[action_node_indices]
        action_std = action_log_std.exp()
        normal = Normal(action_mean, action_std)
        raw_action = action_mean if deterministic else normal.rsample()
        squashed_action = torch.tanh(raw_action)

        full_action = torch.zeros_like(mean)
        full_action[action_node_indices] = squashed_action

        log_prob = normal.log_prob(raw_action)
        log_prob = log_prob - torch.log(1 - squashed_action.pow(2) + EPSILON)
        pooled_log_prob = global_add_pool(
            log_prob.unsqueeze(-1),
            action_batches,
            size=int(graph_batch["batch"].max().item()) + 1,
        ).squeeze(-1)

        full_mean = torch.zeros_like(mean)
        full_mean[action_node_indices] = torch.tanh(action_mean)
        return full_action, pooled_log_prob, full_mean

    def actor_head_parameters(self) -> list[nn.Parameter]:
        if self.decouple_actor_output_heads:
            return list(self.mean_heads.parameters()) + list(self.log_std_heads.parameters())
        return list(self.mean_linear.parameters()) + list(self.log_std_linear.parameters())


class CriticActionGNN(nn.Module):
    def __init__(
        self,
        node_sizes: Dict[str, int],
        feature_dim: int,
        gnn_hidden_dim: int,
        num_gcn_layers: int,
    ) -> None:
        super().__init__()
        self.node_embedding = TypeSpecificEmbedding(node_sizes=node_sizes, feature_dim=feature_dim)
        if num_gcn_layers != 2:
            raise ValueError("Current critic architecture is fixed to 2 GCN layers. Set critic_num_gcn_layers=2.")
        hidden_dim_1 = gnn_hidden_dim // 2
        hidden_dim_2 = gnn_hidden_dim

        self.gcn_conv = GCNConv(feature_dim + 1, hidden_dim_1)
        self.gcn_layers = nn.ModuleList([GCNConv(hidden_dim_1, hidden_dim_2)])
        self.l1 = nn.Linear(hidden_dim_2, 64)
        self.l2 = nn.Linear(64, 1)
        self.apply(weights_init_)

    def forward(self, graph_batch: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        x = self.node_embedding(graph_batch)
        masked_action = torch.zeros_like(action)
        masked_action[graph_batch["action_node_indices"]] = action[graph_batch["action_node_indices"]]
        state_action = torch.cat([x, masked_action.reshape(-1, 1)], dim=1)
        x = F.relu(self.gcn_conv(state_action, graph_batch["edge_index"]))
        for layer in self.gcn_layers:
            x = F.relu(layer(x, graph_batch["edge_index"]))
        pooled = global_mean_pool(x, graph_batch["batch"])
        x = F.relu(self.l1(pooled))
        return self.l2(x).squeeze(-1)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        node_sizes: Dict[str, int],
        feature_dim: int,
        gnn_hidden_dim: int,
        num_gcn_layers: int,
    ) -> None:
        super().__init__()
        self.q1 = CriticActionGNN(
            node_sizes=node_sizes,
            feature_dim=feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
        )
        self.q2 = CriticActionGNN(
            node_sizes=node_sizes,
            feature_dim=feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
        )

    def forward(self, graph_batch: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(graph_batch, action), self.q2(graph_batch, action)


def compute_batch_target_entropy(graph_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    action_batches = graph_batch["batch"][graph_batch["action_node_indices"]]
    counts = global_add_pool(
        torch.ones((action_batches.shape[0], 1), dtype=torch.float32, device=action_batches.device),
        action_batches,
        size=int(graph_batch["batch"].max().item()) + 1,
    ).squeeze(-1)
    return -counts


def map_node_actions_to_env_action(
    park_graph: Dict[str, Any],
    node_action: torch.Tensor,
) -> Dict[str, Any]:
    mapped_action = torch.zeros(park_graph["fixed_action_dim"], dtype=node_action.dtype, device=node_action.device)
    for node_idx, fixed_idx in zip(park_graph["action_node_indices"], park_graph["action_mapper"]):
        mapped_action[fixed_idx] = node_action[node_idx]

    env_action = {"bes": 0.0, "ev": {}}
    env_action["bes"] = float(mapped_action[park_graph["bes_action_index"]].detach().cpu().item())
    for ev_idx in park_graph["ev_indexes"]:
        ev_id = park_graph["node_names"][ev_idx]
        env_action["ev"][ev_id] = float(node_action[ev_idx].detach().cpu().item())
    return env_action


@dataclass
class LocalGNNCSACConfig:
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
    actor_head_lr: float | None = None
    critic_backbone_lr: float | None = None
    critic_head_lr: float | None = None
    gnn_hidden_dim: int = GNN_HIDDEN_DIM
    actor_num_gcn_layers: int = GNN_ACTOR_NUM_LAYERS
    critic_num_gcn_layers: int = GNN_CRITIC_NUM_LAYERS
    feature_dim: int = GNN_FEATURE_DIM
    decouple_actor_output_heads: bool | None = None

    def __post_init__(self) -> None:
        if self.decouple_actor_output_heads is None:
            self.decouple_actor_output_heads = False
        if self.actor_backbone_lr is None:
            self.actor_backbone_lr = FED_BACKBONE_LR if self.enable_federation else NONFED_BACKBONE_LR
        if self.actor_head_lr is None:
            self.actor_head_lr = FED_HEAD_LR if self.enable_federation else NONFED_HEAD_LR
        if self.critic_backbone_lr is None:
            self.critic_backbone_lr = FED_BACKBONE_LR if self.enable_federation and self.federate_critic_backbone else NONFED_BACKBONE_LR
        if self.critic_head_lr is None:
            self.critic_head_lr = FED_HEAD_LR if self.enable_federation and self.federate_critic_backbone else NONFED_HEAD_LR


class LocalGNNCSACAgent:
    def __init__(self, config: LocalGNNCSACConfig) -> None:
        self.config = config
        self.enable_federation = config.enable_federation
        self.device = torch.device(config.update_device)
        self.act_device = torch.device(config.act_device)
        torch.manual_seed(config.seed)
        self.node_sizes = get_node_sizes(config.privacy_mode)

        self.actor = ActorActionGNN(
            node_sizes=self.node_sizes,
            feature_dim=config.feature_dim,
            gnn_hidden_dim=config.gnn_hidden_dim,
            num_gcn_layers=config.actor_num_gcn_layers,
            decouple_actor_output_heads=config.decouple_actor_output_heads,
        ).to(self.device)
        self.critic = CriticNetwork(
            node_sizes=self.node_sizes,
            feature_dim=config.feature_dim,
            gnn_hidden_dim=config.gnn_hidden_dim,
            num_gcn_layers=config.critic_num_gcn_layers,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.cost_critic = CriticNetwork(
            node_sizes=self.node_sizes,
            feature_dim=config.feature_dim,
            gnn_hidden_dim=config.gnn_hidden_dim,
            num_gcn_layers=config.critic_num_gcn_layers,
        ).to(self.device)
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
                {"params": self.actor.gcn_conv.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.gcn_layers.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.actor_head_parameters(), "lr": config.actor_head_lr},
            ]
        )
        self.critic_optimizer = torch.optim.Adam(
            [
                {"params": self.critic.q1.node_embedding.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q1.gcn_conv.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q1.gcn_layers.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q1.l1.parameters(), "lr": config.critic_head_lr},
                {"params": self.critic.q1.l2.parameters(), "lr": config.critic_head_lr},
                {"params": self.critic.q2.node_embedding.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q2.gcn_conv.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q2.gcn_layers.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.critic.q2.l1.parameters(), "lr": config.critic_head_lr},
                {"params": self.critic.q2.l2.parameters(), "lr": config.critic_head_lr},
            ]
        )
        self.cost_critic_optimizer = torch.optim.Adam(
            [
                {"params": self.cost_critic.q1.node_embedding.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q1.gcn_conv.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q1.gcn_layers.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q1.l1.parameters(), "lr": config.critic_head_lr},
                {"params": self.cost_critic.q1.l2.parameters(), "lr": config.critic_head_lr},
                {"params": self.cost_critic.q2.node_embedding.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q2.gcn_conv.parameters(), "lr": config.critic_backbone_lr},
                {"params": self.cost_critic.q2.gcn_layers.parameters(), "lr": config.critic_backbone_lr},
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
            graph_batch = self._single_graph_to_batch(park_graph, device=self.act_device)
            node_action, _, node_mean = self.actor_inference.sample(graph_batch, deterministic=deterministic)
            chosen_action = node_mean if deterministic else node_action
            env_action = self._map_node_actions_to_env_action(park_graph, chosen_action)
            raw_node_action = chosen_action.detach().cpu()
            if return_node_action:
                return env_action, raw_node_action
            return env_action

    def evaluate_cmdp_score(
        self,
        park_graph: Dict[str, Any],
        node_action: torch.Tensor,
    ) -> float:
        with torch.inference_mode():
            graph_batch = self._single_graph_to_batch(park_graph, device=self.device)
            action_tensor = node_action.to(self.device, dtype=torch.float32).reshape(-1)
            q1, q2 = self.critic(graph_batch, action_tensor)
            qc1, qc2 = self.cost_critic(graph_batch, action_tensor)
            q_value = torch.min(q1, q2)
            qc_value = 0.5 * (qc1 + qc2)
            score = q_value - self.lambda_value.detach() * qc_value
            return float(score.detach().cpu().item())

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
        obs_batch = [transition.obs for transition in batch]
        next_obs_batch = [transition.next_obs for transition in batch]
        action_batch = torch.cat(
            [transition.action.to(self.device).reshape(-1) for transition in batch],
            dim=0,
        )
        reward = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device)
        cost = torch.tensor([transition.cost for transition in batch], dtype=torch.float32, device=self.device)
        done = torch.tensor([transition.done for transition in batch], dtype=torch.float32, device=self.device)

        obs_graph_batch = self._batch_graphs(obs_batch, device=self.device)
        next_graph_batch = self._batch_graphs(next_obs_batch, device=self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_graph_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_graph_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha.detach() * next_state_log_pi
            next_q_value = reward + (1.0 - done) * self.config.gamma * min_qf_next_target
            qcf1_next_target, qcf2_next_target = self.cost_critic_target(next_graph_batch, next_state_action)
            mean_qcf_next_target = 0.5 * (qcf1_next_target + qcf2_next_target)
            next_qc_value = cost + (1.0 - done) * self.config.gamma * mean_qcf_next_target

        qf1, qf2 = self.critic(obs_graph_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        critic_loss_mean = qf1_loss + qf2_loss
        critic_proximal_penalty = self._compute_critic_proximal_penalty()
        critic_loss = critic_loss_mean + self.config.critic_proximal_weight * critic_proximal_penalty

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        qcf1, qcf2 = self.cost_critic(obs_graph_batch, action_batch)
        qcf1_loss = F.mse_loss(qcf1, next_qc_value)
        qcf2_loss = F.mse_loss(qcf2, next_qc_value)
        cost_critic_loss_mean = qcf1_loss + qcf2_loss
        cost_critic_proximal_penalty = self._compute_cost_critic_proximal_penalty()
        cost_critic_loss = cost_critic_loss_mean + self.config.critic_proximal_weight * cost_critic_proximal_penalty

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(obs_graph_batch)
        qf1_pi, qf2_pi = self.critic(obs_graph_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        qcf1_pi, qcf2_pi = self.cost_critic(obs_graph_batch, pi)
        mean_qcf_pi = 0.5 * (qcf1_pi + qcf2_pi)
        cost_term = self.lambda_value.detach() * mean_qcf_pi
        actor_loss_mean = ((self.alpha.detach() * log_pi) - min_qf_pi + cost_term).mean()
        actor_proximal_penalty = self._compute_actor_proximal_penalty()
        actor_total_loss = actor_loss_mean + self.config.actor_proximal_weight * actor_proximal_penalty

        self.actor_optimizer.zero_grad()
        actor_total_loss.backward()
        self.actor_optimizer.step()

        target_entropy = compute_batch_target_entropy(obs_graph_batch)
        alpha_loss_mean = -(self.log_alpha * (log_pi + target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss_mean.backward()
        self.alpha_optimizer.step()

        lambda_loss_mean = (self.log_lambda * (self.config.d - mean_qcf_pi.detach())).mean()
        self.lambda_optimizer.zero_grad()
        lambda_loss_mean.backward()
        self.lambda_optimizer.step()

        self.soft_update_targets()
        self._sync_inference_actor()
        return {
            "actor_loss": float(actor_total_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "cost_critic_loss": float(cost_critic_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss_mean.detach().cpu().item()),
            "lambda_loss": float(lambda_loss_mean.detach().cpu().item()),
            "lambda_value": float(self.lambda_value.detach().cpu().item()),
            "mean_qcf_pi": float(mean_qcf_pi.detach().mean().cpu().item()),
        }

    def soft_update_targets(self) -> None:
        for target_param, source_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau
            )
        for target_param, source_param in zip(self.cost_critic_target.parameters(), self.cost_critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + source_param.data * self.config.tau
            )

    def get_shared_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        shared_state = {
            "actor_backbone": {
                **{f"node_embedding.{k}": v.detach().cpu().clone() for k, v in self.actor.node_embedding.state_dict().items()},
                **{f"gcn_conv.{k}": v.detach().cpu().clone() for k, v in self.actor.gcn_conv.state_dict().items()},
                **{f"gcn_layers.{k}": v.detach().cpu().clone() for k, v in self.actor.gcn_layers.state_dict().items()},
            },
        }
        if self.enable_federation and self.config.federate_critic_backbone:
            shared_state["critic_backbone"] = {
                **{f"q1.node_embedding.{k}": v.detach().cpu().clone() for k, v in self.critic.q1.node_embedding.state_dict().items()},
                **{f"q1.gcn_conv.{k}": v.detach().cpu().clone() for k, v in self.critic.q1.gcn_conv.state_dict().items()},
                **{f"q1.gcn_layers.{k}": v.detach().cpu().clone() for k, v in self.critic.q1.gcn_layers.state_dict().items()},
                **{f"q2.node_embedding.{k}": v.detach().cpu().clone() for k, v in self.critic.q2.node_embedding.state_dict().items()},
                **{f"q2.gcn_conv.{k}": v.detach().cpu().clone() for k, v in self.critic.q2.gcn_conv.state_dict().items()},
                **{f"q2.gcn_layers.{k}": v.detach().cpu().clone() for k, v in self.critic.q2.gcn_layers.state_dict().items()},
            }
            shared_state["cost_critic_backbone"] = {
                **{f"q1.node_embedding.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q1.node_embedding.state_dict().items()},
                **{f"q1.gcn_conv.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q1.gcn_conv.state_dict().items()},
                **{f"q1.gcn_layers.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q1.gcn_layers.state_dict().items()},
                **{f"q2.node_embedding.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q2.node_embedding.state_dict().items()},
                **{f"q2.gcn_conv.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q2.gcn_conv.state_dict().items()},
                **{f"q2.gcn_layers.{k}": v.detach().cpu().clone() for k, v in self.cost_critic.q2.gcn_layers.state_dict().items()},
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
            "checkpoint_format": "gnn_csac_v1",
            "park_type": park_type,
            "episode": episode,
            "agent_config": asdict(self.config),
            "state_spec": {
                "node_sizes": dict(self.node_sizes),
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

    def _build_inference_actor(self) -> ActorActionGNN:
        return ActorActionGNN(
            node_sizes=self.node_sizes,
            feature_dim=self.config.feature_dim,
            gnn_hidden_dim=self.config.gnn_hidden_dim,
            num_gcn_layers=self.config.actor_num_gcn_layers,
            decouple_actor_output_heads=self.config.decouple_actor_output_heads,
        ).to(self.act_device)

    def _sync_inference_actor(self) -> None:
        self.actor_inference.load_state_dict(
            {key: value.detach().to(self.act_device) for key, value in self.actor.state_dict().items()}
        )
        self.actor_inference.eval()

    def _load_backbone_module(
        self,
        module: torch.nn.Module,
        shared_block: Dict[str, torch.Tensor],
    ) -> None:
        self._load_named_backbone_module(
            module=module,
            shared_block=shared_block,
            block_prefix="",
        )

    def _load_named_backbone_module(
        self,
        module: torch.nn.Module,
        shared_block: Dict[str, torch.Tensor],
        block_prefix: str,
    ) -> None:
        node_embedding_state = module.node_embedding.state_dict()
        gcn_conv_state = module.gcn_conv.state_dict()
        gcn_layers_state = module.gcn_layers.state_dict()

        loaded_node_embedding_state = {}
        for key, local_value in node_embedding_state.items():
            loaded_node_embedding_state[key] = shared_block[f"{block_prefix}node_embedding.{key}"].detach().to(
                self.device,
                dtype=local_value.dtype,
            )

        loaded_gcn_conv_state = {}
        for key, local_value in gcn_conv_state.items():
            loaded_gcn_conv_state[key] = shared_block[f"{block_prefix}gcn_conv.{key}"].detach().to(
                self.device,
                dtype=local_value.dtype,
            )

        loaded_gcn_layers_state = {}
        for key, local_value in gcn_layers_state.items():
            loaded_gcn_layers_state[key] = shared_block[f"{block_prefix}gcn_layers.{key}"].detach().to(
                self.device,
                dtype=local_value.dtype,
            )

        module.node_embedding.load_state_dict(loaded_node_embedding_state)
        module.gcn_conv.load_state_dict(loaded_gcn_conv_state)
        module.gcn_layers.load_state_dict(loaded_gcn_layers_state)

    def _compute_actor_proximal_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        if not self.enable_federation or not self.global_actor_reference:
            return penalty

        for name, param in self.actor.node_embedding.named_parameters():
            penalty = penalty + (param - self.global_actor_reference[f"node_embedding.{name}"]).pow(2).sum()
        for name, param in self.actor.gcn_conv.named_parameters():
            penalty = penalty + (param - self.global_actor_reference[f"gcn_conv.{name}"]).pow(2).sum()
        for name, param in self.actor.gcn_layers.named_parameters():
            penalty = penalty + (param - self.global_actor_reference[f"gcn_layers.{name}"]).pow(2).sum()
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
        module: torch.nn.Module,
        reference: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if not reference:
            param = next(module.parameters(), None)
            device = torch.device("cpu") if param is None else param.device
            return torch.zeros((), dtype=torch.float32, device=device)

        penalty = torch.zeros((), dtype=torch.float32, device=next(module.parameters()).device)
        for prefix, submodule in (
            ("node_embedding", module.node_embedding),
            ("gcn_conv", module.gcn_conv),
            ("gcn_layers", module.gcn_layers),
        ):
            for name, param in submodule.named_parameters():
                penalty = penalty + (param - reference[f"{prefix}.{name}"]).pow(2).sum()
        return penalty

    def _single_graph_to_batch(
        self,
        graph: Dict[str, Any],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        node_count = len(graph["node_types"])
        node_type_ids = torch.tensor(graph["node_type_ids"], dtype=torch.long, device=device)
        if graph["edge_index"]:
            edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        batch = torch.zeros(node_count, dtype=torch.long, device=device)
        return {
            "node_type_ids": node_type_ids,
            "edge_index": edge_index,
            "batch": batch,
            "action_node_indices": torch.tensor(graph["action_node_indices"], dtype=torch.long, device=device),
            "ev_features": torch.tensor(graph["ev_features"], dtype=torch.float32, device=device) if graph["ev_features"] else torch.zeros((0, self.node_sizes["ev"]), dtype=torch.float32, device=device),
            "cs_features": torch.tensor(graph["cs_features"], dtype=torch.float32, device=device) if graph["cs_features"] else torch.zeros((0, self.node_sizes["cs"]), dtype=torch.float32, device=device),
            "bes_features": torch.tensor(graph["bes_features"], dtype=torch.float32, device=device) if graph["bes_features"] else torch.zeros((0, self.node_sizes["bes"]), dtype=torch.float32, device=device),
            "pv_features": torch.tensor(graph["pv_features"], dtype=torch.float32, device=device) if graph["pv_features"] else torch.zeros((0, self.node_sizes["pv"]), dtype=torch.float32, device=device),
            "external_features": torch.tensor(graph["external_features"], dtype=torch.float32, device=device) if graph["external_features"] else torch.zeros((0, self.node_sizes["external"]), dtype=torch.float32, device=device),
            "ev_indexes": torch.tensor(graph["ev_indexes"], dtype=torch.long, device=device),
            "cs_indexes": torch.tensor(graph["cs_indexes"], dtype=torch.long, device=device),
            "bes_indexes": torch.tensor(graph["bes_indexes"], dtype=torch.long, device=device),
            "pv_indexes": torch.tensor(graph["pv_indexes"], dtype=torch.long, device=device),
            "external_indexes": torch.tensor(graph["external_indexes"], dtype=torch.long, device=device),
        }

    def _batch_graphs(
        self,
        graphs: List[Dict[str, Any]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        node_type_ids_list: List[torch.Tensor] = []
        edge_index_list: List[torch.Tensor] = []
        batch_list: List[torch.Tensor] = []
        ev_features_list: List[torch.Tensor] = []
        cs_features_list: List[torch.Tensor] = []
        bes_features_list: List[torch.Tensor] = []
        pv_features_list: List[torch.Tensor] = []
        external_features_list: List[torch.Tensor] = []
        action_node_indices_list: List[torch.Tensor] = []
        ev_indexes_list: List[torch.Tensor] = []
        cs_indexes_list: List[torch.Tensor] = []
        bes_indexes_list: List[torch.Tensor] = []
        pv_indexes_list: List[torch.Tensor] = []
        external_indexes_list: List[torch.Tensor] = []
        node_offset = 0

        for graph_id, graph in enumerate(graphs):
            node_count = len(graph["node_types"])
            node_type_ids_list.append(torch.tensor(graph["node_type_ids"], dtype=torch.long, device=device))
            batch_list.append(torch.full((node_count,), graph_id, dtype=torch.long, device=device))
            ev_features_list.append(torch.tensor(graph["ev_features"], dtype=torch.float32, device=device) if graph["ev_features"] else torch.zeros((0, self.node_sizes["ev"]), dtype=torch.float32, device=device))
            cs_features_list.append(torch.tensor(graph["cs_features"], dtype=torch.float32, device=device) if graph["cs_features"] else torch.zeros((0, self.node_sizes["cs"]), dtype=torch.float32, device=device))
            bes_features_list.append(torch.tensor(graph["bes_features"], dtype=torch.float32, device=device) if graph["bes_features"] else torch.zeros((0, self.node_sizes["bes"]), dtype=torch.float32, device=device))
            pv_features_list.append(torch.tensor(graph["pv_features"], dtype=torch.float32, device=device) if graph["pv_features"] else torch.zeros((0, self.node_sizes["pv"]), dtype=torch.float32, device=device))
            external_features_list.append(torch.tensor(graph["external_features"], dtype=torch.float32, device=device) if graph["external_features"] else torch.zeros((0, self.node_sizes["external"]), dtype=torch.float32, device=device))
            action_node_indices_list.append(torch.tensor(graph["action_node_indices"], dtype=torch.long, device=device) + node_offset)
            ev_indexes_list.append(torch.tensor(graph["ev_indexes"], dtype=torch.long, device=device) + node_offset)
            cs_indexes_list.append(torch.tensor(graph["cs_indexes"], dtype=torch.long, device=device) + node_offset)
            bes_indexes_list.append(torch.tensor(graph["bes_indexes"], dtype=torch.long, device=device) + node_offset)
            pv_indexes_list.append(torch.tensor(graph["pv_indexes"], dtype=torch.long, device=device) + node_offset)
            external_indexes_list.append(torch.tensor(graph["external_indexes"], dtype=torch.long, device=device) + node_offset)
            if graph["edge_index"]:
                edge_index = torch.tensor(graph["edge_index"], dtype=torch.long, device=device).t().contiguous()
                edge_index_list.append(edge_index + node_offset)
            node_offset += node_count

        return {
            "node_type_ids": torch.cat(node_type_ids_list, dim=0),
            "edge_index": (
                torch.cat(edge_index_list, dim=1)
                if edge_index_list
                else torch.zeros((2, 0), dtype=torch.long, device=device)
            ),
            "batch": torch.cat(batch_list, dim=0),
            "action_node_indices": torch.cat(action_node_indices_list, dim=0),
            "ev_features": torch.cat(ev_features_list, dim=0),
            "cs_features": torch.cat(cs_features_list, dim=0),
            "bes_features": torch.cat(bes_features_list, dim=0),
            "pv_features": torch.cat(pv_features_list, dim=0),
            "external_features": torch.cat(external_features_list, dim=0),
            "ev_indexes": torch.cat(ev_indexes_list, dim=0) if ev_indexes_list else torch.zeros((0,), dtype=torch.long, device=device),
            "cs_indexes": torch.cat(cs_indexes_list, dim=0),
            "bes_indexes": torch.cat(bes_indexes_list, dim=0),
            "pv_indexes": torch.cat(pv_indexes_list, dim=0),
            "external_indexes": torch.cat(external_indexes_list, dim=0),
        }

    @staticmethod
    def _map_node_actions_to_env_action(
        park_graph: Dict[str, Any],
        node_action: torch.Tensor,
    ) -> Dict[str, Any]:
        return map_node_actions_to_env_action(park_graph, node_action)


__all__ = [
    "TypeSpecificEmbedding",
    "ActorActionGNN",
    "CriticActionGNN",
    "CriticNetwork",
    "LocalGNNCSACConfig",
    "LocalGNNCSACAgent",
]
