from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

from agent.central_state import get_central_node_sizes
from agent.replaybuffer import ReplayBuffer


LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0
EPSILON = 1e-6
NODE_TYPE_NAMES = ("cs", "bes", "pv", "external", "ev")
NODE_TYPE_TO_ID = {name: idx for idx, name in enumerate(NODE_TYPE_NAMES)}
FEATURE_DIM = 64
HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 256
BACKBONE_LR = 3e-4
HEAD_LR = 3e-4


def weights_init_(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
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
        feature_dim = next(iter(self.embeddings.values())).out_features
        embedded = torch.zeros(
            (total_nodes, feature_dim),
            dtype=torch.float32,
            device=graph_batch["node_type_ids"].device,
        )
        for node_type_name in NODE_TYPE_NAMES:
            indexes = graph_batch[f"{node_type_name}_indexes"]
            features = graph_batch[f"{node_type_name}_features"]
            if indexes.numel() > 0:
                embedded[indexes] = self.embeddings[node_type_name](features)
        return F.relu(embedded)


class CentralGNNActor(nn.Module):
    def __init__(self, node_sizes: Dict[str, int], feature_dim: int, decouple_actor_output_heads: bool) -> None:
        super().__init__()
        self.decouple_actor_output_heads = decouple_actor_output_heads
        self.node_embedding = TypeSpecificEmbedding(node_sizes=node_sizes, feature_dim=feature_dim)
        self.gcn_conv = GCNConv(feature_dim, HIDDEN_DIM_1)
        self.gcn_layers = nn.ModuleList([GCNConv(HIDDEN_DIM_1, HIDDEN_DIM_2)])
        if self.decouple_actor_output_heads:
            self.mean_heads = nn.ModuleDict({"bes": nn.Linear(HIDDEN_DIM_2, 1), "ev": nn.Linear(HIDDEN_DIM_2, 1)})
            self.log_std_heads = nn.ModuleDict({"bes": nn.Linear(HIDDEN_DIM_2, 1), "ev": nn.Linear(HIDDEN_DIM_2, 1)})
        else:
            self.mean_linear = nn.Linear(HIDDEN_DIM_2, 1)
            self.log_std_linear = nn.Linear(HIDDEN_DIM_2, 1)
        self.apply(weights_init_)

    def _encode(self, graph_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.node_embedding(graph_batch)
        x = F.relu(self.gcn_conv(x, graph_batch["edge_index"]))
        for layer in self.gcn_layers:
            x = F.relu(layer(x, graph_batch["edge_index"]))
        return x

    def forward(self, graph_batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(graph_batch)
        if self.decouple_actor_output_heads:
            mean = torch.zeros((x.shape[0],), dtype=torch.float32, device=x.device)
            log_std = torch.zeros((x.shape[0],), dtype=torch.float32, device=x.device)
            for node_type_name in ("bes", "ev"):
                indexes = graph_batch[f"{node_type_name}_indexes"]
                if indexes.numel() == 0:
                    continue
                mean_by_type = self.mean_heads[node_type_name](x).squeeze(-1)
                log_std_by_type = self.log_std_heads[node_type_name](x).squeeze(-1)
                mean[indexes] = mean_by_type[indexes]
                log_std[indexes] = log_std_by_type[indexes]
        else:
            mean = self.mean_linear(x).squeeze(-1)
            log_std = self.log_std_linear(x).squeeze(-1)
        return mean, torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

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
        normal = Normal(action_mean, action_log_std.exp())
        raw_action = action_mean if deterministic else normal.rsample()
        squashed_action = torch.tanh(raw_action)

        full_action = torch.zeros_like(mean)
        full_action[action_node_indices] = squashed_action
        log_prob = normal.log_prob(raw_action) - torch.log(1 - squashed_action.pow(2) + EPSILON)
        batch_size = int(graph_batch["batch"].max().item()) + 1
        pooled_log_prob = global_add_pool(log_prob.unsqueeze(-1), action_batches, size=batch_size).squeeze(-1)

        full_mean = torch.zeros_like(mean)
        full_mean[action_node_indices] = torch.tanh(action_mean)
        return full_action, pooled_log_prob, full_mean

    def actor_head_parameters(self) -> list[nn.Parameter]:
        if self.decouple_actor_output_heads:
            return list(self.mean_heads.parameters()) + list(self.log_std_heads.parameters())
        return list(self.mean_linear.parameters()) + list(self.log_std_linear.parameters())


class CentralGNNCritic(nn.Module):
    def __init__(self, node_sizes: Dict[str, int], feature_dim: int) -> None:
        super().__init__()
        self.node_embedding = TypeSpecificEmbedding(node_sizes=node_sizes, feature_dim=feature_dim)
        self.gcn_conv = GCNConv(feature_dim + 1, HIDDEN_DIM_1)
        self.gcn_layers = nn.ModuleList([GCNConv(HIDDEN_DIM_1, HIDDEN_DIM_2)])
        self.l1 = nn.Linear(HIDDEN_DIM_2, 64)
        self.l2 = nn.Linear(64, 1)
        self.apply(weights_init_)

    def forward(self, graph_batch: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        x = self.node_embedding(graph_batch)
        masked_action = torch.zeros_like(action)
        masked_action[graph_batch["action_node_indices"]] = action[graph_batch["action_node_indices"]]
        x = torch.cat([x, masked_action.reshape(-1, 1)], dim=1)
        x = F.relu(self.gcn_conv(x, graph_batch["edge_index"]))
        for layer in self.gcn_layers:
            x = F.relu(layer(x, graph_batch["edge_index"]))
        pooled = global_mean_pool(x, graph_batch["batch"])
        return self.l2(F.relu(self.l1(pooled))).squeeze(-1)


class CentralGNNCriticNetwork(nn.Module):
    def __init__(self, node_sizes: Dict[str, int], feature_dim: int) -> None:
        super().__init__()
        self.q1 = CentralGNNCritic(node_sizes=node_sizes, feature_dim=feature_dim)
        self.q2 = CentralGNNCritic(node_sizes=node_sizes, feature_dim=feature_dim)

    def forward(self, graph_batch: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(graph_batch, action), self.q2(graph_batch, action)


@dataclass
class CentralGNNCSACConfig:
    algorithm_variant: str
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
    feature_dim: int = FEATURE_DIM
    decouple_actor_output_heads: bool = True
    actor_backbone_lr: float = BACKBONE_LR
    actor_head_lr: float = HEAD_LR
    critic_backbone_lr: float = BACKBONE_LR
    critic_head_lr: float = HEAD_LR


class CentralGNNCSACAgent:
    def __init__(self, config: CentralGNNCSACConfig) -> None:
        self.config = config
        self.device = torch.device(config.update_device)
        self.act_device = torch.device(config.act_device)
        torch.manual_seed(config.seed)
        self.node_sizes = get_central_node_sizes(config.privacy_mode)

        self.actor = CentralGNNActor(self.node_sizes, config.feature_dim, config.decouple_actor_output_heads).to(self.device)
        self.critic = CentralGNNCriticNetwork(self.node_sizes, config.feature_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.cost_critic = CentralGNNCriticNetwork(self.node_sizes, config.feature_dim).to(self.device)
        self.cost_critic_target = copy.deepcopy(self.cost_critic).to(self.device)
        self.actor_inference = self._build_inference_actor()

        self.actor_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.node_embedding.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.gcn_conv.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.gcn_layers.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.actor_head_parameters(), "lr": config.actor_head_lr},
            ]
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_backbone_lr)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config.critic_backbone_lr)
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
        graph: Dict[str, Any],
        deterministic: bool = False,
        return_node_action: bool = False,
    ) -> Dict[str, Any] | Tuple[Dict[str, Any], torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.inference_mode():
            graph_batch = self._single_graph_to_batch(graph, device=self.act_device)
            node_action, _, node_mean = self.actor_inference.sample(graph_batch, deterministic=deterministic)
            chosen_action = node_mean if deterministic else node_action
            env_action = self._node_action_to_env_action(graph, chosen_action)
            park_node_actions = self._node_action_to_park_node_actions(graph, chosen_action)
            raw_node_action = chosen_action.detach().cpu()
            if return_node_action:
                return env_action, raw_node_action, park_node_actions
            return env_action

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
        action_batch = torch.cat([transition.action.to(self.device).reshape(-1) for transition in batch], dim=0)
        reward = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device)
        cost = torch.tensor([transition.cost for transition in batch], dtype=torch.float32, device=self.device)
        done = torch.tensor([transition.done for transition in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_batch)
            q1_next, q2_next = self.critic_target(next_batch, next_action)
            qcf1_next, qcf2_next = self.cost_critic_target(next_batch, next_action)
            next_q_value = reward + (1.0 - done) * self.config.gamma * (
                torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_pi
            )
            next_cost_value = cost + (1.0 - done) * self.config.gamma * torch.min(qcf1_next, qcf2_next)

        q1, q2 = self.critic(obs_batch, action_batch)
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        qcf1, qcf2 = self.cost_critic(obs_batch, action_batch)
        cost_critic_loss = F.mse_loss(qcf1, next_cost_value) + F.mse_loss(qcf2, next_cost_value)
        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        self.cost_critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(obs_batch)
        q1_pi, q2_pi = self.critic(obs_batch, pi)
        qcf1_pi, qcf2_pi = self.cost_critic(obs_batch, pi)
        mean_qcf_pi = 0.5 * (qcf1_pi + qcf2_pi)
        actor_loss = (
            self.alpha.detach() * log_pi
            - torch.min(q1_pi, q2_pi)
            + self.lambda_value.detach() * mean_qcf_pi
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        target_entropy = self._compute_batch_target_entropy(obs_batch)
        alpha_loss = -(self.log_alpha * (log_pi + target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        lambda_loss = (self.log_lambda * (float(self.config.d) - mean_qcf_pi.detach())).mean()
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()

        self.soft_update_targets()
        self._sync_inference_actor()
        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "cost_critic_loss": float(cost_critic_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "lambda_loss": float(lambda_loss.detach().cpu().item()),
            "lambda_value": float(self.lambda_value.detach().cpu().item()),
            "mean_qcf_pi": float(mean_qcf_pi.detach().mean().cpu().item()),
        }

    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau)
        for target_param, param in zip(self.cost_critic_target.parameters(), self.cost_critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau)

    def export_checkpoint(self, park_type: str, episode: int) -> Dict[str, Any]:
        return {
            "checkpoint_format": "central_gnn_csac_v1",
            "park_type": park_type,
            "episode": episode,
            "agent_config": asdict(self.config),
            "state_spec": {"node_sizes": dict(self.node_sizes)},
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

    def export_evaluation_checkpoint(self, park_type: str, episode: int) -> Dict[str, Any]:
        return {
            "checkpoint_format": "central_gnn_csac_v1",
            "export_kind": "evaluation_only",
            "park_type": park_type,
            "episode": episode,
            "agent_config": {
                "algorithm_variant": self.config.algorithm_variant,
                "privacy_mode": self.config.privacy_mode,
                "feature_dim": self.config.feature_dim,
                "decouple_actor_output_heads": self.config.decouple_actor_output_heads,
                "d": self.config.d,
            },
            "state_spec": {"node_sizes": dict(self.node_sizes)},
            "models": {
                "actor": {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()},
                "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
                "cost_critic": {k: v.detach().cpu().clone() for k, v in self.cost_critic.state_dict().items()},
            },
        }

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.actor.load_state_dict(checkpoint["models"]["actor"])
        self.critic.load_state_dict(checkpoint["models"]["critic"])
        self.critic_target.load_state_dict(checkpoint["models"].get("critic_target", checkpoint["models"]["critic"]))
        self.cost_critic.load_state_dict(checkpoint["models"]["cost_critic"])
        self.cost_critic_target.load_state_dict(
            checkpoint["models"].get("cost_critic_target", checkpoint["models"]["cost_critic"])
        )
        if "optimizers" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["optimizers"]["actor"])
            self.critic_optimizer.load_state_dict(checkpoint["optimizers"]["critic"])
            self.cost_critic_optimizer.load_state_dict(checkpoint["optimizers"]["cost_critic"])
            self.alpha_optimizer.load_state_dict(checkpoint["optimizers"]["alpha"])
            self.lambda_optimizer.load_state_dict(checkpoint["optimizers"]["lambda"])
            self.log_alpha.data.copy_(torch.tensor(checkpoint["temperature"]["log_alpha"], device=self.device))
            self.log_lambda.data.copy_(torch.tensor(checkpoint["temperature"]["log_lambda"], device=self.device))
            self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        self._sync_inference_actor()

    def _build_inference_actor(self) -> CentralGNNActor:
        return CentralGNNActor(
            node_sizes=self.node_sizes,
            feature_dim=self.config.feature_dim,
            decouple_actor_output_heads=self.config.decouple_actor_output_heads,
        ).to(self.act_device)

    def _sync_inference_actor(self) -> None:
        self.actor_inference.load_state_dict(
            {key: value.detach().to(self.act_device) for key, value in self.actor.state_dict().items()}
        )
        self.actor_inference.eval()

    def _single_graph_to_batch(self, graph: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
        node_count = len(graph["node_types"])
        edge_index = (
            torch.tensor(graph["edge_index"], dtype=torch.long, device=device).t().contiguous()
            if graph["edge_index"]
            else torch.zeros((2, 0), dtype=torch.long, device=device)
        )
        return {
            "node_type_ids": torch.tensor([NODE_TYPE_TO_ID[name] for name in graph["node_types"]], dtype=torch.long, device=device),
            "edge_index": edge_index,
            "batch": torch.zeros((node_count,), dtype=torch.long, device=device),
            "action_node_indices": torch.tensor(graph["bes_indexes"] + graph["ev_indexes"], dtype=torch.long, device=device),
            "ev_features": self._features_tensor(graph, "ev", device),
            "cs_features": self._features_tensor(graph, "cs", device),
            "bes_features": self._features_tensor(graph, "bes", device),
            "pv_features": self._features_tensor(graph, "pv", device),
            "external_features": self._features_tensor(graph, "external", device),
            "ev_indexes": torch.tensor(graph["ev_indexes"], dtype=torch.long, device=device),
            "cs_indexes": torch.tensor(graph["cs_indexes"], dtype=torch.long, device=device),
            "bes_indexes": torch.tensor(graph["bes_indexes"], dtype=torch.long, device=device),
            "pv_indexes": torch.tensor(graph["pv_indexes"], dtype=torch.long, device=device),
            "external_indexes": torch.tensor(graph["external_indexes"], dtype=torch.long, device=device),
        }

    def _batch_graphs(self, graphs: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
        batches = []
        offset = 0
        for graph_id, graph in enumerate(graphs):
            batch = self._single_graph_to_batch(graph, device=device)
            node_count = len(graph["node_types"])
            batch["edge_index"] = batch["edge_index"] + offset if batch["edge_index"].numel() > 0 else batch["edge_index"]
            batch["batch"] = torch.full((node_count,), graph_id, dtype=torch.long, device=device)
            for key in ("action_node_indices", "ev_indexes", "cs_indexes", "bes_indexes", "pv_indexes", "external_indexes"):
                batch[key] = batch[key] + offset
            batches.append(batch)
            offset += node_count
        return {
            "node_type_ids": torch.cat([batch["node_type_ids"] for batch in batches], dim=0),
            "edge_index": torch.cat([batch["edge_index"] for batch in batches], dim=1),
            "batch": torch.cat([batch["batch"] for batch in batches], dim=0),
            "action_node_indices": torch.cat([batch["action_node_indices"] for batch in batches], dim=0),
            "ev_features": torch.cat([batch["ev_features"] for batch in batches], dim=0),
            "cs_features": torch.cat([batch["cs_features"] for batch in batches], dim=0),
            "bes_features": torch.cat([batch["bes_features"] for batch in batches], dim=0),
            "pv_features": torch.cat([batch["pv_features"] for batch in batches], dim=0),
            "external_features": torch.cat([batch["external_features"] for batch in batches], dim=0),
            "ev_indexes": torch.cat([batch["ev_indexes"] for batch in batches], dim=0),
            "cs_indexes": torch.cat([batch["cs_indexes"] for batch in batches], dim=0),
            "bes_indexes": torch.cat([batch["bes_indexes"] for batch in batches], dim=0),
            "pv_indexes": torch.cat([batch["pv_indexes"] for batch in batches], dim=0),
            "external_indexes": torch.cat([batch["external_indexes"] for batch in batches], dim=0),
        }

    def _features_tensor(self, graph: Dict[str, Any], node_type: str, device: torch.device) -> torch.Tensor:
        features = graph[f"{node_type}_features"]
        if features:
            return torch.tensor(features, dtype=torch.float32, device=device)
        return torch.zeros((0, self.node_sizes[node_type]), dtype=torch.float32, device=device)

    @staticmethod
    def _compute_batch_target_entropy(graph_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        action_batches = graph_batch["batch"][graph_batch["action_node_indices"]]
        batch_size = int(graph_batch["batch"].max().item()) + 1
        counts = global_add_pool(
            torch.ones((action_batches.shape[0], 1), dtype=torch.float32, device=action_batches.device),
            action_batches,
            size=batch_size,
        ).squeeze(-1)
        return -counts

    @staticmethod
    def _node_action_to_env_action(graph: Dict[str, Any], node_action: torch.Tensor) -> Dict[str, Any]:
        env_action = {"parks": {}}
        ev_mapping = graph.get("ev_node_to_id_and_park", {})
        for park_type in graph["park_order"]:
            slice_info = graph["park_action_slices"][park_type]
            park_action = {"bes": 0.0, "ev": {}}
            if slice_info["bes_index"] is not None:
                park_action["bes"] = float(node_action[int(slice_info["bes_index"])].detach().cpu().item())
            for ev_index in slice_info["ev_indexes"]:
                _, ev_id = ev_mapping.get(int(ev_index), (park_type, graph["node_names"][int(ev_index)]))
                park_action["ev"][ev_id] = float(node_action[int(ev_index)].detach().cpu().item())
            env_action["parks"][park_type] = park_action
        return env_action

    @staticmethod
    def _node_action_to_park_node_actions(graph: Dict[str, Any], node_action: torch.Tensor) -> Dict[str, torch.Tensor]:
        park_actions: Dict[str, torch.Tensor] = {}
        for park_type in graph["park_order"]:
            slice_info = graph["park_action_slices"][park_type]
            park_node_action = torch.zeros((int(slice_info["local_node_count"]),), dtype=torch.float32)
            if slice_info["bes_index"] is not None and slice_info["local_bes_index"] is not None:
                park_node_action[int(slice_info["local_bes_index"])] = node_action[int(slice_info["bes_index"])].detach().cpu()
            for global_ev_index, local_ev_index in zip(slice_info["ev_indexes"], slice_info["local_ev_indexes"]):
                park_node_action[int(local_ev_index)] = node_action[int(global_ev_index)].detach().cpu()
            park_actions[park_type] = park_node_action
        return park_actions


__all__ = ["CentralGNNCSACConfig", "CentralGNNCSACAgent"]
