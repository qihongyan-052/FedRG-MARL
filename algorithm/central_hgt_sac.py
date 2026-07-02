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

from agent.central_state import get_central_node_sizes
from agent.replaybuffer import ReplayBuffer


LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0
EPSILON = 1e-6
NODE_TYPES = ("cs", "bes", "pv", "external", "ev")
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
FEATURE_DIM = 64
ACTOR_HIDDEN_DIM_1 = 128
ACTOR_HIDDEN_DIM_2 = 256
CRITIC_HIDDEN_DIM_1 = 128
CRITIC_HIDDEN_DIM_2 = 256
NUM_HEADS = 2
BACKBONE_LR = 3e-4
HEAD_LR = 3e-4


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
        return {
            node_type: F.relu(self.embeddings[node_type](x_dict[node_type]))
            for node_type in NODE_TYPES
        }


class CentralHGTActor(nn.Module):
    def __init__(self, node_sizes: Dict[str, int], feature_dim: int, decouple_actor_output_heads: bool) -> None:
        super().__init__()
        self.decouple_actor_output_heads = decouple_actor_output_heads
        self.node_embedding = HeteroTypeEmbedding(node_sizes=node_sizes, feature_dim=feature_dim)
        self.hgt_conv1 = HGTConv(
            in_channels={node_type: feature_dim for node_type in NODE_TYPES},
            out_channels=ACTOR_HIDDEN_DIM_1,
            metadata=METADATA,
            heads=NUM_HEADS,
        )
        self.norm1 = nn.ModuleDict({node_type: nn.LayerNorm(ACTOR_HIDDEN_DIM_1) for node_type in NODE_TYPES})
        self.hgt_conv2 = HGTConv(
            in_channels={node_type: ACTOR_HIDDEN_DIM_1 for node_type in NODE_TYPES},
            out_channels=ACTOR_HIDDEN_DIM_2,
            metadata=METADATA,
            heads=NUM_HEADS,
        )
        self.norm2 = nn.ModuleDict({node_type: nn.LayerNorm(ACTOR_HIDDEN_DIM_2) for node_type in NODE_TYPES})
        if self.decouple_actor_output_heads:
            self.mean_heads = nn.ModuleDict({"bes": nn.Linear(ACTOR_HIDDEN_DIM_2, 1), "ev": nn.Linear(ACTOR_HIDDEN_DIM_2, 1)})
            self.log_std_heads = nn.ModuleDict({"bes": nn.Linear(ACTOR_HIDDEN_DIM_2, 1), "ev": nn.Linear(ACTOR_HIDDEN_DIM_2, 1)})
        else:
            self.mean_head = nn.Linear(ACTOR_HIDDEN_DIM_2, 1)
            self.log_std_head = nn.Linear(ACTOR_HIDDEN_DIM_2, 1)
        self.apply(weights_init_)

    def _encode(self, batch: Batch) -> Dict[str, torch.Tensor]:
        x_dict = {node_type: batch[node_type].x for node_type in NODE_TYPES}
        edge_index_dict = {edge_type: batch[edge_type].edge_index for edge_type in EDGE_TYPES}
        x_dict = self.node_embedding(x_dict)
        x_dict = {
            key: F.relu(self.norm1[key](value))
            for key, value in self.hgt_conv1(x_dict, edge_index_dict).items()
        }
        return {
            key: F.relu(self.norm2[key](value))
            for key, value in self.hgt_conv2(x_dict, edge_index_dict).items()
        }

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
            log_prob = normal.log_prob(raw_action) - torch.log(1 - squashed_action.pow(2) + EPSILON)
            pooled = global_add_pool(log_prob.unsqueeze(-1), batch[node_type].batch, size=num_graphs).squeeze(-1)
            log_prob_by_graph = pooled if log_prob_by_graph is None else log_prob_by_graph + pooled
        if log_prob_by_graph is None:
            log_prob_by_graph = torch.zeros((num_graphs,), dtype=torch.float32, device=batch["external"].x.device)
        return action_dict, log_prob_by_graph, mean_action_dict


class CentralHGTCritic(nn.Module):
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
            out_channels=CRITIC_HIDDEN_DIM_1,
            metadata=METADATA,
            heads=NUM_HEADS,
        )
        self.norm1 = nn.ModuleDict({node_type: nn.LayerNorm(CRITIC_HIDDEN_DIM_1) for node_type in NODE_TYPES})
        self.hgt_conv2 = HGTConv(
            in_channels={node_type: CRITIC_HIDDEN_DIM_1 for node_type in NODE_TYPES},
            out_channels=CRITIC_HIDDEN_DIM_2,
            metadata=METADATA,
            heads=NUM_HEADS,
        )
        self.norm2 = nn.ModuleDict({node_type: nn.LayerNorm(CRITIC_HIDDEN_DIM_2) for node_type in NODE_TYPES})
        self.l1 = nn.Linear(CRITIC_HIDDEN_DIM_2, 64)
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
        pooled_sum = torch.zeros((batch_size, CRITIC_HIDDEN_DIM_2), dtype=torch.float32, device=batch["external"].x.device)
        present_counts = torch.zeros((batch_size, 1), dtype=torch.float32, device=batch["external"].x.device)
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
        return self.l2(F.relu(self.l1(pooled_graph))).squeeze(-1)


class CentralHGTCriticNetwork(nn.Module):
    def __init__(self, node_sizes: Dict[str, int], feature_dim: int) -> None:
        super().__init__()
        self.q1 = CentralHGTCritic(node_sizes=node_sizes, feature_dim=feature_dim)
        self.q2 = CentralHGTCritic(node_sizes=node_sizes, feature_dim=feature_dim)

    def forward(self, batch: Batch, action_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(batch, action_dict), self.q2(batch, action_dict)


@dataclass
class CentralHGTSACConfig:
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
    feature_dim: int = FEATURE_DIM
    decouple_actor_output_heads: bool = True
    actor_backbone_lr: float = BACKBONE_LR
    actor_head_lr: float = HEAD_LR
    critic_backbone_lr: float = BACKBONE_LR
    critic_head_lr: float = HEAD_LR


class CentralHGTSACAgent:
    def __init__(self, config: CentralHGTSACConfig) -> None:
        self.config = config
        self.device = torch.device(config.update_device)
        self.act_device = torch.device(config.act_device)
        torch.manual_seed(config.seed)
        self.node_sizes = get_central_node_sizes(config.privacy_mode)

        self.actor = CentralHGTActor(
            node_sizes=self.node_sizes,
            feature_dim=config.feature_dim,
            decouple_actor_output_heads=config.decouple_actor_output_heads,
        ).to(self.device)
        self.critic = CentralHGTCriticNetwork(node_sizes=self.node_sizes, feature_dim=config.feature_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.actor_inference = self._build_inference_actor()

        self.actor_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.node_embedding.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.hgt_conv1.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.norm1.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.hgt_conv2.parameters(), "lr": config.actor_backbone_lr},
                {"params": self.actor.norm2.parameters(), "lr": config.actor_backbone_lr},
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
        graph: Dict[str, Any],
        deterministic: bool = False,
        return_node_action: bool = False,
    ) -> Dict[str, Any] | Tuple[Dict[str, Any], torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.inference_mode():
            batch = self._batch_graphs([graph], device=self.act_device)
            action_dict, _, mean_action_dict = self.actor_inference.sample(batch, deterministic=deterministic)
            chosen = mean_action_dict if deterministic else action_dict
            node_action = self._hetero_action_to_node_action(graph, chosen)
            env_action = self._hetero_action_to_env_action(graph, chosen)
            park_node_actions = self._hetero_action_to_park_node_actions(graph, chosen)
            if return_node_action:
                return env_action, node_action, park_node_actions
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
                "lambda_value": 0.0,
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
        done = torch.tensor([transition.done for transition in batch], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_batch)
            q1_next, q2_next = self.critic_target(next_batch, next_action)
            next_q_value = reward + (1.0 - done) * self.config.gamma * (
                torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_pi
            )

        q1, q2 = self.critic(obs_batch, action_batch)
        critic_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(obs_batch)
        q1_pi, q2_pi = self.critic(obs_batch, pi)
        actor_loss = ((self.alpha.detach() * log_pi) - torch.min(q1_pi, q2_pi)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        target_entropy = self._compute_batch_target_entropy(obs_batch)
        alpha_loss = -(self.log_alpha * (log_pi + target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.soft_update_targets()
        self._sync_inference_actor()
        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "cost_critic_loss": 0.0,
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "lambda_loss": 0.0,
            "lambda_value": 0.0,
            "mean_qcf_pi": 0.0,
        }

    def soft_update_targets(self) -> None:
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau)

    def export_checkpoint(self, park_type: str, episode: int) -> Dict[str, Any]:
        return {
            "checkpoint_format": "central_hgt_sac_v1",
            "park_type": park_type,
            "episode": episode,
            "agent_config": asdict(self.config),
            "state_spec": {"node_sizes": dict(self.node_sizes)},
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
            "temperature": {"log_alpha": float(self.log_alpha.detach().cpu().item())},
            "replay_buffer": self.replay_buffer.state_dict(),
        }

    def export_evaluation_checkpoint(self, park_type: str, episode: int) -> Dict[str, Any]:
        return {
            "checkpoint_format": "central_hgt_sac_v1",
            "export_kind": "evaluation_only",
            "park_type": park_type,
            "episode": episode,
            "agent_config": {
                "algorithm_variant": self.config.algorithm_variant,
                "privacy_mode": self.config.privacy_mode,
                "feature_dim": self.config.feature_dim,
                "decouple_actor_output_heads": self.config.decouple_actor_output_heads,
            },
            "state_spec": {"node_sizes": dict(self.node_sizes)},
            "models": {
                "actor": {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()},
                "critic": {k: v.detach().cpu().clone() for k, v in self.critic.state_dict().items()},
            },
        }

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.actor.load_state_dict(checkpoint["models"]["actor"])
        self.critic.load_state_dict(checkpoint["models"]["critic"])
        target_state = checkpoint["models"].get("critic_target", checkpoint["models"]["critic"])
        self.critic_target.load_state_dict(target_state)
        self.actor_optimizer.load_state_dict(checkpoint["optimizers"]["actor"])
        self.critic_optimizer.load_state_dict(checkpoint["optimizers"]["critic"])
        self.alpha_optimizer.load_state_dict(checkpoint["optimizers"]["alpha"])
        self.log_alpha.data.copy_(torch.tensor(checkpoint["temperature"]["log_alpha"], device=self.device))
        self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        self._sync_inference_actor()

    def _build_inference_actor(self) -> CentralHGTActor:
        return CentralHGTActor(
            node_sizes=self.node_sizes,
            feature_dim=self.config.feature_dim,
            decouple_actor_output_heads=self.config.decouple_actor_output_heads,
        ).to(self.act_device)

    def _sync_inference_actor(self) -> None:
        self.actor_inference.load_state_dict(
            {key: value.detach().to(self.act_device) for key, value in self.actor.state_dict().items()}
        )
        self.actor_inference.eval()

    def _graph_to_heterodata(self, graph: Dict[str, Any], device: torch.device) -> HeteroData:
        data = HeteroData()
        for node_type in NODE_TYPES:
            features = graph[f"{node_type}_features"]
            if features:
                data[node_type].x = torch.tensor(features, dtype=torch.float32, device=device)
            else:
                data[node_type].x = torch.zeros((0, self.node_sizes[node_type]), dtype=torch.float32, device=device)

        global_to_local = {}
        for node_type in NODE_TYPES:
            for local_idx, global_idx in enumerate(graph[f"{node_type}_indexes"]):
                global_to_local[int(global_idx)] = (node_type, local_idx)

        relation_edges: Dict[Tuple[str, str, str], List[List[int]]] = {
            edge_type: [[], []] for edge_type in EDGE_TYPES
        }
        for src_global, dst_global in graph["edge_index"]:
            src_type, src_local = global_to_local[int(src_global)]
            dst_type, dst_local = global_to_local[int(dst_global)]
            relation = self._edge_relation(src_type, dst_type)
            if relation is None:
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
        return Batch.from_data_list([self._graph_to_heterodata(graph, device=device) for graph in graphs])

    @staticmethod
    def _edge_relation(src_type: str, dst_type: str) -> Tuple[str, str, str] | None:
        if src_type == "ev" and dst_type == "cs":
            return ("ev", "to", "cs")
        if src_type == "cs" and dst_type == "ev":
            return ("cs", "rev_to_ev", "ev")
        if src_type == "bes" and dst_type == "cs":
            return ("bes", "to", "cs")
        if src_type == "cs" and dst_type == "bes":
            return ("cs", "rev_to_bes", "bes")
        if src_type == "pv" and dst_type == "cs":
            return ("pv", "to", "cs")
        if src_type == "cs" and dst_type == "pv":
            return ("cs", "rev_to_pv", "pv")
        if src_type == "external" and dst_type == "cs":
            return ("external", "to", "cs")
        if src_type == "cs" and dst_type == "external":
            return ("cs", "rev_to_external", "external")
        return None

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
            for bes_index in graph["bes_indexes"]:
                bes_actions.append(flat_action[int(bes_index)].reshape(1))
            for ev_index in graph["ev_indexes"]:
                ev_actions.append(flat_action[int(ev_index)].reshape(1))
        return {
            "bes": torch.cat(bes_actions, dim=0) if bes_actions else torch.zeros((0,), dtype=torch.float32, device=device),
            "ev": torch.cat(ev_actions, dim=0) if ev_actions else torch.zeros((0,), dtype=torch.float32, device=device),
        }

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
        for offset, bes_index in enumerate(graph["bes_indexes"]):
            node_action[int(bes_index)] = action_dict["bes"][offset].detach().cpu()
        for offset, ev_index in enumerate(graph["ev_indexes"]):
            node_action[int(ev_index)] = action_dict["ev"][offset].detach().cpu()
        return node_action

    @staticmethod
    def _hetero_action_to_env_action(graph: Dict[str, Any], action_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        env_action = {"parks": {}}
        bes_offset = 0
        ev_offset = 0
        ev_mapping = graph.get("ev_node_to_id_and_park", {})
        for park_type in graph["park_order"]:
            slice_info = graph["park_action_slices"][park_type]
            park_action = {"bes": 0.0, "ev": {}}
            if slice_info["bes_index"] is not None:
                park_action["bes"] = float(action_dict["bes"][bes_offset].detach().cpu().item())
                bes_offset += 1
            for ev_index in slice_info["ev_indexes"]:
                _, ev_id = ev_mapping.get(int(ev_index), (park_type, graph["node_names"][int(ev_index)]))
                park_action["ev"][ev_id] = float(
                    action_dict["ev"][ev_offset].detach().cpu().item()
                )
                ev_offset += 1
            env_action["parks"][park_type] = park_action
        return env_action

    @staticmethod
    def _hetero_action_to_park_node_actions(
        graph: Dict[str, Any],
        action_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        node_action = CentralHGTSACAgent._hetero_action_to_node_action(graph, action_dict)
        park_actions: Dict[str, torch.Tensor] = {}
        for park_type in graph["park_order"]:
            slice_info = graph["park_action_slices"][park_type]
            park_node_action = torch.zeros((int(slice_info["local_node_count"]),), dtype=torch.float32)
            if slice_info["bes_index"] is not None and slice_info["local_bes_index"] is not None:
                park_node_action[int(slice_info["local_bes_index"])] = node_action[int(slice_info["bes_index"])]
            for global_ev_index, local_ev_index in zip(slice_info["ev_indexes"], slice_info["local_ev_indexes"]):
                park_node_action[int(local_ev_index)] = node_action[int(global_ev_index)]
            park_actions[park_type] = park_node_action
        return park_actions
