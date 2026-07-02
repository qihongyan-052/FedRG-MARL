from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple
import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch_geometric.nn import global_add_pool, global_mean_pool

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
RELATION_TYPE_NAMES = (
    "bes_to_cs",
    "cs_to_bes",
    "pv_to_cs",
    "cs_to_pv",
    "external_to_cs",
    "cs_to_external",
    "ev_to_cs",
    "cs_to_ev",
)
RELATION_TYPE_TO_ID = {name: idx for idx, name in enumerate(RELATION_TYPE_NAMES)}


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


class SharedPrivateRelationalConv(nn.Module):
    """Relation-specific message passing with shared weights plus local low-rank adapters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adapter_rank: int = 8,
        use_relation_gated_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_relation_gated_fusion = use_relation_gated_fusion
        self.relation_names = RELATION_TYPE_NAMES
        self.shared = nn.ModuleDict(
            {name: nn.Linear(in_channels, out_channels, bias=False) for name in self.relation_names}
        )
        self.adapter_down = nn.ModuleDict(
            {name: nn.Linear(in_channels, adapter_rank, bias=False) for name in self.relation_names}
        )
        self.adapter_up = nn.ModuleDict(
            {name: nn.Linear(adapter_rank, out_channels, bias=False) for name in self.relation_names}
        )
        self.relation_logits = nn.Parameter(torch.zeros(len(self.relation_names), dtype=torch.float32))
        self.root = nn.Linear(in_channels, out_channels, bias=True)
        self.apply(weights_init_)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        relation_outs: List[torch.Tensor] = []
        relation_active: List[torch.Tensor] = []
        if edge_index.numel() > 0:
            src = edge_index[0]
            dst = edge_index[1]
            for relation_name, relation_id in RELATION_TYPE_TO_ID.items():
                relation_out = torch.zeros((x.shape[0], self.out_channels), dtype=x.dtype, device=x.device)
                relation_deg = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
                mask = edge_type_ids == relation_id
                if bool(mask.any()):
                    relation_src = src[mask]
                    relation_dst = dst[mask]
                    messages = self.shared[relation_name](x[relation_src])
                    messages = messages + self.adapter_up[relation_name](
                        self.adapter_down[relation_name](x[relation_src])
                    )
                    relation_out.index_add_(0, relation_dst, messages)
                    relation_deg.index_add_(
                        0,
                        relation_dst,
                        torch.ones((relation_dst.shape[0], 1), dtype=x.dtype, device=x.device),
                    )
                relation_outs.append(relation_out / relation_deg.clamp_min(1.0))
                relation_active.append(relation_deg.gt(0.0))
        if not self.use_relation_gated_fusion and relation_outs:
            stacked_outs = torch.stack(relation_outs, dim=1)
            active = torch.cat(relation_active, dim=1).to(dtype=x.dtype)
            equal_weight = active / active.sum(dim=1, keepdim=True).clamp_min(1.0)
            out = (stacked_outs * equal_weight.unsqueeze(-1)).sum(dim=1)
        elif not self.use_relation_gated_fusion:
            out = torch.zeros((x.shape[0], self.out_channels), dtype=x.dtype, device=x.device)
        elif relation_outs:
            stacked_outs = torch.stack(relation_outs, dim=1)
            active = torch.cat(relation_active, dim=1).to(dtype=x.dtype)
            gate = torch.softmax(self.relation_logits, dim=0).to(dtype=x.dtype, device=x.device).view(1, -1)
            gated = active * gate
            gated = gated / gated.sum(dim=1, keepdim=True).clamp_min(EPSILON)
            out = (stacked_outs * gated.unsqueeze(-1)).sum(dim=1)
        else:
            out = torch.zeros((x.shape[0], self.out_channels), dtype=x.dtype, device=x.device)
        return out + self.root(x)

    def shared_state_dict(self) -> Dict[str, torch.Tensor]:
        return {f"shared.{k}": v.detach().cpu().clone() for k, v in self.shared.state_dict().items()}

    def load_shared_state_dict(
        self,
        shared_state: Dict[str, torch.Tensor],
        prefix: str = "",
        relation_mix: Dict[str, float] | None = None,
        default_mix: float = 1.0,
    ) -> None:
        local_state = self.shared.state_dict()
        loaded_state = {}
        for key, local_value in local_state.items():
            state_key = f"{prefix}shared.{key}"
            if state_key in shared_state:
                global_value = shared_state[state_key].detach().to(
                    local_value.device,
                    dtype=local_value.dtype,
                )
                relation_name = key.split(".", 1)[0]
                mix_ratio = default_mix if relation_mix is None else relation_mix.get(relation_name, default_mix)
                mix_ratio = min(1.0, max(0.0, float(mix_ratio)))
                loaded_state[key] = (1.0 - mix_ratio) * local_value + mix_ratio * global_value
            else:
                loaded_state[key] = local_value
        self.shared.load_state_dict(loaded_state)


class ActorActionGNN(nn.Module):
    def __init__(
        self,
        node_sizes: Dict[str, int],
        feature_dim: int,
        gnn_hidden_dim: int,
        num_gcn_layers: int,
        decouple_actor_output_heads: bool = False,
        use_relation_gated_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.decouple_actor_output_heads = decouple_actor_output_heads
        self.node_embedding = TypeSpecificEmbedding(node_sizes=node_sizes, feature_dim=feature_dim)
        if num_gcn_layers != 2:
            raise ValueError("Current actor architecture is fixed to 2 GCN layers. Set actor_num_gcn_layers=2.")
        hidden_dim_1 = gnn_hidden_dim // 2
        hidden_dim_2 = gnn_hidden_dim

        self.gcn_conv = SharedPrivateRelationalConv(
            feature_dim,
            hidden_dim_1,
            use_relation_gated_fusion=use_relation_gated_fusion,
        )
        self.gcn_layers = nn.ModuleList(
            [
                SharedPrivateRelationalConv(
                    hidden_dim_1,
                    hidden_dim_2,
                    use_relation_gated_fusion=use_relation_gated_fusion,
                )
            ]
        )
        if self.decouple_actor_output_heads:
            self.mean_heads = nn.ModuleDict(
                {
                    "bes": nn.Linear(hidden_dim_2, 1),
                    "ev": nn.Linear(hidden_dim_2, 1),
                }
            )
            self.log_std_heads = nn.ModuleDict(
                {
                    "bes": nn.Linear(hidden_dim_2, 1),
                    "ev": nn.Linear(hidden_dim_2, 1),
                }
            )
        else:
            self.mean_linear = nn.Linear(hidden_dim_2, 1)
            self.log_std_linear = nn.Linear(hidden_dim_2, 1)

    def _encode(
        self,
        graph_batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = self.node_embedding(graph_batch)
        x = F.relu(self.gcn_conv(x, graph_batch["edge_index"], graph_batch["edge_type_ids"]))
        for layer in self.gcn_layers:
            x = F.relu(layer(x, graph_batch["edge_index"], graph_batch["edge_type_ids"]))
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
                mean_by_type = self.mean_heads[node_type_name](x).squeeze(-1)
                log_std_by_type = self.log_std_heads[node_type_name](x).squeeze(-1)
                mean[indexes] = mean_by_type[indexes]
                log_std[indexes] = log_std_by_type[indexes]
        else:
            mean = self.mean_linear(x).squeeze(-1)
            log_std = self.log_std_linear(x).squeeze(-1)
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
        batch_size = int(graph_batch["batch"].max().item()) + 1
        action_node_types = graph_batch["node_type_ids"][action_node_indices]
        ev_action_mask = action_node_types == NODE_TYPE_NAMES.index("ev")
        bes_action_mask = action_node_types == NODE_TYPE_NAMES.index("bes")
        pooled_log_prob = torch.zeros((batch_size,), dtype=log_prob.dtype, device=log_prob.device)
        if bool(ev_action_mask.any()):
            ev_batches = action_batches[ev_action_mask]
            ev_log_prob_sum = global_add_pool(
                log_prob[ev_action_mask].unsqueeze(-1),
                ev_batches,
                size=batch_size,
            ).squeeze(-1)
            ev_counts = global_add_pool(
                torch.ones((ev_batches.shape[0], 1), dtype=log_prob.dtype, device=log_prob.device),
                ev_batches,
                size=batch_size,
            ).squeeze(-1)
            pooled_log_prob = pooled_log_prob + ev_log_prob_sum / ev_counts.clamp_min(1.0)
        if bool(bes_action_mask.any()):
            bes_batches = action_batches[bes_action_mask]
            bes_log_prob = global_add_pool(
                log_prob[bes_action_mask].unsqueeze(-1),
                bes_batches,
                size=batch_size,
            ).squeeze(-1)
            pooled_log_prob = pooled_log_prob + bes_log_prob

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
        output_dim: int = 1,
        use_relation_gated_fusion: bool = True,
        use_typed_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.use_typed_pooling = use_typed_pooling
        self.node_embedding = TypeSpecificEmbedding(node_sizes=node_sizes, feature_dim=feature_dim)
        if num_gcn_layers != 2:
            raise ValueError("Current critic architecture is fixed to 2 GCN layers. Set critic_num_gcn_layers=2.")
        hidden_dim_1 = gnn_hidden_dim // 2
        hidden_dim_2 = gnn_hidden_dim

        self.gcn_conv = SharedPrivateRelationalConv(
            feature_dim + 1,
            hidden_dim_1,
            use_relation_gated_fusion=use_relation_gated_fusion,
        )
        self.gcn_layers = nn.ModuleList(
            [
                SharedPrivateRelationalConv(
                    hidden_dim_1,
                    hidden_dim_2,
                    use_relation_gated_fusion=use_relation_gated_fusion,
                )
            ]
        )
        pooled_dim = hidden_dim_2 * len(NODE_TYPE_NAMES) if self.use_typed_pooling else hidden_dim_2
        self.l1 = nn.Linear(pooled_dim, 64)
        self.l2 = nn.Linear(64, output_dim)
        self.apply(weights_init_)

    def forward(self, graph_batch: Dict[str, torch.Tensor], action: torch.Tensor) -> torch.Tensor:
        x = self.node_embedding(graph_batch)
        masked_action = torch.zeros_like(action)
        masked_action[graph_batch["action_node_indices"]] = action[graph_batch["action_node_indices"]]
        state_action = torch.cat([x, masked_action.reshape(-1, 1)], dim=1)
        x = F.relu(self.gcn_conv(state_action, graph_batch["edge_index"], graph_batch["edge_type_ids"]))
        for layer in self.gcn_layers:
            x = F.relu(layer(x, graph_batch["edge_index"], graph_batch["edge_type_ids"]))
        pooled = (
            self._role_preserving_pool(x, graph_batch)
            if self.use_typed_pooling
            else global_mean_pool(x, graph_batch["batch"])
        )
        x = F.relu(self.l1(pooled))
        return self.l2(x).squeeze(-1)

    @staticmethod
    def _role_preserving_pool(x: torch.Tensor, graph_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch = graph_batch["batch"]
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled_by_type = []
        for node_type_name in NODE_TYPE_NAMES:
            indexes = graph_batch[f"{node_type_name}_indexes"]
            if indexes.numel() == 0:
                pooled_by_type.append(torch.zeros((batch_size, x.shape[1]), dtype=x.dtype, device=x.device))
                continue
            type_batch = batch[indexes]
            summed = global_add_pool(x[indexes], type_batch, size=batch_size)
            counts = global_add_pool(
                torch.ones((indexes.shape[0], 1), dtype=x.dtype, device=x.device),
                type_batch,
                size=batch_size,
            ).clamp_min(1.0)
            pooled_by_type.append(summed / counts)
        return torch.cat(pooled_by_type, dim=1)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        node_sizes: Dict[str, int],
        feature_dim: int,
        gnn_hidden_dim: int,
        num_gcn_layers: int,
        use_relation_gated_fusion: bool = True,
        use_typed_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.q1 = CriticActionGNN(
            node_sizes=node_sizes,
            feature_dim=feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
            use_relation_gated_fusion=use_relation_gated_fusion,
            use_typed_pooling=use_typed_pooling,
        )
        self.q2 = CriticActionGNN(
            node_sizes=node_sizes,
            feature_dim=feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
            use_relation_gated_fusion=use_relation_gated_fusion,
            use_typed_pooling=use_typed_pooling,
        )

    def forward(self, graph_batch: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(graph_batch, action), self.q2(graph_batch, action)


class CostCriticNetwork(nn.Module):
    def __init__(
        self,
        node_sizes: Dict[str, int],
        feature_dim: int,
        gnn_hidden_dim: int,
        num_gcn_layers: int,
        use_relation_gated_fusion: bool = True,
        use_typed_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.q1 = CriticActionGNN(
            node_sizes=node_sizes,
            feature_dim=feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
            use_relation_gated_fusion=use_relation_gated_fusion,
            use_typed_pooling=use_typed_pooling,
        )
        self.q2 = CriticActionGNN(
            node_sizes=node_sizes,
            feature_dim=feature_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            num_gcn_layers=num_gcn_layers,
            use_relation_gated_fusion=use_relation_gated_fusion,
            use_typed_pooling=use_typed_pooling,
        )

    def forward(self, graph_batch: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(graph_batch, action), self.q2(graph_batch, action)


def compute_batch_target_entropy(graph_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    action_batches = graph_batch["batch"][graph_batch["action_node_indices"]]
    batch_size = int(graph_batch["batch"].max().item()) + 1
    action_node_types = graph_batch["node_type_ids"][graph_batch["action_node_indices"]]
    target_entropy = torch.zeros((batch_size,), dtype=torch.float32, device=action_batches.device)
    if bool((action_node_types == NODE_TYPE_NAMES.index("ev")).any()):
        target_entropy = target_entropy - 1.0
    if bool((action_node_types == NODE_TYPE_NAMES.index("bes")).any()):
        target_entropy = target_entropy - 1.0
    return target_entropy


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
class LocalSPRGNNCSACConfig:
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
    d_local: float | None = None
    d_regional: float | None = None
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
    relation_fed_mix: float = 0.5
    relation_fed_mix_by_relation: Dict[str, float] | None = None
    use_relation_gated_fusion: bool = True
    use_critic_typed_pooling: bool = True

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


class LocalSPRGNNCSACAgent:
    def __init__(self, config: LocalSPRGNNCSACConfig) -> None:
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
            use_relation_gated_fusion=config.use_relation_gated_fusion,
        ).to(self.device)
        self.critic = CriticNetwork(
            node_sizes=self.node_sizes,
            feature_dim=config.feature_dim,
            gnn_hidden_dim=config.gnn_hidden_dim,
            num_gcn_layers=config.critic_num_gcn_layers,
            use_relation_gated_fusion=config.use_relation_gated_fusion,
            use_typed_pooling=config.use_critic_typed_pooling,
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        self.cost_critic = CostCriticNetwork(
            node_sizes=self.node_sizes,
            feature_dim=config.feature_dim,
            gnn_hidden_dim=config.gnn_hidden_dim,
            num_gcn_layers=config.critic_num_gcn_layers,
            use_relation_gated_fusion=config.use_relation_gated_fusion,
            use_typed_pooling=config.use_critic_typed_pooling,
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
        self._actor_backbone_group_indices = [0, 1, 2]
        self._actor_head_group_indices = [3]
        self._critic_backbone_group_indices = [0, 1, 2, 5, 6, 7]
        self._critic_head_group_indices = [3, 4, 8, 9]

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

    @property
    def lambda_local_value(self) -> torch.Tensor:
        return self.lambda_value

    @property
    def lambda_regional_value(self) -> torch.Tensor:
        return self.lambda_value

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
            mean_qc = 0.5 * (qc1 + qc2)
            qc_value = self.lambda_value.detach() * mean_qc
            score = q_value - qc_value
            return float(score.detach().cpu().item())

    def get_actor_relation_fed_mask(self) -> Dict[str, bool]:
        mask = {relation_name: True for relation_name in RELATION_TYPE_NAMES}
        if self.config.park_type == "residential":
            mask["pv_to_cs"] = False
            mask["cs_to_pv"] = False
        return mask

    def make_actor_with_external_backbone(self, shared_backbone: Dict[str, torch.Tensor]) -> ActorActionGNN:
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
        self.replay_buffer.push(
            obs,
            action,
            reward,
            cost,
            next_obs,
            done,
        )

    def set_output_head_learning_rates(self, actor_head_lr: float, critic_head_lr: float) -> None:
        for group_idx in self._actor_head_group_indices:
            self.actor_optimizer.param_groups[group_idx]["lr"] = actor_head_lr
        for group_idx in self._critic_head_group_indices:
            self.critic_optimizer.param_groups[group_idx]["lr"] = critic_head_lr
            self.cost_critic_optimizer.param_groups[group_idx]["lr"] = critic_head_lr

    def set_backbone_learning_rates(self, actor_backbone_lr: float, critic_backbone_lr: float) -> None:
        for group_idx in self._actor_backbone_group_indices:
            self.actor_optimizer.param_groups[group_idx]["lr"] = actor_backbone_lr
        for group_idx in self._critic_backbone_group_indices:
            self.critic_optimizer.param_groups[group_idx]["lr"] = critic_backbone_lr
            self.cost_critic_optimizer.param_groups[group_idx]["lr"] = critic_backbone_lr

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
            score_local = self._evaluate_action_score(obs_batch, local_action, local_log_pi)
            candidate_actor = self.make_actor_with_external_backbone(external_backbone)
            candidate_action, candidate_log_pi, _ = candidate_actor.sample(obs_batch, deterministic=True)
            score_candidate = self._evaluate_action_score(obs_batch, candidate_action, candidate_log_pi)
            advantage = score_candidate - score_local
        return float(advantage.detach().cpu().item())

    def _evaluate_action_score(
        self,
        obs_batch: Dict[str, torch.Tensor],
        action: torch.Tensor,
        log_pi: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2 = self.critic(obs_batch, action)
        qc1, qc2 = self.cost_critic(obs_batch, action)
        q_value = torch.min(q1, q2)
        qc_value = 0.5 * (qc1 + qc2)
        return (q_value - self.lambda_value.detach() * qc_value - self.alpha.detach() * log_pi).mean()

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.config.batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "cost_critic_loss": 0.0,
                "alpha_loss": 0.0,
                "lambda_loss": 0.0,
                "lambda_value": float(self.lambda_value.detach().cpu().item()),
                "lambda_local_value": float(self.lambda_local_value.detach().cpu().item()),
                "lambda_regional_value": float(self.lambda_regional_value.detach().cpu().item()),
                "mean_qcf_pi": 0.0,
                "mean_local_qcf_pi": 0.0,
                "mean_regional_qcf_pi": 0.0,
            }

        batch = self.replay_buffer.sample(self.config.batch_size)
        obs_batch = [transition.obs for transition in batch]
        next_obs_batch = [transition.next_obs for transition in batch]
        action_batch = torch.cat(
            [transition.action.to(self.device).reshape(-1) for transition in batch],
            dim=0,
        )
        reward = torch.tensor([transition.reward for transition in batch], dtype=torch.float32, device=self.device)
        cost_targets = torch.tensor([transition.cost for transition in batch], dtype=torch.float32, device=self.device)
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
            next_qc_value = cost_targets + (1.0 - done) * self.config.gamma * mean_qcf_next_target

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

        lambda_loss_mean = (self.log_lambda * (float(self.config.d) - mean_qcf_pi.detach())).mean()
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
            "lambda_local_value": float(self.lambda_local_value.detach().cpu().item()),
            "lambda_regional_value": float(self.lambda_regional_value.detach().cpu().item()),
            "mean_qcf_pi": float(mean_qcf_pi.detach().mean().cpu().item()),
            "mean_local_qcf_pi": float(mean_qcf_pi.detach().mean().cpu().item()),
            "mean_regional_qcf_pi": 0.0,
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
                **self._export_relation_shared_state(self.actor),
            },
        }
        if self.enable_federation and self.config.federate_critic_backbone:
            shared_state["critic_backbone"] = {
                **{f"q1.{k}": v for k, v in self._export_relation_shared_state(self.critic.q1).items()},
                **{f"q2.{k}": v for k, v in self._export_relation_shared_state(self.critic.q2).items()},
            }
            shared_state["cost_critic_backbone"] = {
                **{f"q1.{k}": v for k, v in self._export_relation_shared_state(self.cost_critic.q1).items()},
                **{f"q2.{k}": v for k, v in self._export_relation_shared_state(self.cost_critic.q2).items()},
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
            "checkpoint_format": "sp_rgnn_csac_v1",
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
        temperature = checkpoint["temperature"]
        legacy_lambda = 0.5 * (
            float(temperature.get("log_lambda_local", temperature.get("log_lambda", 0.0)))
            + float(temperature.get("log_lambda_regional", temperature.get("log_lambda", 0.0)))
        )
        self.log_lambda.data.copy_(
            torch.tensor(temperature.get("log_lambda", legacy_lambda), device=self.device)
        )
        self.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
        self._sync_inference_actor()

    def _build_inference_actor(self) -> ActorActionGNN:
        return ActorActionGNN(
            node_sizes=self.node_sizes,
            feature_dim=self.config.feature_dim,
            gnn_hidden_dim=self.config.gnn_hidden_dim,
            num_gcn_layers=self.config.actor_num_gcn_layers,
            decouple_actor_output_heads=self.config.decouple_actor_output_heads,
            use_relation_gated_fusion=self.config.use_relation_gated_fusion,
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
        module.gcn_conv.load_shared_state_dict(
            shared_block,
            prefix=f"{block_prefix}gcn_conv.",
            relation_mix=self.config.relation_fed_mix_by_relation,
            default_mix=self.config.relation_fed_mix,
        )
        for layer_idx, layer in enumerate(module.gcn_layers):
            layer.load_shared_state_dict(
                shared_block,
                prefix=f"{block_prefix}gcn_layers.{layer_idx}.",
                relation_mix=self.config.relation_fed_mix_by_relation,
                default_mix=self.config.relation_fed_mix,
            )

    @staticmethod
    def _export_relation_shared_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
        exported = {
            f"gcn_conv.{key}": value
            for key, value in module.gcn_conv.shared_state_dict().items()
        }
        for layer_idx, layer in enumerate(module.gcn_layers):
            exported.update(
                {
                    f"gcn_layers.{layer_idx}.{key}": value
                    for key, value in layer.shared_state_dict().items()
                }
            )
        return exported

    def _compute_actor_proximal_penalty(self) -> torch.Tensor:
        penalty = torch.zeros((), dtype=torch.float32, device=self.device)
        if not self.enable_federation or not self.global_actor_reference:
            return penalty

        return penalty + self._compute_relation_shared_penalty(self.actor, self.global_actor_reference)

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
        penalty = penalty + LocalSPRGNNCSACAgent._compute_relation_shared_penalty(module, reference)
        return penalty

    @staticmethod
    def _compute_relation_shared_penalty(
        module: torch.nn.Module,
        reference: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        penalty = torch.zeros((), dtype=torch.float32, device=next(module.parameters()).device)
        for name, param in module.gcn_conv.shared.named_parameters():
            penalty = penalty + (param - reference[f"gcn_conv.shared.{name}"]).pow(2).sum()
        for layer_idx, layer in enumerate(module.gcn_layers):
            for name, param in layer.shared.named_parameters():
                penalty = penalty + (param - reference[f"gcn_layers.{layer_idx}.shared.{name}"]).pow(2).sum()
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
            edge_type_ids = torch.tensor(
                graph.get("edge_type_ids", [0] * len(graph["edge_index"])),
                dtype=torch.long,
                device=device,
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_type_ids = torch.zeros((0,), dtype=torch.long, device=device)
        batch = torch.zeros(node_count, dtype=torch.long, device=device)
        return {
            "node_type_ids": node_type_ids,
            "edge_index": edge_index,
            "edge_type_ids": edge_type_ids,
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
        edge_type_ids_list: List[torch.Tensor] = []
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
                edge_type_ids_list.append(
                    torch.tensor(
                        graph.get("edge_type_ids", [0] * len(graph["edge_index"])),
                        dtype=torch.long,
                        device=device,
                    )
                )
            node_offset += node_count

        return {
            "node_type_ids": torch.cat(node_type_ids_list, dim=0),
            "edge_index": (
                torch.cat(edge_index_list, dim=1)
                if edge_index_list
                else torch.zeros((2, 0), dtype=torch.long, device=device)
            ),
            "edge_type_ids": (
                torch.cat(edge_type_ids_list, dim=0)
                if edge_type_ids_list
                else torch.zeros((0,), dtype=torch.long, device=device)
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
    "SharedPrivateRelationalConv",
    "ActorActionGNN",
    "CriticActionGNN",
    "CriticNetwork",
    "CostCriticNetwork",
    "LocalSPRGNNCSACConfig",
    "LocalSPRGNNCSACAgent",
]
