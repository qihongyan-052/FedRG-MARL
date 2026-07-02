from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import csv
import json
import os
import pickle
import random

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from Fed_average.fed_controller import FederatedAveragingCoordinator, FederatedConfig
from Fed_average.learnable_personalized_fed_actor import LearnablePersonalizedFedActorCoordinator
from agent.central_state import build_central_tr_graph, get_central_node_sizes
from agent.state import StateBuilder, get_node_sizes, normalize_privacy_mode
from algorithm.central_gnn_csac import CentralGNNCSACAgent, CentralGNNCSACConfig
from algorithm.central_hgt_sac import CentralHGTSACAgent, CentralHGTSACConfig
from algorithm.gnn_csac import LocalGNNCSACAgent, LocalGNNCSACConfig
from algorithm.gnn_sac import LocalGNNSACAgent, LocalGNNSACConfig
from algorithm.hgt_csac import LocalHGTCSACAgent, LocalHGTCSACConfig
from algorithm.hgt_sac import LocalHGTSACAgent, LocalHGTSACConfig
from algorithm.mlp_csac import LocalMLPCSACAgent, LocalMLPCSACConfig
from algorithm.mlp_sac import LocalMLPSACAgent, LocalMLPSACConfig
from algorithm.mlp_td3 import LocalMLPTD3Agent, LocalMLPTD3Config
from algorithm.sp_rgnn_csac import LocalSPRGNNCSACAgent, LocalSPRGNNCSACConfig
from env.three_park_charging_env import PARK_TYPES, ThreeParkChargingEnv
from visualize.train.plot_episode_reward import (
    generate_park_episode_reward_plot,
    generate_total_episode_reward_plot,
)
from visualize.train.plot_episode_reward_components import generate_episode_reward_components_plot


@dataclass
class TrainingConfig:
    run_name: str = "sp_rgnn_csac-ablation2-7"
    algorithm_variant: str = "sp_rgnn_csac-ablation2"  # gnn_sac/gnn_csac/sp_rgnn_csac/mlp_sac/mlp_td3/mlp_csac/hgt_sac/hgt_csac/sp_rgnn_csac-ablation1
    enable_federation: bool = False   # 是否联邦
    federate_critic_backbone: bool = False    # 是否联邦actor+critic
    privacy_mode: str = "strong"   # strong/none
    enable_fed_distillation: bool = False     # 是否联邦蒸馏
    enable_fed_distill_actor: bool = False
    decouple_actor_output_heads: bool = True   # True为解耦
    use_central_tr_hgt_agent: bool = False
    use_strong_tr_projection_for_nonprivacy: bool = False
    bes_only_mode: bool = False
    resume_training: bool = False    #False/True
    seed: int = 10
    deterministic_training: bool = True
    total_episodes: int = 1000
    federated_warmup_episodes: int = 200
    federation_early_phase_end_episode: int = 500
    federation_mid_phase_end_episode: int = 800
    federation_early_phase_interval: int = 5
    federation_mid_phase_interval: int = 10
    federation_late_phase_interval: int = 20
    fed_logits_lr: float = 5e-3
    rho_logits_lr: float = 2e-3
    fed_logits_diag_init: float = 2.5
    fed_logits_offdiag_init: float = 0.0
    rho_init: float = 0.08
    candidate_gate_margin_r: float = 0.05
    candidate_gate_margin_o: float = 0.05
    candidate_gate_margin_c: float = 0.05
    eta_probe_r: float = 3e-4
    eta_probe_o: float = 3e-4
    eta_probe_c: float = 3e-4
    eta_max_r: float = 0.005
    eta_max_o: float = 0.005
    eta_max_c: float = 0.005
    fed_distill_warmup_episodes: int = 200
    fed_distill_interval_episodes: int = 15
    fed_distill_batch_size: int = 64
    fed_distill_num_candidates: int = 7
    fed_distill_reward_temperature: float = 0.8
    fed_distill_risk_temperature: float = 0.8
    fed_distill_reward_weight: float = 0.2
    fed_distill_risk_weight: float = 0.3
    fed_distill_actor_bes_weight: float = 0.20
    fed_distill_actor_ev_net_weight: float = 0.12
    strong_nonfed_fed_actor_backbone_lr_before_fed_start: float = 3e-4
    strong_nonfed_fed_actor_backbone_lr_after_fed_start: float = 3e-4
    strong_nonfed_fed_actor_local_backbone_lr_before_fed_start: float = 3e-4
    strong_nonfed_fed_actor_local_backbone_lr_after_fed_start: float = 3e-4
    strong_nonfed_fed_critic_backbone_lr_before_fed_start: float = 3e-4
    strong_nonfed_fed_critic_backbone_lr_after_fed_start: float = 3e-4
    strong_nonfed_fed_actor_head_lr_before_fed_start: float = 3e-4
    strong_nonfed_fed_actor_head_lr_after_fed_start: float = 3e-4
    strong_nonfed_fed_critic_head_lr_before_fed_start: float = 3e-4
    strong_nonfed_fed_critic_head_lr_after_fed_start: float = 3e-4
    sp_rgnn_actor_backbone_lr_before_fed_start: float = 2e-4
    sp_rgnn_actor_backbone_lr_after_fed_start: float = 1e-4
    sp_rgnn_critic_backbone_lr_before_fed_start: float = 2e-4
    sp_rgnn_critic_backbone_lr_after_fed_start: float = 1e-4
    sp_rgnn_actor_head_lr_before_fed_start: float = 2e-4
    sp_rgnn_actor_head_lr_after_fed_start: float = 1e-4
    sp_rgnn_critic_head_lr_before_fed_start: float = 2e-4
    sp_rgnn_critic_head_lr_after_fed_start: float = 1e-4
    critic_federated_warmup_episodes: int = 200
    critic_federation_early_phase_end_episode: int = 450
    critic_federation_mid_phase_end_episode: int = 700
    critic_federation_early_phase_interval: int = 30
    critic_federation_mid_phase_interval: int = 60
    critic_federation_late_phase_interval: int = 120
    update_every_steps: int = 4
    gradient_steps_per_update: int = 1
    act_device: str = "cpu"  #cpu/cuda:0，后续复杂这项建议换cuda:0
    update_device: str = "cpu"  #cpu/cuda:0，后续复杂这项保持为cpu
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 128
    replay_size: int = 100000
    target_entropy_scale: float = -1.0
    actor_proximal_weight: float = 2e-4
    critic_proximal_weight: float = 1e-4
    d: float = 30.0
    lambda_lr: float = 3e-4
    tr_probe_ratio_1: float = 0.2
    tr_probe_ratio_2: float = 0.4
    tr_curvature_weight: float = 0.5
    tr_overload_penalty_weight: float = 1.0
    interrupted_save_interval_episodes: int = 50
    final_save_interval_episodes: int = 20

"""
def _validate_episode_limit(config: TrainingConfig, max_episodes: int = 10) -> None:
    if config.total_episodes > max_episodes:
        raise ValueError(
            f"Episode limit exceeded: total_episodes={config.total_episodes}, max_allowed={max_episodes}.\n"
            "This script is currently restricted to short debug runs only.\n"
            "Please set TrainingConfig.total_episodes to 10 or less, or change the limit intentionally."
        )
"""

@dataclass
class RunDirectories:
    log_dir: Path
    results_dir: Path
    full_best_dir: Path
    full_final_dir: Path
    interrupted_dir: Path


FED_METRIC_PARK_LABELS = ("R", "O", "C")
PARK_LABELS = {
    "residential": "R",
    "office": "O",
    "commercial": "C",
}

SP_RGNN_CSAC_VARIANTS = {"sp_rgnn_csac", "sp_rgnn_csac-ablation1", "sp_rgnn_csac-ablation2"}
SP_RGNN_CSAC_ABLATION1 = "sp_rgnn_csac-ablation1"
SP_RGNN_CSAC_ABLATION2 = "sp_rgnn_csac-ablation2"
SP_RGNN_CSAC_NONFED_ABLATIONS = {SP_RGNN_CSAC_ABLATION1, SP_RGNN_CSAC_ABLATION2}


class CSVLogger:
    def __init__(self, path: Path, fieldnames: List[str], append: bool = False) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self._closed = False
        file_exists = path.exists()
        mode = "a" if append and file_exists else "w"
        self.file = path.open(mode, encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if mode == "w":
            self.writer.writeheader()
            self.flush()

    def write_row(self, row: Dict[str, object]) -> None:
        normalized = {field: row.get(field, "") for field in self.fieldnames}
        try:
            self.writer.writerow(normalized)
        except PermissionError as exc:
            raise PermissionError(
                f"failed to write CSV log '{self.path}'. The file may be open in Excel/WPS or locked by another process."
            ) from exc

    def flush(self) -> None:
        if self._closed:
            return
        try:
            self.file.flush()
        except PermissionError as exc:
            raise PermissionError(
                f"failed to flush CSV log '{self.path}'. The file may be open in Excel/WPS or locked by another process."
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.file.flush()
        except PermissionError:
            pass
        try:
            self.file.close()
        except PermissionError:
            pass
        self._closed = True


def _truncate_csv_at_episode(path: Path, fieldnames: List[str], start_episode: int) -> None:
    if start_episode <= 0 or not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
    except PermissionError as exc:
        raise PermissionError(
            f"failed to read CSV log '{path}' before resume. The file may be open in Excel/WPS or locked."
        ) from exc

    kept_rows = []
    removed_count = 0
    for row in rows:
        try:
            episode = int(row.get("episode", ""))
        except (TypeError, ValueError):
            kept_rows.append(row)
            continue
        if episode < start_episode:
            kept_rows.append(row)
        else:
            removed_count += 1

    if removed_count == 0:
        return
    try:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in kept_rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
    except PermissionError as exc:
        raise PermissionError(
            f"failed to rewrite CSV log '{path}' before resume. The file may be open in Excel/WPS or locked."
        ) from exc
    print(
        f"resume log cleanup: removed {removed_count} stale rows with episode >= {start_episode} from {path}"
    )


def _truncate_training_logs_for_resume(run_dirs: RunDirectories, start_episode: int) -> None:
    _truncate_csv_at_episode(run_dirs.log_dir / "reward_log.csv", _reward_log_fields(), start_episode)
    _truncate_csv_at_episode(run_dirs.log_dir / "training_log.csv", _training_log_fields(), start_episode)
    _truncate_csv_at_episode(run_dirs.log_dir / "bes_soc_steps.csv", _bes_soc_step_log_fields(), start_episode)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch_determinism(enabled: bool) -> None:
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
    try:
        torch.use_deterministic_algorithms(enabled)
    except Exception:
        pass


def _capture_rng_state() -> Dict[str, Any]:
    return {
        "python_random": random.getstate(),
        "numpy_random": None if np is None else np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(rng_state: Dict[str, Any]) -> None:
    random_state = rng_state.get("python_random")
    if random_state is not None:
        random.setstate(random_state)
    if np is not None and rng_state.get("numpy_random") is not None:
        np.random.set_state(rng_state["numpy_random"])
    if rng_state.get("torch_cpu") is not None:
        torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def resolve_compute_device(device: str) -> str:
    normalized = device.strip().lower()
    if normalized == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def validate_training_config(config: TrainingConfig) -> None:
    if _uses_central_tr_hgt_agent(config):
        if config.algorithm_variant not in {"hgt_sac", "gnn_csac"}:
            raise RuntimeError("central TR agent is currently implemented only for hgt_sac or gnn_csac.")
        if config.enable_federation:
            raise RuntimeError("central TR agent does not support federation yet.")
        if config.enable_fed_distillation or config.enable_fed_distill_actor:
            raise RuntimeError("central TR agent does not support fed distillation yet.")
        if normalize_privacy_mode(config.privacy_mode) != "none":
            raise RuntimeError("central TR agent currently requires privacy_mode='none'.")
        if config.use_strong_tr_projection_for_nonprivacy:
            raise RuntimeError(
                "central TR agent currently requires use_strong_tr_projection_for_nonprivacy=False."
            )
    if config.enable_federation and config.algorithm_variant in {"mlp_sac", "mlp_td3", "mlp_csac"}:
        raise RuntimeError(
            "MLP federation is not supported in the current system. "
            "The three parks use different MLP state dimensions, so FedAvg cannot directly aggregate park-specific actor/critic backbones."
        )
    if config.algorithm_variant in SP_RGNN_CSAC_NONFED_ABLATIONS and config.enable_federation:
        raise RuntimeError(f"{config.algorithm_variant} removes parameter federation; set enable_federation=False.")
    if config.enable_fed_distillation:
        if config.enable_federation:
            raise RuntimeError("set enable_federation=False when enable_fed_distillation=True.")
        if config.algorithm_variant != "hgt_sac":
            raise RuntimeError("federated distillation is currently implemented only for hgt_sac.")
        if normalize_privacy_mode(config.privacy_mode) != "strong":
            raise RuntimeError("federated distillation currently requires privacy_mode='strong'.")


def _uses_central_tr_hgt_agent(config: TrainingConfig) -> bool:
    return bool(config.use_central_tr_hgt_agent)


def _uses_hgt_head_lr_phase_schedule(config: TrainingConfig) -> bool:
    return (
        config.algorithm_variant == "hgt_sac"
        and normalize_privacy_mode(config.privacy_mode) == "strong"
        and not _uses_central_tr_hgt_agent(config)
        and not config.enable_fed_distillation
    )


def _resolve_hgt_lr_phase_switch_episode(config: TrainingConfig) -> int:
    return max(
        0,
        int(config.federated_warmup_episodes) + max(1, int(config.federation_early_phase_interval)) - 1,
    )


def _apply_hgt_head_lr_phase(
    local_agents: Dict[str, Any],
    config: TrainingConfig,
    episode: int,
) -> None:
    if not _uses_hgt_head_lr_phase_schedule(config):
        return
    switch_episode = _resolve_hgt_lr_phase_switch_episode(config)
    if episode < switch_episode:
        actor_backbone_lr = config.strong_nonfed_fed_actor_backbone_lr_before_fed_start
        actor_local_backbone_lr = config.strong_nonfed_fed_actor_local_backbone_lr_before_fed_start
        critic_backbone_lr = config.strong_nonfed_fed_critic_backbone_lr_before_fed_start
        actor_head_lr = config.strong_nonfed_fed_actor_head_lr_before_fed_start
        critic_head_lr = config.strong_nonfed_fed_critic_head_lr_before_fed_start
    else:
        actor_backbone_lr = config.strong_nonfed_fed_actor_backbone_lr_after_fed_start
        actor_local_backbone_lr = config.strong_nonfed_fed_actor_local_backbone_lr_after_fed_start
        critic_backbone_lr = config.strong_nonfed_fed_critic_backbone_lr_after_fed_start
        actor_head_lr = config.strong_nonfed_fed_actor_head_lr_after_fed_start
        critic_head_lr = config.strong_nonfed_fed_critic_head_lr_after_fed_start
    for agent in local_agents.values():
        if hasattr(agent, "set_backbone_learning_rates"):
            agent.set_backbone_learning_rates(
                actor_backbone_lr=actor_backbone_lr,
                actor_local_backbone_lr=actor_local_backbone_lr,
                critic_backbone_lr=critic_backbone_lr,
            )
        if hasattr(agent, "set_output_head_learning_rates"):
            agent.set_output_head_learning_rates(
                actor_head_lr=actor_head_lr,
                critic_head_lr=critic_head_lr,
            )


def _apply_sp_rgnn_lr_phase(
    local_agents: Dict[str, Any],
    config: TrainingConfig,
    episode: int,
) -> None:
    if config.algorithm_variant not in SP_RGNN_CSAC_VARIANTS:
        return
    switch_episode = _resolve_hgt_lr_phase_switch_episode(config)
    if episode < switch_episode:
        actor_backbone_lr = config.sp_rgnn_actor_backbone_lr_before_fed_start
        critic_backbone_lr = config.sp_rgnn_critic_backbone_lr_before_fed_start
        actor_head_lr = config.sp_rgnn_actor_head_lr_before_fed_start
        critic_head_lr = config.sp_rgnn_critic_head_lr_before_fed_start
    else:
        actor_backbone_lr = config.sp_rgnn_actor_backbone_lr_after_fed_start
        critic_backbone_lr = config.sp_rgnn_critic_backbone_lr_after_fed_start
        actor_head_lr = config.sp_rgnn_actor_head_lr_after_fed_start
        critic_head_lr = config.sp_rgnn_critic_head_lr_after_fed_start
    for agent in local_agents.values():
        if hasattr(agent, "set_backbone_learning_rates"):
            agent.set_backbone_learning_rates(
                actor_backbone_lr=actor_backbone_lr,
                critic_backbone_lr=critic_backbone_lr,
            )
        if hasattr(agent, "set_output_head_learning_rates"):
            agent.set_output_head_learning_rates(
                actor_head_lr=actor_head_lr,
                critic_head_lr=critic_head_lr,
            )


def build_local_agents(config: TrainingConfig) -> Dict[str, Any]:
    agents: Dict[str, Any] = {}
    resolved_act_device = resolve_compute_device(config.act_device)
    resolved_update_device = resolve_compute_device(config.update_device)
    parameter_federation_enabled = config.enable_federation and not config.enable_fed_distillation
    cp_count_by_park = load_cp_count_by_park()
    for idx, park_type in enumerate(PARK_TYPES):
        if config.algorithm_variant == "gnn_sac":
            agents[park_type] = LocalGNNSACAgent(
                LocalGNNSACConfig(
                    park_type=park_type,
                    algorithm_variant=config.algorithm_variant,
                    enable_federation=config.enable_federation,
                    federate_critic_backbone=config.federate_critic_backbone,
                    privacy_mode=config.privacy_mode,
                    decouple_actor_output_heads=config.decouple_actor_output_heads,
                    alpha_lr=config.alpha_lr,
                    gamma=config.gamma,
                    tau=config.tau,
                    batch_size=config.batch_size,
                    replay_size=config.replay_size,
                    target_entropy=compute_park_target_entropy(cp_count_by_park[park_type], config.target_entropy_scale),
                    actor_proximal_weight=config.actor_proximal_weight,
                    critic_proximal_weight=config.critic_proximal_weight,
                    d=config.d,
                    lambda_lr=config.lambda_lr,
                    actor_backbone_lr=config.sp_rgnn_actor_backbone_lr_before_fed_start,
                    actor_head_lr=config.sp_rgnn_actor_head_lr_before_fed_start,
                    critic_backbone_lr=config.sp_rgnn_critic_backbone_lr_before_fed_start,
                    critic_head_lr=config.sp_rgnn_critic_head_lr_before_fed_start,
                    seed=config.seed + idx,
                    act_device=resolved_act_device,
                    update_device=resolved_update_device,
                )
            )
        elif config.algorithm_variant == "mlp_sac":
            agents[park_type] = LocalMLPSACAgent(
                LocalMLPSACConfig(
                    park_type=park_type,
                    cp_count=cp_count_by_park[park_type],
                    algorithm_variant=config.algorithm_variant,
                    enable_federation=config.enable_federation,
                    federate_critic_backbone=config.federate_critic_backbone,
                    privacy_mode=config.privacy_mode,
                    alpha_lr=config.alpha_lr,
                    gamma=config.gamma,
                    tau=config.tau,
                    batch_size=config.batch_size,
                    replay_size=config.replay_size,
                    target_entropy=compute_park_target_entropy(cp_count_by_park[park_type], config.target_entropy_scale),
                    actor_proximal_weight=config.actor_proximal_weight,
                    d=config.d,
                    lambda_lr=config.lambda_lr,
                    seed=config.seed + idx,
                    act_device=resolved_act_device,
                    update_device=resolved_update_device,
                )
            )
        elif config.algorithm_variant == "mlp_td3":
            agents[park_type] = LocalMLPTD3Agent(
                LocalMLPTD3Config(
                    park_type=park_type,
                    cp_count=cp_count_by_park[park_type],
                    algorithm_variant=config.algorithm_variant,
                    enable_federation=config.enable_federation,
                    privacy_mode=config.privacy_mode,
                    gamma=config.gamma,
                    tau=config.tau,
                    batch_size=config.batch_size,
                    replay_size=config.replay_size,
                    actor_proximal_weight=config.actor_proximal_weight,
                    seed=config.seed + idx,
                    act_device=resolved_act_device,
                    update_device=resolved_update_device,
                    d=config.d,
                    lambda_lr=config.lambda_lr,
                )
            )
        elif config.algorithm_variant == "hgt_sac":
            agents[park_type] = LocalHGTSACAgent(
                LocalHGTSACConfig(
                    park_type=park_type,
                    algorithm_variant=config.algorithm_variant,
                    enable_federation=parameter_federation_enabled,
                    federate_critic_backbone=(config.federate_critic_backbone and not config.enable_fed_distillation),
                    privacy_mode=config.privacy_mode,
                    decouple_actor_output_heads=config.decouple_actor_output_heads,
                    enable_auxiliary_risk_critic=config.enable_fed_distillation,
                    alpha_lr=config.alpha_lr,
                    gamma=config.gamma,
                    tau=config.tau,
                    batch_size=config.batch_size,
                    replay_size=config.replay_size,
                    target_entropy=compute_park_target_entropy(cp_count_by_park[park_type], config.target_entropy_scale),
                    actor_proximal_weight=(
                        0.0
                        if (
                            parameter_federation_enabled
                            and not config.federate_critic_backbone
                            and config.algorithm_variant in {"hgt_sac", "hgt_csac"}
                        )
                        else config.actor_proximal_weight
                    ),
                    critic_proximal_weight=config.critic_proximal_weight,
                    d=config.d,
                    lambda_lr=config.lambda_lr,
                    seed=config.seed + idx,
                    act_device=resolved_act_device,
                    update_device=resolved_update_device,
                )
            )
        elif config.algorithm_variant == "hgt_csac":
            agents[park_type] = LocalHGTCSACAgent(
                LocalHGTCSACConfig(
                    park_type=park_type,
                    algorithm_variant=config.algorithm_variant,
                    enable_federation=config.enable_federation,
                    federate_critic_backbone=config.federate_critic_backbone,
                    privacy_mode=config.privacy_mode,
                    decouple_actor_output_heads=config.decouple_actor_output_heads,
                    alpha_lr=config.alpha_lr,
                    gamma=config.gamma,
                    tau=config.tau,
                    batch_size=config.batch_size,
                    replay_size=config.replay_size,
                    target_entropy=compute_park_target_entropy(cp_count_by_park[park_type], config.target_entropy_scale),
                    actor_proximal_weight=(
                        0.0
                        if (
                            config.enable_federation
                            and not config.federate_critic_backbone
                            and config.algorithm_variant in {"hgt_sac", "hgt_csac"}
                        )
                        else config.actor_proximal_weight
                    ),
                    critic_proximal_weight=config.critic_proximal_weight,
                    d=config.d,
                    lambda_lr=config.lambda_lr,
                    seed=config.seed + idx,
                    act_device=resolved_act_device,
                    update_device=resolved_update_device,
                )
            )
        elif config.algorithm_variant == "mlp_csac":
            agents[park_type] = LocalMLPCSACAgent(
                LocalMLPCSACConfig(
                    park_type=park_type,
                    cp_count=cp_count_by_park[park_type],
                    algorithm_variant=config.algorithm_variant,
                    enable_federation=config.enable_federation,
                    federate_critic_backbone=config.federate_critic_backbone,
                    privacy_mode=config.privacy_mode,
                    alpha_lr=config.alpha_lr,
                    gamma=config.gamma,
                    tau=config.tau,
                    batch_size=config.batch_size,
                    replay_size=config.replay_size,
                    target_entropy=compute_park_target_entropy(cp_count_by_park[park_type], config.target_entropy_scale),
                    actor_proximal_weight=config.actor_proximal_weight,
                    d=config.d,
                    lambda_lr=config.lambda_lr,
                    seed=config.seed + idx,
                    act_device=resolved_act_device,
                    update_device=resolved_update_device,
                )
            )
        elif config.algorithm_variant == "gnn_csac":
            agents[park_type] = LocalGNNCSACAgent(
                LocalGNNCSACConfig(
                    park_type=park_type,
                    algorithm_variant=config.algorithm_variant,
                    enable_federation=config.enable_federation,
                    federate_critic_backbone=config.federate_critic_backbone,
                    privacy_mode=config.privacy_mode,
                    decouple_actor_output_heads=config.decouple_actor_output_heads,
                    alpha_lr=config.alpha_lr,
                    gamma=config.gamma,
                    tau=config.tau,
                    batch_size=config.batch_size,
                    replay_size=config.replay_size,
                    target_entropy=compute_park_target_entropy(cp_count_by_park[park_type], config.target_entropy_scale),
                    actor_proximal_weight=config.actor_proximal_weight,
                    critic_proximal_weight=config.critic_proximal_weight,
                    d=config.d,
                    lambda_lr=config.lambda_lr,
                    seed=config.seed + idx,
                    act_device=resolved_act_device,
                    update_device=resolved_update_device,
                )
            )
        elif config.algorithm_variant in SP_RGNN_CSAC_VARIANTS:
            agents[park_type] = LocalSPRGNNCSACAgent(
                LocalSPRGNNCSACConfig(
                    park_type=park_type,
                    algorithm_variant=config.algorithm_variant,
                    enable_federation=False if config.algorithm_variant in SP_RGNN_CSAC_NONFED_ABLATIONS else config.enable_federation,
                    federate_critic_backbone=config.federate_critic_backbone,
                    privacy_mode=config.privacy_mode,
                    decouple_actor_output_heads=config.decouple_actor_output_heads,
                    use_relation_gated_fusion=config.algorithm_variant != SP_RGNN_CSAC_ABLATION2,
                    use_critic_typed_pooling=config.algorithm_variant != SP_RGNN_CSAC_ABLATION2,
                    alpha_lr=config.alpha_lr,
                    gamma=config.gamma,
                    tau=config.tau,
                    batch_size=config.batch_size,
                    replay_size=config.replay_size,
                    target_entropy=compute_park_target_entropy(cp_count_by_park[park_type], config.target_entropy_scale),
                    actor_proximal_weight=config.actor_proximal_weight,
                    critic_proximal_weight=config.critic_proximal_weight,
                    d=config.d,
                    lambda_lr=config.lambda_lr,
                    seed=config.seed + idx,
                    act_device=resolved_act_device,
                    update_device=resolved_update_device,
                )
            )
        else:
            raise ValueError(f"Unsupported algorithm_variant: {config.algorithm_variant}")
    return agents


def build_central_hgt_agent(config: TrainingConfig) -> CentralHGTSACAgent:
    resolved_act_device = resolve_compute_device(config.act_device)
    resolved_update_device = resolve_compute_device(config.update_device)
    total_cp_count = sum(load_cp_count_by_park().values())
    aligned_network_lr = 3e-4
    central_feature_dim = 64
    central_batch_size = 256
    central_replay_size = 200000
    return CentralHGTSACAgent(
        CentralHGTSACConfig(
            algorithm_variant=config.algorithm_variant,
            privacy_mode=config.privacy_mode,
            alpha_lr=config.alpha_lr,
            gamma=config.gamma,
            tau=config.tau,
            batch_size=central_batch_size,
            replay_size=central_replay_size,
            target_entropy=compute_park_target_entropy(total_cp_count, config.target_entropy_scale),
            actor_proximal_weight=config.actor_proximal_weight,
            critic_proximal_weight=config.critic_proximal_weight,
            seed=config.seed,
            act_device=resolved_act_device,
            update_device=resolved_update_device,
            feature_dim=central_feature_dim,
            decouple_actor_output_heads=config.decouple_actor_output_heads,
            actor_backbone_lr=aligned_network_lr,
            actor_head_lr=aligned_network_lr,
            critic_backbone_lr=aligned_network_lr,
            critic_head_lr=aligned_network_lr,
        )
    )


def build_central_gnn_csac_agent(config: TrainingConfig) -> CentralGNNCSACAgent:
    resolved_act_device = resolve_compute_device(config.act_device)
    resolved_update_device = resolve_compute_device(config.update_device)
    total_cp_count = sum(load_cp_count_by_park().values())
    central_feature_dim = 64
    central_batch_size = 256
    central_replay_size = 200000
    return CentralGNNCSACAgent(
        CentralGNNCSACConfig(
            algorithm_variant=config.algorithm_variant,
            privacy_mode=config.privacy_mode,
            alpha_lr=config.alpha_lr,
            gamma=config.gamma,
            tau=config.tau,
            batch_size=central_batch_size,
            replay_size=central_replay_size,
            target_entropy=compute_park_target_entropy(total_cp_count, config.target_entropy_scale),
            actor_proximal_weight=config.actor_proximal_weight,
            critic_proximal_weight=config.critic_proximal_weight,
            seed=config.seed,
            act_device=resolved_act_device,
            update_device=resolved_update_device,
            d=config.d,
            lambda_lr=config.lambda_lr,
            feature_dim=central_feature_dim,
            decouple_actor_output_heads=config.decouple_actor_output_heads,
        )
    )


def build_central_agent(config: TrainingConfig) -> Any:
    if config.algorithm_variant == "gnn_csac":
        return build_central_gnn_csac_agent(config)
    return build_central_hgt_agent(config)


def configure_environment(env: ThreeParkChargingEnv, config: TrainingConfig) -> None:
    env.tr_probe_ratio_1 = config.tr_probe_ratio_1
    env.tr_probe_ratio_2 = config.tr_probe_ratio_2
    env.tr_curvature_weight = config.tr_curvature_weight
    env.transformer_overload_penalty_weight = config.tr_overload_penalty_weight
    env.bes_only_mode = config.bes_only_mode
    env.privacy_mode = normalize_privacy_mode(config.privacy_mode)
    env.use_central_tr_hgt_agent = _uses_central_tr_hgt_agent(config)
    env.use_strong_tr_projection_for_nonprivacy = config.use_strong_tr_projection_for_nonprivacy
    env.state_builder = StateBuilder(config_dir=env.config_dir, privacy_mode=config.privacy_mode)


def load_cp_count_by_park() -> Dict[str, int]:
    topology_path = Path(__file__).resolve().parent / "config_files" / "three_parks_topology_config.json"
    with topology_path.open("r", encoding="utf-8") as file:
        topology = json.load(file)
    return {park["id"]: int(park["cp"]["count"]) for park in topology["parks"]}


def compute_park_target_entropy(cp_count: int, scale: float = -1.0) -> float:
    action_dim = 1 + cp_count
    return float(scale * action_dim)


def build_joint_action(
    local_agents: Dict[str, Any],
    obs: Dict[str, object],
    deterministic: bool = False,
    return_raw_action: bool = False,
) -> Dict[str, object] | tuple[Dict[str, object], Dict[str, torch.Tensor]]:
    action = {"parks": {}}
    raw_node_actions: Dict[str, torch.Tensor] = {}
    for park_type in PARK_TYPES:
        park_graph = obs["park_graphs"][park_type]
        if return_raw_action:
            env_action, node_action = local_agents[park_type].act(
                park_graph,
                deterministic=deterministic,
                return_node_action=True,
            )
            action["parks"][park_type] = env_action
            raw_node_actions[park_type] = node_action
        else:
            action["parks"][park_type] = local_agents[park_type].act(park_graph, deterministic=deterministic)
    if return_raw_action:
        return action, raw_node_actions
    return action


def _fed_distillation_enabled(config: TrainingConfig) -> bool:
    return (
        config.enable_fed_distillation
        and config.algorithm_variant == "hgt_sac"
        and normalize_privacy_mode(config.privacy_mode) == "strong"
    )


def _should_run_fed_distillation(episode: int, config: TrainingConfig) -> bool:
    if not _fed_distillation_enabled(config):
        return False
    completed_episodes = episode + 1
    if completed_episodes <= config.fed_distill_warmup_episodes:
        return False
    interval = max(1, config.fed_distill_interval_episodes)
    return (completed_episodes - config.fed_distill_warmup_episodes) % interval == 0


def _average_tensor_list(tensors: List[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("cannot average an empty tensor list")
    total = tensors[0].clone()
    for tensor in tensors[1:]:
        total = total + tensor
    return total / float(len(tensors))


def _distillation_teacher_weights(student_park: str) -> Dict[str, float]:
    if student_park == "residential":
        return {
            "office": 0.65,
            "commercial": 0.35,
        }
    if student_park == "office":
        return {
            "residential": 0.40,
            "commercial": 0.60,
        }
    return {
        "residential": 0.30,
        "office": 0.70,
    }


def _weighted_average_tensor_map(
    tensors_by_park: Dict[str, torch.Tensor],
    weights_by_park: Dict[str, float],
) -> torch.Tensor:
    if not tensors_by_park:
        raise ValueError("cannot average an empty tensor map")
    total_weight = 0.0
    weighted_sum: torch.Tensor | None = None
    for park_id, tensor in tensors_by_park.items():
        weight = float(weights_by_park.get(park_id, 0.0))
        if weight <= 0.0:
            continue
        weighted = tensor * weight
        weighted_sum = weighted if weighted_sum is None else weighted_sum + weighted
        total_weight += weight
    if weighted_sum is None or total_weight <= 0.0:
        return _average_tensor_list(list(tensors_by_park.values()))
    return weighted_sum / total_weight


def _teacher_confidence_from_scores(
    scores: torch.Tensor,
    temperature: float,
) -> float:
    if scores.numel() == 0:
        return 0.0
    probs = torch.softmax(scores / max(temperature, 1e-6), dim=1)
    q_spread = scores.max(dim=1).values - scores.min(dim=1).values
    spread_conf = torch.sigmoid(q_spread).mean()
    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-8))).sum(dim=1)
    max_entropy = torch.log(
        torch.tensor(float(scores.shape[1]), dtype=torch.float32, device=scores.device)
    )
    normalized_entropy = entropy / torch.clamp(max_entropy, min=1e-8)
    entropy_conf = (1.0 - normalized_entropy).mean()
    confidence = 0.5 * (spread_conf + entropy_conf)
    return float(torch.clamp(confidence, min=0.0, max=1.0).detach().cpu().item())


def _empty_fed_distill_metrics() -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "fed_distill_reward_loss": 0.0,
        "fed_distill_risk_loss": 0.0,
        "fed_distill_total_loss": 0.0,
        "fed_distill_reward_loss_post": 0.0,
        "fed_distill_risk_loss_post": 0.0,
        "fed_distill_actor_bes_loss": 0.0,
        "fed_distill_actor_ev_net_loss": 0.0,
        "fed_distill_actor_total_loss": 0.0,
        "fed_distill_actor_bes_loss_post": 0.0,
        "fed_distill_actor_ev_net_loss_post": 0.0,
        "fed_distill_rounds": 0.0,
    }
    for student_park in PARK_TYPES:
        student_label = PARK_LABELS[student_park]
        for teacher_park in PARK_TYPES:
            if teacher_park == student_park:
                continue
            teacher_label = PARK_LABELS[teacher_park]
            metrics[f"fed_distill_reward_conf_{student_label}{teacher_label}"] = 0.0
            metrics[f"fed_distill_risk_conf_{student_label}{teacher_label}"] = 0.0
            metrics[f"fed_distill_reward_w_{student_label}{teacher_label}"] = 0.0
            metrics[f"fed_distill_risk_w_{student_label}{teacher_label}"] = 0.0
    return metrics


def _run_fed_distillation_round(
    local_agents: Dict[str, Any],
    config: TrainingConfig,
) -> Dict[str, float]:
    metrics = _empty_fed_distill_metrics()
    if not _fed_distillation_enabled(config):
        return metrics

    pair_counts: Dict[str, float] = {
        key: 0.0 for key in metrics.keys()
        if key.startswith("fed_distill_reward_conf_")
        or key.startswith("fed_distill_risk_conf_")
        or key.startswith("fed_distill_reward_w_")
        or key.startswith("fed_distill_risk_w_")
    }
    for student_park in PARK_TYPES:
        student_agent = local_agents[student_park]
        graphs = student_agent.sample_distillation_graphs(config.fed_distill_batch_size)
        if not graphs:
            continue
        candidate_node_actions = student_agent.build_distillation_candidate_node_actions(
            graphs=graphs,
            num_candidates=config.fed_distill_num_candidates,
        )
        teacher_reward_scores: Dict[str, torch.Tensor] = {}
        teacher_risk_scores: Dict[str, torch.Tensor] = {}
        reward_teacher_weights: Dict[str, float] = {}
        risk_teacher_weights: Dict[str, float] = {}
        for teacher_park in PARK_TYPES:
            if teacher_park == student_park:
                continue
            reward_scores, risk_scores = local_agents[teacher_park].evaluate_distillation_preferences(
                graphs=graphs,
                candidate_node_actions=candidate_node_actions,
            )
            teacher_reward_scores[teacher_park] = reward_scores.detach().cpu()
            teacher_risk_scores[teacher_park] = risk_scores.detach().cpu()
        if not teacher_reward_scores:
            continue

        teacher_weights = _distillation_teacher_weights(student_park)
        student_label = PARK_LABELS[student_park]
        for teacher_park, reward_scores in teacher_reward_scores.items():
            teacher_label = PARK_LABELS[teacher_park]
            reward_confidence = _teacher_confidence_from_scores(
                reward_scores,
                config.fed_distill_reward_temperature,
            )
            reward_teacher_weights[teacher_park] = (
                float(teacher_weights.get(teacher_park, 0.0))
                * reward_confidence
            )
            metrics[f"fed_distill_reward_conf_{student_label}{teacher_label}"] += reward_confidence
            pair_counts[f"fed_distill_reward_conf_{student_label}{teacher_label}"] += 1.0
            metrics[f"fed_distill_reward_w_{student_label}{teacher_label}"] += reward_teacher_weights[teacher_park]
            pair_counts[f"fed_distill_reward_w_{student_label}{teacher_label}"] += 1.0
        mean_teacher_reward_scores = _weighted_average_tensor_map(
            teacher_reward_scores,
            reward_teacher_weights,
        )
        teacher_reward_pref = torch.softmax(
            mean_teacher_reward_scores / max(config.fed_distill_reward_temperature, 1e-6),
            dim=1,
        ).to(student_agent.device)

        teacher_risk_pref = None
        if config.fed_distill_risk_weight > 0.0 and teacher_risk_scores:
            for teacher_park, risk_scores in teacher_risk_scores.items():
                teacher_label = PARK_LABELS[teacher_park]
                risk_confidence = _teacher_confidence_from_scores(
                    -risk_scores,
                    config.fed_distill_risk_temperature,
                )
                risk_teacher_weights[teacher_park] = (
                    float(teacher_weights.get(teacher_park, 0.0))
                    * risk_confidence
                )
                metrics[f"fed_distill_risk_conf_{student_label}{teacher_label}"] += risk_confidence
                pair_counts[f"fed_distill_risk_conf_{student_label}{teacher_label}"] += 1.0
                metrics[f"fed_distill_risk_w_{student_label}{teacher_label}"] += risk_teacher_weights[teacher_park]
                pair_counts[f"fed_distill_risk_w_{student_label}{teacher_label}"] += 1.0
            mean_teacher_risk_scores = _weighted_average_tensor_map(
                teacher_risk_scores,
                risk_teacher_weights,
            )
            teacher_risk_pref = torch.softmax(
                (-mean_teacher_risk_scores) / max(config.fed_distill_risk_temperature, 1e-6),
                dim=1,
            ).to(student_agent.device)

        distill_metrics = student_agent.distill_critic_preferences(
            graphs=graphs,
            candidate_node_actions=candidate_node_actions,
            teacher_reward_pref=teacher_reward_pref,
            teacher_risk_pref=teacher_risk_pref,
            reward_temperature=config.fed_distill_reward_temperature,
            risk_temperature=config.fed_distill_risk_temperature,
            reward_weight=config.fed_distill_reward_weight,
            risk_weight=config.fed_distill_risk_weight,
        )
        actor_distill_metrics = {
            "actor_bes_distill_loss": 0.0,
            "actor_ev_net_distill_loss": 0.0,
            "actor_distill_total_loss": 0.0,
            "actor_bes_distill_loss_post": 0.0,
            "actor_ev_net_distill_loss_post": 0.0,
        }
        if config.enable_fed_distill_actor:
            actor_pref_components: List[Tuple[float, torch.Tensor]] = []
            if config.fed_distill_reward_weight > 0.0:
                actor_pref_components.append((config.fed_distill_reward_weight, teacher_reward_pref))
            if config.fed_distill_risk_weight > 0.0 and teacher_risk_pref is not None:
                actor_pref_components.append((config.fed_distill_risk_weight, teacher_risk_pref))
            teacher_actor_pref = teacher_reward_pref
            if actor_pref_components:
                pref_weight_sum = sum(weight for weight, _ in actor_pref_components)
                weighted_pref = sum(weight * pref for weight, pref in actor_pref_components)
                teacher_actor_pref = weighted_pref / max(pref_weight_sum, 1e-6)
                teacher_actor_pref = teacher_actor_pref / torch.clamp(
                    teacher_actor_pref.sum(dim=1, keepdim=True),
                    min=1e-6,
                )

            actor_distill_metrics = student_agent.distill_actor_tendencies(
                graphs=graphs,
                candidate_node_actions=candidate_node_actions,
                teacher_actor_pref=teacher_actor_pref,
                bes_weight=config.fed_distill_actor_bes_weight,
                ev_net_weight=config.fed_distill_actor_ev_net_weight,
            )
        metrics["fed_distill_reward_loss"] += float(distill_metrics["reward_distill_loss"])
        metrics["fed_distill_risk_loss"] += float(distill_metrics["risk_distill_loss"])
        metrics["fed_distill_total_loss"] += float(distill_metrics["distill_total_loss"])
        metrics["fed_distill_reward_loss_post"] += float(distill_metrics["reward_distill_loss_post"])
        metrics["fed_distill_risk_loss_post"] += float(distill_metrics["risk_distill_loss_post"])
        metrics["fed_distill_actor_bes_loss"] += float(actor_distill_metrics["actor_bes_distill_loss"])
        metrics["fed_distill_actor_ev_net_loss"] += float(actor_distill_metrics["actor_ev_net_distill_loss"])
        metrics["fed_distill_actor_total_loss"] += float(actor_distill_metrics["actor_distill_total_loss"])
        metrics["fed_distill_actor_bes_loss_post"] += float(actor_distill_metrics["actor_bes_distill_loss_post"])
        metrics["fed_distill_actor_ev_net_loss_post"] += float(actor_distill_metrics["actor_ev_net_distill_loss_post"])
        metrics["fed_distill_rounds"] += 1.0

    if metrics["fed_distill_rounds"] > 0:
        divisor = metrics["fed_distill_rounds"]
        metrics["fed_distill_reward_loss"] /= divisor
        metrics["fed_distill_risk_loss"] /= divisor
        metrics["fed_distill_total_loss"] /= divisor
        metrics["fed_distill_reward_loss_post"] /= divisor
        metrics["fed_distill_risk_loss_post"] /= divisor
        metrics["fed_distill_actor_bes_loss"] /= divisor
        metrics["fed_distill_actor_ev_net_loss"] /= divisor
        metrics["fed_distill_actor_total_loss"] /= divisor
        metrics["fed_distill_actor_bes_loss_post"] /= divisor
        metrics["fed_distill_actor_ev_net_loss_post"] /= divisor
    for key, count in pair_counts.items():
        if count > 0.0:
            metrics[key] /= count
    return metrics


def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _serialize_actor_head_blocks(actor: Any) -> Dict[str, Dict[str, torch.Tensor]]:
    blocks: Dict[str, Dict[str, torch.Tensor]] = {}
    if hasattr(actor, "mean_linear"):
        blocks["mean_linear"] = _cpu_state_dict(actor.mean_linear)
    if hasattr(actor, "log_std_linear"):
        blocks["log_std_linear"] = _cpu_state_dict(actor.log_std_linear)
    if hasattr(actor, "mean_head"):
        blocks["mean_head"] = _cpu_state_dict(actor.mean_head)
    if hasattr(actor, "log_std_head"):
        blocks["log_std_head"] = _cpu_state_dict(actor.log_std_head)
    if hasattr(actor, "mean_heads"):
        blocks["mean_heads"] = _cpu_state_dict(actor.mean_heads)
    if hasattr(actor, "log_std_heads"):
        blocks["log_std_heads"] = _cpu_state_dict(actor.log_std_heads)
    return blocks


def _serialize_agent(agent: Any, park_type: str, episode: int) -> Dict[str, object]:
    if hasattr(agent, "export_checkpoint"):
        return agent.export_checkpoint(park_type=park_type, episode=episode)
    return {
        "park_type": park_type,
        "episode": episode,
        "agent_config": asdict(agent.config),
        "actor": {
            "node_embedding": _cpu_state_dict(agent.actor.node_embedding),
            "gcn_conv": _cpu_state_dict(agent.actor.gcn_conv),
            "gcn_layers": _cpu_state_dict(agent.actor.gcn_layers),
            **_serialize_actor_head_blocks(agent.actor),
        },
        "critic1": {
            "node_embedding": _cpu_state_dict(agent.critic.q1.node_embedding),
            "gcn_conv": _cpu_state_dict(agent.critic.q1.gcn_conv),
            "gcn_layers": _cpu_state_dict(agent.critic.q1.gcn_layers),
            "l1": _cpu_state_dict(agent.critic.q1.l1),
            "l2": _cpu_state_dict(agent.critic.q1.l2),
            "l3": _cpu_state_dict(agent.critic.q1.l3),
        },
        "critic2": {
            "node_embedding": _cpu_state_dict(agent.critic.q2.node_embedding),
            "gcn_conv": _cpu_state_dict(agent.critic.q2.gcn_conv),
            "gcn_layers": _cpu_state_dict(agent.critic.q2.gcn_layers),
            "l1": _cpu_state_dict(agent.critic.q2.l1),
            "l2": _cpu_state_dict(agent.critic.q2.l2),
            "l3": _cpu_state_dict(agent.critic.q2.l3),
        },
        "cost_critic1": {
            "node_embedding": _cpu_state_dict(agent.cost_critic.q1.node_embedding),
            "gcn_conv": _cpu_state_dict(agent.cost_critic.q1.gcn_conv),
            "gcn_layers": _cpu_state_dict(agent.cost_critic.q1.gcn_layers),
            "l1": _cpu_state_dict(agent.cost_critic.q1.l1),
            "l2": _cpu_state_dict(agent.cost_critic.q1.l2),
            "l3": _cpu_state_dict(agent.cost_critic.q1.l3),
        },
        "cost_critic2": {
            "node_embedding": _cpu_state_dict(agent.cost_critic.q2.node_embedding),
            "gcn_conv": _cpu_state_dict(agent.cost_critic.q2.gcn_conv),
            "gcn_layers": _cpu_state_dict(agent.cost_critic.q2.gcn_layers),
            "l1": _cpu_state_dict(agent.cost_critic.q2.l1),
            "l2": _cpu_state_dict(agent.cost_critic.q2.l2),
            "l3": _cpu_state_dict(agent.cost_critic.q2.l3),
        },
        "target_cost_critic1": {
            "node_embedding": _cpu_state_dict(agent.cost_critic_target.q1.node_embedding),
            "gcn_conv": _cpu_state_dict(agent.cost_critic_target.q1.gcn_conv),
            "gcn_layers": _cpu_state_dict(agent.cost_critic_target.q1.gcn_layers),
            "l1": _cpu_state_dict(agent.cost_critic_target.q1.l1),
            "l2": _cpu_state_dict(agent.cost_critic_target.q1.l2),
            "l3": _cpu_state_dict(agent.cost_critic_target.q1.l3),
        },
        "target_cost_critic2": {
            "node_embedding": _cpu_state_dict(agent.cost_critic_target.q2.node_embedding),
            "gcn_conv": _cpu_state_dict(agent.cost_critic_target.q2.gcn_conv),
            "gcn_layers": _cpu_state_dict(agent.cost_critic_target.q2.gcn_layers),
            "l1": _cpu_state_dict(agent.cost_critic_target.q2.l1),
            "l2": _cpu_state_dict(agent.cost_critic_target.q2.l2),
            "l3": _cpu_state_dict(agent.cost_critic_target.q2.l3),
        },
        "target_critic1": {
            "node_embedding": _cpu_state_dict(agent.critic_target.q1.node_embedding),
            "gcn_conv": _cpu_state_dict(agent.critic_target.q1.gcn_conv),
            "gcn_layers": _cpu_state_dict(agent.critic_target.q1.gcn_layers),
            "l1": _cpu_state_dict(agent.critic_target.q1.l1),
            "l2": _cpu_state_dict(agent.critic_target.q1.l2),
            "l3": _cpu_state_dict(agent.critic_target.q1.l3),
        },
        "target_critic2": {
            "node_embedding": _cpu_state_dict(agent.critic_target.q2.node_embedding),
            "gcn_conv": _cpu_state_dict(agent.critic_target.q2.gcn_conv),
            "gcn_layers": _cpu_state_dict(agent.critic_target.q2.gcn_layers),
            "l1": _cpu_state_dict(agent.critic_target.q2.l1),
            "l2": _cpu_state_dict(agent.critic_target.q2.l2),
            "l3": _cpu_state_dict(agent.critic_target.q2.l3),
        },
        "temperature": {
            "log_alpha": float(agent.log_alpha.detach().cpu().item()),
            "alpha": float(agent.alpha.detach().cpu().item()),
            "log_lambda": float(agent.log_lambda.detach().cpu().item()),
            "lambda": float(agent.lambda_value.detach().cpu().item()),
        },
        "optimizers": {
            "actor": agent.actor_optimizer.state_dict(),
            "critic": agent.critic_optimizer.state_dict(),
            "cost_critic": agent.cost_critic_optimizer.state_dict(),
            "alpha": agent.alpha_optimizer.state_dict(),
            "lambda": agent.lambda_optimizer.state_dict(),
        },
        "replay_buffer": agent.replay_buffer.state_dict(),
    }


def _evaluation_agent_config_keys(checkpoint_format: str) -> tuple[str, ...]:
    common = (
        "park_type",
        "algorithm_variant",
        "enable_federation",
        "privacy_mode",
        "federate_critic_backbone",
    )
    if checkpoint_format in {"mlp_sac_v1", "mlp_td3_v1", "mlp_v1"}:
        return common + ("cp_count", "mlp_hidden_dim")
    if checkpoint_format in {"gnn_sac_v1", "gnn_csac_v1", "sp_rgnn_csac_v1"}:
        keys = common + (
            "feature_dim",
            "decouple_actor_output_heads",
            "gnn_hidden_dim",
            "actor_num_gcn_layers",
            "critic_num_gcn_layers",
        )
        if checkpoint_format == "sp_rgnn_csac_v1":
            keys += (
                "use_relation_gated_fusion",
                "use_critic_typed_pooling",
            )
        return keys
    if checkpoint_format == "hgt_sac_v2":
        return common + (
            "feature_dim",
            "decouple_actor_output_heads",
            "enable_auxiliary_risk_critic",
        )
    if checkpoint_format == "hgt_csac_v1":
        return common + (
            "feature_dim",
            "decouple_actor_output_heads",
        )
    return common


def _evaluation_model_keys(checkpoint_format: str) -> tuple[str, ...]:
    if checkpoint_format in {"mlp_sac_v1", "mlp_td3_v1", "gnn_sac_v1", "hgt_sac_v2"}:
        return ("actor", "critic")
    if checkpoint_format == "sp_rgnn_csac_v1":
        return ("actor", "critic", "cost_critic")
    if checkpoint_format in {"mlp_v1", "gnn_csac_v1", "sp_rgnn_csac_v1", "hgt_csac_v1"}:
        return ("actor", "cost_critic")
    return ("actor",)


def _is_non_export_training_lr_key(key: str) -> bool:
    if key in {"alpha_lr", "entropy_lr"}:
        return True
    return key.endswith("_lr") and (
        key.startswith("actor_")
        or key.startswith("critic_")
        or key.startswith("cost_critic_")
        or key.startswith("risk_critic_")
    )


def _strip_non_export_training_lr_fields(value: object) -> object:
    if isinstance(value, dict):
        cleaned: Dict[object, object] = {}
        for key, item in value.items():
            if isinstance(key, str) and (_is_non_export_training_lr_key(key) or key == "lr"):
                continue
            cleaned[key] = _strip_non_export_training_lr_fields(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_non_export_training_lr_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_non_export_training_lr_fields(item) for item in value)
    return value


def _prepare_checkpoint_for_non_interrupt_model_save(checkpoint: Dict[str, object]) -> Dict[str, object]:
    checkpoint = dict(checkpoint)
    checkpoint["export_kind"] = "evaluation_only"
    checkpoint.pop("optimizers", None)
    checkpoint.pop("replay_buffer", None)
    checkpoint.pop("temperature", None)
    if "agent_config" in checkpoint:
        checkpoint["agent_config"] = _strip_non_export_training_lr_fields(checkpoint["agent_config"])
    return checkpoint


def _serialize_agent_for_non_interrupt_model_save(agent: Any, park_type: str, episode: int) -> Dict[str, object]:
    if hasattr(agent, "export_evaluation_checkpoint"):
        return _prepare_checkpoint_for_non_interrupt_model_save(
            agent.export_evaluation_checkpoint(park_type=park_type, episode=episode)
        )
    checkpoint = _serialize_agent(agent, park_type=park_type, episode=episode)
    checkpoint_format = str(checkpoint.get("checkpoint_format", ""))
    if checkpoint_format not in {
        "mlp_sac_v1",
        "mlp_td3_v1",
        "mlp_v1",
        "gnn_sac_v1",
        "gnn_csac_v1",
        "sp_rgnn_csac_v1",
        "hgt_sac_v2",
        "hgt_csac_v1",
    }:
        return _prepare_checkpoint_for_non_interrupt_model_save(checkpoint)
    agent_config = dict(checkpoint.get("agent_config", {}))
    minimal_agent_config = {
        key: agent_config[key]
        for key in _evaluation_agent_config_keys(checkpoint_format)
        if key in agent_config
    }
    models = dict(checkpoint.get("models", {}))
    minimal_models = {
        key: models[key]
        for key in _evaluation_model_keys(checkpoint_format)
        if key in models
    }
    return _prepare_checkpoint_for_non_interrupt_model_save({
        "checkpoint_format": checkpoint_format,
        "park_type": checkpoint.get("park_type"),
        "episode": checkpoint.get("episode"),
        "agent_config": minimal_agent_config,
        "state_spec": checkpoint.get("state_spec", {}),
        "models": minimal_models,
    })


def _prepare_run_directories(
    root_dir: Path,
    run_name: str,
    enable_federation: bool,
    resume_training: bool,
) -> RunDirectories:
    saved_root = (root_dir / "saved").resolve()
    run_dir = (saved_root / run_name).resolve()
    del resume_training
    log_dir = run_dir / "log"
    results_dir = run_dir / "results"
    models_dir = run_dir / "models"
    model_family_dir = models_dir / ("fed_full" if enable_federation else "local_full")
    full_best_dir = model_family_dir / "best"
    full_final_dir = model_family_dir / "final"
    interrupted_dir = model_family_dir / "interrupted"
    for path in (
        log_dir,
        results_dir,
        full_best_dir,
        full_final_dir,
        interrupted_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return RunDirectories(
        log_dir=log_dir,
        results_dir=results_dir,
        full_best_dir=full_best_dir,
        full_final_dir=full_final_dir,
        interrupted_dir=interrupted_dir,
    )


def _energy_log_fields() -> List[str]:
    fields = ["episode", "step", "time", "weather"]
    per_park_suffixes = [
        "active_ev_count",
        "pv_energy_kwh",
        "raw_bes_action",
        "requested_bes_grid_energy_kwh",
        "cs_projected_bes_grid_energy_kwh",
        "tr_projected_bes_grid_energy_kwh",
        "grid_purchase_energy_kwh",
        "grid_sale_energy_kwh",
        "bes_charge_grid_energy_kwh",
        "bes_discharge_grid_energy_kwh",
        "ev_charge_grid_energy_kwh",
        "ev_discharge_grid_energy_kwh",
        "cs_trunc_charge_kwh",
        "cs_trunc_discharge_kwh",
        "tr_trunc_charge_kwh",
        "tr_trunc_discharge_kwh",
        "departure_debt_energy_kwh",
        "departure_soc_shortfall_energy_kwh",
        "bes_soc",
        "lambda",
    ]
    total_suffixes = [
        "pv_energy_kwh",
        "grid_purchase_energy_kwh",
        "grid_sale_energy_kwh",
        "bes_charge_grid_energy_kwh",
        "bes_discharge_grid_energy_kwh",
        "ev_charge_grid_energy_kwh",
        "ev_discharge_grid_energy_kwh",
        "cs_trunc_charge_kwh",
        "cs_trunc_discharge_kwh",
        "tr_trunc_charge_kwh",
        "tr_trunc_discharge_kwh",
        "departure_debt_energy_kwh",
        "departure_soc_shortfall_energy_kwh",
    ]
    for park_type in PARK_TYPES:
        fields.extend([f"{park_type}_{suffix}" for suffix in per_park_suffixes])
    fields.extend([f"total_{suffix}" for suffix in total_suffixes])
    return fields


def _bes_soc_step_log_fields() -> List[str]:
    fields = ["episode", "step", "time", "weather"]
    for park_type in PARK_TYPES:
        fields.extend(
            [
                f"{park_type}_bes_soc",
                f"{park_type}_bes_soc_delta",
            ]
        )
    return fields


def _current_lambda_snapshot(local_agents: Dict[str, Any]) -> Dict[str, float]:
    return {
        f"{park_type}_lambda": float(local_agents[park_type].lambda_value.detach().cpu().item())
        for park_type in PARK_TYPES
    }


def _reward_log_fields() -> List[str]:
    return [
        "episode",
        "total_grid_purchase_cost",
        "total_grid_sale_revenue",
        "total_v2g_compensation_cost",
        "total_ev_charge_revenue",
        "total_cs_projection_penalty",
        "total_tr_projection_penalty",
        "total_debt_penalty",
        "total_soc_shortfall_penalty",
        "total_bes_terminal_penalty",
    ]


def _training_log_fields() -> List[str]:
    fields = [
        "episode",
        "seed",
        "weather",
        "steps",
        "total_profit_reward",
        "total_constraint_cost",
        "discounted_constraint_cost",
        "mean_step_constraint_cost",
        "total_reward",
        "mean_lambda",
        "total_grid_purchase_cost",
        "total_grid_sale_revenue",
        "total_tr_projection_penalty",
        "total_ev_charge_revenue",
        "total_v2g_compensation_cost",
        "total_cs_projection_penalty",
        "total_user_satisfaction_penalty",
        "total_soc_shortfall_penalty",
        "total_debt_penalty",
        "total_bes_terminal_penalty",
    ]
    fields.extend(
        [
            "fed_actor_scheme",
            "fed_actor_w_RR",
            "fed_actor_w_RO",
            "fed_actor_w_RC",
            "fed_actor_w_OR",
            "fed_actor_w_OO",
            "fed_actor_w_OC",
            "fed_actor_w_CR",
            "fed_actor_w_CO",
            "fed_actor_w_CC",
            "fed_actor_rho_R",
            "fed_actor_rho_O",
            "fed_actor_rho_C",
            "fed_actor_eta_R",
            "fed_actor_eta_O",
            "fed_actor_eta_C",
            "fed_actor_A_RR",
            "fed_actor_A_RO",
            "fed_actor_A_RC",
            "fed_actor_A_OR",
            "fed_actor_A_OO",
            "fed_actor_A_OC",
            "fed_actor_A_CR",
            "fed_actor_A_CO",
            "fed_actor_A_CC",
            "fed_actor_Ac_R",
            "fed_actor_Ac_O",
            "fed_actor_Ac_C",
            "fed_actor_Ac_norm_R",
            "fed_actor_Ac_norm_O",
            "fed_actor_Ac_norm_C",
            "fed_actor_gate_margin_R",
            "fed_actor_gate_margin_O",
            "fed_actor_gate_margin_C",
            "fed_actor_eta_probe_R",
            "fed_actor_eta_probe_O",
            "fed_actor_eta_probe_C",
            "fed_actor_eta_max_R",
            "fed_actor_eta_max_O",
            "fed_actor_eta_max_C",
            "fed_distill_reward_loss",
            "fed_distill_risk_loss",
            "fed_distill_total_loss",
            "fed_distill_reward_loss_post",
            "fed_distill_risk_loss_post",
            "fed_distill_actor_bes_loss",
            "fed_distill_actor_ev_net_loss",
            "fed_distill_actor_total_loss",
            "fed_distill_actor_bes_loss_post",
            "fed_distill_actor_ev_net_loss_post",
            "fed_distill_rounds",
        ]
    )
    for student_park in PARK_TYPES:
        student_label = PARK_LABELS[student_park]
        for teacher_park in PARK_TYPES:
            if teacher_park == student_park:
                continue
            teacher_label = PARK_LABELS[teacher_park]
            fields.extend(
                [
                    f"fed_distill_reward_conf_{student_label}{teacher_label}",
                    f"fed_distill_risk_conf_{student_label}{teacher_label}",
                    f"fed_distill_reward_w_{student_label}{teacher_label}",
                    f"fed_distill_risk_w_{student_label}{teacher_label}",
                ]
            )
    for park_type in PARK_TYPES:
        fields.extend(
            [
                f"{park_type}_reward",
                f"{park_type}_profit_reward",
                f"{park_type}_constraint_cost",
                f"{park_type}_grid_purchase_cost",
                f"{park_type}_grid_sale_revenue",
                f"{park_type}_tr_projection_penalty",
                f"{park_type}_ev_charge_revenue",
                f"{park_type}_v2g_compensation_cost",
                f"{park_type}_cs_projection_penalty",
                f"{park_type}_user_satisfaction_penalty",
                f"{park_type}_soc_shortfall_penalty",
                f"{park_type}_debt_penalty",
                f"{park_type}_bes_terminal_penalty",
            ]
        )
    return fields


def _training_reward_log_fields() -> List[str]:
    fields = [
        "episode",
        "step",
        "time",
        "weather",
        "total_profit_reward",
        "total_constraint_cost",
        "total_training_reward",
        "total_user_satisfaction_penalty",
        "total_cs_projection_penalty",
        "total_tr_projection_penalty",
        "total_bes_terminal_penalty",
        "total_debt_penalty",
    ]
    for park_type in PARK_TYPES:
        fields.extend(
            [
                f"{park_type}_profit_reward",
                f"{park_type}_constraint_cost",
                f"{park_type}_training_reward",
                f"{park_type}_user_satisfaction_penalty",
                f"{park_type}_cs_projection_penalty",
                f"{park_type}_tr_projection_penalty",
                f"{park_type}_bes_terminal_penalty",
                f"{park_type}_debt_penalty",
            ]
        )
    return fields


def _save_local_full_named(target_dir: Path, local_agents: Dict[str, Any], episode: int) -> None:
    for park_type, agent in local_agents.items():
        checkpoint = _serialize_agent_for_non_interrupt_model_save(agent, park_type=park_type, episode=episode)
        torch.save(checkpoint, target_dir / f"{park_type}.pt")


def _save_central_full_named(target_dir: Path, central_agent: Any, episode: int) -> None:
    checkpoint = _serialize_agent_for_non_interrupt_model_save(central_agent, park_type="central", episode=episode)
    torch.save(checkpoint, target_dir / "central.pt")


def _uses_lag_pfed_actor(config: TrainingConfig) -> bool:
    return (
        config.enable_federation
        and not config.enable_fed_distillation
        and not config.federate_critic_backbone
        and config.algorithm_variant in {"hgt_sac", "hgt_csac", "sp_rgnn_csac"}
    )


def _lag_gate_margin_by_park(config: TrainingConfig) -> Dict[str, float]:
    return {
        "residential": float(config.candidate_gate_margin_r),
        "office": float(config.candidate_gate_margin_o),
        "commercial": float(config.candidate_gate_margin_c),
    }


def _lag_eta_probe_by_park(config: TrainingConfig) -> Dict[str, float]:
    return {
        "residential": float(config.eta_probe_r),
        "office": float(config.eta_probe_o),
        "commercial": float(config.eta_probe_c),
    }


def _lag_eta_max_by_park(config: TrainingConfig) -> Dict[str, float]:
    return {
        "residential": float(config.eta_max_r),
        "office": float(config.eta_max_o),
        "commercial": float(config.eta_max_c),
    }


def _empty_fed_actor_metrics() -> Dict[str, object]:
    return {
        "fed_actor_scheme": "",
        "fed_actor_w_RR": "",
        "fed_actor_w_RO": "",
        "fed_actor_w_RC": "",
        "fed_actor_w_OR": "",
        "fed_actor_w_OO": "",
        "fed_actor_w_OC": "",
        "fed_actor_w_CR": "",
        "fed_actor_w_CO": "",
        "fed_actor_w_CC": "",
        "fed_actor_rho_R": "",
        "fed_actor_rho_O": "",
        "fed_actor_rho_C": "",
        "fed_actor_eta_R": "",
        "fed_actor_eta_O": "",
        "fed_actor_eta_C": "",
        "fed_actor_A_RR": "",
        "fed_actor_A_RO": "",
        "fed_actor_A_RC": "",
        "fed_actor_A_OR": "",
        "fed_actor_A_OO": "",
        "fed_actor_A_OC": "",
        "fed_actor_A_CR": "",
        "fed_actor_A_CO": "",
        "fed_actor_A_CC": "",
        "fed_actor_Ac_R": "",
        "fed_actor_Ac_O": "",
        "fed_actor_Ac_C": "",
        "fed_actor_Ac_norm_R": "",
        "fed_actor_Ac_norm_O": "",
        "fed_actor_Ac_norm_C": "",
        "fed_actor_gate_margin_R": "",
        "fed_actor_gate_margin_O": "",
        "fed_actor_gate_margin_C": "",
        "fed_actor_eta_probe_R": "",
        "fed_actor_eta_probe_O": "",
        "fed_actor_eta_probe_C": "",
        "fed_actor_eta_max_R": "",
        "fed_actor_eta_max_O": "",
        "fed_actor_eta_max_C": "",
    }


def _flatten_fed_actor_metrics(metrics: Dict[str, Any] | None, scheme: str) -> Dict[str, object]:
    row = _empty_fed_actor_metrics()
    row["fed_actor_scheme"] = scheme
    if not metrics:
        return row
    weights = metrics.get("fed_weights", [])
    rho = metrics.get("rho", [])
    eta = metrics.get("eta", [])
    source_adv = metrics.get("source_advantage", [])
    candidate_adv = metrics.get("candidate_advantage", [])
    candidate_adv_norm = metrics.get("candidate_advantage_norm", [])
    gate_margin = metrics.get("candidate_gate_margin", [])
    eta_probe = metrics.get("eta_probe", [])
    eta_max = metrics.get("eta_max", [])

    for i, target_label in enumerate(FED_METRIC_PARK_LABELS):
        if i < len(rho):
            row[f"fed_actor_rho_{target_label}"] = rho[i]
        if i < len(eta):
            row[f"fed_actor_eta_{target_label}"] = eta[i]
        if i < len(candidate_adv):
            row[f"fed_actor_Ac_{target_label}"] = candidate_adv[i]
        if i < len(candidate_adv_norm):
            row[f"fed_actor_Ac_norm_{target_label}"] = candidate_adv_norm[i]
        if i < len(gate_margin):
            row[f"fed_actor_gate_margin_{target_label}"] = gate_margin[i]
        if i < len(eta_probe):
            row[f"fed_actor_eta_probe_{target_label}"] = eta_probe[i]
        if i < len(eta_max):
            row[f"fed_actor_eta_max_{target_label}"] = eta_max[i]
        if i < len(weights):
            for j, source_label in enumerate(FED_METRIC_PARK_LABELS):
                if j < len(weights[i]):
                    row[f"fed_actor_w_{target_label}{source_label}"] = weights[i][j]
        if i < len(source_adv):
            for j, source_label in enumerate(FED_METRIC_PARK_LABELS):
                if j < len(source_adv[i]):
                    row[f"fed_actor_A_{target_label}{source_label}"] = source_adv[i][j]
    return row


def _print_fed_actor_metrics(metrics: Dict[str, Any] | None, scheme: str) -> None:
    if not scheme:
        return
    if not metrics:
        print(f"  fed_actor: scheme={scheme} no_aggregate_yet")
        return

    weights = metrics.get("fed_weights", [])
    rho = metrics.get("rho", [])
    eta = metrics.get("eta", [])
    candidate_adv = metrics.get("candidate_advantage", [])
    candidate_adv_norm = metrics.get("candidate_advantage_norm", [])
    gate_margin = metrics.get("candidate_gate_margin", [])
    eta_probe = metrics.get("eta_probe", [])
    eta_max = metrics.get("eta_max", [])
    aggregate_count = metrics.get("aggregate_count", "")

    print(f"  fed_actor: scheme={scheme} aggregate_count={aggregate_count}")
    for i, target_label in enumerate(FED_METRIC_PARK_LABELS):
        row_weights = weights[i] if i < len(weights) else []
        weight_text = ", ".join(
            f"{source_label}={row_weights[j]:.3f}"
            for j, source_label in enumerate(FED_METRIC_PARK_LABELS)
            if j < len(row_weights)
        )
        rho_i = rho[i] if i < len(rho) else 0.0
        eta_i = eta[i] if i < len(eta) else 0.0
        ac_i = candidate_adv[i] if i < len(candidate_adv) else 0.0
        ac_norm_i = candidate_adv_norm[i] if i < len(candidate_adv_norm) else 0.0
        margin_i = gate_margin[i] if i < len(gate_margin) else 0.0
        probe_i = eta_probe[i] if i < len(eta_probe) else 0.0
        eta_max_i = eta_max[i] if i < len(eta_max) else 0.0
        print(
            f"    target_{target_label}: "
            f"weights[{weight_text}] "
            f"rho={rho_i:.3f} eta={eta_i:.4f} "
            f"A_candidate={ac_i:.4f} A_norm={ac_norm_i:.4f} "
            f"margin={margin_i:.3f} probe={probe_i:.4f} eta_max={eta_max_i:.4f}"
        )


def _load_module_blocks(module: torch.nn.Module, blocks: Dict[str, Dict[str, Any]]) -> None:
    module.node_embedding.load_state_dict(blocks["node_embedding"])
    if hasattr(module, "gcn_conv") and "gcn_conv" in blocks:
        module.gcn_conv.load_state_dict(blocks["gcn_conv"])
    if hasattr(module, "gcn_layers") and "gcn_layers" in blocks:
        module.gcn_layers.load_state_dict(blocks["gcn_layers"])
    if hasattr(module, "hgt_conv1") and "hgt_conv1" in blocks:
        module.hgt_conv1.load_state_dict(blocks["hgt_conv1"])
    if hasattr(module, "norm1") and "norm1" in blocks:
        module.norm1.load_state_dict(blocks["norm1"])
    if hasattr(module, "hgt_conv2") and "hgt_conv2" in blocks:
        module.hgt_conv2.load_state_dict(blocks["hgt_conv2"])
    if hasattr(module, "norm2") and "norm2" in blocks:
        module.norm2.load_state_dict(blocks["norm2"])
    if hasattr(module, "mean_linear") and "mean_linear" in blocks:
        module.mean_linear.load_state_dict(blocks["mean_linear"])
    if hasattr(module, "log_std_linear") and "log_std_linear" in blocks:
        module.log_std_linear.load_state_dict(blocks["log_std_linear"])
    if hasattr(module, "mean_head") and "mean_head" in blocks:
        module.mean_head.load_state_dict(blocks["mean_head"])
    if hasattr(module, "log_std_head") and "log_std_head" in blocks:
        module.log_std_head.load_state_dict(blocks["log_std_head"])
    if hasattr(module, "mean_heads") and "mean_heads" in blocks:
        module.mean_heads.load_state_dict(blocks["mean_heads"])
    if hasattr(module, "log_std_heads") and "log_std_heads" in blocks:
        module.log_std_heads.load_state_dict(blocks["log_std_heads"])
    if "l1" in blocks:
        module.l1.load_state_dict(blocks["l1"])
        module.l2.load_state_dict(blocks["l2"])
        module.l3.load_state_dict(blocks["l3"])


def _infer_checkpoint_node_sizes(agent_checkpoint: Dict[str, Any]) -> Dict[str, int]:
    if "state_spec" in agent_checkpoint and "node_sizes" in agent_checkpoint["state_spec"]:
        return {
            key: int(value)
            for key, value in agent_checkpoint["state_spec"]["node_sizes"].items()
        }
    actor_block = agent_checkpoint["actor"]
    return {
        "cs": int(actor_block["node_embedding"]["embeddings.cs.weight"].shape[1]),
        "bes": int(actor_block["node_embedding"]["embeddings.bes.weight"].shape[1]),
        "pv": int(actor_block["node_embedding"]["embeddings.pv.weight"].shape[1]),
        "external": int(actor_block["node_embedding"]["embeddings.external.weight"].shape[1]),
        "ev": int(actor_block["node_embedding"]["embeddings.ev.weight"].shape[1]),
    }


def _restore_agent_from_checkpoint(agent: Any, checkpoint: Dict[str, Any]) -> None:
    if checkpoint.get("checkpoint_kind") == "training_no_replay":
        checkpoint_with_empty_replay = dict(checkpoint)
        checkpoint_with_empty_replay["replay_buffer"] = agent.replay_buffer.state_dict()
        agent.load_checkpoint(checkpoint_with_empty_replay)
        return
    if checkpoint.get("export_kind") == "evaluation_only":
        models = checkpoint["models"]
        agent.actor.load_state_dict(models["actor"])
        if "critic" in models and hasattr(agent, "critic"):
            agent.critic.load_state_dict(models["critic"])
            if hasattr(agent, "critic_target"):
                agent.critic_target.load_state_dict(models["critic"])
        if "cost_critic" in models and hasattr(agent, "cost_critic"):
            agent.cost_critic.load_state_dict(models["cost_critic"])
            if hasattr(agent, "cost_critic_target"):
                agent.cost_critic_target.load_state_dict(models["cost_critic"])
        if "risk_critic" in models and hasattr(agent, "risk_critic"):
            agent.risk_critic.load_state_dict(models["risk_critic"])
            if hasattr(agent, "risk_critic_target"):
                agent.risk_critic_target.load_state_dict(models["risk_critic"])
        agent._sync_inference_actor()
        return
    if checkpoint.get("checkpoint_format") in {
        "mlp_v1",
        "mlp_sac_v1",
        "mlp_td3_v1",
        "hgt_sac_v1",
        "hgt_sac_v2",
        "central_hgt_sac_v1",
        "central_gnn_csac_v1",
        "hgt_csac_v1",
        "gnn_sac_v1",
        "gnn_csac_v1",
        "sp_rgnn_csac_v1",
    }:
        agent.load_checkpoint(checkpoint)
        return
    _load_module_blocks(agent.actor, checkpoint["actor"])
    _load_module_blocks(agent.critic.q1, checkpoint["critic1"])
    _load_module_blocks(agent.critic.q2, checkpoint["critic2"])
    _load_module_blocks(agent.cost_critic.q1, checkpoint["cost_critic1"])
    _load_module_blocks(agent.cost_critic.q2, checkpoint["cost_critic2"])
    _load_module_blocks(agent.critic_target.q1, checkpoint["target_critic1"])
    _load_module_blocks(agent.critic_target.q2, checkpoint["target_critic2"])
    _load_module_blocks(agent.cost_critic_target.q1, checkpoint["target_cost_critic1"])
    _load_module_blocks(agent.cost_critic_target.q2, checkpoint["target_cost_critic2"])
    agent.actor_optimizer.load_state_dict(checkpoint["optimizers"]["actor"])
    agent.critic_optimizer.load_state_dict(checkpoint["optimizers"]["critic"])
    agent.cost_critic_optimizer.load_state_dict(checkpoint["optimizers"]["cost_critic"])
    agent.alpha_optimizer.load_state_dict(checkpoint["optimizers"]["alpha"])
    agent.lambda_optimizer.load_state_dict(checkpoint["optimizers"]["lambda"])
    agent.log_alpha.data.copy_(torch.tensor(checkpoint["temperature"]["log_alpha"], device=agent.device))
    agent.log_lambda.data.copy_(torch.tensor(checkpoint["temperature"]["log_lambda"], device=agent.device))
    agent.replay_buffer.load_state_dict(checkpoint["replay_buffer"])
    agent._sync_inference_actor()


def _interrupt_state_path(run_dirs: RunDirectories) -> Path:
    return run_dirs.interrupted_dir / "training_state.pt"


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _load_interrupt_state(state_path: Path) -> Dict[str, Any] | None:
    try:
        return torch.load(state_path, map_location="cpu", weights_only=False)
    except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as exc:
        print(
            f"resume checkpoint is unreadable and will be ignored: {state_path}\n"
            f"reason: {exc}"
        )
        return None


def _save_interrupt_checkpoint(
    run_dirs: RunDirectories,
    config: TrainingConfig,
    local_agents: Dict[str, Any],
    next_episode: int,
    best_total_reward: float,
    actor_fed_coordinator: Any | None = None,
) -> None:
    state = {
        "config": asdict(config),
        "next_episode": next_episode,
        "best_total_reward": best_total_reward,
        "rng_state": _capture_rng_state(),
        "agents": {
            park_type: _strip_replay_buffer_from_training_checkpoint(
                _serialize_agent(agent, park_type=park_type, episode=max(0, next_episode - 1))
            )
            for park_type, agent in local_agents.items()
        },
    }
    if actor_fed_coordinator is not None and hasattr(actor_fed_coordinator, "export_state"):
        state["actor_fed_coordinator_state"] = actor_fed_coordinator.export_state()
    _atomic_torch_save(state, _interrupt_state_path(run_dirs))


def _save_interrupt_checkpoint_central(
    run_dirs: RunDirectories,
    config: TrainingConfig,
    central_agent: Any,
    next_episode: int,
    best_total_reward: float,
) -> None:
    state = {
        "config": asdict(config),
        "next_episode": next_episode,
        "best_total_reward": best_total_reward,
        "rng_state": _capture_rng_state(),
        "central_agent": _strip_replay_buffer_from_training_checkpoint(
            _serialize_agent(central_agent, park_type="central", episode=max(0, next_episode - 1))
        ),
    }
    _atomic_torch_save(state, _interrupt_state_path(run_dirs))


def _strip_replay_buffer_from_training_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint = dict(checkpoint)
    checkpoint["checkpoint_kind"] = "training_no_replay"
    checkpoint["replay_buffer_omitted"] = True
    checkpoint.pop("replay_buffer", None)
    return checkpoint


def _try_resume_training_state(
    run_dirs: RunDirectories,
    config: TrainingConfig,
    local_agents: Dict[str, Any],
    actor_fed_coordinator: Any | None = None,
) -> tuple[int, float, bool]:
    if not config.resume_training:
        return 0, float("-inf"), False
    state_path = _interrupt_state_path(run_dirs)
    if not state_path.exists():
        print(f"resume requested but checkpoint not found: {state_path}")
        return 0, float("-inf"), False

    state = _load_interrupt_state(state_path)
    if state is None:
        return 0, float("-inf"), False
    saved_config = state.get("config", {})
    saved_privacy_mode = normalize_privacy_mode(saved_config.get("privacy_mode", saved_config.get("state_mode", "strong")))
    if saved_privacy_mode != normalize_privacy_mode(config.privacy_mode):
        raise RuntimeError("resume checkpoint privacy_mode does not match current config")
    if saved_config.get("enable_federation") != config.enable_federation:
        raise RuntimeError("resume checkpoint enable_federation does not match current config")
    if bool(saved_config.get("enable_fed_distillation", False)) != config.enable_fed_distillation:
        raise RuntimeError("resume checkpoint enable_fed_distillation does not match current config")
    if bool(saved_config.get("federate_critic_backbone", False)) != config.federate_critic_backbone:
        raise RuntimeError("resume checkpoint federate_critic_backbone does not match current config")
    if saved_config.get("algorithm_variant", "gnn_csac") != config.algorithm_variant:
        raise RuntimeError("resume checkpoint algorithm_variant does not match current config")
    if bool(saved_config.get("bes_only_mode", False)) != config.bes_only_mode:
        raise RuntimeError("resume checkpoint bes_only_mode does not match current config")
    if saved_config.get("decouple_actor_output_heads", None) != config.decouple_actor_output_heads:
        raise RuntimeError("resume checkpoint decouple_actor_output_heads does not match current config")
    if bool(saved_config.get("use_strong_tr_projection_for_nonprivacy", True)) != bool(
        config.use_strong_tr_projection_for_nonprivacy
    ):
        raise RuntimeError(
            "resume checkpoint use_strong_tr_projection_for_nonprivacy does not match current config"
        )
    current_node_sizes = get_node_sizes(config.privacy_mode)
    checkpoint_node_sizes = _infer_checkpoint_node_sizes(state["agents"][PARK_TYPES[0]])
    if checkpoint_node_sizes != current_node_sizes:
        raise RuntimeError(
            "resume checkpoint node_sizes do not match current code. "
            f"checkpoint={checkpoint_node_sizes}, current={current_node_sizes}. "
            "This usually means you changed state features/dimensions after the checkpoint was saved. "
            "Use a new run_name to restart training, or revert the state definition to the checkpoint version."
        )

    for park_type in PARK_TYPES:
        _restore_agent_from_checkpoint(local_agents[park_type], state["agents"][park_type])
    if actor_fed_coordinator is not None and "actor_fed_coordinator_state" in state:
        actor_fed_coordinator.load_state(state["actor_fed_coordinator_state"])
    if "rng_state" in state:
        _restore_rng_state(state["rng_state"])

    return (
        int(state["next_episode"]),
        float(state.get("best_total_reward", state.get("best_profit_reward", float("-inf")))),
        True,
    )


def _try_resume_training_state_central(
    run_dirs: RunDirectories,
    config: TrainingConfig,
    central_agent: Any,
) -> tuple[int, float, bool]:
    if not config.resume_training:
        return 0, float("-inf"), False
    state_path = _interrupt_state_path(run_dirs)
    if not state_path.exists():
        print(f"resume requested but checkpoint not found: {state_path}")
        return 0, float("-inf"), False

    state = _load_interrupt_state(state_path)
    if state is None:
        return 0, float("-inf"), False
    if "central_agent" not in state:
        raise RuntimeError("resume checkpoint does not contain a central agent state")
    saved_config = state.get("config", {})
    saved_privacy_mode = normalize_privacy_mode(saved_config.get("privacy_mode", saved_config.get("state_mode", "strong")))
    if saved_privacy_mode != normalize_privacy_mode(config.privacy_mode):
        raise RuntimeError("resume checkpoint privacy_mode does not match current config")
    if saved_config.get("algorithm_variant", "gnn_csac") != config.algorithm_variant:
        raise RuntimeError("resume checkpoint algorithm_variant does not match current config")
    if bool(saved_config.get("use_central_tr_hgt_agent", False)) != bool(config.use_central_tr_hgt_agent):
        raise RuntimeError("resume checkpoint use_central_tr_hgt_agent does not match current config")
    if bool(saved_config.get("use_strong_tr_projection_for_nonprivacy", True)) != bool(
        config.use_strong_tr_projection_for_nonprivacy
    ):
        raise RuntimeError(
            "resume checkpoint use_strong_tr_projection_for_nonprivacy does not match current config"
        )
    current_node_sizes = get_central_node_sizes(config.privacy_mode)
    checkpoint_node_sizes = _infer_checkpoint_node_sizes(state["central_agent"])
    if checkpoint_node_sizes != current_node_sizes:
        raise RuntimeError(
            "resume checkpoint node_sizes do not match current central graph definition. "
            f"checkpoint={checkpoint_node_sizes}, current={current_node_sizes}."
        )

    _restore_agent_from_checkpoint(central_agent, state["central_agent"])
    if "rng_state" in state:
        _restore_rng_state(state["rng_state"])
    return (
        int(state["next_episode"]),
        float(state.get("best_total_reward", float("-inf"))),
        True,
    )

def _refresh_training_visualizations(run_dirs: RunDirectories) -> None:
    generate_total_episode_reward_plot(
        training_log_csv=run_dirs.log_dir / "training_log.csv",
        output_path=run_dirs.results_dir / "episode_total_reward_curve.png",
    )
    generate_park_episode_reward_plot(
        training_log_csv=run_dirs.log_dir / "training_log.csv",
        output_path=run_dirs.results_dir / "episode_park_profit_penalty_curve.png",
    )
    generate_episode_reward_components_plot(
        reward_log_csv=run_dirs.log_dir / "reward_log.csv",
        output_path=run_dirs.results_dir / "episode_reward_components.png",
    )


def _run_central_hgt_training(config: TrainingConfig, run_dirs: RunDirectories) -> None:
    if not _uses_central_tr_hgt_agent(config):
        raise RuntimeError("_run_central_hgt_training requires use_central_tr_hgt_agent=True")
    set_global_seed(config.seed)
    env = ThreeParkChargingEnv(seed=config.seed)
    configure_environment(env, config)
    central_agent = build_central_agent(config)
    start_episode, best_total_reward, resumed = _try_resume_training_state_central(
        run_dirs=run_dirs,
        config=config,
        central_agent=central_agent,
    )
    if start_episode >= config.total_episodes:
        print(
            f"resume checkpoint already reached total_episodes: "
            f"start_episode={start_episode}, total_episodes={config.total_episodes}"
        )
        return

    append_logs = resumed and start_episode > 0
    if append_logs:
        _truncate_training_logs_for_resume(run_dirs, start_episode)
    reward_logger = CSVLogger(run_dirs.log_dir / "reward_log.csv", _reward_log_fields(), append=append_logs)
    training_logger = CSVLogger(run_dirs.log_dir / "training_log.csv", _training_log_fields(), append=append_logs)
    bes_soc_step_logger = CSVLogger(
        run_dirs.log_dir / "bes_soc_steps.csv",
        _bes_soc_step_log_fields(),
        append=append_logs,
    )

    try:
        print(
            f"mode=CentralTRHGTSAC "
            f"algorithm_variant={config.algorithm_variant} "
            f"privacy_mode={config.privacy_mode} "
            f"use_strong_tr_projection_for_nonprivacy={config.use_strong_tr_projection_for_nonprivacy} "
            f"decouple_actor_output_heads={config.decouple_actor_output_heads} "
            f"bes_only_mode={config.bes_only_mode} "
            f"resume_training={config.resume_training} "
            f"start_episode={start_episode} "
            f"act_device={config.act_device} "
            f"update_device={config.update_device}"
        )
        for episode in range(start_episode, config.total_episodes):
            episode_seed = config.seed + episode
            obs, reset_info = env.reset(seed=episode_seed)
            previous_bes_soc = {
                park_type: float(env.runtime_states[park_type].bes_soc)
                for park_type in PARK_TYPES
            }
            done = False
            episode_reward = 0.0
            episode_steps = 0
            episode_raw_park_rewards = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_profit_rewards = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_constraint_costs = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_training_profit_rewards = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_training_constraint_costs = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_discounted_training_constraint_costs = {park_type: 0.0 for park_type in PARK_TYPES}
            latest_mean_qcf_pi = 0.0
            episode_lambda_values: List[float] = []
            discounted_constraint_cost = 0.0
            episode_env_metrics: Dict[str, float] = {
                "total_profit_reward": 0.0,
                "total_constraint_cost": 0.0,
                "total_grid_purchase_cost": 0.0,
                "total_grid_sale_revenue": 0.0,
                "total_tr_projection_penalty": 0.0,
                "total_ev_charge_revenue": 0.0,
                "total_v2g_compensation_cost": 0.0,
                "total_cs_projection_penalty": 0.0,
                "total_user_satisfaction_penalty": 0.0,
                "total_soc_shortfall_penalty": 0.0,
                "total_debt_penalty": 0.0,
                "total_bes_terminal_penalty": 0.0,
            }
            for park_type in PARK_TYPES:
                episode_env_metrics.update(
                    {
                        f"{park_type}_grid_purchase_cost": 0.0,
                        f"{park_type}_profit_reward": 0.0,
                        f"{park_type}_constraint_cost": 0.0,
                        f"{park_type}_grid_sale_revenue": 0.0,
                        f"{park_type}_tr_projection_penalty": 0.0,
                        f"{park_type}_ev_charge_revenue": 0.0,
                        f"{park_type}_v2g_compensation_cost": 0.0,
                        f"{park_type}_cs_projection_penalty": 0.0,
                        f"{park_type}_user_satisfaction_penalty": 0.0,
                        f"{park_type}_soc_shortfall_penalty": 0.0,
                        f"{park_type}_debt_penalty": 0.0,
                        f"{park_type}_bes_terminal_penalty": 0.0,
                    }
                )

            while not done:
                central_obs = build_central_tr_graph(obs)
                joint_action, raw_node_action, raw_node_actions = central_agent.act(
                    central_obs,
                    deterministic=False,
                    return_node_action=True,
                )
                next_obs, reward, terminated, truncated, info = env.step(
                    joint_action,
                    raw_node_actions=raw_node_actions,
                )
                done = terminated or truncated
                next_central_obs = build_central_tr_graph(next_obs)
                central_agent.store_transition(
                    obs=central_obs,
                    action=raw_node_action,
                    reward=float(reward),
                    cost=float(info["reward_breakdown"]["constraint_cost"]),
                    next_obs=next_central_obs,
                    done=done,
                )

                for park_type in PARK_TYPES:
                    episode_raw_park_rewards[park_type] += info["park_reward_breakdown"][park_type]["total_reward"]
                    episode_training_profit_rewards[park_type] += info["park_reward_breakdown"][park_type]["profit_reward"]
                    episode_training_constraint_costs[park_type] += info["park_reward_breakdown"][park_type]["constraint_cost"]
                    episode_profit_rewards[park_type] += info["park_reward_breakdown"][park_type]["logging_profit_reward"]
                    episode_constraint_costs[park_type] += info["park_reward_breakdown"][park_type]["logging_constraint_cost"]
                    episode_discounted_training_constraint_costs[park_type] += (
                        (config.gamma ** episode_steps)
                        * float(info["park_reward_breakdown"][park_type]["constraint_cost"])
                    )

                reward_metrics = info["reward_log"]
                reward_row = {
                    "episode": episode,
                    **{field: reward_metrics[field] for field in _reward_log_fields() if field != "episode"},
                }
                reward_logger.write_row(reward_row)
                energy_metrics = info["energy_log"]
                bes_soc_step_row: Dict[str, object] = {
                    "episode": episode,
                    "step": energy_metrics["step"],
                    "time": energy_metrics["time"],
                    "weather": energy_metrics["weather"],
                }
                for park_type in PARK_TYPES:
                    current_bes_soc = float(energy_metrics[f"{park_type}_bes_soc"])
                    bes_soc_step_row[f"{park_type}_bes_soc"] = current_bes_soc
                    bes_soc_step_row[f"{park_type}_bes_soc_delta"] = current_bes_soc - previous_bes_soc[park_type]
                    previous_bes_soc[park_type] = current_bes_soc
                bes_soc_step_logger.write_row(bes_soc_step_row)

                episode_env_metrics["total_profit_reward"] += float(reward_metrics["total_profit_reward"])
                episode_env_metrics["total_constraint_cost"] += float(reward_metrics["total_constraint_cost"])
                discounted_constraint_cost += (config.gamma ** episode_steps) * float(reward_metrics["total_constraint_cost"])
                episode_env_metrics["total_grid_purchase_cost"] += float(reward_metrics["total_grid_purchase_cost"])
                episode_env_metrics["total_grid_sale_revenue"] += float(reward_metrics["total_grid_sale_revenue"])
                episode_env_metrics["total_tr_projection_penalty"] += float(reward_metrics["total_tr_projection_penalty"])
                episode_env_metrics["total_ev_charge_revenue"] += float(reward_metrics["total_ev_charge_revenue"])
                episode_env_metrics["total_v2g_compensation_cost"] += float(reward_metrics["total_v2g_compensation_cost"])
                episode_env_metrics["total_cs_projection_penalty"] += float(reward_metrics["total_cs_projection_penalty"])
                episode_env_metrics["total_user_satisfaction_penalty"] += float(reward_metrics["total_soc_shortfall_penalty"])
                episode_env_metrics["total_soc_shortfall_penalty"] += float(reward_metrics["total_soc_shortfall_penalty"])
                episode_env_metrics["total_debt_penalty"] += float(reward_metrics["total_debt_penalty"])
                episode_env_metrics["total_bes_terminal_penalty"] += float(reward_metrics["total_bes_terminal_penalty"])
                for park_type in PARK_TYPES:
                    episode_env_metrics[f"{park_type}_profit_reward"] += float(reward_metrics[f"{park_type}_profit_reward"])
                    episode_env_metrics[f"{park_type}_constraint_cost"] += float(reward_metrics[f"{park_type}_constraint_cost"])
                    episode_env_metrics[f"{park_type}_grid_purchase_cost"] += float(reward_metrics[f"{park_type}_grid_purchase_cost"])
                    episode_env_metrics[f"{park_type}_grid_sale_revenue"] += float(reward_metrics[f"{park_type}_grid_sale_revenue"])
                    episode_env_metrics[f"{park_type}_tr_projection_penalty"] += float(reward_metrics[f"{park_type}_tr_projection_penalty"])
                    episode_env_metrics[f"{park_type}_ev_charge_revenue"] += float(reward_metrics[f"{park_type}_ev_charge_revenue"])
                    episode_env_metrics[f"{park_type}_v2g_compensation_cost"] += float(reward_metrics[f"{park_type}_v2g_compensation_cost"])
                    episode_env_metrics[f"{park_type}_cs_projection_penalty"] += float(reward_metrics[f"{park_type}_cs_projection_penalty"])
                    episode_env_metrics[f"{park_type}_user_satisfaction_penalty"] += float(reward_metrics[f"{park_type}_soc_shortfall_penalty"])
                    episode_env_metrics[f"{park_type}_soc_shortfall_penalty"] += float(reward_metrics[f"{park_type}_soc_shortfall_penalty"])
                    episode_env_metrics[f"{park_type}_debt_penalty"] += float(reward_metrics[f"{park_type}_debt_penalty"])
                    episode_env_metrics[f"{park_type}_bes_terminal_penalty"] += float(reward_metrics[f"{park_type}_bes_terminal_penalty"])

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                if episode_steps % config.update_every_steps == 0:
                    for _ in range(config.gradient_steps_per_update):
                        update_metrics = central_agent.update()
                        episode_lambda_values.append(update_metrics["lambda_value"])
                        latest_mean_qcf_pi = float(update_metrics["mean_qcf_pi"])

            should_save_final = (
                ((episode + 1) % config.final_save_interval_episodes == 0)
                or (episode == config.total_episodes - 1)
            )
            if should_save_final:
                _save_central_full_named(run_dirs.full_final_dir, central_agent, episode)

            mean_step_constraint_cost = episode_env_metrics["total_constraint_cost"] / max(episode_steps, 1)
            total_reward_from_parks = sum(episode_raw_park_rewards.values())
            if total_reward_from_parks > best_total_reward + 1e-9:
                best_total_reward = total_reward_from_parks
                _save_central_full_named(run_dirs.full_best_dir, central_agent, episode)
            if abs(episode_reward - total_reward_from_parks) > 1e-6:
                raise RuntimeError(
                    "episode total reward is not equal to the sum of park rewards: "
                    f"episode_reward={episode_reward}, park_sum={total_reward_from_parks}"
                )

            training_row: Dict[str, object] = {
                "episode": episode,
                "seed": episode_seed,
                "weather": reset_info["weather"],
                "steps": episode_steps,
                "total_profit_reward": episode_env_metrics["total_profit_reward"],
                "total_constraint_cost": episode_env_metrics["total_constraint_cost"],
                "discounted_constraint_cost": discounted_constraint_cost,
                "mean_step_constraint_cost": mean_step_constraint_cost,
                "total_reward": total_reward_from_parks,
                "mean_lambda": (
                    sum(episode_lambda_values) / len(episode_lambda_values)
                    if episode_lambda_values
                    else float(central_agent.lambda_value.detach().cpu().item())
                ),
            }
            training_row.update(episode_env_metrics)
            for park_type in PARK_TYPES:
                training_row[f"{park_type}_reward"] = episode_raw_park_rewards[park_type]
                training_row[f"{park_type}_profit_reward"] = episode_profit_rewards[park_type]
                training_row[f"{park_type}_constraint_cost"] = episode_constraint_costs[park_type]
            training_row.update(_empty_fed_actor_metrics())
            training_row.update(_empty_fed_distill_metrics())
            training_logger.write_row(training_row)
            reward_logger.flush()
            training_logger.flush()
            bes_soc_step_logger.flush()

            if ((episode + 1) % config.interrupted_save_interval_episodes == 0) or (episode == config.total_episodes - 1):
                _save_interrupt_checkpoint_central(
                    run_dirs=run_dirs,
                    config=config,
                    central_agent=central_agent,
                    next_episode=episode + 1,
                    best_total_reward=best_total_reward,
                )
            if (episode + 1) % 5 == 0:
                _refresh_training_visualizations(run_dirs)

            print(
                f"episode={episode:03d} "
                f"system_total_reward={total_reward_from_parks:.4f} "
                f"system_profit={episode_env_metrics['total_profit_reward']:.4f} "
                f"system_penalty={episode_env_metrics['total_constraint_cost']:.4f} "
                f"system_discounted_cost={discounted_constraint_cost:.4f} "
                f"mean_step_constraint_cost={mean_step_constraint_cost:.4f}"
            )
            print(
                f"  central: "
                f"lambda={float(central_agent.lambda_value.detach().cpu().item()):.4f} "
                f"mean_qcf_pi={latest_mean_qcf_pi:.4f} "
                f"(d={config.d:.4f})"
            )
            for park_type in PARK_TYPES:
                print(
                    f"  {park_type}: "
                    f"train_profit={episode_training_profit_rewards[park_type]:.4f} "
                    f"train_penalty={episode_training_constraint_costs[park_type]:.4f} "
                    f"log_profit={episode_profit_rewards[park_type]:.4f} "
                    f"log_penalty={episode_constraint_costs[park_type]:.4f} "
                    f"discounted_training_cost={episode_discounted_training_constraint_costs[park_type]:.4f}"
                )

        reward_logger.flush()
        training_logger.flush()
        bes_soc_step_logger.flush()
        _refresh_training_visualizations(run_dirs)
    except KeyboardInterrupt:
        print(
            "training interrupted; resume checkpoint is saved every "
            f"{config.interrupted_save_interval_episodes} episodes and at the final episode boundary"
        )
        raise
    finally:
        reward_logger.close()
        training_logger.close()
        bes_soc_step_logger.close()

def run_training(config: TrainingConfig) -> None:
    resolved_act_device = resolve_compute_device(config.act_device)
    resolved_update_device = resolve_compute_device(config.update_device)
    config = TrainingConfig(
        run_name=config.run_name,
        algorithm_variant=config.algorithm_variant,
        enable_federation=config.enable_federation,
        federate_critic_backbone=config.federate_critic_backbone,
        privacy_mode=config.privacy_mode,
        enable_fed_distillation=config.enable_fed_distillation,
        use_central_tr_hgt_agent=config.use_central_tr_hgt_agent,
        decouple_actor_output_heads=config.decouple_actor_output_heads,
        bes_only_mode=config.bes_only_mode,
        resume_training=config.resume_training,
        seed=config.seed,
        deterministic_training=config.deterministic_training,
        total_episodes=config.total_episodes,
        federated_warmup_episodes=config.federated_warmup_episodes,
        federation_early_phase_end_episode=config.federation_early_phase_end_episode,
        federation_mid_phase_end_episode=config.federation_mid_phase_end_episode,
        federation_early_phase_interval=config.federation_early_phase_interval,
        federation_mid_phase_interval=config.federation_mid_phase_interval,
        federation_late_phase_interval=config.federation_late_phase_interval,
        fed_logits_lr=config.fed_logits_lr,
        rho_logits_lr=config.rho_logits_lr,
        fed_logits_diag_init=config.fed_logits_diag_init,
        fed_logits_offdiag_init=config.fed_logits_offdiag_init,
        rho_init=config.rho_init,
        fed_distill_warmup_episodes=config.fed_distill_warmup_episodes,
        fed_distill_interval_episodes=config.fed_distill_interval_episodes,
        fed_distill_batch_size=config.fed_distill_batch_size,
        fed_distill_num_candidates=config.fed_distill_num_candidates,
        fed_distill_reward_temperature=config.fed_distill_reward_temperature,
        fed_distill_risk_temperature=config.fed_distill_risk_temperature,
        fed_distill_reward_weight=config.fed_distill_reward_weight,
        fed_distill_risk_weight=config.fed_distill_risk_weight,
        strong_nonfed_fed_actor_backbone_lr_before_fed_start=config.strong_nonfed_fed_actor_backbone_lr_before_fed_start,
        strong_nonfed_fed_actor_backbone_lr_after_fed_start=config.strong_nonfed_fed_actor_backbone_lr_after_fed_start,
        strong_nonfed_fed_actor_local_backbone_lr_before_fed_start=config.strong_nonfed_fed_actor_local_backbone_lr_before_fed_start,
        strong_nonfed_fed_actor_local_backbone_lr_after_fed_start=config.strong_nonfed_fed_actor_local_backbone_lr_after_fed_start,
        strong_nonfed_fed_critic_backbone_lr_before_fed_start=config.strong_nonfed_fed_critic_backbone_lr_before_fed_start,
        strong_nonfed_fed_critic_backbone_lr_after_fed_start=config.strong_nonfed_fed_critic_backbone_lr_after_fed_start,
        strong_nonfed_fed_actor_head_lr_before_fed_start=config.strong_nonfed_fed_actor_head_lr_before_fed_start,
        strong_nonfed_fed_actor_head_lr_after_fed_start=config.strong_nonfed_fed_actor_head_lr_after_fed_start,
        strong_nonfed_fed_critic_head_lr_before_fed_start=config.strong_nonfed_fed_critic_head_lr_before_fed_start,
        strong_nonfed_fed_critic_head_lr_after_fed_start=config.strong_nonfed_fed_critic_head_lr_after_fed_start,
        sp_rgnn_actor_backbone_lr_before_fed_start=config.sp_rgnn_actor_backbone_lr_before_fed_start,
        sp_rgnn_actor_backbone_lr_after_fed_start=config.sp_rgnn_actor_backbone_lr_after_fed_start,
        sp_rgnn_critic_backbone_lr_before_fed_start=config.sp_rgnn_critic_backbone_lr_before_fed_start,
        sp_rgnn_critic_backbone_lr_after_fed_start=config.sp_rgnn_critic_backbone_lr_after_fed_start,
        sp_rgnn_actor_head_lr_before_fed_start=config.sp_rgnn_actor_head_lr_before_fed_start,
        sp_rgnn_actor_head_lr_after_fed_start=config.sp_rgnn_actor_head_lr_after_fed_start,
        sp_rgnn_critic_head_lr_before_fed_start=config.sp_rgnn_critic_head_lr_before_fed_start,
        sp_rgnn_critic_head_lr_after_fed_start=config.sp_rgnn_critic_head_lr_after_fed_start,
        critic_federated_warmup_episodes=config.critic_federated_warmup_episodes,
        critic_federation_early_phase_end_episode=config.critic_federation_early_phase_end_episode,
        critic_federation_mid_phase_end_episode=config.critic_federation_mid_phase_end_episode,
        critic_federation_early_phase_interval=config.critic_federation_early_phase_interval,
        critic_federation_mid_phase_interval=config.critic_federation_mid_phase_interval,
        critic_federation_late_phase_interval=config.critic_federation_late_phase_interval,
        update_every_steps=config.update_every_steps,
        gradient_steps_per_update=config.gradient_steps_per_update,
        act_device=resolved_act_device,
        update_device=resolved_update_device,
        alpha_lr=config.alpha_lr,
        gamma=config.gamma,
        tau=config.tau,
        batch_size=config.batch_size,
        replay_size=config.replay_size,
        target_entropy_scale=config.target_entropy_scale,
        actor_proximal_weight=config.actor_proximal_weight,
        critic_proximal_weight=config.critic_proximal_weight,
        d=config.d,
        lambda_lr=config.lambda_lr,
        use_strong_tr_projection_for_nonprivacy=config.use_strong_tr_projection_for_nonprivacy,
        tr_probe_ratio_1=config.tr_probe_ratio_1,
        tr_probe_ratio_2=config.tr_probe_ratio_2,
        tr_curvature_weight=config.tr_curvature_weight,
        tr_overload_penalty_weight=config.tr_overload_penalty_weight,
        interrupted_save_interval_episodes=config.interrupted_save_interval_episodes,
        final_save_interval_episodes=config.final_save_interval_episodes,
    )
    configure_torch_determinism(config.deterministic_training)
    validate_training_config(config)
    run_dirs = _prepare_run_directories(
        root_dir=Path(__file__).resolve().parent,
        run_name=config.run_name,
        enable_federation=config.enable_federation,
        resume_training=config.resume_training,
    )
    if _uses_central_tr_hgt_agent(config):
        _run_central_hgt_training(config, run_dirs)
        return

    set_global_seed(config.seed)
    env = ThreeParkChargingEnv(seed=config.seed)
    configure_environment(env, config)
    local_agents = build_local_agents(config)
    actor_fed_coordinator = None
    critic_fed_coordinator = None
    parameter_federation_enabled = config.enable_federation and not _fed_distillation_enabled(config)
    if parameter_federation_enabled:
        actor_fed_config = FederatedConfig(
            warmup_episodes=config.federated_warmup_episodes,
            early_phase_end_episode=config.federation_early_phase_end_episode,
            mid_phase_end_episode=config.federation_mid_phase_end_episode,
            early_phase_interval=config.federation_early_phase_interval,
            mid_phase_interval=config.federation_mid_phase_interval,
            late_phase_interval=config.federation_late_phase_interval,
        )
        if _uses_lag_pfed_actor(config):
            actor_fed_coordinator = LearnablePersonalizedFedActorCoordinator(
                config=actor_fed_config,
                park_ids=list(PARK_TYPES),
                fed_logits_lr=config.fed_logits_lr,
                rho_logits_lr=config.rho_logits_lr,
                fed_logits_diag_init=config.fed_logits_diag_init,
                fed_logits_offdiag_init=config.fed_logits_offdiag_init,
                rho_init=config.rho_init,
                candidate_gate_margin=_lag_gate_margin_by_park(config),
                eta_probe=_lag_eta_probe_by_park(config),
                eta_max=_lag_eta_max_by_park(config),
                device=config.update_device,
            )
        else:
            actor_fed_coordinator = FederatedAveragingCoordinator(actor_fed_config)
        critic_fed_coordinator = FederatedAveragingCoordinator(
            FederatedConfig(
                warmup_episodes=config.critic_federated_warmup_episodes,
                early_phase_end_episode=config.critic_federation_early_phase_end_episode,
                mid_phase_end_episode=config.critic_federation_mid_phase_end_episode,
                early_phase_interval=config.critic_federation_early_phase_interval,
                mid_phase_interval=config.critic_federation_mid_phase_interval,
                late_phase_interval=config.critic_federation_late_phase_interval,
            )
        )
    start_episode, best_total_reward, resumed = _try_resume_training_state(
        run_dirs=run_dirs,
        config=config,
        local_agents=local_agents,
        actor_fed_coordinator=actor_fed_coordinator,
    )
    _apply_hgt_head_lr_phase(local_agents, config, episode=start_episode)
    _apply_sp_rgnn_lr_phase(local_agents, config, episode=start_episode)
    if start_episode >= config.total_episodes:
        print(
            f"resume checkpoint already reached total_episodes: "
            f"start_episode={start_episode}, total_episodes={config.total_episodes}"
        )
        return
    env.attach_local_agents(local_agents)

    append_logs = resumed and start_episode > 0
    if append_logs:
        _truncate_training_logs_for_resume(run_dirs, start_episode)
    reward_logger = CSVLogger(run_dirs.log_dir / "reward_log.csv", _reward_log_fields(), append=append_logs)
    training_logger = CSVLogger(run_dirs.log_dir / "training_log.csv", _training_log_fields(), append=append_logs)
    bes_soc_step_logger = CSVLogger(
        run_dirs.log_dir / "bes_soc_steps.csv",
        _bes_soc_step_log_fields(),
        append=append_logs,
    )
    latest_actor_fed_metrics: Dict[str, Any] | None = None
    latest_fed_distill_metrics: Dict[str, float] = _empty_fed_distill_metrics()

    try:
        if config.algorithm_variant == "gnn_sac":
            training_mode = "FedGNNSAC" if config.enable_federation else "LocalGNNSAC"
        elif config.algorithm_variant == "mlp_sac":
            training_mode = "FedMLPSAC" if config.enable_federation else "LocalMLPSAC"
        elif config.algorithm_variant == "mlp_td3":
            training_mode = "LocalMLPTD3"
        elif config.algorithm_variant == "gnn_csac":
            training_mode = "FedGNNCSAC" if config.enable_federation else "LocalGNNCSAC"
        elif config.algorithm_variant in SP_RGNN_CSAC_VARIANTS:
            training_mode = "FedSPRGNNCSAC" if config.enable_federation else "LocalSPRGNNCSAC"
        elif config.algorithm_variant == "hgt_sac":
            training_mode = "FedDistillHGTSAC" if _fed_distillation_enabled(config) else ("FedHGTSAC" if config.enable_federation else "LocalHGTSAC")
        elif config.algorithm_variant == "hgt_csac":
            training_mode = "FedHGTCSAC" if config.enable_federation else "LocalHGTCSAC"
        elif config.algorithm_variant == "mlp_csac":
            training_mode = "FedMLPCSAC" if config.enable_federation else "LocalMLPCSAC"
        else:
            training_mode = config.algorithm_variant
        print(
            f"mode={training_mode} "
            f"algorithm_variant={config.algorithm_variant} "
            f"privacy_mode={config.privacy_mode} "
            f"enable_fed_distillation={config.enable_fed_distillation} "
            f"enable_fed_distill_actor={config.enable_fed_distill_actor} "
            f"enable_federation={config.enable_federation} "
            f"decouple_actor_output_heads={config.decouple_actor_output_heads} "
            f"federate_critic_backbone={config.federate_critic_backbone} "
            f"bes_only_mode={config.bes_only_mode} "
            f"resume_training={config.resume_training} "
            f"start_episode={start_episode} "
            f"act_device={config.act_device} "
            f"update_device={config.update_device}"
        )
        for episode in range(start_episode, config.total_episodes):
            _apply_hgt_head_lr_phase(local_agents, config, episode=episode)
            _apply_sp_rgnn_lr_phase(local_agents, config, episode=episode)
            latest_fed_distill_metrics = _empty_fed_distill_metrics()
            episode_seed = config.seed + episode
            obs, reset_info = env.reset(seed=episode_seed)
            previous_bes_soc = {
                park_type: float(env.runtime_states[park_type].bes_soc)
                for park_type in PARK_TYPES
            }
            done = False
            episode_reward = 0.0
            episode_steps = 0
            episode_raw_park_rewards = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_profit_rewards = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_constraint_costs = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_training_profit_rewards = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_training_constraint_costs = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_discounted_training_constraint_costs = {park_type: 0.0 for park_type in PARK_TYPES}
            latest_mean_qcf_pi = {park_type: 0.0 for park_type in PARK_TYPES}
            episode_lambda_values: List[float] = []
            discounted_constraint_cost = 0.0
            episode_env_metrics: Dict[str, float] = {
                "total_profit_reward": 0.0,
                "total_constraint_cost": 0.0,
                "total_grid_purchase_cost": 0.0,
                "total_grid_sale_revenue": 0.0,
                "total_tr_projection_penalty": 0.0,
                "total_ev_charge_revenue": 0.0,
                "total_v2g_compensation_cost": 0.0,
                "total_cs_projection_penalty": 0.0,
                "total_user_satisfaction_penalty": 0.0,
                "total_soc_shortfall_penalty": 0.0,
                "total_debt_penalty": 0.0,
                "total_bes_terminal_penalty": 0.0,
            }
            for park_type in PARK_TYPES:
                episode_env_metrics.update(
                    {
                        f"{park_type}_grid_purchase_cost": 0.0,
                        f"{park_type}_profit_reward": 0.0,
                        f"{park_type}_constraint_cost": 0.0,
                        f"{park_type}_grid_sale_revenue": 0.0,
                        f"{park_type}_tr_projection_penalty": 0.0,
                        f"{park_type}_ev_charge_revenue": 0.0,
                        f"{park_type}_v2g_compensation_cost": 0.0,
                        f"{park_type}_cs_projection_penalty": 0.0,
                        f"{park_type}_user_satisfaction_penalty": 0.0,
                        f"{park_type}_soc_shortfall_penalty": 0.0,
                        f"{park_type}_debt_penalty": 0.0,
                        f"{park_type}_bes_terminal_penalty": 0.0,
                    }
                )
            while not done:
                joint_action, raw_node_actions = build_joint_action(
                    local_agents,
                    obs,
                    deterministic=False,
                    return_raw_action=True,
                )
                next_obs, reward, terminated, truncated, info = env.step(
                    joint_action,
                    raw_node_actions=raw_node_actions,
                )
                done = terminated or truncated

                for park_type in PARK_TYPES:
                    if config.algorithm_variant in {"gnn_sac", "mlp_sac", "mlp_td3", "hgt_sac"}:
                        transition_reward = info["park_reward_breakdown"][park_type]["training_total_reward"]
                    else:
                        transition_reward = info["park_reward_breakdown"][park_type]["profit_reward"]
                    park_breakdown = info["park_reward_breakdown"][park_type]
                    local_agents[park_type].store_transition(
                        obs=obs["park_graphs"][park_type],
                        action=raw_node_actions[park_type],
                        reward=transition_reward,
                        cost=park_breakdown["constraint_cost"],
                        next_obs=next_obs["park_graphs"][park_type],
                        done=done,
                    )
                    episode_raw_park_rewards[park_type] += info["park_reward_breakdown"][park_type]["total_reward"]
                    episode_training_profit_rewards[park_type] += info["park_reward_breakdown"][park_type]["profit_reward"]
                    episode_training_constraint_costs[park_type] += info["park_reward_breakdown"][park_type]["constraint_cost"]
                    episode_profit_rewards[park_type] += info["park_reward_breakdown"][park_type]["logging_profit_reward"]
                    episode_constraint_costs[park_type] += info["park_reward_breakdown"][park_type]["logging_constraint_cost"]
                    episode_discounted_training_constraint_costs[park_type] += (
                        (config.gamma ** episode_steps)
                        * float(info["park_reward_breakdown"][park_type]["constraint_cost"])
                    )
                reward_metrics = info["reward_log"]
                reward_row = {
                    "episode": episode,
                    **{field: reward_metrics[field] for field in _reward_log_fields() if field != "episode"},
                }
                reward_logger.write_row(reward_row)
                energy_metrics = info["energy_log"]
                bes_soc_step_row: Dict[str, object] = {
                    "episode": episode,
                    "step": energy_metrics["step"],
                    "time": energy_metrics["time"],
                    "weather": energy_metrics["weather"],
                }
                for park_type in PARK_TYPES:
                    current_bes_soc = float(energy_metrics[f"{park_type}_bes_soc"])
                    bes_soc_step_row[f"{park_type}_bes_soc"] = current_bes_soc
                    bes_soc_step_row[f"{park_type}_bes_soc_delta"] = (
                        current_bes_soc - previous_bes_soc[park_type]
                    )
                    previous_bes_soc[park_type] = current_bes_soc
                bes_soc_step_logger.write_row(bes_soc_step_row)

                episode_env_metrics["total_profit_reward"] += float(reward_metrics["total_profit_reward"])
                episode_env_metrics["total_constraint_cost"] += float(reward_metrics["total_constraint_cost"])
                discounted_constraint_cost += (
                    (config.gamma ** episode_steps) * float(reward_metrics["total_constraint_cost"])
                )
                episode_env_metrics["total_grid_purchase_cost"] += float(reward_metrics["total_grid_purchase_cost"])
                episode_env_metrics["total_grid_sale_revenue"] += float(reward_metrics["total_grid_sale_revenue"])
                episode_env_metrics["total_tr_projection_penalty"] += float(reward_metrics["total_tr_projection_penalty"])
                episode_env_metrics["total_ev_charge_revenue"] += float(reward_metrics["total_ev_charge_revenue"])
                episode_env_metrics["total_v2g_compensation_cost"] += float(reward_metrics["total_v2g_compensation_cost"])
                episode_env_metrics["total_cs_projection_penalty"] += float(reward_metrics["total_cs_projection_penalty"])
                episode_env_metrics["total_user_satisfaction_penalty"] += float(reward_metrics["total_soc_shortfall_penalty"])
                episode_env_metrics["total_soc_shortfall_penalty"] += float(reward_metrics["total_soc_shortfall_penalty"])
                episode_env_metrics["total_debt_penalty"] += float(reward_metrics["total_debt_penalty"])
                episode_env_metrics["total_bes_terminal_penalty"] += float(reward_metrics["total_bes_terminal_penalty"])
                for park_type in PARK_TYPES:
                    episode_env_metrics[f"{park_type}_profit_reward"] += float(reward_metrics[f"{park_type}_profit_reward"])
                    episode_env_metrics[f"{park_type}_constraint_cost"] += float(reward_metrics[f"{park_type}_constraint_cost"])
                    episode_env_metrics[f"{park_type}_grid_purchase_cost"] += float(reward_metrics[f"{park_type}_grid_purchase_cost"])
                    episode_env_metrics[f"{park_type}_grid_sale_revenue"] += float(reward_metrics[f"{park_type}_grid_sale_revenue"])
                    episode_env_metrics[f"{park_type}_tr_projection_penalty"] += float(reward_metrics[f"{park_type}_tr_projection_penalty"])
                    episode_env_metrics[f"{park_type}_ev_charge_revenue"] += float(reward_metrics[f"{park_type}_ev_charge_revenue"])
                    episode_env_metrics[f"{park_type}_v2g_compensation_cost"] += float(reward_metrics[f"{park_type}_v2g_compensation_cost"])
                    episode_env_metrics[f"{park_type}_cs_projection_penalty"] += float(reward_metrics[f"{park_type}_cs_projection_penalty"])
                    episode_env_metrics[f"{park_type}_user_satisfaction_penalty"] += float(reward_metrics[f"{park_type}_soc_shortfall_penalty"])
                    episode_env_metrics[f"{park_type}_soc_shortfall_penalty"] += float(reward_metrics[f"{park_type}_soc_shortfall_penalty"])
                    episode_env_metrics[f"{park_type}_debt_penalty"] += float(reward_metrics[f"{park_type}_debt_penalty"])
                    episode_env_metrics[f"{park_type}_bes_terminal_penalty"] += float(reward_metrics[f"{park_type}_bes_terminal_penalty"])

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                if episode_steps % config.update_every_steps == 0:
                    for park_type in PARK_TYPES:
                        for _ in range(config.gradient_steps_per_update):
                            update_metrics = local_agents[park_type].update()
                            episode_lambda_values.append(update_metrics["lambda_value"])
                            latest_mean_qcf_pi[park_type] = float(update_metrics["mean_qcf_pi"])

            if _should_run_fed_distillation(episode, config):
                latest_fed_distill_metrics = _run_fed_distillation_round(local_agents, config)

            if actor_fed_coordinator is not None and actor_fed_coordinator.should_aggregate(episode):
                if _uses_lag_pfed_actor(config):
                    latest_actor_fed_metrics = actor_fed_coordinator.aggregate(local_agents)
                else:
                    uniform_actor_weights = {
                        park_id: 1.0 / len(PARK_TYPES)
                        for park_id in PARK_TYPES
                    }
                    actor_fed_coordinator.aggregate(
                        local_agents,
                        normalized_weights=uniform_actor_weights,
                        block_weights={"actor_backbone": uniform_actor_weights},
                        selected_blocks=["actor_backbone"],
                    )
                    latest_actor_fed_metrics = None
            if (
                critic_fed_coordinator is not None
                and config.federate_critic_backbone
                and critic_fed_coordinator.should_aggregate(episode)
            ):
                uniform_actor_weights = {
                    park_id: 1.0 / len(PARK_TYPES)
                    for park_id in PARK_TYPES
                }
                critic_blocks = ["critic_backbone"]
                block_weights = {
                    "critic_backbone": uniform_actor_weights,
                }
                if config.algorithm_variant in {"gnn_csac", *SP_RGNN_CSAC_VARIANTS, "mlp_csac", "hgt_csac"}:
                    critic_blocks.append("cost_critic_backbone")
                    block_weights["cost_critic_backbone"] = uniform_actor_weights
                critic_fed_coordinator.aggregate(
                    local_agents,
                    normalized_weights=uniform_actor_weights,
                    block_weights=block_weights,
                    selected_blocks=critic_blocks,
                )

            should_save_final = (
                ((episode + 1) % config.final_save_interval_episodes == 0)
                or (episode == config.total_episodes - 1)
            )
            if should_save_final:
                _save_local_full_named(run_dirs.full_final_dir, local_agents, episode)

            mean_step_constraint_cost = (
                episode_env_metrics["total_constraint_cost"] / max(episode_steps, 1)
            )
            total_reward_from_parks = sum(episode_raw_park_rewards.values())
            is_better_checkpoint = total_reward_from_parks > best_total_reward + 1e-9
            if is_better_checkpoint:
                best_total_reward = total_reward_from_parks
                _save_local_full_named(run_dirs.full_best_dir, local_agents, episode)

            total_reward_consistency_error = episode_reward - total_reward_from_parks
            if abs(total_reward_consistency_error) > 1e-6:
                raise RuntimeError(
                    "episode total reward is not equal to the sum of park rewards: "
                    f"episode_reward={episode_reward}, park_sum={total_reward_from_parks}"
                )

            training_row: Dict[str, object] = {
                "episode": episode,
                "seed": episode_seed,
                "weather": reset_info["weather"],
                "steps": episode_steps,
                "total_profit_reward": episode_env_metrics["total_profit_reward"],
                "total_constraint_cost": episode_env_metrics["total_constraint_cost"],
                "discounted_constraint_cost": discounted_constraint_cost,
                "mean_step_constraint_cost": mean_step_constraint_cost,
                "total_reward": total_reward_from_parks,
                "mean_lambda": (
                    sum(episode_lambda_values) / len(episode_lambda_values)
                    if episode_lambda_values
                    else float(local_agents[PARK_TYPES[0]].lambda_value.detach().cpu().item())
                ),
            }
            training_row.update(episode_env_metrics)
            for park_type in PARK_TYPES:
                training_row[f"{park_type}_reward"] = episode_raw_park_rewards[park_type]
                training_row[f"{park_type}_profit_reward"] = episode_profit_rewards[park_type]
                training_row[f"{park_type}_constraint_cost"] = episode_constraint_costs[park_type]
            training_row.update(
                _flatten_fed_actor_metrics(
                    latest_actor_fed_metrics,
                    scheme="lag_pfed_actor" if _uses_lag_pfed_actor(config) else ("fedavg" if actor_fed_coordinator is not None else ""),
                )
            )
            training_row.update(latest_fed_distill_metrics)
            training_logger.write_row(training_row)
            reward_logger.flush()
            training_logger.flush()
            bes_soc_step_logger.flush()
            if (
                ((episode + 1) % config.interrupted_save_interval_episodes == 0)
                or (episode == config.total_episodes - 1)
            ):
                _save_interrupt_checkpoint(
                    run_dirs=run_dirs,
                    config=config,
                    local_agents=local_agents,
                    next_episode=episode + 1,
                    best_total_reward=best_total_reward,
                    actor_fed_coordinator=actor_fed_coordinator,
                )
            if (episode + 1) % 5 == 0:
                _refresh_training_visualizations(run_dirs)

            print(
                f"episode={episode:03d} "
                f"system_total_reward={total_reward_from_parks:.4f} "
                f"system_profit={episode_env_metrics['total_profit_reward']:.4f} "
                f"system_penalty={episode_env_metrics['total_constraint_cost']:.4f} "
                f"system_discounted_cost={discounted_constraint_cost:.4f} "
                f"mean_step_constraint_cost={mean_step_constraint_cost:.4f}"
            )
            for park_type in PARK_TYPES:
                current_lambda = float(local_agents[park_type].lambda_value.detach().cpu().item())
                print(
                    f"  {park_type}: "
                    f"train_profit={episode_training_profit_rewards[park_type]:.4f} "
                    f"train_penalty={episode_training_constraint_costs[park_type]:.4f} "
                    f"log_profit={episode_profit_rewards[park_type]:.4f} "
                    f"log_penalty={episode_constraint_costs[park_type]:.4f} "
                    f"lambda={current_lambda:.4f} "
                    f"mean_qcf_pi={latest_mean_qcf_pi[park_type]:.4f} "
                    f"discounted_training_cost={episode_discounted_training_constraint_costs[park_type]:.4f} "
                    f"(d={config.d:.4f})"
                )
            _print_fed_actor_metrics(
                latest_actor_fed_metrics,
                scheme="lag_pfed_actor" if _uses_lag_pfed_actor(config) else ("fedavg" if actor_fed_coordinator is not None else ""),
            )
            if latest_fed_distill_metrics.get("fed_distill_rounds", 0.0) > 0:
                print(
                    "  fed_distill: "
                    f"reward_kl_pre={latest_fed_distill_metrics['fed_distill_reward_loss']:.4f} "
                    f"reward_kl_post={latest_fed_distill_metrics['fed_distill_reward_loss_post']:.4f} "
                    f"risk_kl_pre={latest_fed_distill_metrics['fed_distill_risk_loss']:.4f} "
                    f"risk_kl_post={latest_fed_distill_metrics['fed_distill_risk_loss_post']:.4f} "
                    f"actor_bes_pre={latest_fed_distill_metrics['fed_distill_actor_bes_loss']:.4f} "
                    f"actor_bes_post={latest_fed_distill_metrics['fed_distill_actor_bes_loss_post']:.4f} "
                    f"actor_ev_pre={latest_fed_distill_metrics['fed_distill_actor_ev_net_loss']:.4f} "
                    f"actor_ev_post={latest_fed_distill_metrics['fed_distill_actor_ev_net_loss_post']:.4f} "
                    f"critic_total={latest_fed_distill_metrics['fed_distill_total_loss']:.4f} "
                    f"actor_total={latest_fed_distill_metrics['fed_distill_actor_total_loss']:.4f} "
                    f"rounds={int(latest_fed_distill_metrics['fed_distill_rounds'])}"
                )

        reward_logger.flush()
        training_logger.flush()
        bes_soc_step_logger.flush()
        _refresh_training_visualizations(run_dirs)
    except KeyboardInterrupt:
        print(
            "training interrupted; resume checkpoint is saved every "
            f"{config.interrupted_save_interval_episodes} episodes and at the final episode boundary"
        )
        raise
    finally:
        reward_logger.close()
        training_logger.close()
        bes_soc_step_logger.close()


if __name__ == "__main__":
    run_training(TrainingConfig())
