from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List
import csv
import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from algorithm.gnn_csac import LocalGNNCSACAgent, LocalGNNCSACConfig
from algorithm.gnn_sac import LocalGNNSACAgent, LocalGNNSACConfig
from algorithm.hgt_csac import LocalHGTCSACAgent, LocalHGTCSACConfig
from algorithm.central_gnn_csac import CentralGNNCSACAgent
from algorithm.central_hgt_sac import CentralHGTSACAgent
from algorithm.hgt_sac import LocalHGTSACAgent, LocalHGTSACConfig
from algorithm.mlp_csac import LocalMLPCSACAgent, LocalMLPCSACConfig
from algorithm.mlp_sac import LocalMLPSACAgent, LocalMLPSACConfig
from algorithm.mlp_td3 import LocalMLPTD3Agent, LocalMLPTD3Config
from algorithm.sp_rgnn_csac import LocalSPRGNNCSACAgent, LocalSPRGNNCSACConfig
from agent.central_state import build_central_tr_graph
from env.three_park_charging_env import PARK_TYPES, ThreeParkChargingEnv
from agent.state import normalize_privacy_mode
from train_three_park_agent import (
    SP_RGNN_CSAC_ABLATION2,
    SP_RGNN_CSAC_VARIANTS,
    TrainingConfig,
    build_central_agent,
    build_joint_action,
    compute_park_target_entropy,
    configure_environment,
    load_cp_count_by_park,
    set_global_seed,
    validate_training_config,
    _restore_agent_from_checkpoint,
)


@dataclass
class EvaluationConfig:
    run_name: str = "SP_RGNN_CSAC-隐私+参数联邦-2"
    algorithm_variant: str = "sp_rgnn_csac"  # gnn_sac/gnn_csac/sp_rgnn_csac/mlp_sac/mlp_td3/mlp_csac/hgt_sac/hgt_csac/sp_rgnn_csac-ablation1/sp_rgnn_csac-ablation2
    enable_federation: bool = True
    privacy_mode: str = "strong"  # strong/none
    checkpoint_kind: str = "best"  # best/final
    eval_episodes: int = 7
    deterministic: bool = True
    seed: int = 10
    act_device: str = "cpu"
    update_device: str = "cpu"
    save_csv: bool = True
    decouple_actor_output_heads: bool = True
    enable_fed_distillation: bool = False
    use_strong_tr_projection_for_nonprivacy: bool = False
    use_central_tr_hgt_agent: bool = False
    bes_only_mode: bool = False
    tr_probe_ratio_1: float = 0.2
    tr_probe_ratio_2: float = 0.4
    tr_curvature_weight: float = 0.5
    tr_overload_penalty_weight: float = 1.0


def _resolve_model_dir(root_dir: Path, run_name: str, enable_federation: bool, checkpoint_kind: str) -> Path:
    if checkpoint_kind not in {"best", "final"}:
        raise ValueError("checkpoint_kind must be either 'best' or 'final'")
    family = "fed_full" if enable_federation else "local_full"
    return root_dir / "saved" / run_name / "models" / family / checkpoint_kind


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _reference_checkpoint_path(model_dir: Path, use_central_tr_hgt_agent: bool) -> Path:
    return model_dir / ("central.pt" if use_central_tr_hgt_agent else f"{PARK_TYPES[0]}.pt")


def _validate_eval_config_against_saved_models(
    model_dir: Path,
    eval_config: EvaluationConfig,
) -> Dict[str, Any]:
    reference_checkpoint = _load_checkpoint(
        _reference_checkpoint_path(model_dir, eval_config.use_central_tr_hgt_agent)
    )
    reference_agent_cfg = dict(reference_checkpoint["agent_config"])
    mismatches: List[str] = []

    expected_pairs = {
        "algorithm_variant": eval_config.algorithm_variant,
        "privacy_mode": normalize_privacy_mode(eval_config.privacy_mode),
    }
    if not eval_config.use_central_tr_hgt_agent:
        expected_pairs["enable_federation"] = eval_config.enable_federation
    for key, expected_value in expected_pairs.items():
        saved_value = reference_agent_cfg.get(key)
        if key == "privacy_mode":
            saved_value = normalize_privacy_mode(str(reference_agent_cfg.get("privacy_mode", reference_agent_cfg.get("state_mode", "strong"))))
        if saved_value != expected_value:
            mismatches.append(f"{key}: eval={expected_value!r}, saved={saved_value!r}")

    optional_expected_pairs = {
        "decouple_actor_output_heads": eval_config.decouple_actor_output_heads,
    }
    for key, expected_value in optional_expected_pairs.items():
        if expected_value is None:
            continue
        saved_value = reference_agent_cfg.get(key)
        if saved_value != expected_value:
            mismatches.append(f"{key}: eval={expected_value!r}, saved={saved_value!r}")

    if eval_config.use_central_tr_hgt_agent:
        if eval_config.enable_federation:
            mismatches.append("enable_federation: central evaluation requires False")
        checkpoint_format = str(reference_checkpoint.get("checkpoint_format", ""))
        if checkpoint_format not in {"central_hgt_sac_v1", "central_gnn_csac_v1"}:
            mismatches.append(
                "checkpoint_format: eval central route requires "
                f"'central_hgt_sac_v1' or 'central_gnn_csac_v1', saved={checkpoint_format!r}"
            )
        if mismatches:
            raise RuntimeError(
                "evaluation config does not match saved model parameters:\n"
                + "\n".join(mismatches)
            )
        return reference_checkpoint

    for park_type in PARK_TYPES[1:]:
        checkpoint = _load_checkpoint(model_dir / f"{park_type}.pt")
        agent_cfg = dict(checkpoint["agent_config"])
        for key in expected_pairs.keys():
            reference_value = reference_agent_cfg.get(key)
            current_value = agent_cfg.get(key)
            if key == "privacy_mode":
                reference_value = normalize_privacy_mode(str(reference_agent_cfg.get("privacy_mode", reference_agent_cfg.get("state_mode", "strong"))))
                current_value = normalize_privacy_mode(str(agent_cfg.get("privacy_mode", agent_cfg.get("state_mode", "strong"))))
            if current_value != reference_value:
                mismatches.append(
                    f"{park_type}.{key}: saved={current_value!r}, reference={reference_value!r}"
                )
        for key in optional_expected_pairs.keys():
            reference_value = reference_agent_cfg.get(key)
            current_value = agent_cfg.get(key)
            if current_value != reference_value:
                mismatches.append(
                    f"{park_type}.{key}: saved={current_value!r}, reference={reference_value!r}"
                )

    if mismatches:
        raise RuntimeError(
            "evaluation config does not match saved model parameters:\n"
            + "\n".join(mismatches)
        )
    return reference_checkpoint


def _infer_saved_experiment_signature(
    reference_checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    agent_cfg = dict(reference_checkpoint["agent_config"])
    return {
        "algorithm_variant": str(agent_cfg.get("algorithm_variant", "gnn_csac")),
        "privacy_mode": normalize_privacy_mode(str(agent_cfg.get("privacy_mode", agent_cfg.get("state_mode", "strong")))),
        "enable_federation": bool(agent_cfg.get("enable_federation", False)),
        "decouple_actor_output_heads": bool(agent_cfg.get("decouple_actor_output_heads", False)),
        "federate_critic_backbone": bool(agent_cfg.get("federate_critic_backbone", False)),
        "enable_auxiliary_risk_critic": bool(agent_cfg.get("enable_auxiliary_risk_critic", False)),
        "use_central_tr_hgt_agent": False,
    }


def _validate_saved_route_signature(
    saved_signature: Dict[str, Any],
    eval_config: EvaluationConfig,
) -> None:
    mismatches: List[str] = []
    if saved_signature["algorithm_variant"] != eval_config.algorithm_variant:
        mismatches.append(
            f"algorithm_variant: eval={eval_config.algorithm_variant!r}, saved={saved_signature['algorithm_variant']!r}"
        )
    if saved_signature["enable_federation"] != eval_config.enable_federation:
        mismatches.append(
            f"enable_federation: eval={eval_config.enable_federation!r}, saved={saved_signature['enable_federation']!r}"
        )
    if saved_signature["privacy_mode"] != normalize_privacy_mode(eval_config.privacy_mode):
        mismatches.append(
            f"privacy_mode: eval={normalize_privacy_mode(eval_config.privacy_mode)!r}, saved={saved_signature['privacy_mode']!r}"
        )
    if eval_config.decouple_actor_output_heads is not None:
        if saved_signature["decouple_actor_output_heads"] != eval_config.decouple_actor_output_heads:
            mismatches.append(
                "decouple_actor_output_heads: "
                f"eval={eval_config.decouple_actor_output_heads!r}, saved={saved_signature['decouple_actor_output_heads']!r}"
            )
    if bool(saved_signature.get("use_central_tr_hgt_agent", False)) != bool(eval_config.use_central_tr_hgt_agent):
        mismatches.append(
            "use_central_tr_hgt_agent: "
            f"eval={eval_config.use_central_tr_hgt_agent!r}, saved={saved_signature.get('use_central_tr_hgt_agent', False)!r}"
        )
    if mismatches:
        raise RuntimeError(
            "evaluation route does not match saved model route:\n"
            + "\n".join(mismatches)
        )


def _default_agent_runtime_config(
    algorithm_variant: str,
    park_type: str,
    cp_count: int,
    act_device: str,
    update_device: str,
) -> Dict[str, Any]:
    defaults = TrainingConfig()
    runtime: Dict[str, Any] = {
        "park_type": park_type,
        "algorithm_variant": algorithm_variant,
        "enable_federation": False,
        "privacy_mode": "strong",
        "alpha_lr": float(defaults.alpha_lr),
        "gamma": float(defaults.gamma),
        "tau": float(defaults.tau),
        "batch_size": int(defaults.batch_size),
        "replay_size": int(defaults.replay_size),
        "target_entropy": compute_park_target_entropy(cp_count, float(defaults.target_entropy_scale)),
        "actor_proximal_weight": float(defaults.actor_proximal_weight),
        "critic_proximal_weight": float(defaults.critic_proximal_weight),
        "seed": int(defaults.seed),
        "act_device": act_device,
        "update_device": update_device,
        "d": float(defaults.d),
        "lambda_lr": float(defaults.lambda_lr),
        "federate_critic_backbone": False,
    }
    if algorithm_variant in {"mlp_sac", "mlp_td3", "mlp_csac"}:
        runtime["cp_count"] = cp_count
    if algorithm_variant in SP_RGNN_CSAC_VARIANTS:
        runtime["use_relation_gated_fusion"] = algorithm_variant != SP_RGNN_CSAC_ABLATION2
        runtime["use_critic_typed_pooling"] = algorithm_variant != SP_RGNN_CSAC_ABLATION2
    return runtime


def _build_agents_from_saved_models(
    model_dir: Path,
    act_device: str,
    update_device: str,
) -> Dict[str, Any]:
    config_cls_by_variant = {
        "gnn_sac": LocalGNNSACConfig,
        "mlp_sac": LocalMLPSACConfig,
        "mlp_td3": LocalMLPTD3Config,
        "gnn_csac": LocalGNNCSACConfig,
        "sp_rgnn_csac": LocalSPRGNNCSACConfig,
        "sp_rgnn_csac-ablation1": LocalSPRGNNCSACConfig,
        "sp_rgnn_csac-ablation2": LocalSPRGNNCSACConfig,
        "hgt_csac": LocalHGTCSACConfig,
        "hgt_sac": LocalHGTSACConfig,
        "mlp_csac": LocalMLPCSACConfig,
    }
    cp_count_by_park = load_cp_count_by_park()
    local_agents: Dict[str, Any] = {}
    for park_type in PARK_TYPES:
        checkpoint = _load_checkpoint(model_dir / f"{park_type}.pt")
        checkpoint_agent_cfg = dict(checkpoint["agent_config"])
        algorithm_variant = str(checkpoint_agent_cfg.get("algorithm_variant", "gnn_csac"))
        agent_cfg = _default_agent_runtime_config(
            algorithm_variant=algorithm_variant,
            park_type=park_type,
            cp_count=cp_count_by_park[park_type],
            act_device=act_device,
            update_device=update_device,
        )
        agent_cfg.update(checkpoint_agent_cfg)
        agent_cfg["park_type"] = park_type
        agent_cfg["privacy_mode"] = normalize_privacy_mode(
            str(agent_cfg.get("privacy_mode", agent_cfg.get("state_mode", "strong")))
        )
        config_cls = config_cls_by_variant.get(algorithm_variant)
        if config_cls is None:
            raise ValueError(f"Unsupported algorithm_variant in checkpoint: {algorithm_variant}")
        valid_fields = {field.name for field in fields(config_cls)}
        filtered_agent_cfg = {key: value for key, value in agent_cfg.items() if key in valid_fields}
        filtered_agent_cfg.setdefault("algorithm_variant", algorithm_variant)
        if algorithm_variant == "gnn_sac":
            agent = LocalGNNSACAgent(LocalGNNSACConfig(**filtered_agent_cfg))
        elif algorithm_variant == "mlp_sac":
            agent = LocalMLPSACAgent(LocalMLPSACConfig(**filtered_agent_cfg))
        elif algorithm_variant == "mlp_td3":
            agent = LocalMLPTD3Agent(LocalMLPTD3Config(**filtered_agent_cfg))
        elif algorithm_variant == "gnn_csac":
            agent = LocalGNNCSACAgent(LocalGNNCSACConfig(**filtered_agent_cfg))
        elif algorithm_variant in SP_RGNN_CSAC_VARIANTS:
            agent = LocalSPRGNNCSACAgent(LocalSPRGNNCSACConfig(**filtered_agent_cfg))
        elif algorithm_variant == "hgt_csac":
            agent = LocalHGTCSACAgent(LocalHGTCSACConfig(**filtered_agent_cfg))
        elif algorithm_variant == "hgt_sac":
            agent = LocalHGTSACAgent(LocalHGTSACConfig(**filtered_agent_cfg))
        else:
            agent = LocalMLPCSACAgent(LocalMLPCSACConfig(**filtered_agent_cfg))
        _restore_agent_from_checkpoint(agent, checkpoint)
        local_agents[park_type] = agent
    return local_agents


def _infer_saved_experiment_signature_central(
    reference_checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    agent_cfg = dict(reference_checkpoint["agent_config"])
    return {
        "algorithm_variant": str(agent_cfg.get("algorithm_variant", "hgt_sac")),
        "privacy_mode": normalize_privacy_mode(str(agent_cfg.get("privacy_mode", "strong"))),
        "enable_federation": False,
        "decouple_actor_output_heads": bool(agent_cfg.get("decouple_actor_output_heads", False)),
        "federate_critic_backbone": False,
        "enable_auxiliary_risk_critic": False,
        "use_central_tr_hgt_agent": True,
    }


def _build_central_agent_from_saved_model(
    model_dir: Path,
    eval_training_config: TrainingConfig,
) -> CentralHGTSACAgent | CentralGNNCSACAgent:
    checkpoint = _load_checkpoint(model_dir / "central.pt")
    central_agent = build_central_agent(eval_training_config)
    _restore_agent_from_checkpoint(central_agent, checkpoint)
    return central_agent


def _build_env_training_config(
    eval_config: EvaluationConfig,
    saved_signature: Dict[str, Any],
) -> TrainingConfig:
    return TrainingConfig(
        run_name=eval_config.run_name,
        algorithm_variant=str(saved_signature["algorithm_variant"]),
        enable_federation=eval_config.enable_federation,
        federate_critic_backbone=False,
        privacy_mode=str(saved_signature["privacy_mode"]),
        use_strong_tr_projection_for_nonprivacy=bool(eval_config.use_strong_tr_projection_for_nonprivacy),
        enable_fed_distillation=False,
        enable_fed_distill_actor=False,
        use_central_tr_hgt_agent=bool(saved_signature.get("use_central_tr_hgt_agent", False)),
        decouple_actor_output_heads=bool(saved_signature["decouple_actor_output_heads"]),
        bes_only_mode=bool(eval_config.bes_only_mode),
        resume_training=False,
        seed=eval_config.seed,
        deterministic_training=True,
        total_episodes=1,
        act_device=eval_config.act_device,
        update_device=eval_config.update_device,
        tr_probe_ratio_1=float(eval_config.tr_probe_ratio_1),
        tr_probe_ratio_2=float(eval_config.tr_probe_ratio_2),
        tr_curvature_weight=float(eval_config.tr_curvature_weight),
        tr_overload_penalty_weight=float(eval_config.tr_overload_penalty_weight),
    )


def _step_eval_fields() -> List[str]:
    fields = [
        "episode",
        "seed",
        "weather",
        "step",
        "time",
        "total_profit_reward",
        "total_user_payment_cost",
        "total_grid_purchase_cost",
        "total_grid_sale_revenue",
        "total_v2g_compensation_cost",
        "total_constraint_cost",
        "total_immediate_reward",
    ]
    for park_type in PARK_TYPES:
        fields.extend(
            [
                f"{park_type}_profit_reward",
                f"{park_type}_user_payment_cost",
                f"{park_type}_grid_purchase_cost",
                f"{park_type}_grid_sale_revenue",
                f"{park_type}_v2g_compensation_cost",
                f"{park_type}_constraint_cost",
                f"{park_type}_immediate_reward",
                f"{park_type}_park_grid_exchange_kwh",
                f"{park_type}_bes_grid_energy_kwh",
                f"{park_type}_ev_grid_energy_total_kwh",
                f"{park_type}_bes_soc",
                f"{park_type}_ev_grid_energy_by_id",
            ]
        )
    return fields


def _serialize_ev_energy_sequence(ev_energy_by_id: Dict[str, float]) -> str:
    return json.dumps(
        {ev_id: float(ev_energy_by_id[ev_id]) for ev_id in sorted(ev_energy_by_id.keys())},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _build_step_eval_row(
    episode: int,
    episode_seed: int,
    reward_row: Dict[str, Any],
    energy_row: Dict[str, Any],
    transition_info: Dict[str, Any],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "episode": episode,
        "seed": episode_seed,
        "weather": reward_row["weather"],
        "step": reward_row["step"],
        "time": reward_row["time"],
        "total_profit_reward": reward_row["total_profit_reward"],
        "total_user_payment_cost": reward_row["total_ev_charge_revenue"],
        "total_grid_purchase_cost": reward_row["total_grid_purchase_cost"],
        "total_grid_sale_revenue": reward_row["total_grid_sale_revenue"],
        "total_v2g_compensation_cost": reward_row["total_v2g_compensation_cost"],
        "total_constraint_cost": reward_row["total_constraint_cost"],
        "total_immediate_reward": reward_row["total_immediate_reward"],
    }
    for park_type in PARK_TYPES:
        executed = transition_info["executed_flows"][park_type]
        ev_energy_total = sum(float(value) for value in executed["ev_grid_energy_by_id"].values())
        row[f"{park_type}_profit_reward"] = reward_row[f"{park_type}_profit_reward"]
        row[f"{park_type}_user_payment_cost"] = reward_row[f"{park_type}_ev_charge_revenue"]
        row[f"{park_type}_grid_purchase_cost"] = reward_row[f"{park_type}_grid_purchase_cost"]
        row[f"{park_type}_grid_sale_revenue"] = reward_row[f"{park_type}_grid_sale_revenue"]
        row[f"{park_type}_v2g_compensation_cost"] = reward_row[f"{park_type}_v2g_compensation_cost"]
        row[f"{park_type}_constraint_cost"] = reward_row[f"{park_type}_constraint_cost"]
        row[f"{park_type}_immediate_reward"] = reward_row[f"{park_type}_immediate_reward"]
        row[f"{park_type}_park_grid_exchange_kwh"] = executed["park_grid_exchange_kwh"]
        row[f"{park_type}_bes_grid_energy_kwh"] = executed["bes_grid_energy_kwh"]
        row[f"{park_type}_ev_grid_energy_total_kwh"] = ev_energy_total
        row[f"{park_type}_bes_soc"] = energy_row[f"{park_type}_bes_soc"]
        row[f"{park_type}_ev_grid_energy_by_id"] = _serialize_ev_energy_sequence(executed["ev_grid_energy_by_id"])
    return row


def _write_step_eval_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_step_eval_fields())
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in writer.fieldnames})


def run_evaluation(config: EvaluationConfig) -> None:
    validate_training_config(
        TrainingConfig(
            run_name=config.run_name,
            algorithm_variant=config.algorithm_variant,
            enable_federation=config.enable_federation,
            privacy_mode=config.privacy_mode,
            enable_fed_distillation=config.enable_fed_distillation,
            use_central_tr_hgt_agent=config.use_central_tr_hgt_agent,
            use_strong_tr_projection_for_nonprivacy=config.use_strong_tr_projection_for_nonprivacy,
            act_device=config.act_device,
            update_device=config.update_device,
        )
    )
    root_dir = Path(__file__).resolve().parent
    model_dir = _resolve_model_dir(root_dir, config.run_name, config.enable_federation, config.checkpoint_kind)

    first_checkpoint = _validate_eval_config_against_saved_models(model_dir, config)
    if config.use_central_tr_hgt_agent:
        saved_signature = _infer_saved_experiment_signature_central(first_checkpoint)
    else:
        saved_signature = _infer_saved_experiment_signature(first_checkpoint)
    _validate_saved_route_signature(saved_signature, config)
    eval_training_config = _build_env_training_config(config, saved_signature=saved_signature)

    set_global_seed(config.seed)
    local_agents: Dict[str, Any] | None = None
    central_agent: CentralHGTSACAgent | CentralGNNCSACAgent | None = None
    if config.use_central_tr_hgt_agent:
        central_agent = _build_central_agent_from_saved_model(
            model_dir=model_dir,
            eval_training_config=eval_training_config,
        )
    else:
        local_agents = _build_agents_from_saved_models(
            model_dir=model_dir,
            act_device=config.act_device,
            update_device=config.update_device,
        )
    env = ThreeParkChargingEnv(seed=config.seed)
    configure_environment(env, eval_training_config)
    if local_agents is not None:
        env.attach_local_agents(local_agents)

    step_rows: List[Dict[str, object]] = []

    print(
        f"evaluate run_name={config.run_name} "
        f"mode={'central' if config.use_central_tr_hgt_agent else 'local'} "
        f"algorithm_variant={saved_signature['algorithm_variant']} "
        f"enable_federation={saved_signature['enable_federation']} "
        f"privacy_mode={saved_signature['privacy_mode']} "
        f"decouple_actor_output_heads={saved_signature['decouple_actor_output_heads']} "
        f"use_strong_tr_projection_for_nonprivacy={config.use_strong_tr_projection_for_nonprivacy} "
        f"act_device={config.act_device} "
        f"update_device={config.update_device} "
        f"deterministic={config.deterministic}"
    )

    for episode in range(config.eval_episodes):
        episode_seed = config.seed + episode
        obs, reset_info = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0.0
        episode_steps = 0
        discounted_constraint_cost = 0.0
        metrics: Dict[str, float] = {
            "total_profit_reward": 0.0,
            "total_constraint_cost": 0.0,
            "total_grid_purchase_cost": 0.0,
            "total_grid_sale_revenue": 0.0,
            "total_ev_charge_revenue": 0.0,
            "total_v2g_compensation_cost": 0.0,
            "total_cs_projection_penalty": 0.0,
            "total_tr_projection_penalty": 0.0,
            "total_soc_shortfall_penalty": 0.0,
            "total_debt_penalty": 0.0,
            "total_bes_terminal_penalty": 0.0,
        }
        park_rewards = {park_type: 0.0 for park_type in PARK_TYPES}
        for park_type in PARK_TYPES:
            metrics.update(
                {
                    f"{park_type}_profit_reward": 0.0,
                    f"{park_type}_constraint_cost": 0.0,
                    f"{park_type}_grid_purchase_cost": 0.0,
                    f"{park_type}_grid_sale_revenue": 0.0,
                    f"{park_type}_ev_charge_revenue": 0.0,
                    f"{park_type}_v2g_compensation_cost": 0.0,
                    f"{park_type}_cs_projection_penalty": 0.0,
                    f"{park_type}_tr_projection_penalty": 0.0,
                    f"{park_type}_soc_shortfall_penalty": 0.0,
                    f"{park_type}_debt_penalty": 0.0,
                    f"{park_type}_bes_terminal_penalty": 0.0,
                }
            )

        while not done:
            if config.use_central_tr_hgt_agent:
                if central_agent is None:
                    raise RuntimeError("central evaluation requested but central agent was not built")
                central_obs = build_central_tr_graph(obs, privacy_mode=saved_signature["privacy_mode"])
                joint_action, _raw_node_action, raw_node_actions = central_agent.act(
                    central_obs,
                    deterministic=config.deterministic,
                    return_node_action=True,
                )
            else:
                if local_agents is None:
                    raise RuntimeError("local evaluation requested but local agents were not built")
                joint_action, raw_node_actions = build_joint_action(
                    local_agents,
                    obs,
                    deterministic=config.deterministic,
                    return_raw_action=True,
                )
            next_obs, reward, terminated, truncated, info = env.step(
                joint_action,
                raw_node_actions=raw_node_actions,
            )
            done = terminated or truncated
            reward_row = info["reward_log"]
            energy_row = info["energy_log"]

            step_rows.append(
                _build_step_eval_row(
                    episode=episode,
                    episode_seed=episode_seed,
                    reward_row=reward_row,
                    energy_row=energy_row,
                    transition_info=info["transition"],
                )
            )
            print(
                f"eval_episode={episode:03d} "
                f"step={int(reward_row['step']):03d} "
                f"time={reward_row['time']} "
                f"profit={float(reward_row['total_profit_reward']):.4f} "
                f"constraint={float(reward_row['total_constraint_cost']):.4f}"
            )

            episode_reward += float(reward)
            metrics["total_profit_reward"] += float(reward_row["total_profit_reward"])
            metrics["total_constraint_cost"] += float(reward_row["total_constraint_cost"])
            discounted_constraint_cost += (
                (eval_training_config.gamma ** episode_steps) * float(reward_row["total_constraint_cost"])
            )
            metrics["total_grid_purchase_cost"] += float(reward_row["total_grid_purchase_cost"])
            metrics["total_grid_sale_revenue"] += float(reward_row["total_grid_sale_revenue"])
            metrics["total_ev_charge_revenue"] += float(reward_row["total_ev_charge_revenue"])
            metrics["total_v2g_compensation_cost"] += float(reward_row["total_v2g_compensation_cost"])
            metrics["total_cs_projection_penalty"] += float(reward_row["total_cs_projection_penalty"])
            metrics["total_tr_projection_penalty"] += float(reward_row["total_tr_projection_penalty"])
            metrics["total_soc_shortfall_penalty"] += float(reward_row["total_soc_shortfall_penalty"])
            metrics["total_debt_penalty"] += float(reward_row["total_debt_penalty"])
            metrics["total_bes_terminal_penalty"] += float(reward_row["total_bes_terminal_penalty"])

            for park_type in PARK_TYPES:
                park_rewards[park_type] += float(reward_row[f"{park_type}_immediate_reward"])
                metrics[f"{park_type}_profit_reward"] += float(reward_row[f"{park_type}_profit_reward"])
                metrics[f"{park_type}_constraint_cost"] += float(reward_row[f"{park_type}_constraint_cost"])
                metrics[f"{park_type}_grid_purchase_cost"] += float(reward_row[f"{park_type}_grid_purchase_cost"])
                metrics[f"{park_type}_grid_sale_revenue"] += float(reward_row[f"{park_type}_grid_sale_revenue"])
                metrics[f"{park_type}_ev_charge_revenue"] += float(reward_row[f"{park_type}_ev_charge_revenue"])
                metrics[f"{park_type}_v2g_compensation_cost"] += float(reward_row[f"{park_type}_v2g_compensation_cost"])
                metrics[f"{park_type}_cs_projection_penalty"] += float(reward_row[f"{park_type}_cs_projection_penalty"])
                metrics[f"{park_type}_tr_projection_penalty"] += float(reward_row[f"{park_type}_tr_projection_penalty"])
                metrics[f"{park_type}_soc_shortfall_penalty"] += float(reward_row[f"{park_type}_soc_shortfall_penalty"])
                metrics[f"{park_type}_debt_penalty"] += float(reward_row[f"{park_type}_debt_penalty"])
                metrics[f"{park_type}_bes_terminal_penalty"] += float(reward_row[f"{park_type}_bes_terminal_penalty"])

            episode_steps += 1
            obs = next_obs

        print(
            f"eval_episode={episode:03d} "
            f"reward={episode_reward:.4f} "
            f"profit={metrics['total_profit_reward']:.4f} "
            f"constraint={metrics['total_constraint_cost']:.4f} "
            f"discounted_constraint={discounted_constraint_cost:.4f}"
        )

    if config.save_csv:
        output_path = root_dir / "saved" / config.run_name / "evaluates" / "evaluation_steps.csv"
        _write_step_eval_csv(output_path, step_rows)
        print(f"saved evaluation csv: {output_path}")

    completed_episodes = max(1, config.eval_episodes)
    mean_reward = sum(float(metrics_row["total_immediate_reward"]) for metrics_row in step_rows) / completed_episodes
    mean_profit = sum(float(metrics_row["total_profit_reward"]) for metrics_row in step_rows) / completed_episodes
    mean_constraint = sum(float(metrics_row["total_constraint_cost"]) for metrics_row in step_rows) / completed_episodes
    mean_discounted_constraint = sum(
        (eval_training_config.gamma ** int(metrics_row["step"])) * float(metrics_row["total_constraint_cost"])
        for metrics_row in step_rows
    ) / completed_episodes
    print(
        f"evaluation_summary "
        f"mean_reward={mean_reward:.4f} "
        f"mean_profit={mean_profit:.4f} "
        f"mean_constraint={mean_constraint:.4f} "
        f"mean_discounted_constraint={mean_discounted_constraint:.4f}"
    )


if __name__ == "__main__":
    run_evaluation(EvaluationConfig())
