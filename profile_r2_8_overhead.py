from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
import torch

from Fed_average.fed_controller import FederatedAveragingCoordinator, FederatedConfig
from Fed_average.learnable_personalized_fed_actor import LearnablePersonalizedFedActorCoordinator
from env.three_park_charging_env import PARK_TYPES, ThreeParkChargingEnv
from evaluate_three_park_agent import (
    EvaluationConfig,
    _build_agents_from_saved_models,
    _build_env_training_config,
    _infer_saved_experiment_signature,
    _resolve_model_dir,
    _validate_eval_config_against_saved_models,
    _validate_saved_route_signature,
)
from train_three_park_agent import build_joint_action, configure_environment, set_global_seed
from tr_coordination.strong_privacy_coordinator import (
    DEFAULT_SECURE_AGGREGATION_SCALE,
    DEFAULT_SECURE_MASK_BOUND,
    secure_masked_sum,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "saved" / "reviewer2_comment8" / "profiling"
N_VALUES = (3, 5, 10, 20, 50)
FED_WARMUP = 10
FED_REPEATS = 100
FED_REPEATS_BY_N = {3: 100, 5: 20, 10: 5, 20: 2, 50: 1}
SECURE_WARMUP = 10
SECURE_REPEATS = 1000
ONLINE_WARMUP_STEPS = 10
ONLINE_REPEATS = 1000
EPS = 1e-12


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: str
    dtype: str
    numel: int
    bytes: int


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _elapsed_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _module_parameter_bytes(module: torch.nn.Module) -> int:
    return sum(_tensor_nbytes(parameter) for parameter in module.parameters())


def _module_parameter_count(module: torch.nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in module.parameters())


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _describe(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.mean(ordered),
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "median": statistics.median(ordered),
        "p95": ordered[p95_index],
        "min": ordered[0],
        "max": ordered[-1],
    }


class PeakRSSSampler:
    def __init__(self, interval_s: float = 0.0005) -> None:
        self.interval_s = interval_s
        self.process = psutil.Process(os.getpid())
        self.start_rss = 0
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRSSSampler":
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss

        def sample() -> None:
            while not self._stop.is_set():
                self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
                self._stop.wait(self.interval_s)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak_rss / (1024.0 * 1024.0)

    @property
    def delta_mb(self) -> float:
        return max(0, self.peak_rss - self.start_rss) / (1024.0 * 1024.0)


def _hardware_software() -> Dict[str, Any]:
    cpu_name = platform.processor()
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            cpu_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    cuda_available = torch.cuda.is_available()
    return {
        "cpu": cpu_name or "unable to confirm",
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": psutil.virtual_memory().total / (1024.0**3),
        "gpu": torch.cuda.get_device_name(0) if cuda_available else "none",
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_build": torch.version.cuda or "none",
        "cuda_available": cuda_available,
        "profiling_device": "cpu",
        "pytorch_intraop_threads": torch.get_num_threads(),
        "pytorch_interop_threads": torch.get_num_interop_threads(),
    }


def _load_fedrg_agents() -> tuple[Dict[str, Any], Any]:
    config = EvaluationConfig(
        run_name="SP_RGNN_CSAC-隐私+参数联邦-2",
        algorithm_variant="sp_rgnn_csac",
        enable_federation=True,
        checkpoint_kind="best",
        deterministic=True,
        seed=10,
        act_device="cpu",
        update_device="cpu",
        save_csv=False,
    )
    model_dir = _resolve_model_dir(ROOT, config.run_name, config.enable_federation, config.checkpoint_kind)
    checkpoint = _validate_eval_config_against_saved_models(model_dir, config)
    signature = _infer_saved_experiment_signature(checkpoint)
    _validate_saved_route_signature(signature, config)
    env_config = _build_env_training_config(config, signature)
    set_global_seed(config.seed)
    agents = _build_agents_from_saved_models(model_dir, config.act_device, config.update_device)
    return agents, env_config


def build_parameter_inventory(agent: Any) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    shared = agent.get_shared_state()["actor_backbone"]
    inventory: List[Dict[str, Any]] = []
    for name, tensor in shared.items():
        inventory.append(
            {
                "tensor_name": name,
                "shape": str(tuple(tensor.shape)),
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
                "element_size_bytes": int(tensor.element_size()),
                "bytes": _tensor_nbytes(tensor),
            }
        )
    shared_count = sum(int(row["numel"]) for row in inventory)
    shared_bytes = sum(int(row["bytes"]) for row in inventory)
    actor_count = _module_parameter_count(agent.actor)
    actor_bytes = _module_parameter_bytes(agent.actor)
    reward_critic_count = _module_parameter_count(agent.critic)
    reward_critic_bytes = _module_parameter_bytes(agent.critic)
    cost_critic_count = _module_parameter_count(agent.cost_critic)
    cost_critic_bytes = _module_parameter_bytes(agent.cost_critic)
    scalar_count = int(agent.log_alpha.numel() + agent.log_lambda.numel())
    scalar_bytes = _tensor_nbytes(agent.log_alpha) + _tensor_nbytes(agent.log_lambda)
    local_count = actor_count + reward_critic_count + cost_critic_count + scalar_count
    local_bytes = actor_bytes + reward_critic_bytes + cost_critic_bytes + scalar_bytes
    summary = {
        "shared_tensor_count": len(inventory),
        "federated_parameter_count": shared_count,
        "shared_payload_bytes": shared_bytes,
        "shared_payload_mb_decimal": shared_bytes / 1_000_000.0,
        "shared_payload_mib": shared_bytes / (1024.0**2),
        "full_actor_parameter_count": actor_count,
        "full_actor_bytes": actor_bytes,
        "full_actor_mb_decimal": actor_bytes / 1_000_000.0,
        "reward_critic_parameter_count": reward_critic_count,
        "cost_critic_parameter_count": cost_critic_count,
        "full_local_optimization_parameter_count": local_count,
        "full_local_optimization_bytes": local_bytes,
        "federated_to_actor_pct": 100.0 * shared_count / actor_count,
        "federated_to_local_pct": 100.0 * shared_count / local_count,
        "definition": "actor + online reward critics + online cost critics + log_alpha + log_lambda; target networks and inference copy excluded",
    }
    return inventory, summary


def build_communication_rows(shared_params: int, shared_bytes: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n in N_VALUES:
        for method in ("FedAvg-SP-RGNN-CSAC", "FedProx-SP-RGNN-CSAC", "FedRG-MARL"):
            upload = n * shared_bytes
            if method == "FedRG-MARL":
                source_download = n * (n - 1) * shared_bytes
                candidate_download = n * shared_bytes
                download = source_download + candidate_download
                topology_note = "N(N-1) source backbones for target-local scoring + one personalized candidate per target"
            else:
                source_download = 0
                candidate_download = n * shared_bytes
                download = candidate_download
                topology_note = "one aggregated shared backbone broadcast per client"
            total = upload + download
            rows.append(
                {
                    "method": method,
                    "N": n,
                    "num_shared_params_per_client": shared_params,
                    "shared_payload_bytes": shared_bytes,
                    "upload_bytes_per_round": upload,
                    "download_bytes_per_round": download,
                    "source_scoring_download_bytes": source_download,
                    "candidate_or_global_download_bytes": candidate_download,
                    "total_logical_bytes_per_round": total,
                    "bytes_per_park_per_round": total / n,
                    "total_logical_mb_decimal": total / 1_000_000.0,
                    "accounting_boundary": "logical distributed mapping; current implementation is single-process and performs no socket transfer",
                    "topology_note": topology_note,
                }
            )
    return rows


class OnlineProfilingEnv(ThreeParkChargingEnv):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.stage_ms: Dict[str, float] = {}

    def reset_stage_times(self) -> None:
        self.stage_ms = {
            "action_decoding_ms": 0.0,
            "local_cs_projection_ms": 0.0,
            "regional_masking_aggregation_ms": 0.0,
            "regional_tr_total_ms": 0.0,
            "final_local_execution_ms": 0.0,
        }

    def _decode_actions(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter_ns()
        result = super()._decode_actions(*args, **kwargs)
        self.stage_ms["action_decoding_ms"] += _elapsed_ms(start)
        return result

    def _run_cs_projection(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter_ns()
        result = super()._run_cs_projection(*args, **kwargs)
        self.stage_ms["local_cs_projection_ms"] += _elapsed_ms(start)
        return result

    def _secure_aggregate_by_park(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter_ns()
        result = super()._secure_aggregate_by_park(*args, **kwargs)
        self.stage_ms["regional_masking_aggregation_ms"] += _elapsed_ms(start)
        return result

    def _run_tr_projection(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter_ns()
        result = super()._run_tr_projection(*args, **kwargs)
        self.stage_ms["regional_tr_total_ms"] += _elapsed_ms(start)
        return result

    def _apply_projected_actions(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter_ns()
        result = super()._apply_projected_actions(*args, **kwargs)
        self.stage_ms["final_local_execution_ms"] += _elapsed_ms(start)
        return result


def profile_online_latency(
    agents: Dict[str, Any],
    env_config: Any,
    warmup_steps: int = ONLINE_WARMUP_STEPS,
    repeats: int = ONLINE_REPEATS,
) -> List[Dict[str, Any]]:
    env = OnlineProfilingEnv(seed=10)
    configure_environment(env, env_config)
    env.attach_local_agents(agents)
    obs, _ = env.reset(seed=10)
    episode_seed = 10
    measured_rows: List[Dict[str, Any]] = []
    total_steps = warmup_steps + repeats
    for absolute_step in range(total_steps):
        if env.done:
            episode_seed += 1
            obs, _ = env.reset(seed=episode_seed)
        env.reset_stage_times()
        _sync(torch.device("cpu"))
        total_start = time.perf_counter_ns()
        inference_start = time.perf_counter_ns()
        joint_action, raw_node_actions = build_joint_action(
            agents,
            obs,
            deterministic=True,
            return_raw_action=True,
        )
        inference_ms = _elapsed_ms(inference_start)
        next_obs, reward, terminated, truncated, info = env.step(
            joint_action,
            raw_node_actions=raw_node_actions,
        )
        _sync(torch.device("cpu"))
        total_ms = _elapsed_ms(total_start)
        done = terminated or truncated

        for park_type in PARK_TYPES:
            breakdown = info["park_reward_breakdown"][park_type]
            agents[park_type].store_transition(
                obs=obs["park_graphs"][park_type],
                action=raw_node_actions[park_type],
                reward=float(breakdown["profit_reward"]),
                cost=float(breakdown["constraint_cost"]),
                next_obs=next_obs["park_graphs"][park_type],
                done=done,
            )

        if absolute_step >= warmup_steps:
            tr_excluding_masks = max(
                0.0,
                env.stage_ms["regional_tr_total_ms"]
                - env.stage_ms["regional_masking_aggregation_ms"],
            )
            accounted = (
                inference_ms
                + env.stage_ms["action_decoding_ms"]
                + env.stage_ms["local_cs_projection_ms"]
                + env.stage_ms["regional_tr_total_ms"]
                + env.stage_ms["final_local_execution_ms"]
            )
            measured_rows.append(
                {
                    "repeat_id": absolute_step - warmup_steps,
                    "episode_seed": episode_seed,
                    "environment_step": int(info["reward_log"]["step"]),
                    "device": "cpu",
                    "local_actor_inference_ms": inference_ms,
                    "action_decoding_feasibility_ms": env.stage_ms["action_decoding_ms"],
                    "local_cs_projection_ms": env.stage_ms["local_cs_projection_ms"],
                    "regional_masking_aggregation_ms": env.stage_ms["regional_masking_aggregation_ms"],
                    "regional_tr_coordination_excluding_masks_ms": tr_excluding_masks,
                    "regional_tr_coordination_total_ms": env.stage_ms["regional_tr_total_ms"],
                    "final_local_execution_update_ms": env.stage_ms["final_local_execution_ms"],
                    "other_environment_logging_reward_state_ms": max(0.0, total_ms - accounted),
                    "end_to_end_decision_ms": total_ms,
                    "tr_triggered": int(bool(info["tr_projection"]["triggered"])),
                }
            )
        obs = next_obs
    return measured_rows


class ProfilingClient:
    """Same federation interface, real shared tensors and real target scoring code."""

    def __init__(self, client_id: str, template_agent: Any, shared_state: Dict[str, torch.Tensor]) -> None:
        self.client_id = client_id
        self.template_agent = template_agent
        self.shared_state = {key: value.detach().cpu().clone() for key, value in shared_state.items()}
        self.global_reference: Dict[str, torch.Tensor] = {}

    def get_shared_state(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "actor_backbone": {
                key: value.detach().cpu().clone() for key, value in self.shared_state.items()
            }
        }

    def get_actor_relation_fed_mask(self) -> Dict[str, bool]:
        return self.template_agent.get_actor_relation_fed_mask()

    def evaluate_external_backbone_advantage(self, backbone: Dict[str, torch.Tensor]) -> float:
        return self.template_agent.evaluate_external_backbone_advantage(backbone)

    def evaluate_candidate_backbone_advantage(self, backbone: Dict[str, torch.Tensor]) -> float:
        return self.template_agent.evaluate_candidate_backbone_advantage(backbone)

    def soft_load_actor_backbone(self, candidate: Dict[str, torch.Tensor], eta: float) -> None:
        self.template_agent.soft_load_actor_backbone(candidate, eta)
        if eta > 0.0:
            self.shared_state = {
                key: ((1.0 - eta) * self.shared_state[key] + eta * candidate[key]).detach().cpu()
                for key in self.shared_state
            }

    def set_global_actor_reference(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        self.global_reference = {
            key: value.detach().cpu().clone()
            for key, value in state["actor_backbone"].items()
        }

    def load_shared_state(self, state: Dict[str, Dict[str, torch.Tensor]]) -> None:
        global_state = state["actor_backbone"]
        mix = float(self.template_agent.config.relation_fed_mix)
        self.shared_state = {
            key: ((1.0 - mix) * self.shared_state[key] + mix * global_state[key]).detach().cpu()
            for key in self.shared_state
        }


def _make_clients(n: int, agents: Dict[str, Any]) -> Dict[str, ProfilingClient]:
    templates = [agents[park_type] for park_type in PARK_TYPES]
    base_states = [agent.get_shared_state()["actor_backbone"] for agent in templates]
    clients: Dict[str, ProfilingClient] = {}
    for index in range(n):
        template = templates[index % len(templates)]
        state = {
            key: value.detach().cpu().clone()
            for key, value in base_states[index % len(base_states)].items()
        }
        clients[f"park_{index:03d}"] = ProfilingClient(f"park_{index:03d}", template, state)
    return clients


def _new_personalized_coordinator(client_ids: Sequence[str]) -> LearnablePersonalizedFedActorCoordinator:
    return LearnablePersonalizedFedActorCoordinator(
        config=FederatedConfig(),
        park_ids=list(client_ids),
        fed_logits_lr=5e-3,
        rho_logits_lr=2e-3,
        fed_logits_diag_init=2.5,
        fed_logits_offdiag_init=0.0,
        rho_init=0.08,
        candidate_gate_margin=0.05,
        eta_probe=3e-4,
        eta_max=0.005,
        device="cpu",
    )


def _fedrg_profiled_round(
    coordinator: LearnablePersonalizedFedActorCoordinator,
    clients: Dict[str, ProfilingClient],
) -> Dict[str, float]:
    device = coordinator.device
    _sync(device)
    total_start = time.perf_counter_ns()

    preparation_start = time.perf_counter_ns()
    local_states = {
        client_id: clients[client_id].get_shared_state()["actor_backbone"]
        for client_id in coordinator.park_ids
    }
    relation_masks = {
        client_id: clients[client_id].get_actor_relation_fed_mask()
        for client_id in coordinator.park_ids
    }
    preparation_ms = _elapsed_ms(preparation_start)

    source_start = time.perf_counter_ns()
    source_advantage = coordinator.compute_source_advantage_matrix(clients, local_states)
    source_score_ms = _elapsed_ms(source_start)

    weight_start = time.perf_counter_ns()
    source_advantage_norm = coordinator._rowwise_normalize_by_mean_abs(source_advantage)
    coordinator.update_fed_logits(source_advantage_norm)
    fed_weights = torch.softmax(coordinator.fed_logits, dim=1).detach()
    weight_update_ms = _elapsed_ms(weight_start)

    candidate_start = time.perf_counter_ns()
    candidates = coordinator.build_personalized_candidates(local_states, fed_weights, relation_masks)
    candidate_construction_ms = _elapsed_ms(candidate_start)

    candidate_score_start = time.perf_counter_ns()
    candidate_advantage = coordinator.compute_candidate_advantages(clients, candidates)
    candidate_score_ms = _elapsed_ms(candidate_score_start)

    gating_start = time.perf_counter_ns()
    candidate_advantage_norm = coordinator._normalize_candidate_advantages(
        candidate_advantage,
        source_advantage,
    )
    coordinator.update_rho_logits(candidate_advantage_norm)
    rho = torch.sigmoid(coordinator.rho_logits).detach()
    acceptance_gating_ms = _elapsed_ms(gating_start)

    loading_start = time.perf_counter_ns()
    coordinator.soft_load_candidates(clients, candidates, candidate_advantage_norm, rho)
    soft_loading_ms = _elapsed_ms(loading_start)
    _sync(device)
    total_ms = _elapsed_ms(total_start)
    return {
        "preparation_time_ms": preparation_ms,
        "source_score_time_ms": source_score_ms,
        "weight_update_time_ms": weight_update_ms,
        "candidate_construction_time_ms": candidate_construction_ms,
        "candidate_score_time_ms": candidate_score_ms,
        "acceptance_gating_time_ms": acceptance_gating_ms,
        "loading_time_ms": soft_loading_ms,
        "score_time_ms": source_score_ms + candidate_score_ms,
        "aggregation_time_ms": weight_update_ms + acceptance_gating_ms,
        "candidate_time_ms": candidate_construction_ms,
        "total_federation_time_ms": total_ms,
    }


def _fedavg_profiled_round(
    coordinator: FederatedAveragingCoordinator,
    clients: Dict[str, ProfilingClient],
) -> Dict[str, float]:
    start = time.perf_counter_ns()
    coordinator.aggregate(clients)
    total = _elapsed_ms(start)
    return {
        "preparation_time_ms": 0.0,
        "source_score_time_ms": 0.0,
        "weight_update_time_ms": 0.0,
        "candidate_construction_time_ms": 0.0,
        "candidate_score_time_ms": 0.0,
        "acceptance_gating_time_ms": 0.0,
        "loading_time_ms": 0.0,
        "score_time_ms": 0.0,
        "aggregation_time_ms": total,
        "candidate_time_ms": 0.0,
        "total_federation_time_ms": total,
    }


def profile_federation_scalability(
    agents: Dict[str, Any],
    shared_params: int,
    shared_bytes: int,
    communication_rows: Sequence[Mapping[str, Any]],
    warmup: int = FED_WARMUP,
    repeats: int | Mapping[int, int] = FED_REPEATS_BY_N,
    methods: Sequence[str] = (
        "FedAvg-SP-RGNN-CSAC",
        "FedProx-SP-RGNN-CSAC",
        "FedRG-MARL",
    ),
) -> List[Dict[str, Any]]:
    communication = {
        (str(row["method"]), int(row["N"])): row for row in communication_rows
    }
    rows: List[Dict[str, Any]] = []
    for n in N_VALUES:
        for method in methods:
            repeat_count = (
                int(repeats[n] if isinstance(repeats, Mapping) else repeats)
                if method == "FedRG-MARL"
                else FED_REPEATS
            )
            clients = _make_clients(n, agents)
            if method == "FedRG-MARL":
                coordinator: Any = _new_personalized_coordinator(list(clients))
                round_fn: Callable[[], Dict[str, float]] = lambda: _fedrg_profiled_round(coordinator, clients)
            else:
                coordinator = FederatedAveragingCoordinator(FederatedConfig())
                round_fn = lambda: _fedavg_profiled_round(coordinator, clients)

            print(f"federation benchmark method={method} N={n} warmup={warmup} repeats={repeat_count}")
            if method == "FedRG-MARL":
                local_states = {
                    client_id: clients[client_id].get_shared_state()["actor_backbone"]
                    for client_id in coordinator.park_ids
                }
                relation_masks = {
                    client_id: clients[client_id].get_actor_relation_fed_mask()
                    for client_id in coordinator.park_ids
                }
                first_client = coordinator.park_ids[0]
                second_client = coordinator.park_ids[1]
                for _ in range(warmup):
                    clients[first_client].evaluate_external_backbone_advantage(local_states[second_client])
                    clients[first_client].evaluate_candidate_backbone_advantage(local_states[second_client])
                    synthetic_advantage = torch.zeros((n, n), dtype=torch.float32)
                    coordinator.update_fed_logits(synthetic_advantage)
                    weights = torch.softmax(coordinator.fed_logits, dim=1).detach()
                    coordinator.build_personalized_candidates(local_states, weights, relation_masks)
                    coordinator.update_rho_logits(torch.zeros((n,), dtype=torch.float32))
            else:
                for _ in range(warmup):
                    round_fn()

            repeat_metrics: List[Dict[str, float]] = []
            sampler: PeakRSSSampler | None = None
            for repeat_id in range(repeat_count):
                if repeat_id == 0:
                    with PeakRSSSampler() as first_sampler:
                        metrics = round_fn()
                    sampler = first_sampler
                else:
                    metrics = round_fn()
                repeat_metrics.append(metrics)
                row = {
                    "method": method,
                    "N": n,
                    "repeat_id": repeat_id,
                    "requested_formal_repeats": FED_REPEATS,
                    "actual_full_round_repeats": repeat_count,
                    "repeat_protocol_note": (
                        "100 full rounds" if repeat_count >= FED_REPEATS else
                        "adaptive repeats: real O(N^2) scoring made 100 full rounds impractical; no values fabricated"
                    ),
                    "device": "cpu",
                    "num_shared_params": shared_params,
                    "shared_payload_bytes": shared_bytes,
                    "upload_bytes_per_round": int(communication[(method, n)]["upload_bytes_per_round"]),
                    "download_bytes_per_round": int(communication[(method, n)]["download_bytes_per_round"]),
                    "total_logical_bytes_per_round": int(communication[(method, n)]["total_logical_bytes_per_round"]),
                    "number_of_target_source_evaluations": n * (n - 1) if method == "FedRG-MARL" else 0,
                    "number_of_candidate_evaluations": n if method == "FedRG-MARL" else 0,
                    "temporary_candidate_model_bytes": n * shared_bytes if method == "FedRG-MARL" else shared_bytes,
                    **metrics,
                    "cpu_peak_memory_mb": 0.0,
                    "cpu_peak_memory_delta_mb": 0.0,
                    "gpu_peak_memory_mb": 0.0,
                }
                rows.append(row)

            if sampler is None:
                raise RuntimeError("at least one federation repeat is required")
            for row in rows[-repeat_count:]:
                row["cpu_peak_memory_mb"] = sampler.peak_mb
                row["cpu_peak_memory_delta_mb"] = sampler.delta_mb
                if torch.cuda.is_available():
                    row["gpu_peak_memory_mb"] = 0.0

            total_stats = _describe([item["total_federation_time_ms"] for item in repeat_metrics])
            print(
                f"  total_ms mean={total_stats['mean']:.6f} "
                f"p95={total_stats['p95']:.6f} peak_rss_mb={sampler.peak_mb:.2f}"
            )
    return rows


def _instrument_masked_aggregation(
    values_by_quantity: Sequence[Mapping[str, float]],
    rng: random.Random,
) -> Dict[str, float]:
    mask_generation_start = time.perf_counter_ns()
    park_ids = sorted(values_by_quantity[0])
    masks_by_quantity: List[List[tuple[str, str, int]]] = []
    for _values in values_by_quantity:
        masks: List[tuple[str, str, int]] = []
        for left_index, left in enumerate(park_ids):
            for right in park_ids[left_index + 1 :]:
                masks.append((left, right, rng.randint(-DEFAULT_SECURE_MASK_BOUND, DEFAULT_SECURE_MASK_BOUND)))
        masks_by_quantity.append(masks)
    mask_generation_ms = _elapsed_ms(mask_generation_start)

    construction_start = time.perf_counter_ns()
    all_messages: List[Dict[str, int]] = []
    for values, masks in zip(values_by_quantity, masks_by_quantity):
        messages = {
            park_id: int(round(values[park_id] * DEFAULT_SECURE_AGGREGATION_SCALE))
            for park_id in park_ids
        }
        for left, right, mask in masks:
            messages[left] += mask
            messages[right] -= mask
        all_messages.append(messages)
    construction_ms = _elapsed_ms(construction_start)

    sum_start = time.perf_counter_ns()
    results = [sum(messages.values()) / DEFAULT_SECURE_AGGREGATION_SCALE for messages in all_messages]
    secure_sum_ms = _elapsed_ms(sum_start)
    expected = [sum(values.values()) for values in values_by_quantity]
    if any(abs(actual - target) > 2e-6 for actual, target in zip(results, expected)):
        raise RuntimeError("instrumented additive masks did not cancel")
    python_object_bytes = sum(
        sys.getsizeof(value) for messages in all_messages for value in messages.values()
    )
    return {
        "mask_generation_time_ms": mask_generation_ms,
        "masked_message_construction_time_ms": construction_ms,
        "secure_sum_time_ms": secure_sum_ms,
        "python_integer_message_object_bytes": float(python_object_bytes),
    }


def profile_secure_aggregation(
    warmup: int = SECURE_WARMUP,
    repeats: int = SECURE_REPEATS,
    quantity_count: int = 4,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n in N_VALUES:
        park_ids = [f"park_{index:03d}" for index in range(n)]
        values_by_quantity = [
            {
                park_id: ((quantity_index + 1) * (index + 1) * 0.125) * (-1.0 if index % 3 == 0 else 1.0)
                for index, park_id in enumerate(park_ids)
            }
            for quantity_index in range(quantity_count)
        ]

        def plain_once() -> None:
            for values in values_by_quantity:
                sum(values.values())

        actual_rng = random.Random(20260817 + n)

        def secure_once() -> None:
            for values in values_by_quantity:
                secure_masked_sum(values, actual_rng)

        instrument_rng = random.Random(20260818 + n)
        for _ in range(warmup):
            plain_once()
            secure_once()
            _instrument_masked_aggregation(values_by_quantity, instrument_rng)

        print(f"secure aggregation benchmark N={n} K={quantity_count} repeats={repeats}")
        for repeat_id in range(repeats):
            plain_start = time.perf_counter_ns()
            plain_once()
            plain_ms = _elapsed_ms(plain_start)

            secure_start = time.perf_counter_ns()
            secure_once()
            secure_ms = _elapsed_ms(secure_start)
            components = _instrument_masked_aggregation(values_by_quantity, instrument_rng)
            overhead = secure_ms - plain_ms
            rows.append(
                {
                    "N": n,
                    "repeat_id": repeat_id,
                    "number_of_secure_aggregated_quantities": quantity_count,
                    "number_of_pairwise_masks": quantity_count * n * (n - 1) // 2,
                    "masked_values_received_by_coordinator": quantity_count * n,
                    "masked_value_runtime_type": "Python int (fixed-point value scaled by 1e6)",
                    "plain_aggregation_time_ms": plain_ms,
                    **components,
                    "total_secure_aggregation_time_ms": secure_ms,
                    "secure_aggregation_overhead_ms": overhead,
                    "secure_aggregation_overhead_percent": 100.0 * overhead / max(plain_ms, EPS),
                    "logical_masked_payload_bytes_per_step": quantity_count * n * 8,
                    "logical_payload_encoding_assumption": "signed int64; code has no serializer/network transport",
                    "cpu_peak_memory_mb": 0.0,
                    "cpu_peak_memory_delta_mb": 0.0,
                }
            )

        with PeakRSSSampler() as sampler:
            for _ in range(max(100, repeats)):
                secure_once()
        for row in rows[-repeats:]:
            row["cpu_peak_memory_mb"] = sampler.peak_mb
            row["cpu_peak_memory_delta_mb"] = sampler.delta_mb
        secure_stats = _describe(
            [float(row["total_secure_aggregation_time_ms"]) for row in rows[-repeats:]]
        )
        print(
            f"  secure_ms mean={secure_stats['mean']:.6f} "
            f"p95={secure_stats['p95']:.6f} masks={quantity_count * n * (n - 1) // 2}"
        )
    return rows


def _group_stats(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    value_field: str,
) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[float]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(float(row[value_field]))
    result: List[Dict[str, Any]] = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        result.append(
            {
                **{field: value for field, value in zip(group_fields, key)},
                **_describe(values),
            }
        )
    return result


def _plot_federation_runtime(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    stats = _group_stats(rows, ("method", "N"), "total_federation_time_ms")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in ("FedAvg-SP-RGNN-CSAC", "FedProx-SP-RGNN-CSAC", "FedRG-MARL"):
        subset = sorted((row for row in stats if row["method"] == method), key=lambda row: int(row["N"]))
        ax.errorbar(
            [int(row["N"]) for row in subset],
            [row["mean"] for row in subset],
            yerr=[row["std"] for row in subset],
            marker="o",
            capsize=3,
            label=method.replace("-SP-RGNN-CSAC", ""),
        )
    ax.set_xlabel("Number of parks, N")
    ax.set_ylabel("In-process federation round time (ms)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_communication(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method in ("FedAvg-SP-RGNN-CSAC", "FedProx-SP-RGNN-CSAC", "FedRG-MARL"):
        subset = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["N"]))
        ax.plot(
            [int(row["N"]) for row in subset],
            [float(row["total_logical_mb_decimal"]) for row in subset],
            marker="o",
            label=method.replace("-SP-RGNN-CSAC", ""),
        )
    ax.set_xlabel("Number of parks, N")
    ax.set_ylabel("Logical communication per round (MB)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_secure_runtime(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    plain = _group_stats(rows, ("N",), "plain_aggregation_time_ms")
    secure = _group_stats(rows, ("N",), "total_secure_aggregation_time_ms")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot([int(row["N"]) for row in plain], [row["mean"] for row in plain], marker="o", label="Plain sum")
    ax.plot([int(row["N"]) for row in secure], [row["mean"] for row in secure], marker="o", label="Additive masking")
    ax.set_xlabel("Number of parks, N")
    ax.set_ylabel("Aggregation time for K=4 scalars (ms)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _format_stats(stats: Mapping[str, float]) -> str:
    return (
        f"mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
        f"median={stats['median']:.6f}, p95={stats['p95']:.6f}, max={stats['max']:.6f}"
    )


def write_summary(
    path: Path,
    inventory: Sequence[Mapping[str, Any]],
    parameter_summary: Mapping[str, Any],
    communication_rows: Sequence[Mapping[str, Any]],
    federation_rows: Sequence[Mapping[str, Any]],
    secure_rows: Sequence[Mapping[str, Any]],
    online_rows: Sequence[Mapping[str, Any]],
    hardware: Mapping[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("Reviewer 2 Comment #8: communication/computation overhead and scalability profiling")
    lines.append("")
    lines.append("Scope: checkpoint-only, single-process CPU profiling; no training and no network sockets.")
    lines.append("")
    lines.append("HARDWARE / SOFTWARE")
    for key, value in hardware.items():
        lines.append(f"{key}: {value}")

    lines.append("")
    lines.append("FEDERATION PARAMETER INVENTORY")
    for key, value in parameter_summary.items():
        lines.append(f"{key}: {value}")
    lines.append("tensor keys:")
    for row in inventory:
        lines.append(
            f"  {row['tensor_name']} shape={row['shape']} dtype={row['dtype']} "
            f"numel={row['numel']} bytes={row['bytes']}"
        )
    lines.append(
        "Excluded by get_shared_state/_export_relation_shared_state: private adapters, node/type embeddings, "
        "relation_logits, root transforms, action heads, reward critics, cost critics, log_alpha, log_lambda, "
        "optimizer states, replay buffer, and raw trajectories."
    )

    lines.append("")
    lines.append("ACTUAL FEDRG DATA FLOW")
    lines.append("1. Each client exposes get_shared_state()['actor_backbone'] to the in-process coordinator.")
    lines.append("2. Coordinator calls each target agent on every non-self source backbone: N(N-1) evaluations.")
    lines.append("3. Coordinator updates target-specific fed_logits and constructs one weighted candidate per target.")
    lines.append("4. Each target evaluates its candidate; coordinator updates rho/gating; target soft-loads the candidate.")
    lines.append("5. Current code uses Python calls/shared memory only; no measured network latency or throughput.")

    lines.append("")
    lines.append("COMMUNICATION (logical distributed mapping, decimal MB)")
    for row in communication_rows:
        if int(row["N"]) in (3, 50):
            lines.append(
                f"{row['method']} N={row['N']}: upload={int(row['upload_bytes_per_round'])/1e6:.6f} MB, "
                f"download={int(row['download_bytes_per_round'])/1e6:.6f} MB, "
                f"total={int(row['total_logical_bytes_per_round'])/1e6:.6f} MB"
            )

    lines.append("")
    lines.append("FEDERATION RUNTIME (ms)")
    lines.append(
        "Repeat protocol: all timed component functions were warmed up 10 times. "
        "FedAvg/FedProx used 10 complete-round warm-ups; FedRG used 10 warm-ups per real component "
        "because complete-round cost grows quadratically."
    )
    repeat_counts = {
        n: len({
            int(row["repeat_id"])
            for row in federation_rows
            if row["method"] == "FedRG-MARL" and int(row["N"]) == n
        })
        for n in N_VALUES
    }
    lines.append(
        "FedRG full-round formal repeats: "
        + ", ".join(f"N={n}: {repeat_counts[n]}" for n in N_VALUES)
        + ". The requested target was 100 repeats; N>3 uses fewer real repeats because one N=50 "
          "round takes about 331 s. No timings were extrapolated or fabricated."
    )
    lines.append(
        "Consequently, dispersion statistics for N=5/10/20 are based on 20/5/2 rounds; "
        "N=50 has one observation, so its std/median/p95 do not estimate run-to-run variability."
    )
    for method in ("FedAvg-SP-RGNN-CSAC", "FedProx-SP-RGNN-CSAC", "FedRG-MARL"):
        for n in N_VALUES:
            values = [
                float(row["total_federation_time_ms"])
                for row in federation_rows
                if row["method"] == method and int(row["N"]) == n
            ]
            memory = next(
                float(row["cpu_peak_memory_mb"])
                for row in federation_rows
                if row["method"] == method and int(row["N"]) == n
            )
            lines.append(f"{method} N={n}: {_format_stats(_describe(values))}, peak_RSS={memory:.3f} MB")

    lines.append("")
    lines.append("PERSONALIZED SOURCE EVALUATIONS")
    for n in N_VALUES:
        lines.append(f"N={n}: target-source={n*(n-1)}, candidate={n}")
    lines.append("Source scoring is O(N^2 * C_score(B)); candidate construction is O(N^2 * P).")
    lines.append("Logical FedRG communication under target-local scoring is N(N+1)P bytes = O(N^2 P).")
    lines.append("FedAvg/FedProx aggregation and communication are O(NP); FedProx changes local loss, not federation messages.")

    lines.append("")
    lines.append("SECURE AGGREGATION")
    lines.append("Implementation level: fixed-point additive-mask simulation using Python random.Random and Python ints.")
    lines.append("Not included: cryptographic key exchange, PRG setup, authentication, dropout recovery, public-key operations, sockets.")
    lines.append("K=4 in a triggered TR step: raw exchange, controllable capacity, preference capacity, responsibility.")
    lines.append("Pairwise masks per triggered step: K*N(N-1)/2; masked scalar messages: K*N.")
    for n in N_VALUES:
        subset = [row for row in secure_rows if int(row["N"]) == n]
        plain = _describe([float(row["plain_aggregation_time_ms"]) for row in subset])
        secure = _describe([float(row["total_secure_aggregation_time_ms"]) for row in subset])
        overhead = _describe([float(row["secure_aggregation_overhead_ms"]) for row in subset])
        lines.append(
            f"N={n}: masks={4*n*(n-1)//2}, plain_mean={plain['mean']:.6f} ms, "
            f"masked_mean={secure['mean']:.6f} ms, absolute_overhead_mean={overhead['mean']:.6f} ms, "
            f"logical_int64_payload={4*n*8} bytes"
        )
    lines.append("Relative percentages are unstable because the plain Python sum baseline is near timer resolution; use absolute overhead.")

    lines.append("")
    lines.append("ONLINE LATENCY (1000 real three-park decision steps, ms)")
    online_fields = (
        "local_actor_inference_ms",
        "action_decoding_feasibility_ms",
        "local_cs_projection_ms",
        "regional_masking_aggregation_ms",
        "regional_tr_coordination_excluding_masks_ms",
        "final_local_execution_update_ms",
        "end_to_end_decision_ms",
    )
    for field in online_fields:
        lines.append(f"{field}: {_format_stats(_describe([float(row[field]) for row in online_rows]))}")
    end_to_end = _describe([float(row["end_to_end_decision_ms"]) for row in online_rows])
    lines.append(f"Scheduling interval: 900000 ms; mean latency fraction={100*end_to_end['mean']/900000:.9f}%")

    lines.append("")
    lines.append("SCALING INTERPRETATION")
    lines.append("This is a systems-level benchmark using real model/scoring code and replicated client interfaces, not N-park scheduling validation.")
    lines.append(
        "The results support only a limited small/moderate-federation feasibility statement (for example N<=10 "
        "in offline federation on this machine). They do not support broad scalability: the straightforward "
        "pairwise personalized scorer takes about 331 s at N=50 and has O(N^2) computation and logical communication."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_profile(
    fed_warmup: int = FED_WARMUP,
    fed_repeats: int | Mapping[int, int] = FED_REPEATS_BY_N,
    secure_warmup: int = SECURE_WARMUP,
    secure_repeats: int = SECURE_REPEATS,
    online_warmup: int = ONLINE_WARMUP_STEPS,
    online_repeats: int = ONLINE_REPEATS,
) -> Dict[str, Path]:
    torch.set_num_threads(1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    hardware = _hardware_software()
    agents, env_config = _load_fedrg_agents()
    inventory, parameter_summary = build_parameter_inventory(agents["residential"])
    communication_rows = build_communication_rows(
        int(parameter_summary["federated_parameter_count"]),
        int(parameter_summary["shared_payload_bytes"]),
    )

    print(f"online latency warmup={online_warmup} repeats={online_repeats}")
    online_rows = profile_online_latency(agents, env_config, online_warmup, online_repeats)
    if min(len(agent.replay_buffer) for agent in agents.values()) < agents["residential"].config.batch_size:
        raise RuntimeError("online profiling did not populate the real scoring replay batch")
    federation_rows = profile_federation_scalability(
        agents,
        int(parameter_summary["federated_parameter_count"]),
        int(parameter_summary["shared_payload_bytes"]),
        communication_rows,
        fed_warmup,
        fed_repeats,
    )
    secure_rows = profile_secure_aggregation(secure_warmup, secure_repeats, quantity_count=4)

    paths = {
        "inventory": OUTPUT_DIR / "r2_8_parameter_inventory.csv",
        "communication": OUTPUT_DIR / "r2_8_communication_scalability.csv",
        "federation": OUTPUT_DIR / "r2_8_federation_scalability.csv",
        "secure": OUTPUT_DIR / "r2_8_secure_aggregation_scalability.csv",
        "online": OUTPUT_DIR / "r2_8_online_latency.csv",
        "summary": OUTPUT_DIR / "r2_8_summary.txt",
        "hardware": OUTPUT_DIR / "r2_8_hardware_software.json",
        "config": OUTPUT_DIR / "r2_8_profiling_config.json",
        "federation_plot": OUTPUT_DIR / "federation_runtime_vs_num_parks.png",
        "communication_plot": OUTPUT_DIR / "communication_vs_num_parks.png",
        "secure_plot": OUTPUT_DIR / "secure_aggregation_runtime_vs_num_parks.png",
    }
    _write_csv(paths["inventory"], inventory)
    _write_csv(paths["communication"], communication_rows)
    _write_csv(paths["federation"], federation_rows)
    _write_csv(paths["secure"], secure_rows)
    _write_csv(paths["online"], online_rows)
    paths["hardware"].write_text(json.dumps(hardware, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["config"].write_text(
        json.dumps(
            {
                "N_values": N_VALUES,
                "fed_warmup": fed_warmup,
                "fed_repeats": fed_repeats,
                "secure_warmup": secure_warmup,
                "secure_repeats": secure_repeats,
                "secure_quantity_count_triggered_step": 4,
                "online_warmup_steps": online_warmup,
                "online_repeats": online_repeats,
                "checkpoint": "SP_RGNN_CSAC-隐私+参数联邦-2/models/fed_full/best",
                "profiling_scope": "single-process CPU, no socket/network transfer",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_federation_runtime(federation_rows, paths["federation_plot"])
    _plot_communication(communication_rows, paths["communication_plot"])
    _plot_secure_runtime(secure_rows, paths["secure_plot"])
    write_summary(
        paths["summary"],
        inventory,
        parameter_summary,
        communication_rows,
        federation_rows,
        secure_rows,
        online_rows,
        hardware,
    )
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reviewer 2 Comment #8 overhead/scalability profiling")
    parser.add_argument("--fed-warmup", type=int, default=FED_WARMUP)
    parser.add_argument(
        "--uniform-fed-repeats",
        type=int,
        default=None,
        help="Override the documented adaptive full-round repeat counts with one count for every N.",
    )
    parser.add_argument("--secure-warmup", type=int, default=SECURE_WARMUP)
    parser.add_argument("--secure-repeats", type=int, default=SECURE_REPEATS)
    parser.add_argument("--online-warmup", type=int, default=ONLINE_WARMUP_STEPS)
    parser.add_argument("--online-repeats", type=int, default=ONLINE_REPEATS)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outputs = run_profile(
        fed_warmup=args.fed_warmup,
        fed_repeats=(args.uniform_fed_repeats if args.uniform_fed_repeats is not None else FED_REPEATS_BY_N),
        secure_warmup=args.secure_warmup,
        secure_repeats=args.secure_repeats,
        online_warmup=args.online_warmup,
        online_repeats=args.online_repeats,
    )
    for name, output in outputs.items():
        print(f"{name}: {output}")
