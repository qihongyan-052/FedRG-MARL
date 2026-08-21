from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from env.three_park_charging_env import PARK_TYPES, ThreeParkChargingEnv
from evaluate_three_park_agent import (
    EvaluationConfig,
    _build_agents_from_saved_models,
    _build_env_training_config,
    _build_step_eval_row,
    _infer_saved_experiment_signature,
    _resolve_model_dir,
    _step_eval_fields,
    _validate_eval_config_against_saved_models,
    _validate_saved_route_signature,
)
from train_three_park_agent import build_joint_action, configure_environment, set_global_seed


EPS = 1e-9
DEFAULT_SCENARIO_COUNT = 30
DEFAULT_BASE_SEED = 10


@dataclass(frozen=True)
class MethodSpec:
    method: str
    run_name: str
    algorithm_variant: str
    enable_federation: bool
    checkpoint_kind: str = "best"

    @property
    def checkpoint_id(self) -> str:
        family = "fed_full" if self.enable_federation else "local_full"
        return f"{self.run_name}/models/{family}/{self.checkpoint_kind}"


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        method="GNN-SAC",
        run_name="gnn_sac-隐私+不联邦-2",
        algorithm_variant="gnn_sac",
        enable_federation=False,
    ),
    MethodSpec(
        method="CSAC-GNN",
        run_name="gnn_csac-隐私+不联邦-2",
        algorithm_variant="gnn_csac",
        enable_federation=False,
    ),
    MethodSpec(
        method="FedRG-MARL",
        run_name="SP_RGNN_CSAC-隐私+参数联邦-2",
        algorithm_variant="sp_rgnn_csac",
        enable_federation=True,
    ),
)


class ConstraintPressureEvaluationEnv(ThreeParkChargingEnv):
    """Expose existing projection diagnostics without changing projection behavior."""

    def _build_projection_trace_row(
        self,
        cs_results: Dict[str, Any],
        tr_summary: Any,
    ) -> Dict[str, Any]:
        row = super()._build_projection_trace_row(cs_results, tr_summary)
        for park_type in PARK_TYPES:
            cs_result = cs_results[park_type]
            tr_result = tr_summary.park_results_by_id[park_type]
            row[f"{park_type}_raw_requested_exchange_kwh"] = cs_result.raw_net_after_pv_kwh
            row[f"{park_type}_cs_projected_exchange_kwh"] = cs_result.projected_net_after_pv_kwh
            row[f"{park_type}_tr_projected_exchange_kwh"] = tr_result.projected_park_net_kwh
        return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fields = list(fieldnames or (list(rows[0].keys()) if rows else []))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=resolved_fields)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _scenario_fingerprint(env: ThreeParkChargingEnv) -> str:
    payload = {
        "daily_pv": env.daily_pv,
        "daily_episode": env.daily_episode,
        "initial_bes_soc": {
            park_type: env.runtime_states[park_type].bes_soc for park_type in PARK_TYPES
        },
        "grid_price_table": env.grid_price_table,
    }
    canonical = json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def _conditional_mean(values: Iterable[float], tol: float) -> float:
    positive = [value for value in values if value > tol]
    return _safe_mean(positive)


def _build_park_step_rows(
    method: MethodSpec,
    scenario_id: str,
    evaluation_seed: int,
    timestep: int,
    info: Dict[str, Any],
    tol: float,
) -> List[Dict[str, Any]]:
    trace = info["projection_trace"]
    reward_row = info["reward_log"]
    transition = info["transition"]
    rows: List[Dict[str, Any]] = []
    for park_type in PARK_TYPES:
        hat_g = float(trace[f"{park_type}_raw_requested_exchange_kwh"])
        tilde_g = float(trace[f"{park_type}_cs_projected_exchange_kwh"])
        regional_projected_g = float(trace[f"{park_type}_tr_projected_exchange_kwh"])
        g = float(transition["executed_flows"][park_type]["park_grid_exchange_kwh"])
        cs_limit = float(trace[f"{park_type}_cs_limit_kwh"])
        cs_exceedance = max(abs(hat_g) - cs_limit, 0.0)
        cs_adjustment = abs(hat_g - tilde_g)
        tr_adjustment = abs(tilde_g - g)
        rows.append(
            {
                "method": method.method,
                "checkpoint_id": method.checkpoint_id,
                "scenario_id": scenario_id,
                "evaluation_seed": evaluation_seed,
                "timestep": timestep,
                "park_id": park_type,
                "physical_unit": "kWh/step",
                "hat_g": hat_g,
                "cs_limit": cs_limit,
                "cs_exceedance": cs_exceedance,
                "cs_violation": int(cs_exceedance > tol),
                "tilde_g": tilde_g,
                "cs_adjustment": cs_adjustment,
                "cs_adjustment_flag": int(cs_adjustment > tol),
                "g": g,
                "regional_projected_g": regional_projected_g,
                "execution_residual": g - regional_projected_g,
                "tr_adjustment": tr_adjustment,
                "normalized_cs_exceedance": cs_exceedance / max(cs_limit, EPS),
                "normalized_cs_adjustment": cs_adjustment / max(cs_limit, EPS),
                "reward": float(reward_row[f"{park_type}_immediate_reward"]),
                "constraint_cost": float(reward_row[f"{park_type}_constraint_cost"]),
                "profit": float(reward_row[f"{park_type}_profit_reward"]),
            }
        )
    return rows


def _build_regional_step_row(
    method: MethodSpec,
    scenario_id: str,
    evaluation_seed: int,
    timestep: int,
    park_rows: Sequence[Mapping[str, Any]],
    info: Dict[str, Any],
    tol: float,
) -> Dict[str, Any]:
    trace = info["projection_trace"]
    tr_pre_exchange = sum(float(row["tilde_g"]) for row in park_rows)
    tr_limit = float(trace["tr_limit_kwh"])
    tr_overload = max(abs(tr_pre_exchange) - tr_limit, 0.0)
    total_tr_adjustment = sum(float(row["tr_adjustment"]) for row in park_rows)
    tr_final_exchange = sum(float(row["g"]) for row in park_rows)
    aggregate_flexibility = float(trace["tr_total_capacity_kwh"])
    tr_residual = float(trace["tr_infeasible_residual_kwh"])
    return {
        "method": method.method,
        "checkpoint_id": method.checkpoint_id,
        "scenario_id": scenario_id,
        "evaluation_seed": evaluation_seed,
        "timestep": timestep,
        "physical_unit": "kWh/step",
        "tr_pre_exchange": tr_pre_exchange,
        "tr_limit": tr_limit,
        "tr_overload": tr_overload,
        "tr_violation": int(tr_overload > tol),
        "total_tr_adjustment": total_tr_adjustment,
        "tr_final_exchange": tr_final_exchange,
        "final_tr_violation": int(abs(tr_final_exchange) > tr_limit + tol),
        "normalized_tr_overload": tr_overload / max(tr_limit, EPS),
        "normalized_tr_adjustment": total_tr_adjustment / max(tr_limit, EPS),
        "aggregate_flexibility": aggregate_flexibility,
        "tr_residual": tr_residual,
        "severe_event": int(tr_residual > tol),
        "tr_triggered": int(bool(trace["tr_triggered"])),
    }


def _build_scenario_summary(
    method: MethodSpec,
    scenario_id: str,
    evaluation_seed: int,
    park_rows: Sequence[Mapping[str, Any]],
    regional_rows: Sequence[Mapping[str, Any]],
    original_rows: Sequence[Mapping[str, Any]],
    tol: float,
) -> Dict[str, Any]:
    cs_exceedances = [float(row["cs_exceedance"]) for row in park_rows]
    cs_adjustments = [float(row["cs_adjustment"]) for row in park_rows]
    tr_overloads = [float(row["tr_overload"]) for row in regional_rows]
    tr_adjustments = [float(row["total_tr_adjustment"]) for row in regional_rows]
    step_count = max(len(regional_rows), 1)
    return {
        "method": method.method,
        "checkpoint_id": method.checkpoint_id,
        "scenario_id": scenario_id,
        "evaluation_seed": evaluation_seed,
        "cs_violation_rate": _safe_mean(float(row["cs_violation"]) for row in park_rows),
        "mean_cs_exceedance_kwh": _safe_mean(cs_exceedances),
        "conditional_mean_cs_exceedance_kwh": _conditional_mean(cs_exceedances, tol),
        "cs_adjustment_rate": _safe_mean(float(row["cs_adjustment_flag"]) for row in park_rows),
        "mean_cs_adjustment_kwh": _safe_mean(cs_adjustments),
        "tr_overload_rate": _safe_mean(float(row["tr_violation"]) for row in regional_rows),
        "mean_tr_overload_kwh": _safe_mean(tr_overloads),
        "conditional_mean_tr_overload_kwh": _conditional_mean(tr_overloads, tol),
        "mean_tr_adjustment_per_step_kwh": _safe_mean(tr_adjustments),
        "total_tr_adjustment_per_day_kwh": sum(tr_adjustments),
        "severe_event_count": sum(int(row["severe_event"]) for row in regional_rows),
        "severe_event_rate": sum(int(row["severe_event"]) for row in regional_rows) / step_count,
        "final_tr_violation_rate": sum(int(row["final_tr_violation"]) for row in regional_rows) / step_count,
        "total_profit": sum(float(row["total_profit_reward"]) for row in original_rows),
        "user_cost": sum(float(row["total_user_payment_cost"]) for row in original_rows),
        "constraint_cost": sum(float(row["total_constraint_cost"]) for row in original_rows),
    }


def _build_method_summary(scenario_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rate_metrics = {
        "cs_violation_rate",
        "cs_adjustment_rate",
        "tr_overload_rate",
        "severe_event_rate",
        "final_tr_violation_rate",
    }
    excluded = {"method", "checkpoint_id", "scenario_id", "evaluation_seed"}
    metric_names = [key for key in scenario_rows[0].keys() if key not in excluded]
    summary_rows: List[Dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in scenario_rows}):
        method_rows = [row for row in scenario_rows if row["method"] == method]
        checkpoint_ids = sorted({str(row["checkpoint_id"]) for row in method_rows})
        if len(checkpoint_ids) != 1:
            raise RuntimeError(f"method {method} has multiple checkpoint IDs: {checkpoint_ids}")
        for metric in metric_names:
            values = [float(row[metric]) for row in method_rows]
            scale = 100.0 if metric in rate_metrics else 1.0
            unit = "%" if metric in rate_metrics else ("kWh" if "kwh" in metric else "count_or_cost")
            scaled = [value * scale for value in values]
            summary_rows.append(
                {
                    "method": method,
                    "checkpoint_id": checkpoint_ids[0],
                    "scenario_count": len(method_rows),
                    "metric": f"{metric}_pct" if metric in rate_metrics else metric,
                    "unit": unit,
                    "mean": statistics.mean(scaled),
                    "std": statistics.stdev(scaled) if len(scaled) > 1 else 0.0,
                    "median": statistics.median(scaled),
                    "min": min(scaled),
                    "max": max(scaled),
                }
            )
    return summary_rows


def _select_methods(names: Sequence[str] | None) -> tuple[MethodSpec, ...]:
    if not names:
        return METHODS
    requested = set(names)
    selected = tuple(method for method in METHODS if method.method in requested)
    missing = requested.difference(method.method for method in selected)
    if missing:
        raise ValueError(f"unknown method(s): {sorted(missing)}")
    return selected


def run_constraint_pressure_evaluation(
    scenario_count: int = DEFAULT_SCENARIO_COUNT,
    base_seed: int = DEFAULT_BASE_SEED,
    output_dir: Path | None = None,
    method_names: Sequence[str] | None = None,
) -> Dict[str, Path]:
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive")
    root_dir = Path(__file__).resolve().parent
    output_dir = output_dir or root_dir / "saved" / "reviewer2_comment5" / "evaluates"
    selected_methods = _select_methods(method_names)

    all_park_rows: List[Dict[str, Any]] = []
    all_regional_rows: List[Dict[str, Any]] = []
    all_scenario_rows: List[Dict[str, Any]] = []
    reference_fingerprints: Dict[str, str] = {}
    manifest_rows: List[Dict[str, Any]] = []

    for method in selected_methods:
        config = EvaluationConfig(
            run_name=method.run_name,
            algorithm_variant=method.algorithm_variant,
            enable_federation=method.enable_federation,
            checkpoint_kind=method.checkpoint_kind,
            eval_episodes=scenario_count,
            deterministic=True,
            seed=base_seed,
            save_csv=True,
        )
        model_dir = _resolve_model_dir(root_dir, config.run_name, config.enable_federation, config.checkpoint_kind)
        checkpoint = _validate_eval_config_against_saved_models(model_dir, config)
        saved_signature = _infer_saved_experiment_signature(checkpoint)
        _validate_saved_route_signature(saved_signature, config)
        env_config = _build_env_training_config(config, saved_signature)

        set_global_seed(base_seed)
        local_agents = _build_agents_from_saved_models(
            model_dir=model_dir,
            act_device=config.act_device,
            update_device=config.update_device,
        )
        env = ConstraintPressureEvaluationEnv(seed=base_seed)
        configure_environment(env, env_config)
        env.attach_local_agents(local_agents)
        tol = float(env.reward_consistency_tolerance)
        original_method_rows: List[Dict[str, Any]] = []

        print(f"R2.5 evaluating method={method.method} checkpoint={method.checkpoint_id}")
        for scenario_index in range(scenario_count):
            evaluation_seed = base_seed + scenario_index
            scenario_id = f"scenario_{scenario_index:03d}"
            obs, reset_info = env.reset(seed=evaluation_seed)
            fingerprint = _scenario_fingerprint(env)
            if scenario_id not in reference_fingerprints:
                reference_fingerprints[scenario_id] = fingerprint
                manifest_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "evaluation_seed": evaluation_seed,
                        "weather": reset_info["weather"],
                        "scenario_sha256": fingerprint,
                    }
                )
            elif reference_fingerprints[scenario_id] != fingerprint:
                raise RuntimeError(
                    f"stochastic scenario mismatch for {scenario_id}: "
                    f"expected={reference_fingerprints[scenario_id]}, actual={fingerprint}"
                )

            done = False
            timestep = 0
            scenario_park_rows: List[Dict[str, Any]] = []
            scenario_regional_rows: List[Dict[str, Any]] = []
            scenario_original_rows: List[Dict[str, Any]] = []
            while not done:
                joint_action, raw_node_actions = build_joint_action(
                    local_agents,
                    obs,
                    deterministic=config.deterministic,
                    return_raw_action=True,
                )
                next_obs, _reward, terminated, truncated, info = env.step(
                    joint_action,
                    raw_node_actions=raw_node_actions,
                )
                done = terminated or truncated
                original_row = _build_step_eval_row(
                    episode=scenario_index,
                    episode_seed=evaluation_seed,
                    reward_row=info["reward_log"],
                    energy_row=info["energy_log"],
                    transition_info=info["transition"],
                )
                park_rows = _build_park_step_rows(
                    method=method,
                    scenario_id=scenario_id,
                    evaluation_seed=evaluation_seed,
                    timestep=timestep,
                    info=info,
                    tol=tol,
                )
                regional_row = _build_regional_step_row(
                    method=method,
                    scenario_id=scenario_id,
                    evaluation_seed=evaluation_seed,
                    timestep=timestep,
                    park_rows=park_rows,
                    info=info,
                    tol=tol,
                )
                scenario_original_rows.append(original_row)
                scenario_park_rows.extend(park_rows)
                scenario_regional_rows.append(regional_row)
                obs = next_obs
                timestep += 1

            if timestep != 96:
                raise RuntimeError(f"{method.method} {scenario_id} completed {timestep} steps instead of 96")
            all_park_rows.extend(scenario_park_rows)
            all_regional_rows.extend(scenario_regional_rows)
            original_method_rows.extend(scenario_original_rows)
            all_scenario_rows.append(
                _build_scenario_summary(
                    method=method,
                    scenario_id=scenario_id,
                    evaluation_seed=evaluation_seed,
                    park_rows=scenario_park_rows,
                    regional_rows=scenario_regional_rows,
                    original_rows=scenario_original_rows,
                    tol=tol,
                )
            )
            print(
                f"  completed {scenario_id} seed={evaluation_seed} "
                f"weather={reset_info['weather']} fingerprint={fingerprint[:12]}"
            )

        _write_csv(
            output_dir / "original_evaluation" / method.method / "evaluation_steps.csv",
            original_method_rows,
            fieldnames=_step_eval_fields(),
        )

    method_summary_rows = _build_method_summary(all_scenario_rows)
    paths = {
        "park": output_dir / "r2_5_park_step_metrics.csv",
        "regional": output_dir / "r2_5_regional_step_metrics.csv",
        "scenario": output_dir / "r2_5_scenario_summary.csv",
        "summary": output_dir / "r2_5_summary_by_method.csv",
        "manifest": output_dir / "r2_5_scenario_manifest.csv",
    }
    _write_csv(paths["park"], all_park_rows)
    _write_csv(paths["regional"], all_regional_rows)
    _write_csv(paths["scenario"], all_scenario_rows)
    _write_csv(paths["summary"], method_summary_rows)
    _write_csv(paths["manifest"], manifest_rows)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reviewer 2 Comment #5 constraint-pressure evaluation")
    parser.add_argument("--scenarios", type=int, default=DEFAULT_SCENARIO_COUNT)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[method.method for method in METHODS],
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generated = run_constraint_pressure_evaluation(
        scenario_count=args.scenarios,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        method_names=args.methods,
    )
    for name, path in generated.items():
        print(f"{name}: {path}")
