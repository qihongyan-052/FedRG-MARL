from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict
import json

from algorithm.greedy_max_charge import GreedyMaxChargeConfig, LocalGreedyMaxChargeAgent
from env.three_park_charging_env import PARK_TYPES, ThreeParkChargingEnv
from evaluate_three_park_agent import _build_step_eval_row, _write_step_eval_csv
from train_three_park_agent import TrainingConfig, build_joint_action, configure_environment, set_global_seed


@dataclass(frozen=True)
class GreedyMaxChargeEvaluationConfig:
    run_name: str = "max_charge-strong"
    privacy_mode: str = "strong"
    seed: int = 40
    eval_episodes: int = 30
    save_csv: bool = True


def _build_local_agents(config: GreedyMaxChargeEvaluationConfig) -> Dict[str, LocalGreedyMaxChargeAgent]:
    del config
    return {
        park_type: LocalGreedyMaxChargeAgent(GreedyMaxChargeConfig(park_type=park_type))
        for park_type in PARK_TYPES
    }


def _write_config(path: Path, config: GreedyMaxChargeEvaluationConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")


def run_greedy_max_charge_evaluation(config: GreedyMaxChargeEvaluationConfig) -> None:
    root_dir = Path(__file__).resolve().parent
    output_dir = root_dir / "saved" / config.run_name / "evaluates"
    set_global_seed(config.seed)

    env = ThreeParkChargingEnv(seed=config.seed)
    configure_environment(
        env,
        TrainingConfig(
            run_name=config.run_name,
            algorithm_variant="greedy_max_charge",
            privacy_mode=config.privacy_mode,
            enable_federation=False,
            use_central_tr_hgt_agent=False,
        ),
    )
    local_agents = _build_local_agents(config)
    env.attach_local_agents(local_agents)
    step_rows: list[Dict[str, object]] = []

    print(
        f"evaluate run_name={config.run_name} "
        f"algorithm_variant=greedy_max_charge "
        f"privacy_mode={config.privacy_mode}"
    )

    for episode in range(config.eval_episodes):
        episode_seed = config.seed + episode
        obs, reset_info = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0.0
        episode_profit = 0.0
        episode_constraint = 0.0
        while not done:
            joint_action, raw_node_actions = build_joint_action(
                local_agents,
                obs,
                deterministic=True,
                return_raw_action=True,
            )
            next_obs, reward, terminated, truncated, info = env.step(
                joint_action,
                raw_node_actions=raw_node_actions,
            )
            done = terminated or truncated
            reward_row = info["reward_log"]
            step_rows.append(
                _build_step_eval_row(
                    episode=episode,
                    episode_seed=episode_seed,
                    reward_row=reward_row,
                    energy_row=info["energy_log"],
                    transition_info=info["transition"],
                )
            )
            episode_reward += float(reward)
            episode_profit += float(reward_row["total_profit_reward"])
            episode_constraint += float(reward_row["total_constraint_cost"])
            obs = next_obs

        print(
            f"eval_episode={episode:03d} "
            f"weather={reset_info['weather']} "
            f"reward={episode_reward:.4f} "
            f"profit={episode_profit:.4f} "
            f"constraint={episode_constraint:.4f}"
        )

    if config.save_csv:
        output_path = output_dir / "evaluation_steps.csv"
        _write_step_eval_csv(output_path, step_rows)
        _write_config(output_dir / "greedy_max_charge_config.json", config)
        print(f"saved evaluation csv: {output_path}")

    completed_episodes = max(1, config.eval_episodes)
    mean_reward = sum(float(row["total_immediate_reward"]) for row in step_rows) / completed_episodes
    mean_profit = sum(float(row["total_profit_reward"]) for row in step_rows) / completed_episodes
    mean_constraint = sum(float(row["total_constraint_cost"]) for row in step_rows) / completed_episodes
    print(
        f"evaluation_summary "
        f"mean_reward={mean_reward:.4f} "
        f"mean_profit={mean_profit:.4f} "
        f"mean_constraint={mean_constraint:.4f}"
    )


if __name__ == "__main__":
    run_greedy_max_charge_evaluation(GreedyMaxChargeEvaluationConfig())
