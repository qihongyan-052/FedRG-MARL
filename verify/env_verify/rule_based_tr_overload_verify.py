from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from env.three_park_charging_env import PARK_TYPES, ThreeParkChargingEnv
from train_three_park_agent import (
    CSVLogger,
    _energy_log_fields,
    _reward_log_fields,
    _training_log_fields,
    set_global_seed,
)
from visualize.train.plot_episode_reward import (
    generate_park_episode_reward_plot,
    generate_total_episode_reward_plot,
)
from visualize.train.plot_episode_reward_components import generate_episode_reward_components_plot
from visualize.train.realtime_projection_csv import realtime_projection_log_fields


@dataclass
class VerifyConfig:
    run_name: str = "rule_based_tr_overload_verify"
    seed: int = 13
    total_episodes: int = 1


def _prepare_verify_directories(root_dir: Path, run_name: str) -> tuple[Path, Path, Path]:
    run_dir = root_dir / "saved" / run_name
    log_dir = run_dir / "log"
    results_dir = run_dir / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, log_dir, results_dir


def _build_overload_joint_action(obs: Dict[str, object]) -> Dict[str, object]:
    joint_action = {"parks": {}}
    for park_type in PARK_TYPES:
        park_graph = obs["park_graphs"][park_type]
        ev_actions = {ev_id: 1.0 for ev_id in park_graph["active_ev_ids"]}
        joint_action["parks"][park_type] = {
            "bes": 1.0,
            "ev": ev_actions,
        }
    return joint_action


def _refresh_verify_visualizations(log_dir: Path, results_dir: Path) -> None:
    generate_total_episode_reward_plot(
        training_log_csv=log_dir / "training_log.csv",
        output_path=results_dir / "episode_total_reward_curve.png",
    )
    generate_park_episode_reward_plot(
        training_log_csv=log_dir / "training_log.csv",
        output_path=results_dir / "episode_park_reward_curve.png",
    )
    generate_episode_reward_components_plot(
        reward_log_csv=log_dir / "reward_log.csv",
        output_path=results_dir / "episode_reward_components.png",
    )


def run_rule_based_tr_overload_verify(config: VerifyConfig) -> None:
    root_dir = Path(__file__).resolve().parents[2]
    _, log_dir, results_dir = _prepare_verify_directories(root_dir=root_dir, run_name=config.run_name)

    set_global_seed(config.seed)
    env = ThreeParkChargingEnv(seed=config.seed)

    energy_logger = CSVLogger(log_dir / "energy_log.csv", _energy_log_fields())
    reward_logger = CSVLogger(log_dir / "reward_log.csv", _reward_log_fields())
    training_logger = CSVLogger(log_dir / "training_log.csv", _training_log_fields())
    realtime_projection_logger = CSVLogger(
        log_dir / "realtime_projection_log.csv",
        realtime_projection_log_fields(),
    )

    try:
        print("env_verify_policy=force_import_overload ev=1 bes=1")
        for episode in range(config.total_episodes):
            episode_seed = config.seed + episode
            obs, reset_info = env.reset(seed=episode_seed)
            done = False
            episode_reward = 0.0
            episode_steps = 0
            episode_park_rewards = {park_type: 0.0 for park_type in PARK_TYPES}
            tr_trigger_count = 0

            while not done:
                joint_action = _build_overload_joint_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(joint_action)
                done = terminated or truncated

                for park_type in PARK_TYPES:
                    episode_park_rewards[park_type] += info["park_reward_breakdown"][park_type]["total_reward"]

                if info["tr_projection"]["triggered"]:
                    tr_trigger_count += 1

                energy_logger.write_row({"episode": episode, **info["energy_log"]})
                reward_logger.write_row({"episode": episode, **info["reward_log"]})
                realtime_projection_logger.write_row({"episode": episode, **info["projection_trace"]})

                obs = next_obs
                episode_reward += reward
                episode_steps += 1

            training_row: Dict[str, object] = {
                "episode": episode,
                "seed": episode_seed,
                "weather": reset_info["weather"],
                "steps": episode_steps,
                "total_profit_reward": 0.0,
                "total_constraint_cost": 0.0,
                "total_reward": episode_reward,
                "mean_lambda": 0.0,
                "total_grid_purchase_cost": 0.0,
                "total_grid_sale_revenue": 0.0,
                "total_tr_projection_penalty": 0.0,
                "total_ev_charge_revenue": 0.0,
                "total_v2g_compensation_cost": 0.0,
                "total_cs_projection_penalty": 0.0,
                "total_user_satisfaction_penalty": 0.0,
                "total_debt_penalty": 0.0,
            }
            for park_type in PARK_TYPES:
                training_row[f"{park_type}_reward"] = episode_park_rewards[park_type]
                training_row[f"{park_type}_profit_reward"] = 0.0
                training_row[f"{park_type}_constraint_cost"] = 0.0
                training_row[f"{park_type}_grid_purchase_cost"] = 0.0
                training_row[f"{park_type}_grid_sale_revenue"] = 0.0
                training_row[f"{park_type}_tr_projection_penalty"] = 0.0
                training_row[f"{park_type}_ev_charge_revenue"] = 0.0
                training_row[f"{park_type}_v2g_compensation_cost"] = 0.0
                training_row[f"{park_type}_cs_projection_penalty"] = 0.0
                training_row[f"{park_type}_user_satisfaction_penalty"] = 0.0
                training_row[f"{park_type}_debt_penalty"] = 0.0
            training_logger.write_row(training_row)

            if (episode + 1) % 5 == 0:
                _refresh_verify_visualizations(log_dir, results_dir)

            print(
                f"episode={episode:03d} "
                f"residential_reward={episode_park_rewards['residential']:.4f} "
                f"office_reward={episode_park_rewards['office']:.4f} "
                f"commercial_reward={episode_park_rewards['commercial']:.4f} "
                f"tr_trigger_count={tr_trigger_count} "
                f"total_reward={episode_reward:.4f}"
            )

        _refresh_verify_visualizations(log_dir, results_dir)
    finally:
        energy_logger.close()
        reward_logger.close()
        training_logger.close()
        realtime_projection_logger.close()


if __name__ == "__main__":
    run_rule_based_tr_overload_verify(VerifyConfig())
