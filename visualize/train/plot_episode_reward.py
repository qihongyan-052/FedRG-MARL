from __future__ import annotations

from pathlib import Path
import csv
import io

import matplotlib.pyplot as plt

def _moving_average(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
    result: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    raw_text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    sanitized_text = raw_text.replace("\x00", "")
    if not sanitized_text.strip():
        return []
    return [row for row in csv.DictReader(io.StringIO(sanitized_text)) if any((value or "").strip() for value in row.values())]


def _metric_from_row(row: dict[str, str], key: str, fallback_key: str | None = None) -> float:
    explicit_value = row.get(key)
    if explicit_value not in (None, ""):
        return float(explicit_value)
    if fallback_key is None:
        return 0.0
    fallback_value = row.get(fallback_key)
    return float(fallback_value) if fallback_value not in (None, "") else 0.0


def _load_episode_metrics(
    training_log_csv: str | Path,
) -> tuple[list[int], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float], list[float]]:
    training_log_csv = Path(training_log_csv)
    rows = _read_rows(training_log_csv)
    if not rows:
        return [], [], [], [], [], [], [], [], [], [], []

    episodes = [int(row["episode"]) for row in rows]
    total_reward = [_metric_from_row(row, "total_reward") for row in rows]
    residential_reward = [_metric_from_row(row, "residential_reward") for row in rows]
    office_reward = [_metric_from_row(row, "office_reward") for row in rows]
    commercial_reward = [_metric_from_row(row, "commercial_reward") for row in rows]
    residential_profit = [_metric_from_row(row, "residential_profit_reward", "residential_reward") for row in rows]
    office_profit = [_metric_from_row(row, "office_profit_reward", "office_reward") for row in rows]
    commercial_profit = [_metric_from_row(row, "commercial_profit_reward", "commercial_reward") for row in rows]
    residential_penalty = [_metric_from_row(row, "residential_constraint_cost") for row in rows]
    office_penalty = [_metric_from_row(row, "office_constraint_cost") for row in rows]
    commercial_penalty = [_metric_from_row(row, "commercial_constraint_cost") for row in rows]
    return (
        episodes,
        total_reward,
        residential_reward,
        office_reward,
        commercial_reward,
        residential_profit,
        office_profit,
        commercial_profit,
        residential_penalty,
        office_penalty,
        commercial_penalty,
    )


def generate_total_episode_reward_plot(training_log_csv: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path)
    episodes, total_reward, residential_reward, office_reward, commercial_reward, _, _, _, _, _, _ = _load_episode_metrics(training_log_csv)
    if not episodes:
        return

    total_reward_ma = _moving_average(total_reward, window=min(5, len(total_reward)))
    residential_reward_ma = _moving_average(residential_reward, window=min(5, len(residential_reward)))
    office_reward_ma = _moving_average(office_reward, window=min(5, len(office_reward)))
    commercial_reward_ma = _moving_average(commercial_reward, window=min(5, len(commercial_reward)))

    plt.figure(figsize=(12, 7))
    plt.plot(episodes, total_reward_ma, label="total_reward (MA5)", color="#0d3b66", linewidth=3.2)
    plt.plot(episodes, residential_reward_ma, label="residential_reward (MA5)", color="#e76f51", linewidth=1.8, linestyle="--", alpha=0.9)
    plt.plot(episodes, office_reward_ma, label="office_reward (MA5)", color="#2a9d8f", linewidth=1.8, linestyle="--", alpha=0.9)
    plt.plot(episodes, commercial_reward_ma, label="commercial_reward (MA5)", color="#f4a261", linewidth=1.8, linestyle="--", alpha=0.9)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Total Episode Reward Curve")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def generate_park_episode_reward_plot(training_log_csv: str | Path, output_path: str | Path) -> None:
    output_path = Path(output_path)
    episodes, _, _, _, _, residential_profit, office_profit, commercial_profit, residential_penalty, office_penalty, commercial_penalty = _load_episode_metrics(training_log_csv)
    if not episodes:
        return

    residential_profit_ma = _moving_average(residential_profit, window=min(5, len(residential_profit)))
    office_profit_ma = _moving_average(office_profit, window=min(5, len(office_profit)))
    commercial_profit_ma = _moving_average(commercial_profit, window=min(5, len(commercial_profit)))
    residential_penalty_ma = _moving_average(residential_penalty, window=min(5, len(residential_penalty)))
    office_penalty_ma = _moving_average(office_penalty, window=min(5, len(office_penalty)))
    commercial_penalty_ma = _moving_average(commercial_penalty, window=min(5, len(commercial_penalty)))

    plt.figure(figsize=(12, 7))
    plt.plot(episodes, residential_profit_ma, label="residential_profit_reward (MA5)", color="#e76f51", linewidth=2.0)
    plt.plot(episodes, office_profit_ma, label="office_profit_reward (MA5)", color="#2a9d8f", linewidth=2.0)
    plt.plot(episodes, commercial_profit_ma, label="commercial_profit_reward (MA5)", color="#f4a261", linewidth=2.0)
    plt.plot(episodes, residential_penalty_ma, label="residential_constraint_cost (MA5)", color="#c1121f", linewidth=2.0, linestyle="--")
    plt.plot(episodes, office_penalty_ma, label="office_constraint_cost (MA5)", color="#1d3557", linewidth=2.0, linestyle="--")
    plt.plot(episodes, commercial_penalty_ma, label="commercial_constraint_cost (MA5)", color="#6a4c93", linewidth=2.0, linestyle="--")
    plt.xlabel("Episode")
    plt.ylabel("Episode-Summed Value")
    plt.title("Park Profit and Penalty Curves")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
