from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import io

import matplotlib.pyplot as plt


PROFIT_COLUMNS = {
    "total_profit_reward": ("System Total Profit", "#0d3b66", 3.0),
    "residential_profit_reward": ("Residential Park Profit", "#e76f51", 2.0),
    "office_profit_reward": ("Office Park Profit", "#2a9d8f", 2.0),
    "commercial_profit_reward": ("Commercial Park Profit", "#f4a261", 2.0),
}


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    raw_text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    sanitized_text = raw_text.replace("\x00", "")
    if not sanitized_text.strip():
        return []
    return [
        row
        for row in csv.DictReader(io.StringIO(sanitized_text))
        if any((value or "").strip() for value in row.values())
    ]


def _moving_average(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
    result: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        result.append(sum(chunk) / len(chunk))
    return result


def _aggregate_profit_by_episode(rows: list[dict[str, str]], column: str) -> tuple[list[int], list[float]]:
    by_episode: dict[int, float] = defaultdict(float)
    for row in rows:
        by_episode[int(row["episode"])] += float(row[column])
    episodes = sorted(by_episode.keys())
    profits = [by_episode[episode] for episode in episodes]
    return episodes, profits


def _plot_combined_curves(
    curve_data: dict[str, tuple[list[int], list[float]]],
    output_path: Path,
) -> None:
    plt.figure(figsize=(13, 8))
    for column, (label, color, line_width) in PROFIT_COLUMNS.items():
        episodes, profits = curve_data[column]
        profit_ma = _moving_average(profits, window=min(9, len(profits)))
        plt.plot(episodes, profit_ma, color=color, linewidth=line_width, label=f"{label} (MA9)")

    plt.xlabel("Episode")
    plt.ylabel("Profit")
    plt.title("Profit Curves of Three Parks and Overall System")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def generate_profit_curves(training_reward_csv: str | Path, output_dir: str | Path) -> None:
    csv_path = Path(training_reward_csv)
    output_dir = Path(output_dir)
    rows = _read_rows(csv_path)
    if not rows:
        return

    curve_data: dict[str, tuple[list[int], list[float]]] = {}
    for column in PROFIT_COLUMNS:
        curve_data[column] = _aggregate_profit_by_episode(rows, column)

    _plot_combined_curves(curve_data, output_dir / "combined_profit_curves.png")


if __name__ == "__main__":
    generate_profit_curves(
        training_reward_csv=Path("saved/test-5/log/training_reward.csv"),
        output_dir=Path("图片"),
    )
