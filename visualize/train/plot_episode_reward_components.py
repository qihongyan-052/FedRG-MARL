from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import io

import matplotlib.pyplot as plt

COMPONENT_COLUMNS = [
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

def _component_plot_config() -> dict[str, str]:
    return {
        "total_grid_purchase_cost": "grid_purchase_cost",
        "total_grid_sale_revenue": "grid_sale_revenue",
        "total_v2g_compensation_cost": "v2g_compensation_cost",
        "total_ev_charge_revenue": "ev_charge_revenue",
        "total_cs_projection_penalty": "cs_projection_penalty",
        "total_tr_projection_penalty": "tr_projection_penalty",
        "total_debt_penalty": "debt_penalty",
        "total_soc_shortfall_penalty": "soc_shortfall_penalty",
        "total_bes_terminal_penalty": "bes_terminal_penalty",
    }


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    raw_text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    sanitized_text = raw_text.replace("\x00", "")
    if not sanitized_text.strip():
        return []
    return [row for row in csv.DictReader(io.StringIO(sanitized_text)) if any((value or "").strip() for value in row.values())]


def generate_episode_reward_components_plot(reward_log_csv: str | Path, output_path: str | Path) -> None:
    reward_log_csv = Path(reward_log_csv)
    output_path = Path(output_path)
    rows = _read_rows(reward_log_csv)
    if not rows:
        return
    component_labels = _component_plot_config()

    by_episode: dict[int, dict[str, float]] = defaultdict(lambda: {col: 0.0 for col in COMPONENT_COLUMNS})
    for row in rows:
        episode = int(row["episode"])
        for col in COMPONENT_COLUMNS:
            by_episode[episode][col] += float(row[col])

    episodes = sorted(by_episode.keys())
    plt.figure(figsize=(13, 8))
    for col in COMPONENT_COLUMNS:
        label = component_labels[col]
        values = [by_episode[episode][col] for episode in episodes]
        plt.plot(episodes, values, linewidth=1.8, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Episode-Summed Value")
    plt.title("Episode Reward Components")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
