# -*- coding: utf-8 -*-
"""
Plot a representative-day TR coordination figure from evaluation_steps.csv.

Important:
    evaluation_steps.csv stores post-execution park exchanges only. It does not
    contain pre-TR exchange, TR trigger flags, or park-level TR truncation fields.
    Therefore this script plots:
      (a) post-TR regional exchange with transformer limits,
      (b) post-TR transformer utilization and residual overload,
      (c) post-TR park-level exchange contributions.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


INPUT_CSV = Path(
    r"F:\第二篇小论文——代码脚本\saved\SP_RGNN_CSAC-隐私+参数联邦-2\evaluates\evaluation_steps.csv"
)
OUTPUT_DIR = Path(r"D:\All_文档资料\第二篇小论文——资料\论文配图\实验图\结果")
OUTPUT_PATH = OUTPUT_DIR / "TR_coordination_post_execution_SP_RGNN_CSAC.png"

EPISODE_TO_PLOT = 0
STEP_MINUTES = 15
START_HOUR = 8
EXPECTED_STEPS = 96

TR_LIMIT_KW = 900.0
TR_LIMIT_KWH = TR_LIMIT_KW * STEP_MINUTES / 60.0

PARKS = [
    ("residential", "Residential park", "#4e79a7"),
    ("office", "Office park", "#59a14f"),
    ("commercial", "Commercial park", "#e15759"),
]


def read_episode_rows(csv_path: Path, episode: int) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header row")
        required = {
            "episode",
            "time",
            "residential_park_grid_exchange_kwh",
            "office_park_grid_exchange_kwh",
            "commercial_park_grid_exchange_kwh",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        rows = [row for row in reader if int(row["episode"]) == episode]
    if len(rows) != EXPECTED_STEPS:
        raise ValueError(f"expected {EXPECTED_STEPS} rows for episode {episode}, got {len(rows)}")
    return rows


def build_hour_positions(time_labels: List[str]) -> List[float]:
    positions: List[float] = []
    for label in time_labels:
        hour_text, minute_text = label.split(":")
        hour = int(hour_text)
        minute = int(minute_text)
        mapped_hour = hour if hour >= START_HOUR else hour + 24
        positions.append(mapped_hour + minute / 60.0)
    return positions


def step_plot(ax, x_values, y_values, **kwargs) -> None:
    ax.step(x_values, y_values, where="mid", **kwargs)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Times New Roman", "Arial", "Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = read_episode_rows(INPUT_CSV, EPISODE_TO_PLOT)
    time_labels = [row["time"] for row in rows]
    x = build_hour_positions(time_labels)

    park_exchange: Dict[str, List[float]] = {}
    for park_id, _, _ in PARKS:
        column = f"{park_id}_park_grid_exchange_kwh"
        park_exchange[park_id] = [float(row[column]) for row in rows]

    total_exchange = [
        sum(park_exchange[park_id][idx] for park_id, _, _ in PARKS)
        for idx in range(len(rows))
    ]
    utilization = [abs(value) / TR_LIMIT_KWH for value in total_exchange]
    residual_overload = [max(0.0, abs(value) - TR_LIMIT_KWH) for value in total_exchange]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10.5), dpi=320, sharex=True)

    # (a) Post-TR regional exchange.
    ax = axes[0]
    step_plot(
        ax,
        x,
        total_exchange,
        color="#d62728",
        linewidth=2.0,
        label="Post-TR regional net exchange",
    )
    ax.axhline(TR_LIMIT_KWH, color="#303030", linestyle="--", linewidth=1.2, label="+/- TR limit")
    ax.axhline(-TR_LIMIT_KWH, color="#303030", linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color="#606060", linewidth=0.9)
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("(a) Regional net exchange after TR coordination")
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    # (b) Utilization and residual overload after coordination.
    ax = axes[1]
    step_plot(
        ax,
        x,
        utilization,
        color="#1f77b4",
        linewidth=1.8,
        label="TR utilization |exchange| / limit",
    )
    ax.axhline(1.0, color="#303030", linestyle="--", linewidth=1.2, label="limit")
    ax.set_ylabel("Utilization")
    ax.set_title("(b) Transformer utilization and residual overload after coordination")
    ax.set_ylim(bottom=0.0)
    ax2 = ax.twinx()
    ax2.bar(
        x,
        residual_overload,
        width=0.12,
        color="#ff7f0e",
        alpha=0.35,
        label="Residual overload",
    )
    max_residual = max(residual_overload) if residual_overload else 0.0
    ax2.set_ylim(0.0, max(1.0, max_residual * 1.15))
    ax2.set_ylabel("Residual overload (kWh)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=9, frameon=True)

    # (c) Park-level contributions to the post-TR regional exchange.
    ax = axes[2]
    for park_id, label, color in PARKS:
        step_plot(
            ax,
            x,
            park_exchange[park_id],
            color=color,
            linewidth=1.7,
            label=label,
        )
    ax.axhline(0.0, color="#606060", linewidth=0.9)
    ax.set_ylabel("Energy (kWh)")
    ax.set_title("(c) Park-level post-TR exchange contributions")
    ax.legend(loc="upper left", ncol=3, fontsize=9, frameon=True)

    hour_ticks = list(range(START_HOUR, START_HOUR + 25))
    hour_labels = [str(hour if hour < 24 else hour - 24) for hour in hour_ticks]
    for ax in axes:
        ax.grid(True, which="major", color="#9aa0a6", alpha=0.35, linewidth=0.8)
        ax.grid(True, which="minor", color="#c7cbd1", alpha=0.20, linewidth=0.45)
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))
        ax.set_axisbelow(True)
    axes[-1].set_xlim(START_HOUR, START_HOUR + 24)
    axes[-1].set_xticks(hour_ticks)
    axes[-1].set_xticklabels(hour_labels)
    axes[-1].set_xlabel("Time (h)")

    fig.suptitle(
        "Representative-day post-execution TR coordination behavior of SP-RGNN-CSAC",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.972))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=320, bbox_inches="tight")
    plt.close(fig)

    print(f"generated: {OUTPUT_PATH}")
    print(
        "note: evaluation_steps.csv has no pre-TR demand or trigger columns; "
        "this figure uses post-execution exchanges only."
    )


if __name__ == "__main__":
    main()
