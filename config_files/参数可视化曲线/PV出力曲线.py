from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 当前训练/环境主流程使用的 PV 基准文件
USER_CSV_PATH = Path(__file__).resolve().parents[1] / "pv_4weather.csv"

STD_RATIO = 0.05
TRUNC_RATIO = 0.10
BAND_MODE = "both"  # "sigma", "trunc", "both"
SAVE_NAME = "commercial_pv_weather_visualization.png"

WEATHER_COLORS = {
    "sunny": "tab:orange",
    "cloudy": "tab:blue",
    "overcast": "tab:gray",
    "rainy": "tab:green",
}

WEATHER_LABELS = {
    "sunny": "Sunny",
    "cloudy": "Cloudy",
    "overcast": "Overcast",
    "rainy": "Rainy",
}


def resolve_csv_path() -> Path:
    candidates = [
        USER_CSV_PATH,
        Path(__file__).resolve().parent / "pv_4weather.csv",
        Path(__file__).resolve().parents[1] / "pv_4weather.csv",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Cannot find pv_4weather.csv. Tried:\n{tried}")


def load_base_profile(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = ["time", "sunny", "cloudy", "overcast", "rainy"]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Actual columns: {list(df.columns)}")

    df = df[expected_cols].copy()
    for col in expected_cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[expected_cols[1:]].isna().any().any():
        raise ValueError("PV profile contains non-numeric weather values.")
    return df


def build_time_axis(df: pd.DataFrame):
    labels = df["time"].astype(str).tolist()
    x = np.arange(len(labels))
    hour_ticks = [idx for idx, t in enumerate(labels) if t.endswith(":00")]
    hour_ticklabels = [labels[idx] for idx in hour_ticks]
    return x, hour_ticks, hour_ticklabels


def plot_weather_profiles(df: pd.DataFrame, save_path: Path) -> None:
    x, hour_ticks, hour_ticklabels = build_time_axis(df)

    fig, ax = plt.subplots(figsize=(14, 7))
    for weather in ["sunny", "cloudy", "overcast", "rainy"]:
        mu = df[weather].to_numpy(dtype=float)
        sigma = STD_RATIO * mu
        lower_sigma = np.maximum(mu - sigma, 0.0)
        upper_sigma = mu + sigma

        lower_trunc = np.maximum(mu * (1.0 - TRUNC_RATIO), 0.0)
        upper_trunc = mu * (1.0 + TRUNC_RATIO)

        color = WEATHER_COLORS[weather]
        label = WEATHER_LABELS[weather]

        if BAND_MODE in ("trunc", "both"):
            ax.fill_between(x, lower_trunc, upper_trunc, alpha=0.12, color=color)
        if BAND_MODE in ("sigma", "both"):
            ax.fill_between(x, lower_sigma, upper_sigma, alpha=0.22, color=color)
        ax.plot(x, mu, linewidth=2.2, color=color, label=label)

    suffix = {
        "sigma": "(mean with +-1 sigma band)",
        "trunc": "(mean with +-10% truncation band)",
        "both": "(mean with +-1 sigma and +-10% bands)",
    }[BAND_MODE]

    ax.set_title(f"Commercial PV Base Profiles Under Four Weather Types {suffix}")
    ax.set_xlabel("Time")
    ax.set_ylabel("PV Output")
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels(hour_ticklabels, rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    csv_path = resolve_csv_path()
    df = load_base_profile(csv_path)
    save_path = csv_path.parent / SAVE_NAME
    plot_weather_profiles(df, save_path)
    print(f"CSV path: {csv_path}")
    print(f"Saved figure: {save_path}")


if __name__ == "__main__":
    main()
