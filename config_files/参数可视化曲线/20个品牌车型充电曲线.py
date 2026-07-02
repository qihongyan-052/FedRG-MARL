# -*- coding: utf-8 -*-
# 可视化 20 类 EV 原型的三张图：
# 1）P-SOC：功率-荷电状态曲线
# 2）SOC-t：荷电状态-时间曲线
# 3）P-t：功率-时间曲线
#
# 本版对应“增强区分版”参数：
# - 一部分车型被设成明显的慢充型
# - 一部分车型被设成明显的快充型
# 这样在 SOC-t 图上更容易一眼看出充电快慢差异

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ====== 你只需要改这里 ======
JSON_PATH = r"F:\第二篇小论文——代码脚本\config_files\ev_20_brand_models.json"
# ===========================

SOC_GRID = np.linspace(0, 1, 501)
SOC_START = 0.10
SOC_END = 1.00
DT_MINUTES = 0.2

PLOT_P_SOC = True
PLOT_SOC_T = True
PLOT_P_T = True


def set_matplotlib_font():
    """设置中文显示，尽量避免乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def resolve_json_path(json_path: str) -> str:
    """如果没写 .json 后缀，则自动补上。"""
    path = json_path.strip()
    if not path.lower().endswith(".json"):
        path = path + ".json"
    return path


def load_json(json_path: str):
    """读取 JSON 文件并返回 data, evs, actual_path。"""
    actual_path = resolve_json_path(json_path)

    if not os.path.exists(actual_path):
        raise FileNotFoundError(
            f"找不到文件：{actual_path}\n"
            f"请检查文件名、后缀名和路径是否正确。"
        )

    with open(actual_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    evs = data.get("ev_models", [])
    if not evs:
        raise ValueError("JSON 中未找到 ev_models 列表，请检查文件结构。")

    return data, evs, actual_path


def tail_start_soc(ev: dict) -> float:
    """
    计算开始进入尾段恒功率的 SOC。
    由线性下降段与尾段恒功率的交点决定。
    """
    soc_knee = float(ev["soc_knee"])
    soc_tail = float(ev["soc_tail_start"])
    soc_tail = np.clip(soc_tail, soc_knee, 1.0)
    return float(soc_tail)


def charging_power_piecewise(ev: dict, soc: np.ndarray) -> np.ndarray:
    """
    三段式充电功率模型：
    1）SOC <= soc_knee：P = p_ch_max_kw
    2）soc_knee < SOC < soc_tail_start：P 线性下降
    3）SOC >= soc_tail_start：P = tail_power_kw
    """
    p_max = float(ev["p_ch_max_kw"])
    soc_knee = float(ev["soc_knee"])
    tail_power = float(ev["tail_power_kw"])
    soc_tail = tail_start_soc(ev)

    soc = np.asarray(soc, dtype=float)
    power = np.full_like(soc, tail_power, dtype=float)

    mask_cc = soc <= soc_knee
    power[mask_cc] = p_max

    mask_taper = (soc > soc_knee) & (soc < soc_tail)
    if np.any(mask_taper) and soc_tail > soc_knee:
        ratio = (soc_tail - soc[mask_taper]) / (soc_tail - soc_knee)
        power[mask_taper] = tail_power + (p_max - tail_power) * ratio

    mask_tail = soc >= soc_tail
    power[mask_tail] = tail_power

    return np.clip(power, 0.0, p_max)


def simulate_charging(ev: dict, soc0: float, soc_end: float, dt_minutes: float):
    """
    最大功率充电仿真。
    充电效率计入 SOC 演化：
        E_next = E + P * eta_ch * dt
    """
    capacity = float(ev["battery_capacity_kwh"])
    eta_ch = float(ev["eta_ch"])
    dt_h = dt_minutes / 60.0

    energy = soc0 * capacity
    energy_end = soc_end * capacity

    t_list = [0.0]
    e_list = [energy]
    soc_list = [energy / capacity]
    p_list = [charging_power_piecewise(ev, np.array([soc_list[-1]]))[0]]

    max_steps = 500000
    for _ in range(max_steps):
        if energy >= energy_end - 1e-9:
            break

        soc = energy / capacity
        power = charging_power_piecewise(ev, np.array([soc]))[0]
        if power < 1e-9:
            break

        energy = min(energy_end, energy + power * eta_ch * dt_h)

        t_list.append(t_list[-1] + dt_h)
        e_list.append(energy)
        soc_list.append(energy / capacity)
        p_list.append(power)

    return np.array(t_list), np.array(soc_list), np.array(e_list), np.array(p_list)


def legend_outside():
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)


def plot_power_vs_soc(evs: list):
    """画第 1 张图：P-SOC。"""
    plt.figure(figsize=(10, 6))
    for ev in evs:
        name = ev.get("id", "EV")
        power = charging_power_piecewise(ev, SOC_GRID)
        plt.plot(SOC_GRID, power, linewidth=1.5, label=name)

    plt.xlabel("SOC")
    plt.ylabel("P(SOC) (kW)")
    plt.title("20 EV: Power vs SOC (Separated Fast/Slow Profiles)")
    plt.grid(True, alpha=0.25)
    legend_outside()
    plt.tight_layout()


def plot_soc_vs_time(evs: list):
    """画第 2 张图：SOC-t。"""
    plt.figure(figsize=(10, 6))
    for ev in evs:
        name = ev.get("id", "EV")
        t, soc, e, p = simulate_charging(ev, SOC_START, SOC_END, DT_MINUTES)
        plt.plot(t, soc, linewidth=1.6, label=name)

    plt.xlabel("Time (hours)")
    plt.ylabel("SOC(t)")
    plt.title(f"20 EV: SOC vs Time (Separated Fast/Slow Profiles, start={SOC_START} -> end={SOC_END})")
    plt.grid(True, alpha=0.25)
    legend_outside()
    plt.tight_layout()


def plot_power_vs_time(evs: list):
    """画第 3 张图：P-t。"""
    plt.figure(figsize=(10, 6))
    for ev in evs:
        name = ev.get("id", "EV")
        t, soc, e, p = simulate_charging(ev, SOC_START, SOC_END, DT_MINUTES)
        plt.plot(t, p, linewidth=1.5, label=name)

    plt.xlabel("Time (hours)")
    plt.ylabel("P(t) (kW)")
    plt.title(f"20 EV: Power vs Time (Separated Fast/Slow Profiles, start={SOC_START} -> end={SOC_END})")
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.25)
    legend_outside()
    plt.tight_layout()


def main():
    set_matplotlib_font()

    data, evs, actual_path = load_json(JSON_PATH)
    print("JSON_PATH =", actual_path)
    print(f"成功读取 {len(evs)} 个 EV 原型。")

    if "notes_cn" in data:
        print("检测到中文说明，已正常加载。")

    if PLOT_P_SOC:
        plot_power_vs_soc(evs)
    if PLOT_SOC_T:
        plot_soc_vs_time(evs)
    if PLOT_P_T:
        plot_power_vs_time(evs)

    plt.show()


if __name__ == "__main__":
    main()
