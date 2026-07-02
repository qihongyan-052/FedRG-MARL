from __future__ import annotations

from pathlib import Path
import base64
import math
import textwrap
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "sp_rgnn_csac_system_architecture_v4.png"
SVG_PATH = OUT_DIR / "sp_rgnn_csac_system_architecture_v4_exact.svg"

W, H = 4200, 2600
S = 2

C = {
    "bg": "#F7F9FC",
    "ink": "#111827",
    "muted": "#64748B",
    "border": "#B6C2D1",
    "white": "#FFFFFF",
    "obs": "#EAF2FF",
    "agent": "#EAF8F0",
    "exec": "#FFF3E6",
    "fed": "#F4ECFF",
    "private": "#FEE2E2",
    "blue": "#1D4ED8",
    "green": "#047857",
    "purple": "#7C3AED",
    "orange": "#D97706",
    "red": "#DC2626",
    "line": "#475467",
    "ev": "#DBEAFE",
    "bes": "#DCFCE7",
    "pv": "#FEF3C7",
    "es": "#FCE7F3",
    "tr": "#EDE9FE",
}

FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_CN = "C:/Windows/Fonts/msyh.ttc"


def sc(v: float) -> int:
    return int(round(v * S))


def sb(b: Sequence[float]) -> tuple[int, int, int, int]:
    return tuple(sc(x) for x in b)  # type: ignore[return-value]


def ft(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    p = FONT_BOLD if bold else FONT
    if not Path(p).exists():
        p = FONT_CN
    return ImageFont.truetype(p, size * S)


def text_box(d: ImageDraw.ImageDraw, b: Sequence[float], text: str, size: int, *, bold=False, fill=None, max_chars: int | None = None) -> None:
    font = ft(size, bold)
    lines: list[str] = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=max_chars, break_long_words=False) if max_chars else [part])
    lines = lines or [""]
    dims = [d.textbbox((0, 0), line, font=font) for line in lines]
    widths = [bb[2] - bb[0] for bb in dims]
    heights = [bb[3] - bb[1] for bb in dims]
    gap = 6 * S
    total_h = sum(heights) + gap * (len(lines) - 1)
    x1, y1, x2, y2 = sb(b)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        d.text((x1 + (x2 - x1 - w) / 2, y), line, font=font, fill=fill or C["ink"])
        y += h + gap


def label(d: ImageDraw.ImageDraw, x: float, y: float, text: str, size: int, *, bold=False, fill=None, anchor="la") -> None:
    d.text((sc(x), sc(y)), text, font=ft(size, bold), fill=fill or C["ink"], anchor=anchor)


def rounded(d: ImageDraw.ImageDraw, b: Sequence[float], fill: str, outline: str = C["border"], r: int = 24, w: int = 3) -> None:
    d.rounded_rectangle(sb(b), radius=sc(r), fill=fill, outline=outline, width=sc(w))


def rect(d: ImageDraw.ImageDraw, b: Sequence[float], fill: str, outline: str = C["border"], w: int = 2) -> None:
    d.rectangle(sb(b), fill=fill, outline=outline, width=sc(w))


def arrow(d: ImageDraw.ImageDraw, pts: Sequence[tuple[float, float]], color: str, *, w: int = 5, dashed=False, head=True) -> None:
    pp = [(sc(x), sc(y)) for x, y in pts]
    if dashed:
        for a, b in zip(pp, pp[1:]):
            dash(d, a, b, color, sc(w))
    else:
        d.line(pp, fill=color, width=sc(w), joint="curve")
    if head and len(pp) >= 2:
        arrow_head(d, pp[-2], pp[-1], color, sc(w))


def dash(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, w: int) -> None:
    ax, ay = a
    bx, by = b
    dist = math.hypot(bx - ax, by - ay)
    if dist == 0:
        return
    ux, uy = (bx - ax) / dist, (by - ay) / dist
    t = 0.0
    dash_len, gap = 30 * S, 16 * S
    while t < dist:
        t2 = min(t + dash_len, dist)
        d.line([(int(ax + ux * t), int(ay + uy * t)), (int(ax + ux * t2), int(ay + uy * t2))], fill=color, width=w)
        t += dash_len + gap


def arrow_head(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, w: int) -> None:
    ax, ay = a
    bx, by = b
    angle = math.atan2(by - ay, bx - ax)
    size = max(20 * S, w * 3)
    d.polygon(
        [
            (bx, by),
            (bx - size * math.cos(angle - math.pi / 7), by - size * math.sin(angle - math.pi / 7)),
            (bx - size * math.cos(angle + math.pi / 7), by - size * math.sin(angle + math.pi / 7)),
        ],
        fill=color,
    )


def badge(d: ImageDraw.ImageDraw, x: float, y: float, n: str, color: str) -> None:
    d.ellipse(sb((x - 22, y - 22, x + 22, y + 22)), fill=color, outline=C["white"], width=sc(3))
    text_box(d, (x - 20, y - 20, x + 20, y + 20), n, 20, bold=True, fill=C["white"])


def mini(d: ImageDraw.ImageDraw, b: Sequence[float], title: str, fill: str, outline: str = C["border"], size: int = 18) -> None:
    rounded(d, b, fill, outline, r=13, w=2)
    text_box(d, b, title, size, bold=True, max_chars=18)


def node(d: ImageDraw.ImageDraw, x: float, y: float, t: str, fill: str, r: int = 34) -> None:
    d.ellipse(sb((x - r, y - r, x + r, y + r)), fill=fill, outline="#718096", width=sc(2))
    text_box(d, (x - r, y - r, x + r, y + r), t, 17, bold=True)


def graph_icon(d: ImageDraw.ImageDraw, cx: float, cy: float, scale: float = 1.0) -> None:
    items = [
        ("EV", cx - 115 * scale, cy - 76 * scale, C["ev"]),
        ("BES", cx + 115 * scale, cy - 76 * scale, C["bes"]),
        ("PV", cx - 115 * scale, cy + 76 * scale, C["pv"]),
        ("ES", cx + 115 * scale, cy + 76 * scale, C["es"]),
        ("TR", cx, cy + 132 * scale, C["tr"]),
    ]
    for _, x, y, _ in items:
        arrow(d, [(x, y), (cx, cy)], "#9AA8BB", w=3, head=False)
        arrow(d, [(cx, cy), (x, y)], "#9AA8BB", w=3, head=False)
    node(d, cx, cy, "CS", C["white"], 42)
    for name, x, y, fill in items:
        node(d, x, y, name, fill, 34)


def lock(d: ImageDraw.ImageDraw, x: float, y: float) -> None:
    rounded(d, (x - 18, y, x + 18, y + 32), C["private"], C["red"], r=6, w=2)
    d.arc(sb((x - 13, y - 28, x + 13, y + 14)), 180, 360, fill=C["red"], width=sc(4))


def network(d: ImageDraw.ImageDraw, x: float, y: float, layers: Sequence[int], color: str, label_text: str) -> None:
    dx = 62
    for li, n in enumerate(layers):
        yy0 = y - (n - 1) * 26 / 2
        for ni in range(n):
            cx, cy = x + li * dx, yy0 + ni * 26
            if li < len(layers) - 1:
                nn = layers[li + 1]
                ny0 = y - (nn - 1) * 26 / 2
                for nj in range(nn):
                    d.line([(sc(cx + 9), sc(cy)), (sc(x + (li + 1) * dx - 9), sc(ny0 + nj * 26))], fill="#B2BDCC", width=sc(1))
            d.ellipse(sb((cx - 10, cy - 10, cx + 10, cy + 10)), fill=color, outline="#475467", width=sc(1))
    label(d, x + (len(layers) - 1) * dx / 2, y + 78, label_text, 16, bold=True, anchor="mm", fill=C["muted"])


def panel_title(d: ImageDraw.ImageDraw, x: float, y: float, title: str, subtitle: str) -> None:
    label(d, x, y, title, 31, bold=True)
    label(d, x, y + 40, subtitle, 20, fill=C["muted"])


def draw_local_panel(d: ImageDraw.ImageDraw) -> None:
    rounded(d, (70, 190, 780, 1500), C["obs"], "#8DB3E8", r=28, w=3)
    panel_title(d, 110, 240, "(a) Local observations", "one local graph per park")
    for name, y in [("Residential", 350), ("Office", 735), ("Commercial", 1120)]:
        rounded(d, (120, y, 730, y + 310), C["white"], "#9EB9DB", r=18, w=2)
        label(d, 150, y + 40, f"{name} park", 23, bold=True)
        graph_icon(d, 310, y + 168, 0.78)
        mini(d, (485, y + 78, 690, y + 150), "local raw data", C["private"], C["red"])
        lock(d, 670, y + 88)
        mini(d, (485, y + 194, 690, y + 270), "local graph\nobservation", C["ev"], "#6B91C9")


def draw_agent_panel(d: ImageDraw.ImageDraw) -> None:
    rounded(d, (900, 190, 2820, 1500), C["agent"], "#7EBB8D", r=28, w=3)
    panel_title(d, 940, 240, "(b) Local SP-RGNN-CSAC agent architecture", "expanded mechanism subfigures")

    # encoder
    rounded(d, (960, 340, 1220, 565), C["white"], "#7EBB8D", r=18, w=2)
    label(d, 990, 382, "Type-specific encoders", 22, bold=True)
    labels = ["EV", "BES", "PV", "CS", "ES", "TR"]
    fills = [C["ev"], C["bes"], C["pv"], C["white"], C["es"], C["tr"]]
    for i, (lab, fill) in enumerate(zip(labels, fills)):
        rect(d, (995 + i * 34, 430 - i * 6, 1060 + i * 34, 468 - i * 6), fill, "#8796AA", w=1)
    label(d, 1090, 525, "heterogeneous node features", 17, fill=C["muted"], anchor="mm")

    # shared-private relation
    rounded(d, (1300, 315, 1940, 685), C["white"], "#7EBB8D", r=18, w=2)
    label(d, 1335, 360, "Shared-private relation transform", 23, bold=True)
    rels = ["EV-CS", "BES-CS", "PV-CS", "ES-CS", "TR-CS"]
    for i, r in enumerate(rels):
        yy = 410 + i * 52
        mini(d, (1335, yy, 1430, yy + 36), r, "#F8FAFC", "#97A3B7", size=14)
        mini(d, (1475, yy - 4, 1625, yy + 40), "shared", "#DFF3E7", "#4E9F69", size=15)
        mini(d, (1680, yy - 4, 1840, yy + 40), "private\nadapter", "#FFF1DB", C["orange"], size=14)
        arrow(d, [(1430, yy + 18), (1475, yy + 18)], C["blue"], w=2)
        arrow(d, [(1625, yy + 18), (1680, yy + 18)], C["blue"], w=2)
    label(d, 1640, 660, "shared blocks transfer; adapters personalize", 16, fill=C["muted"], anchor="mm")

    # relation gate
    rounded(d, (2020, 315, 2755, 685), C["white"], "#7EBB8D", r=18, w=2)
    label(d, 2055, 360, "Relation-wise normalization and gate", 23, bold=True)
    for i, (lab, fill, knob) in enumerate(zip(["EV-CS", "BES-CS", "PV-CS", "ES-CS", "TR-CS"], [C["ev"], C["bes"], C["pv"], C["es"], C["tr"]], [90, 42, 64, 30, 76])):
        yy = 418 + i * 42
        rect(d, (2065, yy, 2305, yy + 24), fill, "#8796AA", w=1)
        label(d, 2080, yy + 20, lab, 14, bold=True)
        d.line([(sc(2350), sc(yy + 12)), (sc(2495), sc(yy + 12))], fill="#A8B3C2", width=sc(4))
        d.ellipse(sb((2350 + knob - 9, yy + 3, 2350 + knob + 9, yy + 21)), fill=C["green"], outline=C["ink"], width=sc(1))
        arrow(d, [(2495, yy + 12), (2580, 520)], C["green"], w=2, head=False)
    mini(d, (2580, 465, 2715, 585), "fused\nrelation\nstate", "#E8F6EE", "#4E9F69", size=18)

    # actor network
    rounded(d, (960, 785, 1535, 1115), C["white"], "#7EBB8D", r=18, w=2)
    label(d, 995, 830, "Actor policy network", 23, bold=True)
    network(d, 1045, 965, [4, 5, 4, 2], "#BBF7D0", "mean / log-std")
    mini(d, (1360, 880, 1500, 950), "EV actions", C["ev"], "#6B91C9")
    mini(d, (1360, 1005, 1500, 1075), "BES action", C["bes"], "#5EA875")
    arrow(d, [(1260, 950), (1360, 915)], C["blue"], w=3)
    arrow(d, [(1260, 980), (1360, 1040)], C["blue"], w=3)

    # critic network
    rounded(d, (1630, 775, 2755, 1130), C["white"], "#6BB89B", r=18, w=2)
    label(d, 1665, 820, "Role-preserving reward/cost critics", 23, bold=True)
    for i, role in enumerate(["CS", "BES", "PV", "ES", "EV"]):
        mini(d, (1670 + i * 90, 875, 1740 + i * 90, 920), role, "#F8FAFC", "#8796AA", size=14)
    mini(d, (2165, 873, 2290, 923), "concat", "#E8F6EE", "#4E9F69", size=16)
    for i in range(5):
        arrow(d, [(1740 + i * 90, 898), (2165, 898)], C["green"], w=2, head=False)
    network(d, 1695, 1025, [3, 4, 2], "#BFDBFE", "Q-r1")
    network(d, 1905, 1025, [3, 4, 2], "#BFDBFE", "Q-r2")
    network(d, 2115, 1025, [3, 4, 2], "#FDE68A", "Q-c1")
    network(d, 2325, 1025, [3, 4, 2], "#FDE68A", "Q-c2")

    # learning update
    rounded(d, (960, 1220, 2755, 1440), C["white"], "#6BB89B", r=18, w=2)
    label(d, 995, 1265, "Local constrained SAC update", 23, bold=True)
    mini(d, (1015, 1320, 1215, 1395), "replay buffer", "#E8F6EE", "#6BB89B", size=18)
    mini(d, (1340, 1320, 1565, 1395), "actor update", "#E8F6EE", "#6BB89B", size=18)
    mini(d, (1690, 1320, 1945, 1395), "critic update", "#E8F6EE", "#6BB89B", size=18)
    mini(d, (2070, 1320, 2360, 1395), "temperature /\nmultiplier update", "#E8F6EE", "#6BB89B", size=17)
    mini(d, (2465, 1320, 2705, 1395), "local policy\nrefresh", "#E8F6EE", "#6BB89B", size=18)
    for x in [1215, 1565, 1945, 2360]:
        arrow(d, [(x, 1357), (x + 125, 1357)], C["green"], w=3, dashed=True)

    # internal arrows
    arrow(d, [(1220, 450), (1300, 450)], C["blue"], w=4)
    arrow(d, [(1940, 500), (2020, 500)], C["blue"], w=4)
    arrow(d, [(2380, 685), (2380, 735), (1110, 735), (1110, 785)], C["blue"], w=4)
    arrow(d, [(1810, 1220), (1810, 1130)], C["green"], w=4, dashed=True)
    arrow(d, [(1460, 1220), (1260, 1115)], C["green"], w=4, dashed=True)
    arrow(d, [(2585, 1320), (1430, 1115)], C["green"], w=4, dashed=True)


def draw_fed_panel(d: ImageDraw.ImageDraw) -> None:
    rounded(d, (2930, 190, 4130, 1055), C["fed"], "#B59BE8", r=28, w=3)
    panel_title(d, 2970, 240, "(c) Personalized relation federation", "slow time-scale parameter branch")
    for i, park in enumerate(["Residential", "Office", "Commercial"]):
        yy = 360 + i * 120
        mini(d, (2990, yy, 3160, yy + 64), f"{park}\nshared blocks", "#DFF3E7", "#4E9F69", size=16)
        lock(d, 3230, yy + 15)
        label(d, 3270, yy + 43, "private modules blocked", 17, fill=C["red"], anchor="lm")
        arrow(d, [(3160, yy + 32), (3520, 585)], C["purple"], w=3, dashed=True)
    mini(d, (3520, 500, 3745, 670), "personalized\naggregation\nserver", C["white"], "#9A7EDB", size=20)
    label(d, 3835, 470, "compatibility weights", 17, bold=True, anchor="mm")
    heat = [["#6D28D9", "#C4B5FD", "#DDD6FE"], ["#C4B5FD", "#6D28D9", "#BCA8FA"], ["#DDD6FE", "#BCA8FA", "#6D28D9"]]
    for r in range(3):
        for c in range(3):
            rect(d, (3780 + c * 44, 500 + r * 44, 3820 + c * 44, 540 + r * 44), heat[r][c], C["white"], w=1)
    mini(d, (3340, 810, 3910, 930), "return park-specific shared relation references", C["white"], "#9A7EDB", size=21)
    arrow(d, [(3745, 585), (3780, 565)], C["purple"], w=3, dashed=True)
    arrow(d, [(3845, 632), (3630, 810)], C["purple"], w=3, dashed=True)
    label(d, 3535, 1000, "No raw trajectories, actions, rewards/costs, critics, replay buffers, or adapters are uploaded.", 20, fill=C["muted"], anchor="mm")


def draw_exec_panel(d: ImageDraw.ImageDraw) -> None:
    rounded(d, (360, 1640, 4130, 2355), C["exec"], "#E5A454", r=28, w=3)
    panel_title(d, 405, 1690, "(d) Safe execution and regional coordination", "execution-time physical feasibility and TR feedback")
    stages = [
        ("Actor\nactions", "policy output"),
        ("Action\nmapping", "requested EV/BES energy"),
        ("Device\nbounds", "SoC / power / V2G"),
        ("CS capacity-safe\nadjustment", "local station limit"),
        ("Secure\naggregation", "aggregate exchange only"),
        ("TR\ncoordination", "shared transformer pressure"),
        ("Final\nexecution", "EV/BES/PV flows"),
        ("Feedback\nsignals", "reward / cost / TR"),
    ]
    fills = [C["white"], C["white"], C["bes"], C["pv"], C["tr"], "#FFE4C7", C["ev"], C["white"]]
    x0, y0 = 510, 1840
    for i, ((title, sub), fill) in enumerate(zip(stages, fills)):
        x = x0 + i * 430
        mini(d, (x, y0, x + 270, y0 + 145), title, fill, "#C58A43", size=19)
        label(d, x + 135, y0 + 205, sub, 17, fill=C["muted"], anchor="mm")
        if i < len(stages) - 1:
            arrow(d, [(x + 270, y0 + 72), (x0 + (i + 1) * 430, y0 + 72)], C["orange"], w=5)
    # shield and transformer icons
    sx, sy = x0 + 4 * 430 + 135, y0 + 72
    d.polygon([(sc(sx), sc(sy - 52)), (sc(sx + 45), sc(sy - 25)), (sc(sx + 28), sc(sy + 46)), (sc(sx), sc(sy + 66)), (sc(sx - 28), sc(sy + 46)), (sc(sx - 45), sc(sy - 25))], fill="#D8B4FE", outline=C["purple"])
    label(d, sx, sy + 8, "sum", 20, bold=True, fill=C["purple"], anchor="mm")
    tx, ty = x0 + 5 * 430 + 135, y0 + 72
    for j in range(3):
        d.arc(sb((tx - 55 + j * 35, ty - 35, tx - 10 + j * 35, ty + 35)), 90, 270, fill=C["orange"], width=sc(5))


def draw_connections(d: ImageDraw.ImageDraw) -> None:
    # Six external, port-style connections only.
    # 1 observation to agent
    for y in [580, 965, 1350]:
        arrow(d, [(730, y), (835, y), (835, 450), (960, 450)], C["blue"], w=4)
    badge(d, 835, 430, "1", C["blue"])
    label(d, 820, 390, "local graph observation", 18, fill=C["blue"], anchor="mm")

    # 2 action to execution, routed outside panels
    arrow(d, [(1500, 950), (2870, 950), (2870, 1600), (510, 1600), (510, 1840)], C["blue"], w=5)
    badge(d, 2868, 950, "2", C["blue"])
    label(d, 2885, 990, "actions", 18, fill=C["blue"])

    # 3 reward/cost to replay buffer/update
    arrow(d, [(3825, 1985), (3825, 2415), (1115, 2415), (1115, 1440)], C["green"], w=5, dashed=True)
    badge(d, 3825, 2135, "3", C["green"])
    label(d, 2500, 2430, "reward/cost transition to local replay buffer", 19, fill=C["green"], anchor="mm")

    # 4 TR feedback to next observation
    arrow(d, [(3825, 1840), (3825, 1570), (420, 1570), (420, 1430)], C["blue"], w=5)
    badge(d, 3825, 1700, "4", C["blue"])
    label(d, 2060, 1560, "TR feedback enters next local graph", 19, fill=C["blue"], anchor="mm")

    # 5 shared relation upload
    arrow(d, [(1620, 360), (1620, 155), (3180, 155), (3180, 360)], C["purple"], w=5, dashed=True)
    badge(d, 2460, 155, "5", C["purple"])
    label(d, 2460, 128, "selected shared relation parameters", 19, fill=C["purple"], anchor="mm")

    # 6 personalized reference return
    arrow(d, [(3340, 870), (2890, 870), (2890, 1285), (2360, 1285)], C["purple"], w=5, dashed=True)
    badge(d, 2890, 870, "6", C["purple"])
    label(d, 2880, 910, "personalized shared reference", 18, fill=C["purple"], anchor="rm")


def draw_legend(d: ImageDraw.ImageDraw) -> None:
    rounded(d, (2930, 1125, 4130, 1500), C["white"], C["border"], r=20, w=2)
    label(d, 2970, 1175, "Legend", 26, bold=True)
    arrow(d, [(3000, 1240), (3180, 1240)], C["blue"], w=5)
    label(d, 3215, 1248, "online decision / execution / TR feedback", 20, fill=C["muted"], anchor="lm")
    arrow(d, [(3000, 1305), (3180, 1305)], C["green"], w=5, dashed=True)
    label(d, 3215, 1313, "local constrained SAC learning", 20, fill=C["muted"], anchor="lm")
    arrow(d, [(3000, 1370), (3180, 1370)], C["purple"], w=5, dashed=True)
    label(d, 3215, 1378, "periodic personalized relation federation", 20, fill=C["muted"], anchor="lm")
    lock(d, 3035, 1425)
    label(d, 3090, 1457, "locked items stay local and are not uploaded", 20, fill=C["muted"], anchor="lm")


def main() -> None:
    img = Image.new("RGB", (W * S, H * S), C["bg"])
    d = ImageDraw.Draw(img)
    label(d, 90, 75, "SP-RGNN-CSAC System Architecture", 48, bold=True)
    label(d, 90, 130, "Main closed loop plus expanded mechanism subfigures for the paper figure", 25, fill=C["muted"])

    draw_local_panel(d)
    draw_agent_panel(d)
    draw_fed_panel(d)
    draw_exec_panel(d)
    draw_connections(d)
    draw_legend(d)

    img = img.resize((W, H), Image.Resampling.LANCZOS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(PNG_PATH)
    data = base64.b64encode(PNG_PATH.read_bytes()).decode("ascii")
    SVG_PATH.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><image width="{W}" height="{H}" href="data:image/png;base64,{data}"/></svg>',
        encoding="utf-8",
    )
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()
