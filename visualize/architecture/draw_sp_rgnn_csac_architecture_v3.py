from __future__ import annotations

from pathlib import Path
import base64
import math
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "sp_rgnn_csac_system_architecture_v3.png"
SVG_PATH = OUT_DIR / "sp_rgnn_csac_system_architecture_v3_exact.svg"

W, H = 3900, 2450
S = 2

COL = {
    "bg": "#F7F9FC",
    "ink": "#142033",
    "muted": "#667085",
    "border": "#B7C4D6",
    "white": "#FFFFFF",
    "blue": "#1D4ED8",
    "green": "#047857",
    "purple": "#7C3AED",
    "orange": "#D97706",
    "red": "#DC2626",
    "park": "#EAF2FF",
    "actor": "#EAF8F0",
    "learn": "#EAF8F2",
    "fed": "#F4ECFF",
    "exec": "#FFF4E6",
    "soft_blue": "#DBEAFE",
    "soft_green": "#DCFCE7",
    "soft_yellow": "#FEF3C7",
    "soft_pink": "#FCE7F3",
    "soft_purple": "#EDE9FE",
    "soft_red": "#FEE2E2",
    "line": "#475467",
}

FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_CN = "C:/Windows/Fonts/msyh.ttc"


def sf(v: float) -> int:
    return int(round(v * S))


def bx(b: Sequence[float]) -> tuple[int, int, int, int]:
    return tuple(sf(x) for x in b)  # type: ignore[return-value]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    p = FONT_BOLD if bold else FONT
    if not Path(p).exists():
        p = FONT_CN
    return ImageFont.truetype(p, size * S)


def label(d: ImageDraw.ImageDraw, x: float, y: float, text: str, size: int, *, bold=False, fill=None, anchor="la") -> None:
    d.text((sf(x), sf(y)), text, font=font(size, bold), fill=fill or COL["ink"], anchor=anchor)


def rounded(d: ImageDraw.ImageDraw, b: Sequence[float], fill: str, outline: str = COL["border"], r: int = 24, w: int = 3) -> None:
    d.rounded_rectangle(bx(b), radius=sf(r), fill=fill, outline=outline, width=sf(w))


def rect(d: ImageDraw.ImageDraw, b: Sequence[float], fill: str, outline: str = COL["border"], w: int = 2) -> None:
    d.rectangle(bx(b), fill=fill, outline=outline, width=sf(w))


def line(d: ImageDraw.ImageDraw, pts: Sequence[tuple[float, float]], color: str, w: int = 4, dashed=False, arrow=True) -> None:
    points = [(sf(x), sf(y)) for x, y in pts]
    if dashed:
        for a, b in zip(points, points[1:]):
            dashed_line(d, a, b, color, sf(w))
    else:
        d.line(points, fill=color, width=sf(w), joint="curve")
    if arrow and len(points) >= 2:
        arrow_head(d, points[-2], points[-1], color, sf(w))


def dashed_line(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, w: int) -> None:
    ax, ay = a
    bx_, by_ = b
    dist = math.hypot(bx_ - ax, by_ - ay)
    if dist <= 0:
        return
    ux, uy = (bx_ - ax) / dist, (by_ - ay) / dist
    t = 0
    dash, gap = 28 * S, 16 * S
    while t < dist:
        t2 = min(t + dash, dist)
        d.line([(int(ax + ux * t), int(ay + uy * t)), (int(ax + ux * t2), int(ay + uy * t2))], fill=color, width=w)
        t += dash + gap


def arrow_head(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, w: int) -> None:
    ax, ay = a
    bx_, by_ = b
    ang = math.atan2(by_ - ay, bx_ - ax)
    size = max(20 * S, w * 3)
    p1 = (bx_, by_)
    p2 = (bx_ - size * math.cos(ang - math.pi / 7), by_ - size * math.sin(ang - math.pi / 7))
    p3 = (bx_ - size * math.cos(ang + math.pi / 7), by_ - size * math.sin(ang + math.pi / 7))
    d.polygon([p1, p2, p3], fill=color)


def center_text(d: ImageDraw.ImageDraw, b: Sequence[float], text: str, size: int, *, bold=False, fill=None) -> None:
    f = font(size, bold)
    lines = text.split("\n")
    hs = []
    ws = []
    for t in lines:
        bb = d.textbbox((0, 0), t, font=f)
        ws.append(bb[2] - bb[0])
        hs.append(bb[3] - bb[1])
    gap = 6 * S
    total = sum(hs) + gap * (len(lines) - 1)
    x1, y1, x2, y2 = bx(b)
    y = y1 + (y2 - y1 - total) / 2
    for t, ww, hh in zip(lines, ws, hs):
        d.text((x1 + (x2 - x1 - ww) / 2, y), t, font=f, fill=fill or COL["ink"])
        y += hh + gap


def mini_card(d: ImageDraw.ImageDraw, b: Sequence[float], title: str, fill: str, outline: str, size=19) -> None:
    rounded(d, b, fill, outline, r=12, w=2)
    center_text(d, b, title, size, bold=True)


def neuron_net(d: ImageDraw.ImageDraw, x: float, y: float, layers: Sequence[int], *, color: str, label_text: str = "") -> None:
    dx = 62
    radius = 10
    maxn = max(layers)
    for li, n in enumerate(layers):
        yy0 = y - (n - 1) * 26 / 2
        for ni in range(n):
            cx, cy = x + li * dx, yy0 + ni * 26
            if li < len(layers) - 1:
                next_n = layers[li + 1]
                ny0 = y - (next_n - 1) * 26 / 2
                for nj in range(next_n):
                    d.line([(sf(cx + radius), sf(cy)), (sf(x + (li + 1) * dx - radius), sf(ny0 + nj * 26))], fill="#AAB7C6", width=sf(1))
            d.ellipse(bx((cx - radius, cy - radius, cx + radius, cy + radius)), fill=color, outline="#4B5563", width=sf(1))
    if label_text:
        label(d, x + (len(layers) - 1) * dx / 2, y + maxn * 18 + 22, label_text, 17, bold=True, anchor="mm")


def draw_lock(d: ImageDraw.ImageDraw, x: float, y: float) -> None:
    rounded(d, (x - 18, y, x + 18, y + 32), COL["soft_red"], COL["red"], r=6, w=2)
    d.arc(bx((x - 13, y - 28, x + 13, y + 14)), 180, 360, fill=COL["red"], width=sf(4))


def graph_subfig(d: ImageDraw.ImageDraw, x: float, y: float, scale: float = 1.0) -> None:
    cx, cy = x, y
    nodes = [
        ("EV", cx - 115 * scale, cy - 80 * scale, COL["soft_blue"]),
        ("BES", cx + 115 * scale, cy - 80 * scale, COL["soft_green"]),
        ("PV", cx - 115 * scale, cy + 75 * scale, COL["soft_yellow"]),
        ("ES", cx + 115 * scale, cy + 75 * scale, COL["soft_pink"]),
        ("TR", cx, cy + 132 * scale, COL["soft_purple"]),
    ]
    for name, nx, ny, c in nodes:
        line(d, [(nx, ny), (cx, cy)], "#98A6BA", w=3, arrow=False)
        line(d, [(cx, cy), (nx, ny)], "#98A6BA", w=3, arrow=False)
    node_circle(d, cx, cy, "CS", COL["white"], 42)
    for name, nx, ny, c in nodes:
        node_circle(d, nx, ny, name, c, 34)


def node_circle(d: ImageDraw.ImageDraw, x: float, y: float, text: str, fill: str, r: int) -> None:
    d.ellipse(bx((x - r, y - r, x + r, y + r)), fill=fill, outline="#718096", width=sf(2))
    center_text(d, (x - r, y - r, x + r, y + r), text, 17, bold=True)


def shared_private_block(d: ImageDraw.ImageDraw, b: Sequence[float]) -> None:
    rounded(d, b, COL["white"], "#7EBB8D", r=18, w=2)
    x1, y1, x2, y2 = b
    label(d, x1 + 22, y1 + 36, "Shared-private relation transform", 23, bold=True)
    # Relation lanes
    rels = ["EV-CS", "BES-CS", "PV-CS", "ES-CS", "TR-CS"]
    for i, r in enumerate(rels):
        yy = y1 + 78 + i * 58
        mini_card(d, (x1 + 30, yy, x1 + 120, yy + 38), r, "#F8FAFC", "#9AA7B7", size=15)
        mini_card(d, (x1 + 155, yy - 4, x1 + 295, yy + 42), "shared", "#DFF3E7", "#4E9F69", size=15)
        mini_card(d, (x1 + 335, yy - 4, x1 + 480, yy + 42), "private\nadapter", "#FFF1DB", "#D97706", size=14)
        line(d, [(x1 + 120, yy + 19), (x1 + 155, yy + 19)], COL["blue"], w=2)
        line(d, [(x1 + 295, yy + 19), (x1 + 335, yy + 19)], COL["blue"], w=2)
        line(d, [(x1 + 480, yy + 19), (x1 + 520, yy + 19)], COL["blue"], w=2)
    label(d, x1 + 320, y2 - 24, "shared blocks may federate; adapters stay local", 16, fill=COL["muted"], anchor="mm")


def relation_gate_block(d: ImageDraw.ImageDraw, b: Sequence[float]) -> None:
    rounded(d, b, COL["white"], "#7EBB8D", r=18, w=2)
    x1, y1, x2, y2 = b
    label(d, x1 + 22, y1 + 36, "Relation-channel normalization and gate", 23, bold=True)
    colors = [COL["soft_blue"], COL["soft_green"], COL["soft_yellow"], COL["soft_pink"], COL["soft_purple"]]
    names = ["EV", "BES", "PV", "ES", "TR"]
    for i, (n, c) in enumerate(zip(names, colors)):
        yy = y1 + 88 + i * 44
        rect(d, (x1 + 42, yy, x1 + 245, yy + 24), c, "#8796AA", w=1)
        label(d, x1 + 60, yy + 20, n + "-CS channel", 15, bold=True)
        # gate slider
        d.line([(sf(x1 + 290), sf(yy + 12)), (sf(x1 + 420), sf(yy + 12))], fill="#A6B2C2", width=sf(4))
        knob_x = x1 + 310 + [80, 35, 55, 22, 68][i]
        d.ellipse(bx((knob_x - 9, yy + 3, knob_x + 9, yy + 21)), fill=COL["green"], outline=COL["ink"], width=sf(1))
    mini_card(d, (x1 + 485, y1 + 122, x1 + 610, y1 + 238), "fused\ngraph\nstate", "#E8F6EE", "#4E9F69", size=18)
    for i in range(5):
        yy = y1 + 100 + i * 44
        line(d, [(x1 + 420, yy), (x1 + 485, y1 + 180)], COL["green"], w=2, arrow=False)


def actor_network_block(d: ImageDraw.ImageDraw, b: Sequence[float]) -> None:
    rounded(d, b, COL["white"], "#7EBB8D", r=18, w=2)
    x1, y1, x2, y2 = b
    label(d, x1 + 22, y1 + 36, "Actor policy network", 23, bold=True)
    neuron_net(d, x1 + 95, y1 + 165, [4, 5, 4, 2], color="#BBF7D0", label_text="mean / log-std")
    mini_card(d, (x1 + 360, y1 + 100, x1 + 500, y1 + 165), "EV actions", COL["soft_blue"], "#6B91C9", size=18)
    mini_card(d, (x1 + 360, y1 + 205, x1 + 500, y1 + 270), "BES action", COL["soft_green"], "#5EA875", size=18)
    line(d, [(x1 + 285, y1 + 155), (x1 + 360, y1 + 132)], COL["blue"], w=3)
    line(d, [(x1 + 285, y1 + 175), (x1 + 360, y1 + 237)], COL["blue"], w=3)


def critic_network_block(d: ImageDraw.ImageDraw, b: Sequence[float]) -> None:
    rounded(d, b, COL["white"], "#6BB89B", r=18, w=2)
    x1, y1, x2, y2 = b
    label(d, x1 + 22, y1 + 36, "Role-preserving twin critics", 23, bold=True)
    # Type pools
    pools = ["CS", "BES", "PV", "ES", "EV"]
    for i, p in enumerate(pools):
        mini_card(d, (x1 + 35 + i * 86, y1 + 72, x1 + 105 + i * 86, y1 + 118), p, "#F8FAFC", "#8796AA", size=15)
    mini_card(d, (x1 + 500, y1 + 70, x1 + 610, y1 + 120), "concat", "#E8F6EE", "#4E9F69", size=16)
    for i in range(5):
        line(d, [(x1 + 105 + i * 86, y1 + 95), (x1 + 500, y1 + 95)], COL["green"], w=2, arrow=False)
    # Twin reward and cost nets
    neuron_net(d, x1 + 105, y1 + 215, [3, 4, 2], color="#BFDBFE", label_text="Q-r1")
    neuron_net(d, x1 + 315, y1 + 215, [3, 4, 2], color="#BFDBFE", label_text="Q-r2")
    neuron_net(d, x1 + 525, y1 + 215, [3, 4, 2], color="#FDE68A", label_text="Q-c1")
    neuron_net(d, x1 + 735, y1 + 215, [3, 4, 2], color="#FDE68A", label_text="Q-c2")
    label(d, x1 + 250, y1 + 335, "reward value", 17, fill=COL["muted"], anchor="mm")
    label(d, x1 + 670, y1 + 335, "constraint-risk value", 17, fill=COL["muted"], anchor="mm")


def federation_block(d: ImageDraw.ImageDraw, b: Sequence[float]) -> None:
    rounded(d, b, COL["fed"], "#B59BE8", r=24, w=3)
    x1, y1, x2, y2 = b
    label(d, x1 + 28, y1 + 45, "Personalized relation federation", 29, bold=True)
    # Upload puzzle blocks
    for i, name in enumerate(["R", "O", "C"]):
        yy = y1 + 110 + i * 95
        mini_card(d, (x1 + 45, yy, x1 + 170, yy + 58), f"{name}\nshared", "#DFF3E7", "#4E9F69", size=16)
        draw_lock(d, x1 + 222, yy + 10)
        label(d, x1 + 252, yy + 38, "private parts blocked", 16, fill=COL["red"], anchor="lm")
        line(d, [(x1 + 170, yy + 29), (x1 + 360, y1 + 250)], COL["purple"], w=3, dashed=True)
    mini_card(d, (x1 + 360, y1 + 180, x1 + 560, y1 + 320), "federation\nserver", "#FFFFFF", "#9A7EDB", size=22)
    # heatmap
    hx, hy = x1 + 625, y1 + 160
    label(d, hx, hy - 25, "personalized weights", 17, bold=True)
    heat = [["#6D28D9", "#C4B5FD", "#DDD6FE"], ["#C4B5FD", "#6D28D9", "#BCA8FA"], ["#DDD6FE", "#BCA8FA", "#6D28D9"]]
    for r in range(3):
        for c in range(3):
            rect(d, (hx + c * 42, hy + r * 42, hx + c * 42 + 38, hy + r * 42 + 38), heat[r][c], "#FFFFFF", w=1)
    mini_card(d, (x1 + 605, y1 + 340, x1 + 830, y1 + 425), "park-specific\nshared references", "#FFFFFF", "#9A7EDB", size=18)
    line(d, [(x1 + 560, y1 + 250), (hx, hy + 63)], COL["purple"], w=3, dashed=True)
    line(d, [(hx + 126, hy + 63), (x1 + 710, y1 + 340)], COL["purple"], w=3, dashed=True)


def execution_block(d: ImageDraw.ImageDraw, b: Sequence[float]) -> None:
    rounded(d, b, COL["exec"], "#E5A454", r=24, w=3)
    x1, y1, x2, y2 = b
    label(d, x1 + 28, y1 + 45, "Safe execution and regional coordination", 29, bold=True)
    stages = [
        ("raw actions", "policy output"),
        ("device\nbounds", "SoC / power\nV2G feasibility"),
        ("CS safe\nadjust", "local capacity\ncorrection"),
        ("secure\naggregate", "sum only"),
        ("TR\ncoordination", "shared transformer\npressure"),
        ("execute\nflows", "EV / BES / PV\npark exchange"),
        ("feedback", "reward / cost\nTR signal"),
    ]
    colors = [COL["white"], COL["soft_green"], COL["soft_yellow"], COL["soft_purple"], "#FFE4C7", COL["soft_blue"], COL["white"]]
    x = x1 + 80
    for i, ((t, s), c) in enumerate(zip(stages, colors)):
        mini_card(d, (x + i * 250, y1 + 130, x + i * 250 + 190, y1 + 260), t, c, "#C58A43", size=18)
        label(d, x + i * 250 + 95, y1 + 302, s, 16, fill=COL["muted"], anchor="mm")
        if i < len(stages) - 1:
            line(d, [(x + i * 250 + 190, y1 + 195), (x + (i + 1) * 250, y1 + 195)], COL["orange"], w=4)
    # icons
    # shield on secure aggregate
    sx, sy = x + 3 * 250 + 95, y1 + 190
    d.polygon([(sf(sx), sf(sy - 50)), (sf(sx + 45), sf(sy - 25)), (sf(sx + 30), sf(sy + 45)), (sf(sx), sf(sy + 65)), (sf(sx - 30), sf(sy + 45)), (sf(sx - 45), sf(sy - 25))], fill="#D8B4FE", outline="#7C3AED")
    label(d, sx, sy + 5, "Σ", 30, bold=True, anchor="mm", fill=COL["purple"])
    # transformer coil
    tx, ty = x + 4 * 250 + 95, y1 + 190
    for j in range(3):
        d.arc(bx((tx - 50 + j * 32, ty - 30, tx - 10 + j * 32, ty + 30)), 90, 270, fill=COL["orange"], width=sf(4))


def main() -> None:
    img = Image.new("RGB", (W * S, H * S), COL["bg"])
    d = ImageDraw.Draw(img)
    label(d, 90, 70, "SP-RGNN-CSAC System Architecture", 48, bold=True)
    label(d, 90, 125, "Visual mechanism diagram: local relational graphs, shared-private policy, constrained critics, safe execution, and personalized federation", 25, fill=COL["muted"])

    # Main panels
    rounded(d, (70, 190, 760, 1530), COL["park"], "#8DB3E8", r=28, w=3)
    rounded(d, (840, 190, 2510, 1530), COL["actor"], "#7EBB8D", r=28, w=3)
    label(d, 105, 240, "1. Local park observations", 31, bold=True)
    label(d, 880, 240, "2. SP-RGNN-CSAC local agent internals", 31, bold=True)

    # Local graph parks
    for i, (name, y) in enumerate([("Residential park", 330), ("Office park", 720), ("Commercial park", 1110)]):
        rounded(d, (110, y, 720, y + 315), COL["white"], "#9CB8DD", r=20, w=2)
        label(d, 140, y + 42, name, 24, bold=True)
        graph_subfig(d, 310, y + 170, 0.8)
        mini_card(d, (480, y + 80, 675, y + 150), "local\nraw data", COL["soft_red"], "#DC2626", size=17)
        draw_lock(d, 655, y + 88)
        mini_card(d, (480, y + 190, 675, y + 270), "local graph\nobservation", COL["soft_blue"], "#6B91C9", size=17)
        line(d, [(675, y + 230), (815, y + 230), (815, 410), (920, 410)], COL["blue"], w=4)

    # SP-RGNN mechanism blocks
    mini_card(d, (920, 330, 1115, 490), "type-specific\nnode encoders", COL["white"], "#7EBB8D", size=20)
    # small encoder stacks
    for j, c in enumerate([COL["soft_blue"], COL["soft_green"], COL["soft_yellow"], COL["soft_pink"], COL["soft_purple"]]):
        rect(d, (950 + j * 30, 410 - j * 8, 1015 + j * 30, 445 - j * 8), c, "#8796AA", w=1)
    shared_private_block(d, (1180, 300, 1760, 650))
    relation_gate_block(d, (1835, 300, 2470, 650))
    actor_network_block(d, (920, 725, 1465, 1055))
    critic_network_block(d, (1535, 730, 2470, 1110))
    mini_card(d, (920, 1180, 1135, 1355), "local replay\nbuffer", COL["white"], "#6BB89B", size=21)
    # buffer cylinder hint
    d.ellipse(bx((948, 1210, 1108, 1260)), fill="#E8F6EE", outline="#6BB89B", width=sf(2))
    d.rectangle(bx((948, 1235, 1108, 1305)), fill="#E8F6EE", outline="#6BB89B", width=sf(2))
    d.ellipse(bx((948, 1280, 1108, 1330)), fill="#E8F6EE", outline="#6BB89B", width=sf(2))
    mini_card(d, (1210, 1190, 1515, 1355), "constrained SAC\nupdate", COL["white"], "#6BB89B", size=21)
    # update gauges
    d.arc(bx((1275, 1245, 1345, 1315)), 180, 350, fill=COL["green"], width=sf(5))
    d.arc(bx((1378, 1245, 1448, 1315)), 180, 350, fill=COL["orange"], width=sf(5))
    label(d, 1310, 1340, "temperature", 14, fill=COL["muted"], anchor="mm")
    label(d, 1413, 1340, "multiplier", 14, fill=COL["muted"], anchor="mm")
    mini_card(d, (1615, 1190, 1950, 1355), "actor reference\nregularization", COL["white"], "#6BB89B", size=20)
    mini_card(d, (2030, 1190, 2470, 1355), "updated local policy", COL["white"], "#6BB89B", size=21)
    neuron_net(d, 2165, 1280, [3, 4, 2], color="#BBF7D0")

    # Internal arrows
    line(d, [(1115, 410), (1180, 410)], COL["blue"], w=4)
    line(d, [(1760, 475), (1835, 475)], COL["blue"], w=4)
    line(d, [(2150, 650), (2150, 690), (1040, 690), (1040, 725)], COL["blue"], w=4)
    line(d, [(1465, 900), (2585, 1650)], COL["blue"], w=4)
    line(d, [(1135, 1265), (1210, 1265)], COL["green"], w=4, dashed=True)
    line(d, [(1515, 1265), (1615, 1265)], COL["green"], w=4, dashed=True)
    line(d, [(1950, 1265), (2030, 1265)], COL["green"], w=4, dashed=True)
    line(d, [(2210, 1190), (1310, 1055)], COL["green"], w=4, dashed=True)
    line(d, [(1040, 1180), (1040, 1055)], COL["green"], w=4, dashed=True)
    line(d, [(1535, 920), (1465, 920)], COL["green"], w=3, dashed=True)

    # Right federation panel
    federation_block(d, (2600, 190, 3815, 830))
    line(d, [(1500, 360), (2600, 330)], COL["purple"], w=5, dashed=True)
    line(d, [(3425, 615), (2525, 615), (2525, 1240), (1950, 1240)], COL["purple"], w=5, dashed=True)

    # Bottom execution
    execution_block(d, (260, 1630, 3815, 2250))
    # route action to execution without crossing internals
    line(d, [(1465, 900), (2530, 900), (2530, 1760), (340, 1760)], COL["blue"], w=5)
    # feedback to local observation
    line(d, [(3650, 1760), (3650, 1585), (410, 1585), (410, 1425)], COL["blue"], w=5)
    label(d, 1930, 1575, "TR feedback returns to the next local graph", 21, fill=COL["blue"], anchor="mm")
    # reward/cost to replay
    line(d, [(3650, 1985), (3650, 2310), (1035, 2310), (1035, 1355)], COL["green"], w=5, dashed=True)
    label(d, 2290, 2320, "reward and cost become local transitions", 21, fill=COL["green"], anchor="mm")

    # Legend
    rounded(d, (2600, 900, 3815, 1530), COL["white"], COL["border"], r=22, w=2)
    label(d, 2635, 950, "How to read the subfigures", 27, bold=True)
    line(d, [(2660, 1025), (2810, 1025)], COL["blue"], w=5)
    label(d, 2840, 1032, "online decision and execution flow", 21, fill=COL["muted"], anchor="lm")
    line(d, [(2660, 1095), (2810, 1095)], COL["green"], w=5, dashed=True)
    label(d, 2840, 1102, "local constrained learning flow", 21, fill=COL["muted"], anchor="lm")
    line(d, [(2660, 1165), (2810, 1165)], COL["purple"], w=5, dashed=True)
    label(d, 2840, 1172, "periodic relation-parameter federation", 21, fill=COL["muted"], anchor="lm")
    mini_card(d, (2650, 1250, 2825, 1320), "green blocks", "#DFF3E7", "#4E9F69", size=17)
    label(d, 2850, 1295, "shared relation parameters allowed to federate", 20, fill=COL["muted"], anchor="lm")
    mini_card(d, (2650, 1360, 2825, 1430), "orange blocks", "#FFF1DB", "#D97706", size=17)
    label(d, 2850, 1405, "park-private adapters and local differences", 20, fill=COL["muted"], anchor="lm")

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

