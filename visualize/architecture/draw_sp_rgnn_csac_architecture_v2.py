from __future__ import annotations

from pathlib import Path
import math
import textwrap
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "sp_rgnn_csac_system_architecture_v2.png"

W, H = 3400, 2150
S = 2

COLORS = {
    "bg": "#F7F9FC",
    "ink": "#172033",
    "muted": "#5B677A",
    "border": "#B9C4D2",
    "white": "#FFFFFF",
    "park": "#EAF2FF",
    "graph": "#F8FBFF",
    "core": "#ECF8F0",
    "exec": "#FFF3E6",
    "learn": "#EAF8F2",
    "fed": "#F4ECFF",
    "private": "#FFF1F1",
    "online": "#1D4ED8",
    "learn_line": "#047857",
    "fed_line": "#7C3AED",
    "safety": "#D97706",
    "red": "#DC2626",
    "green": "#16A34A",
    "yellow": "#FBBF24",
    "purple": "#A78BFA",
    "blue_soft": "#DBEAFE",
    "green_soft": "#DCFCE7",
    "yellow_soft": "#FEF3C7",
    "pink_soft": "#FCE7F3",
    "purple_soft": "#EDE9FE",
}

FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_CN = "C:/Windows/Fonts/msyh.ttc"


def fnt(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT
    if not Path(path).exists():
        path = FONT_CN
    return ImageFont.truetype(path, size * S)


def sc(v: float) -> int:
    return int(round(v * S))


def box_scaled(box: Sequence[float]) -> tuple[int, int, int, int]:
    return tuple(sc(v) for v in box)  # type: ignore[return-value]


def text_wh(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0], b[3] - b[1]


def wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    for part in text.split("\n"):
        out.extend(textwrap.wrap(part, width=width, break_long_words=False) or [""])
    return out


def label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    size: int,
    *,
    bold: bool = False,
    fill: str = COLORS["ink"],
    anchor: str = "la",
) -> None:
    draw.text((sc(xy[0]), sc(xy[1])), text, font=fnt(size, bold), fill=fill, anchor=anchor)


def centered(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    text: str,
    size: int,
    *,
    bold: bool = False,
    fill: str = COLORS["ink"],
    width: int | None = None,
    gap: int = 6,
) -> None:
    font = fnt(size, bold)
    lines = wrap(text, width) if width else text.split("\n")
    line_h = [text_wh(draw, line, font)[1] for line in lines]
    total_h = sum(line_h) + max(0, len(lines) - 1) * gap * S
    x1, y1, x2, y2 = box_scaled(box)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, h in zip(lines, line_h):
        w, _ = text_wh(draw, line, font)
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=font, fill=fill)
        y += h + gap * S


def rounded(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    fill: str,
    outline: str = COLORS["border"],
    *,
    radius: int = 24,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(box_scaled(box), radius=sc(radius), fill=fill, outline=outline, width=sc(width))


def card(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    title: str,
    subtitle: str = "",
    *,
    fill: str = COLORS["white"],
    outline: str = COLORS["border"],
    title_size: int = 25,
    body_size: int = 20,
    body_width: int = 26,
) -> None:
    rounded(draw, box, fill, outline, radius=18, width=2)
    x1, y1, x2, y2 = box
    centered(draw, (x1 + 10, y1 + 12, x2 - 10, y1 + 54), title, title_size, bold=True, width=18)
    if subtitle:
        centered(
            draw,
            (x1 + 16, y1 + 58, x2 - 16, y2 - 12),
            subtitle,
            body_size,
            fill=COLORS["muted"],
            width=body_width,
            gap=4,
        )


def arrow(
    draw: ImageDraw.ImageDraw,
    pts: Sequence[tuple[float, float]],
    color: str,
    *,
    width: int = 5,
    dashed: bool = False,
    head: bool = True,
) -> None:
    points = [(sc(x), sc(y)) for x, y in pts]
    if dashed:
        for a, b in zip(points, points[1:]):
            dashed_line(draw, a, b, color, sc(width))
    else:
        draw.line(points, fill=color, width=sc(width), joint="curve")
    if head and len(points) >= 2:
        head_arrow(draw, points[-2], points[-1], color, sc(width))


def dashed_line(
    draw: ImageDraw.ImageDraw,
    a: tuple[int, int],
    b: tuple[int, int],
    color: str,
    width: int,
    dash: int = 26,
    gap: int = 16,
) -> None:
    ax, ay = a
    bx, by = b
    dist = math.hypot(bx - ax, by - ay)
    if dist <= 0:
        return
    ux, uy = (bx - ax) / dist, (by - ay) / dist
    t = 0.0
    while t < dist:
        t2 = min(t + dash * S, dist)
        draw.line(
            [(int(ax + ux * t), int(ay + uy * t)), (int(ax + ux * t2), int(ay + uy * t2))],
            fill=color,
            width=width,
        )
        t += (dash + gap) * S


def head_arrow(draw: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, width: int) -> None:
    ax, ay = a
    bx, by = b
    angle = math.atan2(by - ay, bx - ax)
    size = max(sc(20), width * 3)
    p1 = (bx, by)
    p2 = (bx - size * math.cos(angle - math.pi / 7), by - size * math.sin(angle - math.pi / 7))
    p3 = (bx - size * math.cos(angle + math.pi / 7), by - size * math.sin(angle + math.pi / 7))
    draw.polygon([p1, p2, p3], fill=color)


def dot_node(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fill: str) -> None:
    x, y = xy
    draw.ellipse(box_scaled((x - 34, y - 28, x + 34, y + 28)), fill=fill, outline="#8796AA", width=sc(2))
    centered(draw, (x - 34, y - 24, x + 34, y + 24), text, 18, bold=True)


def local_graph(draw: ImageDraw.ImageDraw, cx: float, cy: float, scale: float = 1.0) -> None:
    nodes = [
        ("EV", cx - 108 * scale, cy - 70 * scale, COLORS["blue_soft"]),
        ("BES", cx + 108 * scale, cy - 70 * scale, COLORS["green_soft"]),
        ("PV", cx - 108 * scale, cy + 70 * scale, COLORS["yellow_soft"]),
        ("ES", cx + 108 * scale, cy + 70 * scale, COLORS["pink_soft"]),
        ("TR", cx, cy + 125 * scale, COLORS["purple_soft"]),
    ]
    for _, x, y, _ in nodes:
        arrow(draw, [(x, y), (cx, cy)], "#97A6BA", width=3, head=False)
        arrow(draw, [(cx, cy), (x, y)], "#97A6BA", width=3, head=False)
    dot_node(draw, (cx, cy), "CS", COLORS["white"])
    for node in nodes:
        dot_node(draw, (node[1], node[2]), node[0], node[3])


def lock(draw: ImageDraw.ImageDraw, x: float, y: float) -> None:
    draw.rounded_rectangle(box_scaled((x - 18, y - 2, x + 18, y + 30)), radius=sc(6), fill=COLORS["private"], outline=COLORS["red"], width=sc(2))
    draw.arc(box_scaled((x - 13, y - 28, x + 13, y + 14)), 180, 360, fill=COLORS["red"], width=sc(4))


def pill(draw: ImageDraw.ImageDraw, box: Sequence[float], text: str, fill: str, outline: str) -> None:
    rounded(draw, box, fill, outline, radius=40, width=2)
    centered(draw, box, text, 19, bold=True, width=18)


def draw_shared_private_subfigure(draw: ImageDraw.ImageDraw, box: Sequence[float]) -> None:
    rounded(draw, box, COLORS["white"], "#8BC49B", radius=18, width=2)
    x1, y1, x2, y2 = box
    label(draw, (x1 + 20, y1 + 38), "Shared-private relation transform", 24, bold=True)
    # Shared block
    rounded(draw, (x1 + 35, y1 + 80, x1 + 210, y1 + 185), "#E7F5EC", "#64A878", radius=12, width=2)
    centered(draw, (x1 + 45, y1 + 92, x1 + 200, y1 + 172), "Shared\nrelation weights", 19, bold=True, width=16)
    # Private adapters
    for idx, name in enumerate(["R adapter", "O adapter", "C adapter"]):
        yy = y1 + 68 + idx * 50
        rounded(draw, (x1 + 270, yy, x1 + 435, yy + 38), "#FFF7ED", "#F59E0B", radius=9, width=2)
        centered(draw, (x1 + 280, yy + 2, x1 + 425, yy + 36), name, 17, bold=True)
    arrow(draw, [(x1 + 210, y1 + 132), (x1 + 270, y1 + 132)], COLORS["online"], width=3)
    label(draw, (x1 + 35, y2 - 28), "transferable physical relation knowledge + park-specific adaptation", 18, fill=COLORS["muted"])


def draw_relation_channels(draw: ImageDraw.ImageDraw, box: Sequence[float]) -> None:
    rounded(draw, box, COLORS["white"], "#8BC49B", radius=18, width=2)
    x1, y1, x2, y2 = box
    label(draw, (x1 + 20, y1 + 38), "Relation channels", 24, bold=True)
    names = ["EV-CS", "BES-CS", "PV-CS", "ES-CS", "TR-CS"]
    colors = [COLORS["blue_soft"], COLORS["green_soft"], COLORS["yellow_soft"], COLORS["pink_soft"], COLORS["purple_soft"]]
    start = x1 + 32
    for i, name in enumerate(names):
        pill(draw, (start + i * 104, y1 + 80, start + i * 104 + 88, y1 + 132), name, colors[i], "#8796AA")
    card(draw, (x1 + 72, y1 + 165, x1 + 260, y1 + 265), "Relation-wise\nnormalization", "avoid dense EV messages\nswamping sparse channels", title_size=20, body_size=16, body_width=20)
    card(draw, (x1 + 330, y1 + 165, x1 + 520, y1 + 265), "Learnable\nrelation gate", "weighted fusion of active\nphysical channels", title_size=20, body_size=16, body_width=20)
    arrow(draw, [(x1 + 260, y1 + 215), (x1 + 330, y1 + 215)], COLORS["online"], width=3)


def draw_critic_subfigure(draw: ImageDraw.ImageDraw, box: Sequence[float]) -> None:
    rounded(draw, box, COLORS["white"], "#8BC49B", radius=18, width=2)
    x1, y1, x2, y2 = box
    label(draw, (x1 + 20, y1 + 38), "Role-preserving critic readout", 24, bold=True)
    for i, name in enumerate(["CS", "BES", "PV", "ES", "EV"]):
        pill(draw, (x1 + 32 + i * 98, y1 + 82, x1 + 112 + i * 98, y1 + 130), name, COLORS["graph"], "#8796AA")
    arrow(draw, [(x1 + 115, y1 + 172), (x1 + 455, y1 + 172)], COLORS["learn_line"], width=3, head=False)
    centered(draw, (x1 + 80, y1 + 145, x2 - 80, y1 + 210), "separate type summaries are concatenated\nbefore reward and cost value estimation", 19, fill=COLORS["muted"], width=48)


def draw() -> Image.Image:
    img = Image.new("RGB", (W * S, H * S), COLORS["bg"])
    draw = ImageDraw.Draw(img)

    label(draw, (90, 70), "SP-RGNN-CSAC System Architecture", 46, bold=True)
    label(draw, (90, 122), "Privacy-preserving cooperative EV scheduling across heterogeneous park microgrids", 26, fill=COLORS["muted"])

    # Containers
    local_box = (70, 190, 760, 1430)
    core_box = (850, 190, 2300, 1430)
    fed_box = (2385, 190, 3295, 935)
    exec_box = (850, 1510, 3295, 2045)
    rounded(draw, local_box, COLORS["park"], "#8FB4E8", radius=28, width=3)
    rounded(draw, core_box, COLORS["core"], "#87C797", radius=28, width=3)
    rounded(draw, fed_box, COLORS["fed"], "#B599EA", radius=28, width=3)
    rounded(draw, exec_box, COLORS["exec"], "#E9A55A", radius=28, width=3)

    label(draw, (105, 235), "A. Decentralized park agents", 31, bold=True)
    label(draw, (880, 235), "B. Shared-private relational constrained SAC", 31, bold=True)
    label(draw, (2420, 235), "C. Personalized relation federation", 31, bold=True)
    label(draw, (880, 1555), "D. Safe execution and regional transformer coordination", 31, bold=True)

    # Park agents
    parks = [("Residential", 325), ("Office", 700), ("Commercial", 1075)]
    for name, y in parks:
        rounded(draw, (105, y, 725, y + 300), COLORS["white"], "#9CB8DD", radius=20, width=2)
        label(draw, (135, y + 38), f"{name} park local boundary", 24, bold=True)
        local_graph(draw, 295, y + 165, 0.82)
        card(draw, (460, y + 72, 695, y + 165), "Private local data", "EV sessions, states,\nactions, rewards, costs", title_size=19, body_size=16, body_width=22)
        card(draw, (460, y + 185, 695, y + 275), "Private learners", "critics, replay buffer,\nadapters stay local", title_size=19, body_size=16, body_width=22)
        lock(draw, 680, y + 92)
    label(draw, (120, 1378), "Each park builds its own local relational graph; raw data from other parks is not observed.", 21, fill=COLORS["muted"])

    # Core subfigures
    card(draw, (900, 310, 1145, 460), "Type-specific\nnode encoders", "EV, BES, PV, CS, ES,\nTR feedback features", title_size=22, body_size=18, body_width=25)
    draw_shared_private_subfigure(draw, (1215, 300, 1775, 520))
    draw_relation_channels(draw, (1835, 300, 2265, 610))
    card(draw, (900, 570, 1145, 720), "Graph latent\nrepresentation", "relation-aware physical\nstate embedding", title_size=22, body_size=18, body_width=25)
    card(draw, (1215, 590, 1485, 740), "Actor network", "continuous EV/BES\nscheduling actions", title_size=23, body_size=18, body_width=24)
    draw_critic_subfigure(draw, (1545, 675, 2265, 895))
    card(draw, (900, 900, 1145, 1060), "Local replay\nbuffer", "transition samples\nfrom executed steps", title_size=22, body_size=18, body_width=24)
    card(draw, (1215, 895, 1485, 1058), "Twin reward\ncritics", "economic value\nestimation", title_size=22, body_size=18, body_width=22)
    card(draw, (1215, 1105, 1485, 1268), "Twin cost\ncritics", "constraint-risk\nestimation", title_size=22, body_size=18, body_width=22)
    card(draw, (1545, 1035, 1845, 1215), "Constrained SAC\nlocal update", "updates actor, critics,\ntemperature and multiplier", title_size=23, body_size=18, body_width=28)
    card(draw, (1905, 1040, 2265, 1215), "Federation reference\nregularization", "keeps shared relation\nparameters stable after upload", title_size=22, body_size=17, body_width=31)

    # Core internal arrows
    arrow(draw, [(1145, 385), (1215, 385)], COLORS["online"], width=4)
    arrow(draw, [(1775, 410), (1835, 410)], COLORS["online"], width=4)
    arrow(draw, [(2050, 610), (2050, 660), (1020, 660), (1020, 570)], COLORS["online"], width=4)
    arrow(draw, [(1145, 645), (1215, 665)], COLORS["online"], width=4)
    arrow(draw, [(1485, 670), (2340, 670), (2340, 1485), (1035, 1485), (1035, 1635)], COLORS["online"], width=5)
    arrow(draw, [(1145, 980), (1215, 980)], COLORS["learn_line"], width=4, dashed=True)
    arrow(draw, [(1145, 980), (1215, 1188)], COLORS["learn_line"], width=4, dashed=True)
    arrow(draw, [(1485, 976), (1545, 1105)], COLORS["learn_line"], width=4, dashed=True)
    arrow(draw, [(1485, 1188), (1545, 1138)], COLORS["learn_line"], width=4, dashed=True)
    arrow(draw, [(1845, 1120), (1905, 1120)], COLORS["learn_line"], width=4, dashed=True)
    arrow(draw, [(1695, 1035), (1360, 740)], COLORS["learn_line"], width=4, dashed=True)

    # Local observations to core
    for _, y in parks:
        arrow(draw, [(725, y + 150), (820, y + 150), (820, 385), (900, 385)], COLORS["online"], width=4)

    # Federation panel
    card(draw, (2430, 315, 2705, 470), "Uploaded", "selected actor-side\nshared relation transforms", title_size=23, body_size=18, body_width=24)
    card(draw, (2790, 315, 3165, 470), "Not uploaded", "raw states/actions, rewards/costs,\ncritics, replay buffers,\nprivate adapters and heads", title_size=23, body_size=17, body_width=31)
    lock(draw, 3135, 345)
    card(draw, (2510, 555, 2850, 735), "Personalized aggregation", "learned cross-park compatibility\nand relation availability masks", title_size=24, body_size=18, body_width=34)
    card(draw, (2915, 555, 3235, 735), "Returned", "park-specific shared\nrelation references", title_size=24, body_size=18, body_width=28)
    arrow(draw, [(1775, 332), (2430, 392)], COLORS["fed_line"], width=5, dashed=True)
    arrow(draw, [(2705, 392), (2510, 645)], COLORS["fed_line"], width=5, dashed=True)
    arrow(draw, [(2850, 645), (2915, 645)], COLORS["fed_line"], width=5, dashed=True)
    arrow(draw, [(2915, 660), (2340, 660), (2340, 520), (1775, 500)], COLORS["fed_line"], width=5, dashed=True)
    label(draw, (2425, 865), "Slow time-scale branch: knowledge transfer without centralizing trajectories.", 22, fill=COLORS["muted"])

    # Execution panel
    xs = [910, 1240, 1580, 1935, 2290, 2650, 2990]
    exec_cards = [
        ("Action mapping", "Actor actions become\nrequested EV/BES energy"),
        ("Device bounds", "SoC, power and V2G\nfeasibility per device"),
        ("CS capacity-safe\nadjustment", "local station exchange\nkept within CS limit"),
        ("Secure aggregation", "only aggregate exchange\nand capacity statistics"),
        ("TR coordination", "shared transformer import/export\ncapacity regulation"),
        ("Final execution", "EV/BES/PV flows and\npark grid exchange"),
        ("Feedback signals", "reward, cost and\nTR coordination feedback"),
    ]
    for x, (title, sub) in zip(xs, exec_cards):
        card(draw, (x, 1635, x + 250, 1835), title, sub, title_size=21, body_size=17, body_width=24)
    for i in range(len(xs) - 1):
        arrow(draw, [(xs[i] + 250, 1735), (xs[i + 1], 1735)], COLORS["safety"], width=5)
    label(draw, (2140, 1902), "Execution-time cooperation handles physical constraints; it is separate from the federation server.", 22, fill=COLORS["muted"], anchor="mm")

    # Feedback routing: outside containers, no crossing through subfigures
    arrow(draw, [(3115, 1635), (3115, 1465), (500, 1465), (500, 1375)], COLORS["online"], width=5)
    label(draw, (1770, 1455), "TR feedback returns to next-step local graph", 22, fill=COLORS["online"], anchor="mm")
    arrow(draw, [(3115, 1835), (3115, 2075), (990, 2075), (990, 1060)], COLORS["learn_line"], width=5, dashed=True)
    label(draw, (1990, 2087), "reward and cost are stored as local transitions", 22, fill=COLORS["learn_line"], anchor="mm")

    # Legend
    rounded(draw, (2480, 1000, 3275, 1395), COLORS["white"], COLORS["border"], radius=20, width=2)
    label(draw, (2515, 1045), "Line legend and privacy meaning", 25, bold=True)
    arrow(draw, [(2530, 1105), (2680, 1105)], COLORS["online"], width=5)
    label(draw, (2710, 1112), "online decision / execution / next observation", 21, fill=COLORS["muted"], anchor="lm")
    arrow(draw, [(2530, 1165), (2680, 1165)], COLORS["learn_line"], width=5, dashed=True)
    label(draw, (2710, 1172), "local constrained SAC learning", 21, fill=COLORS["muted"], anchor="lm")
    arrow(draw, [(2530, 1225), (2680, 1225)], COLORS["fed_line"], width=5, dashed=True)
    label(draw, (2710, 1232), "periodic personalized federation", 21, fill=COLORS["muted"], anchor="lm")
    label(draw, (2515, 1310), "Two central-looking modules are different:", 21, bold=True)
    label(draw, (2515, 1342), "TR coordination handles physical exchange; federation handles parameters.", 20, fill=COLORS["muted"])

    img = img.resize((W, H), Image.Resampling.LANCZOS)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = draw()
    img.save(PNG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()

