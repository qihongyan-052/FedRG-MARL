from __future__ import annotations

from pathlib import Path
from typing import Sequence
import math

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent / "modules_v6"
PNG = OUT_DIR / "core_agent_module_v6.png"

S = 2
W, H = 2400, 1500

C = {
    "bg": "#F8FAFC",
    "ink": "#111827",
    "muted": "#64748B",
    "border": "#CBD5E1",
    "white": "#FFFFFF",
    "blue": "#1D4ED8",
    "green": "#047857",
    "purple": "#7C3AED",
    "orange": "#D97706",
    "red": "#DC2626",
    "panel": "#ECF8F0",
    "shared": "#DCFCE7",
    "private": "#FFF7ED",
    "actor": "#DBEAFE",
    "critic": "#FEF3C7",
    "cost": "#FDE68A",
    "role": "#F8FAFC",
    "lock": "#FEE2E2",
}

FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_CN = "C:/Windows/Fonts/msyh.ttc"


def sc(v: float) -> int:
    return int(round(v * S))


def sb(b: Sequence[float]) -> tuple[int, int, int, int]:
    return tuple(sc(x) for x in b)  # type: ignore[return-value]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT
    if not Path(path).exists():
        path = FONT_CN
    return ImageFont.truetype(path, size * S)


def label(d: ImageDraw.ImageDraw, x: float, y: float, text: str, size: int, *, bold=False, fill=None, anchor="la") -> None:
    d.text((sc(x), sc(y)), text, font=font(size, bold), fill=fill or C["ink"], anchor=anchor)


def centered(d: ImageDraw.ImageDraw, b: Sequence[float], text: str, size: int, *, bold=False, fill=None) -> None:
    f = font(size, bold)
    lines = text.split("\n")
    boxes = [d.textbbox((0, 0), line, font=f) for line in lines]
    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    gap = 5 * S
    total = sum(heights) + gap * (len(lines) - 1)
    x1, y1, x2, y2 = sb(b)
    y = y1 + (y2 - y1 - total) / 2
    for line, w, h in zip(lines, widths, heights):
        d.text((x1 + (x2 - x1 - w) / 2, y), line, font=f, fill=fill or C["ink"])
        y += h + gap


def rounded(d: ImageDraw.ImageDraw, b: Sequence[float], fill: str, outline: str = C["border"], r: int = 18, w: int = 2) -> None:
    d.rounded_rectangle(sb(b), radius=sc(r), fill=fill, outline=outline, width=sc(w))


def arrow(d: ImageDraw.ImageDraw, pts: Sequence[tuple[float, float]], color: str, *, w: int = 4, dashed=False, head=True) -> None:
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
    if dist <= 0:
        return
    ux, uy = (bx - ax) / dist, (by - ay) / dist
    t = 0.0
    while t < dist:
        t2 = min(t + 26 * S, dist)
        d.line([(int(ax + ux * t), int(ay + uy * t)), (int(ax + ux * t2), int(ay + uy * t2))], fill=color, width=w)
        t += 42 * S


def arrow_head(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, w: int) -> None:
    ax, ay = a
    bx, by = b
    angle = math.atan2(by - ay, bx - ax)
    size = max(18 * S, w * 3)
    d.polygon(
        [
            (bx, by),
            (bx - size * math.cos(angle - math.pi / 7), by - size * math.sin(angle - math.pi / 7)),
            (bx - size * math.cos(angle + math.pi / 7), by - size * math.sin(angle + math.pi / 7)),
        ],
        fill=color,
    )


def block(d: ImageDraw.ImageDraw, b: Sequence[float], text: str, fill: str, outline: str = C["border"], size: int = 18) -> None:
    rounded(d, b, fill, outline, r=12, w=2)
    centered(d, b, text, size, bold=True)


def lane_label(d: ImageDraw.ImageDraw, x: float, y: float, text: str, color: str) -> None:
    rounded(d, (x, y, x + 130, y + 36), "#FFFFFF", color, r=18, w=2)
    centered(d, (x, y, x + 130, y + 36), text, 15, bold=True, fill=color)


def lock(d: ImageDraw.ImageDraw, x: float, y: float) -> None:
    rounded(d, (x - 16, y, x + 16, y + 30), C["lock"], C["red"], r=6, w=2)
    d.arc(sb((x - 12, y - 26, x + 12, y + 12)), 180, 360, fill=C["red"], width=sc(4))


def draw_relation_encoder(d: ImageDraw.ImageDraw) -> None:
    rounded(d, (85, 210, 760, 1160), C["white"], "#86B894", r=22, w=2)
    label(d, 120, 260, "Relational graph encoder", 27, bold=True)
    label(d, 120, 296, "message channels stay separate before fusion", 18, fill=C["muted"])

    # Type embedding input
    block(d, (130, 370, 315, 470), "type-specific\nembedding", "#F8FAFC", "#94A3B8")

    rels = [
        ("EV-CS", "#DBEAFE"),
        ("BES-CS", "#DCFCE7"),
        ("PV-CS", "#FEF3C7"),
        ("ES-CS", "#FCE7F3"),
        ("TR-CS", "#EDE9FE"),
    ]
    for i, (name, fill) in enumerate(rels):
        y = 545 + i * 88
        lane_label(d, 125, y + 16, name, C["blue"])
        block(d, (290, y, 430, y + 68), "shared\ntransform", C["shared"], "#4E9F69", size=15)
        block(d, (465, y, 605, y + 68), "private\nadapter", C["private"], C["orange"], size=15)
        arrow(d, [(255, y + 34), (290, y + 34)], C["blue"], w=2)
        arrow(d, [(430, y + 34), (465, y + 34)], C["blue"], w=2)
        arrow(d, [(605, y + 34), (675, 790)], C["green"], w=2, head=False)

    block(d, (635, 720, 725, 860), "relation\nfusion", "#E8F6EE", "#4E9F69", size=16)
    arrow(d, [(315, 420), (375, 545)], C["blue"], w=3)
    block(d, (285, 1030, 610, 1100), "Only green shared transforms are federation candidates", "#F8FAFC", "#94A3B8", size=16)


def draw_actor(d: ImageDraw.ImageDraw) -> None:
    rounded(d, (840, 210, 1500, 665), C["white"], "#86B4DD", r=22, w=2)
    label(d, 875, 260, "Actor network", 27, bold=True)
    label(d, 875, 296, "clean layer blocks instead of full neuron edges", 18, fill=C["muted"])

    xs = [900, 1080, 1260]
    names = ["graph\nembedding", "policy\ntrunk", "stochastic\naction layer"]
    fills = ["#F8FAFC", "#EAF2FF", "#DBEAFE"]
    for x, name, fill in zip(xs, names, fills):
        block(d, (x, 395, x + 140, 500), name, fill, "#6B91C9", size=17)
    arrow(d, [(1040, 448), (1080, 448)], C["blue"], w=3)
    arrow(d, [(1220, 448), (1260, 448)], C["blue"], w=3)

    block(d, (1410, 340, 1480, 420), "EV\nactions", C["actor"], "#6B91C9", size=14)
    block(d, (1410, 485, 1480, 565), "BES\naction", "#DCFCE7", "#5EA875", size=14)
    arrow(d, [(1400, 430), (1410, 380)], C["blue"], w=3)
    arrow(d, [(1400, 470), (1410, 525)], C["blue"], w=3)


def draw_critics(d: ImageDraw.ImageDraw) -> None:
    rounded(d, (840, 720, 1500, 1160), C["white"], "#86B894", r=22, w=2)
    label(d, 875, 770, "Role-preserving twin critics", 27, bold=True)
    label(d, 875, 806, "typed pooling keeps EV/BES/PV/CS/ES roles distinct", 18, fill=C["muted"])

    roles = ["CS", "BES", "PV", "ES", "EV"]
    for i, role in enumerate(roles):
        block(d, (890 + i * 88, 875, 955 + i * 88, 925), role, "#F8FAFC", "#94A3B8", size=14)
        arrow(d, [(955 + i * 88, 900), (1330, 900)], C["green"], w=1, head=False)
    block(d, (1330, 870, 1440, 930), "concat", "#E8F6EE", "#4E9F69", size=15)

    q_blocks = [
        ("Q-r1", 900, 1010, "#BFDBFE"),
        ("Q-r2", 1045, 1010, "#BFDBFE"),
        ("Q-c1", 1190, 1010, "#FDE68A"),
        ("Q-c2", 1335, 1010, "#FDE68A"),
    ]
    for name, x, y, fill in q_blocks:
        block(d, (x, y, x + 105, y + 78), name, fill, "#94A3B8", size=18)
        arrow(d, [(1385, 930), (x + 52, y)], C["green"], w=2, head=False)
    label(d, 1010, 1118, "reward value", 16, fill=C["muted"], anchor="mm")
    label(d, 1290, 1118, "constraint risk", 16, fill=C["muted"], anchor="mm")


def draw_learning(d: ImageDraw.ImageDraw) -> None:
    rounded(d, (1580, 210, 2305, 1160), C["white"], "#86B894", r=22, w=2)
    label(d, 1615, 260, "Local constrained SAC update", 27, bold=True)
    label(d, 1615, 296, "updates stay inside each park", 18, fill=C["muted"])

    stages = [
        ("transition", "obs, action,\nreward, cost"),
        ("replay\nbuffer", "local\nsamples"),
        ("critic\nupdates", "reward and\ncost critics"),
        ("actor\nupdate", "policy\nimprovement"),
        ("temperature /\nmultiplier", "exploration and\nconstraint pressure"),
        ("local actor\nrefresh", "next-step\npolicy"),
    ]
    y0 = 410
    for i, (title, sub) in enumerate(stages):
        y = y0 + i * 110
        block(d, (1645, y, 1815, y + 72), title, "#E8F6EE", "#6BB89B", size=17)
        label(d, 1915, y + 45, sub, 16, fill=C["muted"], anchor="lm")
        if i < len(stages) - 1:
            arrow(d, [(1730, y + 72), (1730, y + 110)], C["green"], w=3, dashed=True)
    lock(d, 2210, 1015)
    label(d, 2160, 1070, "no server-side\nSAC update", 16, fill=C["red"], anchor="mm")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W * S, H * S), C["bg"])
    d = ImageDraw.Draw(img)

    rounded(d, (40, 40, W - 40, H - 40), C["panel"], "#86B894", r=28, w=3)
    label(d, 90, 105, "Core local SP-RGNN-CSAC agent module - cleaned network style", 36, bold=True)
    label(d, 90, 152, "This is a style test: network internals are shown as aligned layer blocks and typed branches, not dense neuron plots.", 21, fill=C["muted"])

    draw_relation_encoder(d)
    draw_actor(d)
    draw_critics(d)
    draw_learning(d)

    # Clear, minimal internal coupling lines.
    arrow(d, [(725, 790), (840, 448)], C["blue"], w=4)
    arrow(d, [(725, 790), (840, 900)], C["green"], w=4, dashed=True)
    arrow(d, [(1500, 1045), (1580, 630)], C["green"], w=4, dashed=True)
    arrow(d, [(1815, 1018), (2040, 1018), (2040, 665), (1500, 448)], C["green"], w=4, dashed=True)

    img = img.resize((W, H), Image.Resampling.LANCZOS)
    img.save(PNG)
    print(PNG)


if __name__ == "__main__":
    main()
