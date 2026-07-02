from __future__ import annotations

from pathlib import Path
import base64
import math
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent / "modules_v7"
PNG_PATH = OUT_DIR / "module_a_local_relational_observation_n_parks.png"
SVG_PATH = OUT_DIR / "module_a_local_relational_observation_n_parks_exact.svg"

S = 2
W, H = 2100, 1180

C = {
    "bg": "#F7F9FC",
    "ink": "#111827",
    "muted": "#64748B",
    "border": "#B8C4D3",
    "white": "#FFFFFF",
    "panel": "#EAF2FF",
    "park": "#FFFFFF",
    "park_border": "#8FB4E8",
    "blue": "#1D4ED8",
    "line": "#94A3B8",
    "private": "#FEE2E2",
    "red": "#DC2626",
    "ev": "#DBEAFE",
    "bes": "#DCFCE7",
    "pv": "#FEF3C7",
    "es": "#FCE7F3",
    "tr": "#EDE9FE",
    "cs": "#FFFFFF",
    "out": "#EEF6FF",
}

FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_CN = "C:/Windows/Fonts/msyh.ttc"


def sc(v: float) -> int:
    return int(round(v * S))


def sb(b: Sequence[float]) -> tuple[int, int, int, int]:
    return tuple(sc(x) for x in b)  # type: ignore[return-value]


def ft(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT
    if not Path(path).exists():
        path = FONT_CN
    return ImageFont.truetype(path, size * S)


def rounded(d: ImageDraw.ImageDraw, b: Sequence[float], fill: str, outline: str, r: int = 22, w: int = 2) -> None:
    d.rounded_rectangle(sb(b), radius=sc(r), fill=fill, outline=outline, width=sc(w))


def label(
    d: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    size: int,
    *,
    bold: bool = False,
    fill: str | None = None,
    anchor: str = "la",
) -> None:
    d.text((sc(x), sc(y)), text, font=ft(size, bold), fill=fill or C["ink"], anchor=anchor)


def centered(
    d: ImageDraw.ImageDraw,
    b: Sequence[float],
    text: str,
    size: int,
    *,
    bold: bool = False,
    fill: str | None = None,
) -> None:
    font = ft(size, bold)
    lines = text.split("\n")
    boxes = [d.textbbox((0, 0), line, font=font) for line in lines]
    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    gap = 5 * S
    total_h = sum(heights) + gap * max(0, len(lines) - 1)
    x1, y1, x2, y2 = sb(b)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, width, height in zip(lines, widths, heights):
        d.text((x1 + (x2 - x1 - width) / 2, y), line, font=font, fill=fill or C["ink"])
        y += height + gap


def arrow(
    d: ImageDraw.ImageDraw,
    pts: Sequence[tuple[float, float]],
    color: str,
    *,
    w: int = 3,
    dashed: bool = False,
    head: bool = True,
) -> None:
    points = [(sc(x), sc(y)) for x, y in pts]
    if dashed:
        for a, b in zip(points, points[1:]):
            dashed_line(d, a, b, color, sc(w))
    else:
        d.line(points, fill=color, width=sc(w), joint="curve")
    if head and len(points) >= 2:
        arrow_head(d, points[-2], points[-1], color, sc(w))


def dashed_line(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, w: int) -> None:
    ax, ay = a
    bx, by = b
    dist = math.hypot(bx - ax, by - ay)
    if dist <= 0:
        return
    ux, uy = (bx - ax) / dist, (by - ay) / dist
    t = 0.0
    while t < dist:
        t2 = min(t + 20 * S, dist)
        d.line([(int(ax + ux * t), int(ay + uy * t)), (int(ax + ux * t2), int(ay + uy * t2))], fill=color, width=w)
        t += 34 * S


def arrow_head(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, w: int) -> None:
    ax, ay = a
    bx, by = b
    angle = math.atan2(by - ay, bx - ax)
    size = max(15 * S, w * 3)
    d.polygon(
        [
            (bx, by),
            (bx - size * math.cos(angle - math.pi / 7), by - size * math.sin(angle - math.pi / 7)),
            (bx - size * math.cos(angle + math.pi / 7), by - size * math.sin(angle + math.pi / 7)),
        ],
        fill=color,
    )


def node(d: ImageDraw.ImageDraw, x: float, y: float, text: str, fill: str, r: int = 34) -> None:
    d.ellipse(sb((x - r, y - r, x + r, y + r)), fill=fill, outline="#64748B", width=sc(2))
    centered(d, (x - r, y - r, x + r, y + r), text, 17, bold=True)


def small_tag(d: ImageDraw.ImageDraw, b: Sequence[float], text: str, fill: str, outline: str) -> None:
    rounded(d, b, fill, outline, r=12, w=1)
    centered(d, b, text, 14, bold=True, fill=outline)


def lock_icon(d: ImageDraw.ImageDraw, x: float, y: float) -> None:
    rounded(d, (x - 15, y, x + 15, y + 28), C["private"], C["red"], r=6, w=2)
    d.arc(sb((x - 11, y - 23, x + 11, y + 11)), 180, 360, fill=C["red"], width=sc(3))


def local_graph_template(d: ImageDraw.ImageDraw, cx: float, cy: float, scale: float = 1.0) -> None:
    positions = [
        ("EV", cx - 112 * scale, cy - 86 * scale, C["ev"]),
        ("BES", cx + 112 * scale, cy - 86 * scale, C["bes"]),
        ("PV", cx - 112 * scale, cy + 80 * scale, C["pv"]),
        ("ES", cx + 112 * scale, cy + 80 * scale, C["es"]),
        ("TR", cx, cy + 150 * scale, C["tr"]),
    ]
    for _, x, y, _ in positions:
        arrow(d, [(x, y), (cx, cy)], C["line"], w=2, head=False)
        arrow(d, [(cx, cy), (x, y)], C["line"], w=2, head=False)
    node(d, cx, cy, "CS", C["cs"], int(42 * scale))
    for text, x, y, fill in positions:
        radius = int(34 * scale)
        label_text = "TR\nfeedback" if text == "TR" else text
        node(d, x, y, label_text, fill, radius)


def park_card(d: ImageDraw.ImageDraw, x: float, y: float, title: str, graph_label: str, type_hint: str) -> tuple[float, float]:
    rounded(d, (x, y, x + 455, y + 555), C["park"], C["park_border"], r=24, w=2)
    label(d, x + 28, y + 42, title, 25, bold=True)
    label(d, x + 28, y + 76, type_hint, 16, fill=C["muted"])
    rounded(d, (x + 22, y + 102, x + 433, y + 420), "#F8FBFF", "#D4E2F4", r=18, w=1)
    local_graph_template(d, x + 228, y + 258, 0.82)
    small_tag(d, (x + 54, y + 450, x + 185, y + 492), "private", C["private"], C["red"])
    lock_icon(d, x + 204, y + 456)
    small_tag(d, (x + 247, y + 450, x + 403, y + 492), graph_label, C["out"], C["blue"])
    return x + 325, y + 492


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W * S, H * S), C["bg"])
    d = ImageDraw.Draw(img)

    rounded(d, (40, 40, W - 40, H - 40), C["panel"], C["border"], r=30, w=3)
    label(d, 90, 110, "(a) Local relational observation construction for N parks", 38, bold=True)
    label(
        d,
        90,
        158,
        "Each park independently constructs a private local graph from local resources and regional TR feedback.",
        22,
        fill=C["muted"],
    )

    out1 = park_card(d, 120, 265, "Park 1", "G_1,t^reg", "heterogeneous park type")
    outi = park_card(d, 700, 265, "Park i", "G_i,t^reg", "representative local agent")
    outn = park_card(d, 1280, 265, "Park N", "G_N,t^reg", "heterogeneous park type")
    label(d, 1220, 545, "...", 42, bold=True, fill=C["muted"], anchor="mm")

    # Output collection port.
    rounded(d, (650, 900, 1450, 1015), C["white"], C["blue"], r=20, w=2)
    centered(d, (675, 918, 1425, 978), "Local relational graph observations", 24, bold=True, fill=C["blue"])
    centered(d, (675, 970, 1425, 1005), "{ G_i,t^reg } for i = 1,...,N", 19, fill=C["muted"])

    for ox, oy in [out1, outi, outn]:
        arrow(d, [(ox, oy), (ox, 850), (1050, 850), (1050, 900)], C["blue"], w=4)

    # Privacy note.
    rounded(d, (185, 1045, 1915, 1118), C["white"], "#D4E2F4", r=18, w=1)
    lock_icon(d, 230, 1066)
    label(
        d,
        270,
        1088,
        "Raw EV sessions, local states, actions, rewards, costs and trajectories are not exchanged across parks.",
        22,
        fill=C["muted"],
        anchor="lm",
    )

    img = img.resize((W, H), Image.Resampling.LANCZOS)
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
