from __future__ import annotations

from pathlib import Path
import base64
import math
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent / "modules_v7"
PNG_PATH = OUT_DIR / "module_b_shared_private_relational_encoder.png"
SVG_PATH = OUT_DIR / "module_b_shared_private_relational_encoder_exact.svg"

S = 2
W, H = 2450, 1420

C = {
    "bg": "#F7F9FC",
    "panel": "#ECF8F0",
    "ink": "#111827",
    "muted": "#64748B",
    "border": "#B8C4D3",
    "white": "#FFFFFF",
    "blue": "#1D4ED8",
    "green": "#047857",
    "orange": "#D97706",
    "red": "#DC2626",
    "line": "#94A3B8",
    "ev": "#DBEAFE",
    "bes": "#DCFCE7",
    "pv": "#FEF3C7",
    "es": "#FCE7F3",
    "tr": "#EDE9FE",
    "cs": "#FFFFFF",
    "shared": "#DCFCE7",
    "private": "#FFF7ED",
    "norm": "#F8FAFC",
    "gate": "#E8F6EE",
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


def label(d: ImageDraw.ImageDraw, x: float, y: float, text: str, size: int, *, bold=False, fill=None, anchor="la") -> None:
    d.text((sc(x), sc(y)), text, font=ft(size, bold), fill=fill or C["ink"], anchor=anchor)


def centered(d: ImageDraw.ImageDraw, b: Sequence[float], text: str, size: int, *, bold=False, fill=None) -> None:
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


def arrow(d: ImageDraw.ImageDraw, pts: Sequence[tuple[float, float]], color: str, *, w: int = 3, head=True) -> None:
    points = [(sc(x), sc(y)) for x, y in pts]
    d.line(points, fill=color, width=sc(w), joint="curve")
    if head and len(points) >= 2:
        arrow_head(d, points[-2], points[-1], color, sc(w))


def arrow_head(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, w: int) -> None:
    ax, ay = a
    bx, by = b
    angle = math.atan2(by - ay, bx - ax)
    size = max(14 * S, w * 3)
    d.polygon(
        [
            (bx, by),
            (bx - size * math.cos(angle - math.pi / 7), by - size * math.sin(angle - math.pi / 7)),
            (bx - size * math.cos(angle + math.pi / 7), by - size * math.sin(angle + math.pi / 7)),
        ],
        fill=color,
    )


def block(d: ImageDraw.ImageDraw, b: Sequence[float], text: str, fill: str, outline: str, size: int = 16) -> None:
    rounded(d, b, fill, outline, r=11, w=2)
    centered(d, b, text, size, bold=True)


def node(d: ImageDraw.ImageDraw, x: float, y: float, text: str, fill: str, r: int = 30) -> None:
    d.ellipse(sb((x - r, y - r, x + r, y + r)), fill=fill, outline="#64748B", width=sc(2))
    centered(d, (x - r, y - r, x + r, y + r), text, 15, bold=True)


def lock_icon(d: ImageDraw.ImageDraw, x: float, y: float) -> None:
    rounded(d, (x - 13, y, x + 13, y + 24), "#FEE2E2", C["red"], r=5, w=2)
    d.arc(sb((x - 10, y - 19, x + 10, y + 9)), 180, 360, fill=C["red"], width=sc(3))


def local_graph_subfigure(d: ImageDraw.ImageDraw, b: Sequence[float]) -> tuple[float, float]:
    rounded(d, b, C["white"], "#8BB99A", r=18, w=2)
    x1, y1, x2, y2 = b
    label(d, x1 + 24, y1 + 38, "Input local graph", 22, bold=True)
    label(d, x1 + 24, y1 + 66, "G_i,t^reg", 17, fill=C["muted"])
    cx, cy = (x1 + x2) / 2, y1 + 215
    nodes = [
        ("EV", cx - 95, cy - 70, C["ev"]),
        ("BES", cx + 95, cy - 70, C["bes"]),
        ("PV", cx - 95, cy + 70, C["pv"]),
        ("ES", cx + 95, cy + 70, C["es"]),
        ("TR\nfeedback", cx, cy + 125, C["tr"]),
    ]
    for _, x, y, _ in nodes:
        arrow(d, [(x, y), (cx, cy)], C["line"], w=2, head=False)
        arrow(d, [(cx, cy), (x, y)], C["line"], w=2, head=False)
    node(d, cx, cy, "CS", C["cs"], 36)
    for text, x, y, fill in nodes:
        node(d, x, y, text, fill, 29)
    return x2, cy


def type_encoder_subfigure(d: ImageDraw.ImageDraw, b: Sequence[float]) -> tuple[float, float]:
    rounded(d, b, C["white"], "#8BB99A", r=18, w=2)
    x1, y1, x2, y2 = b
    label(d, x1 + 24, y1 + 38, "Type-specific encoders", 22, bold=True)
    encoders = [
        ("EV enc", C["ev"]),
        ("BES enc", C["bes"]),
        ("PV enc", C["pv"]),
        ("CS enc", C["cs"]),
        ("ES enc", C["es"]),
        ("TR enc", C["tr"]),
    ]
    for i, (name, fill) in enumerate(encoders):
        y = y1 + 90 + i * 52
        block(d, (x1 + 36, y, x1 + 160, y + 36), name, fill, "#8391A3", size=13)
        arrow(d, [(x1 + 160, y + 18), (x1 + 245, y2 - 130)], C["blue"], w=2, head=False)
    block(d, (x1 + 245, y2 - 178, x1 + 420, y2 - 82), "typed node\nembeddings", C["gate"], "#4E9F69", size=16)
    return x2, y2 - 130


def relation_transform_subfigure(d: ImageDraw.ImageDraw, b: Sequence[float]) -> tuple[float, float]:
    rounded(d, b, C["white"], "#8BB99A", r=18, w=2)
    x1, y1, x2, y2 = b
    label(d, x1 + 24, y1 + 38, "Shared-private relation transform", 22, bold=True)
    label(d, x1 + 24, y1 + 66, "one lane per physical relation", 16, fill=C["muted"])
    rels = ["EV-CS", "BES-CS", "PV-CS", "ES-CS", "TR-CS"]
    lane_y0 = y1 + 115
    for i, rel in enumerate(rels):
        y = lane_y0 + i * 94
        if i > 0:
            d.line([(sc(x1 + 25), sc(y - 18)), (sc(x2 - 25), sc(y - 18))], fill="#E2E8F0", width=sc(1))
        block(d, (x1 + 36, y, x1 + 145, y + 46), rel, C["norm"], "#94A3B8", size=14)
        block(d, (x1 + 205, y - 8, x1 + 385, y + 54), "shared\ntransform", C["shared"], "#4E9F69", size=15)
        label(d, x1 + 295, y + 75, "federatable", 13, fill=C["green"], anchor="mm")
        block(d, (x1 + 455, y - 8, x1 + 635, y + 54), "private\nadapter", C["private"], C["orange"], size=15)
        lock_icon(d, x1 + 655, y + 11)
        label(d, x1 + 545, y + 75, "local", 13, fill=C["orange"], anchor="mm")
        block(d, (x1 + 735, y - 4, x1 + 890, y + 50), "relation\nmessage", "#EEF6FF", C["blue"], size=14)
        arrow(d, [(x1 + 145, y + 23), (x1 + 205, y + 23)], C["blue"], w=2)
        arrow(d, [(x1 + 385, y + 23), (x1 + 455, y + 23)], C["blue"], w=2)
        arrow(d, [(x1 + 635, y + 23), (x1 + 735, y + 23)], C["blue"], w=2)
    return x2, (y1 + y2) / 2


def gate_subfigure(d: ImageDraw.ImageDraw, b: Sequence[float]) -> tuple[float, float]:
    rounded(d, b, C["white"], "#8BB99A", r=18, w=2)
    x1, y1, x2, y2 = b
    label(d, x1 + 24, y1 + 38, "Relation normalization + gate", 22, bold=True)
    label(d, x1 + 24, y1 + 66, "balance dense and sparse channels", 16, fill=C["muted"])
    rels = ["EV-CS", "BES-CS", "PV-CS", "ES-CS", "TR-CS"]
    fills = [C["ev"], C["bes"], C["pv"], C["es"], C["tr"]]
    knob_offsets = [115, 55, 82, 35, 96]
    for i, (rel, fill, knob) in enumerate(zip(rels, fills, knob_offsets)):
        y = y1 + 122 + i * 70
        block(d, (x1 + 36, y, x1 + 125, y + 38), "Norm", C["norm"], "#94A3B8", size=13)
        block(d, (x1 + 155, y, x1 + 295, y + 38), rel, fill, "#8391A3", size=13)
        d.line([(sc(x1 + 335), sc(y + 19)), (sc(x1 + 505), sc(y + 19))], fill="#A8B3C2", width=sc(5))
        d.ellipse(sb((x1 + 335 + knob - 10, y + 8, x1 + 335 + knob + 10, y + 30)), fill=C["green"], outline=C["ink"], width=sc(1))
        arrow(d, [(x1 + 505, y + 19), (x1 + 610, y1 + 300)], C["green"], w=2, head=False)
    block(d, (x1 + 610, y1 + 232, x1 + 745, y1 + 368), "learnable\nrelation\ngate", C["gate"], "#4E9F69", size=16)
    block(d, (x1 + 805, y1 + 250, x1 + 965, y1 + 350), "local graph\nrepresentation", "#EEF6FF", C["blue"], size=16)
    label(d, x1 + 885, y1 + 380, "z_i,t", 18, fill=C["blue"], anchor="mm")
    arrow(d, [(x1 + 745, y1 + 300), (x1 + 805, y1 + 300)], C["green"], w=3)
    return x2, y1 + 300


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W * S, H * S), C["bg"])
    d = ImageDraw.Draw(img)
    rounded(d, (40, 40, W - 40, H - 40), C["panel"], C["border"], r=30, w=3)
    label(d, 90, 110, "(b) Shared-private relational graph encoder", 38, bold=True)
    label(
        d,
        90,
        158,
        "Relation-specific messages are transformed, normalized and gated before producing the local graph representation.",
        22,
        fill=C["muted"],
    )

    p1 = local_graph_subfigure(d, (95, 245, 455, 690))
    p2 = type_encoder_subfigure(d, (540, 245, 1010, 690))
    p3 = relation_transform_subfigure(d, (95, 775, 1060, 1320))
    p4 = gate_subfigure(d, (1160, 775, 2288, 1320))

    # Clean, high-level flow lines between subfigures only.
    arrow(d, [(455, 468), (540, 468)], C["blue"], w=4)
    arrow(d, [(775, 690), (775, 775)], C["blue"], w=4)
    arrow(d, [(1060, 1048), (1160, 1048)], C["blue"], w=4)

    # Emphasis note.
    rounded(d, (1120, 245, 2288, 690), C["white"], "#D7E6D9", r=18, w=1)
    label(d, 1160, 300, "Visual reading guide", 24, bold=True)
    label(d, 1160, 350, "1. A local graph enters the encoder.", 19, fill=C["muted"])
    label(d, 1160, 398, "2. Node types are embedded separately.", 19, fill=C["muted"])
    label(d, 1160, 446, "3. Each physical relation owns a message lane.", 19, fill=C["muted"])
    label(d, 1160, 494, "4. Shared transforms are transferable; private adapters stay local.", 19, fill=C["muted"])
    label(d, 1160, 542, "5. Normalized relation messages are fused by a learnable gate.", 19, fill=C["muted"])

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
