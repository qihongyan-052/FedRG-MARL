from __future__ import annotations

from pathlib import Path
import math
import textwrap
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
PNG_PATH = OUT_DIR / "sp_rgnn_csac_system_architecture.png"
SVG_PATH = OUT_DIR / "sp_rgnn_csac_system_architecture.svg"

W, H = 2600, 1600
S = 2

COLORS = {
    "bg": "#F7F9FB",
    "ink": "#1F2937",
    "muted": "#667085",
    "line": "#475467",
    "online": "#2563EB",
    "learn": "#059669",
    "fed": "#7C3AED",
    "safety": "#EA580C",
    "agent": "#EAF2FF",
    "actor": "#EEF7F1",
    "exec": "#FFF3E7",
    "learn_bg": "#EAF8F2",
    "fed_bg": "#F3EAFF",
    "private": "#FEECEC",
    "white": "#FFFFFF",
    "border": "#CBD5E1",
}

FONT_MAIN = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_SC = "C:/Windows/Fonts/msyh.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_MAIN
    if not Path(path).exists():
        path = FONT_SC
    return ImageFont.truetype(path, size * S)


def scaled_box(box: Sequence[float]) -> tuple[int, int, int, int]:
    return tuple(int(v * S) for v in box)  # type: ignore[return-value]


def create_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W * S, H * S), COLORS["bg"])
    return img, ImageDraw.Draw(img)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrapped_lines(text: str, width_chars: int) -> list[str]:
    lines: list[str] = []
    for part in text.split("\n"):
        if not part:
            lines.append("")
        else:
            lines.extend(textwrap.wrap(part, width=width_chars, break_long_words=False))
    return lines


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    text: str,
    size: int,
    *,
    bold: bool = False,
    fill: str = COLORS["ink"],
    width_chars: int | None = None,
    line_gap: int = 6,
) -> None:
    fnt = font(size, bold)
    lines = wrapped_lines(text, width_chars) if width_chars else text.split("\n")
    heights = [text_size(draw, line, fnt)[1] for line in lines]
    total_h = sum(heights) + max(0, len(lines) - 1) * line_gap * S
    x1, y1, x2, y2 = scaled_box(box)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, h in zip(lines, heights):
        tw, _ = text_size(draw, line, fnt)
        draw.text((x1 + (x2 - x1 - tw) / 2, y), line, font=fnt, fill=fill)
        y += h + line_gap * S


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    size: int = 24,
    *,
    bold: bool = False,
    fill: str = COLORS["ink"],
    anchor: str = "la",
) -> None:
    draw.text((xy[0] * S, xy[1] * S), text, font=font(size, bold), fill=fill, anchor=anchor)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    *,
    fill: str,
    outline: str = COLORS["border"],
    radius: int = 26,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(
        scaled_box(box),
        radius=radius * S,
        fill=fill,
        outline=outline,
        width=width * S,
    )


def rect_box(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    *,
    fill: str,
    outline: str = COLORS["border"],
    width: int = 3,
) -> None:
    draw.rectangle(scaled_box(box), fill=fill, outline=outline, width=width * S)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    pts: Sequence[tuple[float, float]],
    *,
    color: str,
    width: int = 5,
    dashed: bool = False,
    arrow: bool = True,
) -> None:
    points = [(int(x * S), int(y * S)) for x, y in pts]
    if dashed:
        for a, b in zip(points, points[1:]):
            draw_dashed_segment(draw, a, b, color=color, width=width * S)
    else:
        draw.line(points, fill=color, width=width * S, joint="curve")
    if arrow and len(points) >= 2:
        draw_arrow_head(draw, points[-2], points[-1], color=color, width=width * S)


def draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    a: tuple[int, int],
    b: tuple[int, int],
    *,
    color: str,
    width: int,
    dash: int = 24,
    gap: int = 16,
) -> None:
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay)
    if length == 0:
        return
    ux, uy = (bx - ax) / length, (by - ay) / length
    dist = 0.0
    while dist < length:
        end = min(dist + dash * S, length)
        p1 = (int(ax + ux * dist), int(ay + uy * dist))
        p2 = (int(ax + ux * end), int(ay + uy * end))
        draw.line([p1, p2], fill=color, width=width)
        dist += (dash + gap) * S


def draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    a: tuple[int, int],
    b: tuple[int, int],
    *,
    color: str,
    width: int,
) -> None:
    ax, ay = a
    bx, by = b
    angle = math.atan2(by - ay, bx - ax)
    size = max(18 * S, width * 3)
    left = (
        bx - size * math.cos(angle - math.pi / 7),
        by - size * math.sin(angle - math.pi / 7),
    )
    right = (
        bx - size * math.cos(angle + math.pi / 7),
        by - size * math.sin(angle + math.pi / 7),
    )
    draw.polygon([(bx, by), left, right], fill=color)


def draw_sub_box(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    title: str,
    subtitle: str = "",
    *,
    fill: str = COLORS["white"],
    outline: str = COLORS["border"],
    title_size: int = 23,
    sub_size: int = 19,
    width_chars: int = 24,
) -> None:
    rounded_box(draw, box, fill=fill, outline=outline, radius=16, width=2)
    x1, y1, x2, y2 = box
    draw_centered_text(draw, (x1 + 10, y1 + 10, x2 - 10, y1 + 46), title, title_size, bold=True)
    if subtitle:
        draw_centered_text(
            draw,
            (x1 + 14, y1 + 52, x2 - 14, y2 - 10),
            subtitle,
            sub_size,
            fill=COLORS["muted"],
            width_chars=width_chars,
            line_gap=4,
        )


def draw_local_graph(draw: ImageDraw.ImageDraw, cx: float, cy: float) -> None:
    node_specs = [
        ("EV", cx - 110, cy - 90, "#DBEAFE"),
        ("BES", cx + 110, cy - 90, "#DCFCE7"),
        ("PV", cx - 110, cy + 78, "#FEF3C7"),
        ("ES", cx + 110, cy + 78, "#FCE7F3"),
        ("TR", cx, cy + 132, "#EDE9FE"),
    ]
    cs = (cx, cy, "#FFFFFF")
    for label, x, y, color in node_specs:
        draw_arrow(draw, [(x, y), (cx, cy)], color="#94A3B8", width=3, arrow=False)
    for label, x, y, color in node_specs:
        draw.ellipse(scaled_box((x - 34, y - 28, x + 34, y + 28)), fill=color, outline="#94A3B8", width=2 * S)
        draw_centered_text(draw, (x - 34, y - 24, x + 34, y + 24), label, 20, bold=True)
    draw.ellipse(scaled_box((cs[0] - 42, cs[1] - 34, cs[0] + 42, cs[1] + 34)), fill=cs[2], outline="#64748B", width=3 * S)
    draw_centered_text(draw, (cs[0] - 42, cs[1] - 28, cs[0] + 42, cs[1] + 28), "CS", 22, bold=True)
    draw_label(draw, (cx, cy + 190), "Local graph only", 19, fill=COLORS["muted"], anchor="mm")


def draw_lock(draw: ImageDraw.ImageDraw, x: float, y: float, scale: float = 1.0) -> None:
    sx = S * scale
    draw.rounded_rectangle(
        (int((x - 18) * S), int((y - 2) * S), int((x + 18) * S), int((y + 28) * S)),
        radius=int(5 * sx),
        fill=COLORS["private"],
        outline="#DC2626",
        width=max(1, int(2 * sx)),
    )
    draw.arc(
        (int((x - 13) * S), int((y - 27) * S), int((x + 13) * S), int((y + 13) * S)),
        180,
        360,
        fill="#DC2626",
        width=max(1, int(4 * sx)),
    )


def draw_figure() -> Image.Image:
    img, draw = create_canvas()

    # Header
    draw_label(draw, (80, 70), "SP-RGNN-CSAC System Architecture", 42, bold=True, anchor="la")
    draw_label(
        draw,
        (80, 120),
        "Online execution, local constrained learning, and periodic personalized relation federation",
        25,
        fill=COLORS["muted"],
        anchor="la",
    )

    # Main containers
    obs_box = (70, 205, 510, 660)
    actor_box = (600, 205, 1185, 660)
    exec_box = (1280, 205, 2050, 660)
    agents_box = (70, 760, 710, 1335)
    learn_box = (820, 770, 1570, 1335)
    fed_box = (1710, 785, 2470, 1335)

    rounded_box(draw, obs_box, fill=COLORS["agent"], outline="#93B8E8")
    rounded_box(draw, actor_box, fill=COLORS["actor"], outline="#89C99C")
    rounded_box(draw, exec_box, fill=COLORS["exec"], outline="#F0A35C")
    rounded_box(draw, agents_box, fill=COLORS["agent"], outline="#93B8E8")
    rounded_box(draw, learn_box, fill=COLORS["learn_bg"], outline="#81C7A8")
    rounded_box(draw, fed_box, fill=COLORS["fed_bg"], outline="#B49AEE")

    draw_label(draw, (95, 245), "1  Local Relational Observation", 28, bold=True)
    draw_local_graph(draw, 290, 415)
    draw_label(draw, (95, 615), "EV/BES/PV/ES/TR feedback ↔ CS relation channels", 22, fill=COLORS["muted"])

    draw_label(draw, (625, 245), "2  Shared-Private Relational Actor", 28, bold=True)
    draw_sub_box(draw, (640, 295, 820, 405), "Type-specific\nembedding")
    draw_sub_box(
        draw,
        (870, 295, 1095, 405),
        "Shared-private\nrelation block",
        "shared transform\n+ private adapter",
        sub_size=18,
    )
    draw_sub_box(
        draw,
        (640, 475, 820, 585),
        "Learnable\nrelation-gated\nfusion",
        "",
        title_size=22,
    )
    draw_sub_box(draw, (870, 475, 1095, 585), "Actor network", "continuous actions", sub_size=20)
    draw_arrow(draw, [(820, 350), (870, 350)], color=COLORS["online"], width=4)
    draw_arrow(draw, [(980, 405), (980, 475)], color=COLORS["online"], width=4)
    draw_arrow(draw, [(820, 530), (870, 530)], color=COLORS["online"], width=4)
    draw_label(draw, (1130, 531), "actions", 22, fill=COLORS["online"], anchor="lm")

    draw_label(draw, (1305, 245), "3  Safe Execution and Regional Feedback", 28, bold=True)
    draw_sub_box(draw, (1315, 300, 1515, 410), "Device bounds", "EV/BES executable energy", width_chars=20)
    draw_sub_box(draw, (1570, 300, 1770, 410), "CS capacity-safe\nadjustment", "local station limit", width_chars=20)
    draw_sub_box(draw, (1825, 300, 2015, 410), "Secure\naggregation", "aggregate exchange only", width_chars=20)
    draw_sub_box(draw, (1465, 500, 1695, 610), "TR coordination", "shared transformer constraint", width_chars=24)
    draw_sub_box(draw, (1745, 500, 1995, 610), "Return signals", "reward / cost / TR feedback", width_chars=26)
    draw_arrow(draw, [(1515, 355), (1570, 355)], color=COLORS["safety"], width=4)
    draw_arrow(draw, [(1770, 355), (1825, 355)], color=COLORS["safety"], width=4)
    draw_arrow(draw, [(1920, 410), (1580, 500)], color=COLORS["safety"], width=4)
    draw_arrow(draw, [(1695, 555), (1745, 555)], color=COLORS["safety"], width=4)

    # Main online flow
    draw_arrow(draw, [(510, 430), (600, 430)], color=COLORS["online"], width=6)
    draw_arrow(draw, [(1185, 430), (1280, 430)], color=COLORS["online"], width=6)
    draw_arrow(
        draw,
        [(1870, 610), (1870, 705), (300, 705), (300, 660)],
        color=COLORS["online"],
        width=5,
    )
    draw_label(draw, (1188, 705), "TR feedback enters next local graph", 23, fill=COLORS["online"], anchor="mm")

    # Local agents
    draw_label(draw, (95, 800), "4  Local Park Agents and Privacy Boundary", 28, bold=True)
    agent_rows = [
        ("Residential park", 860),
        ("Office park", 1010),
        ("Commercial park", 1160),
    ]
    for title, y in agent_rows:
        draw_sub_box(draw, (105, y, 365, y + 105), title, "local agent", width_chars=18)
        draw_sub_box(draw, (390, y, 675, y + 105), "Private modules", "adapters, critics, replay buffer,\nraw trajectories", width_chars=28)
        draw_lock(draw, 640, y + 25, 0.85)
    draw_label(draw, (105, 1288), "Only selected shared relation parameters leave this boundary.", 22, fill=COLORS["muted"])

    # Learning
    draw_label(draw, (845, 810), "5  Local Constrained Actor-Critic Learning", 28, bold=True)
    draw_sub_box(draw, (860, 870, 1060, 990), "Transition", "observation, action,\nreward, cost, next observation", width_chars=23)
    draw_sub_box(draw, (1130, 870, 1340, 990), "Local replay\nbuffer")
    draw_sub_box(draw, (1400, 870, 1540, 990), "Reward\ncritics")
    draw_sub_box(draw, (1400, 1045, 1540, 1165), "Cost\ncritics")
    draw_sub_box(draw, (1070, 1105, 1320, 1245), "Constrained SAC\nupdate", "local policy and critic update", width_chars=28)
    draw_arrow(draw, [(1060, 930), (1130, 930)], color=COLORS["learn"], width=4, dashed=True)
    draw_arrow(draw, [(1340, 930), (1400, 930)], color=COLORS["learn"], width=4, dashed=True)
    draw_arrow(draw, [(1340, 930), (1400, 1105)], color=COLORS["learn"], width=4, dashed=True)
    draw_arrow(draw, [(1400, 930), (1320, 1150)], color=COLORS["learn"], width=4, dashed=True)
    draw_arrow(draw, [(1400, 1105), (1320, 1185)], color=COLORS["learn"], width=4, dashed=True)
    draw_arrow(
        draw,
        [(1070, 1175), (750, 1175), (750, 660)],
        color=COLORS["learn"],
        width=5,
        dashed=True,
    )
    draw_label(draw, (760, 1145), "update local actor", 22, fill=COLORS["learn"], anchor="lm")

    # Execution returns to learning and agents
    draw_arrow(draw, [(1900, 610), (1900, 725), (930, 725), (930, 870)], color=COLORS["learn"], width=5, dashed=True)
    draw_label(draw, (1450, 725), "reward / cost form local transition", 22, fill=COLORS["learn"], anchor="mm")
    draw_arrow(draw, [(1800, 610), (1800, 740), (390, 740), (390, 760)], color=COLORS["line"], width=4)

    # Federation
    draw_label(draw, (1735, 825), "6  Personalized Relation Federation", 28, bold=True)
    draw_sub_box(draw, (1750, 895, 1990, 1025), "Selected shared\nrelation parameters", "actor-side only", width_chars=22)
    draw_sub_box(draw, (2055, 895, 2295, 1025), "Federation server", "personalized aggregation", width_chars=23)
    draw_sub_box(draw, (1905, 1095, 2225, 1225), "Personalized shared\nparameters returned", "different candidate for each park", width_chars=30)
    draw_arrow(draw, [(1990, 960), (2055, 960)], color=COLORS["fed"], width=5, dashed=True)
    draw_arrow(draw, [(2175, 1025), (2080, 1095)], color=COLORS["fed"], width=5, dashed=True)
    draw_arrow(draw, [(1905, 1160), (1180, 1160), (1030, 660)], color=COLORS["fed"], width=5, dashed=True)
    draw_arrow(draw, [(675, 1065), (1750, 960)], color=COLORS["fed"], width=5, dashed=True)
    draw_label(draw, (2030, 1282), "Not uploaded: raw states/actions, rewards/costs,\ncritics, replay buffers, private adapters", 23, fill="#6D28D9", anchor="mm")

    # Legend
    legend = (2080, 205, 2470, 455)
    rounded_box(draw, legend, fill=COLORS["white"], outline=COLORS["border"], radius=18, width=2)
    draw_label(draw, (2105, 245), "Arrow legend", 25, bold=True)
    draw_arrow(draw, [(2110, 300), (2210, 300)], color=COLORS["online"], width=5)
    draw_label(draw, (2230, 307), "online execution", 21, fill=COLORS["muted"], anchor="lm")
    draw_arrow(draw, [(2110, 350), (2210, 350)], color=COLORS["learn"], width=5, dashed=True)
    draw_label(draw, (2230, 357), "local learning", 21, fill=COLORS["muted"], anchor="lm")
    draw_arrow(draw, [(2110, 400), (2210, 400)], color=COLORS["fed"], width=5, dashed=True)
    draw_label(draw, (2230, 407), "periodic federation", 21, fill=COLORS["muted"], anchor="lm")

    return img


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(x: float, y: float, text: str, size: int, *, weight: str = "400", fill: str = COLORS["ink"], anchor: str = "start") -> str:
    return f'<text x="{x}" y="{y}" font-family="Segoe UI, Microsoft YaHei, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(text)}</text>'


def svg_rect(box: Sequence[float], fill: str, stroke: str, radius: int = 18, width: int = 2) -> str:
    x1, y1, x2, y2 = box
    return f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'


def svg_arrow(points: Sequence[tuple[float, float]], color: str, *, dashed: bool = False, width: int = 4, marker: str = "arrow") -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="14 10"' if dashed else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{dash} marker-end="url(#{marker})"/>'


def write_svg() -> None:
    # The SVG is a compact editable companion. The PNG is the visually verified primary export.
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
        '<defs>',
        f'<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth"><path d="M2,2 L10,6 L2,10 Z" fill="{COLORS["line"]}"/></marker>',
        f'<marker id="blueArrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth"><path d="M2,2 L10,6 L2,10 Z" fill="{COLORS["online"]}"/></marker>',
        f'<marker id="greenArrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth"><path d="M2,2 L10,6 L2,10 Z" fill="{COLORS["learn"]}"/></marker>',
        f'<marker id="purpleArrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth"><path d="M2,2 L10,6 L2,10 Z" fill="{COLORS["fed"]}"/></marker>',
        "</defs>",
        svg_text(80, 70, "SP-RGNN-CSAC System Architecture", 42, weight="700"),
        svg_text(80, 120, "Online execution, local constrained learning, and periodic personalized relation federation", 25, fill=COLORS["muted"]),
    ]
    for box, fill, stroke, title in [
        ((70, 205, 510, 660), COLORS["agent"], "#93B8E8", "1  Local Relational Observation"),
        ((600, 205, 1185, 660), COLORS["actor"], "#89C99C", "2  Shared-Private Relational Actor"),
        ((1280, 205, 2050, 660), COLORS["exec"], "#F0A35C", "3  Safe Execution and Regional Feedback"),
        ((70, 760, 710, 1335), COLORS["agent"], "#93B8E8", "4  Local Park Agents and Privacy Boundary"),
        ((820, 770, 1570, 1335), COLORS["learn_bg"], "#81C7A8", "5  Local Constrained Actor-Critic Learning"),
        ((1710, 785, 2470, 1335), COLORS["fed_bg"], "#B49AEE", "6  Personalized Relation Federation"),
    ]:
        parts.append(svg_rect(box, fill, stroke, radius=26, width=3))
        parts.append(svg_text(box[0] + 25, box[1] + 40, title, 28, weight="700"))
    # Minimal editable labels and arrows. Use PNG for detailed sublayout.
    labels = [
        (290, 430, "EV/BES/PV/ES/TR feedback ↔ CS", COLORS["ink"], "middle"),
        (895, 370, "type embedding → shared-private relation block", COLORS["ink"], "middle"),
        (895, 540, "learnable relation-gated fusion → actor actions", COLORS["ink"], "middle"),
        (1665, 370, "device bounds → CS capacity-safe adjustment → secure aggregation", COLORS["ink"], "middle"),
        (1730, 565, "TR coordination → reward / cost / TR feedback", COLORS["ink"], "middle"),
        (390, 1065, "three local park agents; private modules stay local", COLORS["ink"], "middle"),
        (1195, 1040, "transition → replay buffer → reward/cost critics → constrained SAC", COLORS["ink"], "middle"),
        (2090, 1055, "selected shared relation parameters only", COLORS["ink"], "middle"),
    ]
    for x, y, text, fill, anchor in labels:
        parts.append(svg_text(x, y, text, 25, fill=fill, anchor=anchor))
    parts.extend(
        [
            svg_arrow([(510, 430), (600, 430)], COLORS["online"], width=6, marker="blueArrow"),
            svg_arrow([(1185, 430), (1280, 430)], COLORS["online"], width=6, marker="blueArrow"),
            svg_arrow([(1870, 610), (1870, 705), (300, 705), (300, 660)], COLORS["online"], width=5, marker="blueArrow"),
            svg_arrow([(1900, 610), (1900, 725), (930, 725), (930, 870)], COLORS["learn"], dashed=True, width=5, marker="greenArrow"),
            svg_arrow([(1070, 1175), (750, 1175), (750, 660)], COLORS["learn"], dashed=True, width=5, marker="greenArrow"),
            svg_arrow([(675, 1065), (1750, 960)], COLORS["fed"], dashed=True, width=5, marker="purpleArrow"),
            svg_arrow([(1905, 1160), (1180, 1160), (1030, 660)], COLORS["fed"], dashed=True, width=5, marker="purpleArrow"),
        ]
    )
    parts.append("</svg>")
    SVG_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = draw_figure()
    img = img.resize((W, H), Image.Resampling.LANCZOS)
    img.save(PNG_PATH)
    write_svg()
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    main()

