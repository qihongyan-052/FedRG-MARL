from __future__ import annotations

from pathlib import Path
import math
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent / "modules_v5"
S = 2

C = {
    "bg": "#F7F9FC",
    "ink": "#111827",
    "muted": "#64748B",
    "border": "#B6C2D1",
    "white": "#FFFFFF",
    "blue": "#1D4ED8",
    "green": "#047857",
    "purple": "#7C3AED",
    "orange": "#D97706",
    "red": "#DC2626",
    "obs": "#EAF2FF",
    "agent": "#EAF8F0",
    "learn": "#EAF8F2",
    "exec": "#FFF3E6",
    "fed": "#F4ECFF",
    "ev": "#DBEAFE",
    "bes": "#DCFCE7",
    "pv": "#FEF3C7",
    "es": "#FCE7F3",
    "tr": "#EDE9FE",
    "private": "#FEE2E2",
    "adapter": "#FFF1DB",
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


def label(d: ImageDraw.ImageDraw, x: float, y: float, text: str, size: int, *, bold=False, fill=None, anchor="la") -> None:
    d.text((sc(x), sc(y)), text, font=ft(size, bold), fill=fill or C["ink"], anchor=anchor)


def centered(d: ImageDraw.ImageDraw, b: Sequence[float], text: str, size: int, *, bold=False, fill=None) -> None:
    f = ft(size, bold)
    lines = text.split("\n")
    boxes = [d.textbbox((0, 0), line, font=f) for line in lines]
    widths = [box[2] - box[0] for box in boxes]
    heights = [box[3] - box[1] for box in boxes]
    gap = 6 * S
    total_h = sum(heights) + gap * max(0, len(lines) - 1)
    x1, y1, x2, y2 = sb(b)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        d.text((x1 + (x2 - x1 - w) / 2, y), line, font=f, fill=fill or C["ink"])
        y += h + gap


def rounded(d: ImageDraw.ImageDraw, b: Sequence[float], fill: str, outline: str = C["border"], r: int = 22, w: int = 3) -> None:
    d.rounded_rectangle(sb(b), radius=sc(r), fill=fill, outline=outline, width=sc(w))


def rect(d: ImageDraw.ImageDraw, b: Sequence[float], fill: str, outline: str = C["border"], w: int = 2) -> None:
    d.rectangle(sb(b), fill=fill, outline=outline, width=sc(w))


def arrow(d: ImageDraw.ImageDraw, pts: Sequence[tuple[float, float]], color: str, *, w: int = 4, dashed=False, head=True) -> None:
    pp = [(sc(x), sc(y)) for x, y in pts]
    if dashed:
        for a, b in zip(pp, pp[1:]):
            dash(d, a, b, color, sc(w))
    else:
        d.line(pp, fill=color, width=sc(w), joint="curve")
    if head and len(pp) >= 2:
        head_arrow(d, pp[-2], pp[-1], color, sc(w))


def dash(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, w: int) -> None:
    ax, ay = a
    bx, by = b
    dist = math.hypot(bx - ax, by - ay)
    if dist <= 0:
        return
    ux, uy = (bx - ax) / dist, (by - ay) / dist
    t = 0.0
    while t < dist:
        t2 = min(t + 28 * S, dist)
        d.line([(int(ax + ux * t), int(ay + uy * t)), (int(ax + ux * t2), int(ay + uy * t2))], fill=color, width=w)
        t += 44 * S


def head_arrow(d: ImageDraw.ImageDraw, a: tuple[int, int], b: tuple[int, int], color: str, w: int) -> None:
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


def card(d: ImageDraw.ImageDraw, b: Sequence[float], text: str, fill: str, outline: str, size: int = 18) -> None:
    rounded(d, b, fill, outline, r=14, w=2)
    centered(d, b, text, size, bold=True)


def node(d: ImageDraw.ImageDraw, x: float, y: float, text: str, fill: str, r: int = 36) -> None:
    d.ellipse(sb((x - r, y - r, x + r, y + r)), fill=fill, outline="#718096", width=sc(2))
    centered(d, (x - r, y - r, x + r, y + r), text, 18, bold=True)


def lock(d: ImageDraw.ImageDraw, x: float, y: float) -> None:
    rounded(d, (x - 18, y, x + 18, y + 32), C["private"], C["red"], r=6, w=2)
    d.arc(sb((x - 13, y - 28, x + 13, y + 14)), 180, 360, fill=C["red"], width=sc(4))


def network(d: ImageDraw.ImageDraw, x: float, y: float, layers: Sequence[int], color: str, name: str) -> None:
    dx = 70
    for li, n in enumerate(layers):
        y0 = y - (n - 1) * 30 / 2
        for ni in range(n):
            cx, cy = x + li * dx, y0 + ni * 30
            if li < len(layers) - 1:
                nn = layers[li + 1]
                ny0 = y - (nn - 1) * 30 / 2
                for nj in range(nn):
                    d.line([(sc(cx + 9), sc(cy)), (sc(x + (li + 1) * dx - 9), sc(ny0 + nj * 30))], fill="#B5BFCC", width=sc(1))
            d.ellipse(sb((cx - 10, cy - 10, cx + 10, cy + 10)), fill=color, outline="#475467", width=sc(1))
    label(d, x + (len(layers) - 1) * dx / 2, y + 88, name, 17, bold=True, fill=C["muted"], anchor="mm")


def canvas(w: int, h: int, title: str, subtitle: str, fill: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (w * S, h * S), C["bg"])
    d = ImageDraw.Draw(img)
    rounded(d, (40, 40, w - 40, h - 40), fill, C["border"], r=28, w=3)
    label(d, 90, 105, title, 38, bold=True)
    label(d, 90, 152, subtitle, 22, fill=C["muted"])
    return img, d


def draw_local_observation() -> Image.Image:
    w, h = 1800, 1120
    img, d = canvas(w, h, "Module A. Local Relational Observation", "Each park builds its own graph; TR feedback is retained as a CS-related relation.", C["obs"])
    for i, (name, x) in enumerate([("Residential", 190), ("Office", 685), ("Commercial", 1180)]):
        rounded(d, (x, 245, x + 430, 860), C["white"], "#9EB9DB", r=22, w=2)
        label(d, x + 28, 292, f"{name} park", 24, bold=True)
        cx, cy = x + 215, 525
        nodes = [
            ("EV", cx - 120, cy - 85, C["ev"]),
            ("BES", cx + 120, cy - 85, C["bes"]),
            ("PV", cx - 120, cy + 85, C["pv"]),
            ("ES", cx + 120, cy + 85, C["es"]),
            ("TR", cx, cy + 155, C["tr"]),
        ]
        for _, nx, ny, _ in nodes:
            arrow(d, [(nx, ny), (cx, cy)], "#9AA8BB", w=3, head=False)
            arrow(d, [(cx, cy), (nx, ny)], "#9AA8BB", w=3, head=False)
        node(d, cx, cy, "CS", C["white"], 45)
        for t, nx, ny, c in nodes:
            node(d, nx, ny, t, c, 36)
        card(d, (x + 55, 780, x + 375, 835), "local graph observation", C["ev"], "#6B91C9", size=18)
    card(d, (300, 930, 1500, 1010), "Privacy boundary: raw EV sessions, local states, actions and costs do not enter other parks' graphs.", C["white"], "#9EB9DB", size=21)
    return img.resize((w, h), Image.Resampling.LANCZOS)


def draw_relational_encoder() -> Image.Image:
    w, h = 2200, 1300
    img, d = canvas(w, h, "Module B. Shared-Private Relational Graph Encoder", "Relation-specific messages are normalized and fused before policy/value networks.", C["agent"])
    card(d, (110, 300, 360, 470), "type-specific\nnode encoders", C["white"], "#7EBB8D", size=22)
    for i, fill in enumerate([C["ev"], C["bes"], C["pv"], C["white"], C["es"], C["tr"]]):
        rect(d, (145 + i * 34, 405 - i * 8, 215 + i * 34, 445 - i * 8), fill, "#8796AA", w=1)
    rounded(d, (465, 245, 1135, 875), C["white"], "#7EBB8D", r=22, w=2)
    label(d, 500, 300, "Shared-private relation transform", 25, bold=True)
    rels = ["EV-CS", "BES-CS", "PV-CS", "ES-CS", "TR-CS"]
    for i, rel in enumerate(rels):
        y = 365 + i * 88
        card(d, (510, y, 625, y + 48), rel, "#F8FAFC", "#97A3B7", size=15)
        card(d, (685, y - 7, 855, y + 55), "shared\ntransform", "#DFF3E7", "#4E9F69", size=15)
        card(d, (930, y - 7, 1085, y + 55), "private\nadapter", C["adapter"], C["orange"], size=15)
        arrow(d, [(625, y + 24), (685, y + 24)], C["blue"], w=3)
        arrow(d, [(855, y + 24), (930, y + 24)], C["blue"], w=3)
    label(d, 800, 830, "green blocks are transferable; orange blocks capture park-specific behavior", 18, fill=C["muted"], anchor="mm")
    rounded(d, (1240, 245, 2050, 875), C["white"], "#7EBB8D", r=22, w=2)
    label(d, 1280, 300, "Relation-wise normalization and learnable gate", 25, bold=True)
    for i, (rel, fill, knob) in enumerate(zip(rels, [C["ev"], C["bes"], C["pv"], C["es"], C["tr"]], [120, 55, 82, 40, 105])):
        y = 370 + i * 78
        rect(d, (1300, y, 1580, y + 35), fill, "#8796AA", w=1)
        label(d, 1320, y + 27, rel + " channel", 16, bold=True)
        d.line([(sc(1640), sc(y + 18)), (sc(1845), sc(y + 18))], fill="#A8B3C2", width=sc(5))
        d.ellipse(sb((1640 + knob - 12, y + 6, 1640 + knob + 12, y + 30)), fill=C["green"], outline=C["ink"], width=sc(1))
        arrow(d, [(1845, y + 18), (1935, 565)], C["green"], w=2, head=False)
    card(d, (1935, 495, 2020, 635), "fused\nstate", "#E8F6EE", "#4E9F69", size=17)
    arrow(d, [(360, 385), (465, 560)], C["blue"], w=4)
    arrow(d, [(1135, 560), (1240, 560)], C["blue"], w=4)
    card(d, (515, 970, 1985, 1060), "Innovation focus: heterogeneous physical relations are kept separate first, then adaptively fused.", C["white"], "#7EBB8D", size=22)
    return img.resize((w, h), Image.Resampling.LANCZOS)


def draw_actor_critic_learning() -> Image.Image:
    w, h = 2300, 1400
    img, d = canvas(w, h, "Module C. Actor-Critic and Local Constrained Learning", "Actor outputs scheduling actions; reward/cost critics guide local constrained SAC updates.", C["learn"])
    rounded(d, (110, 270, 760, 730), C["white"], "#6BB89B", r=22, w=2)
    label(d, 150, 325, "Actor policy network", 27, bold=True)
    network(d, 210, 520, [4, 5, 4, 2], "#BBF7D0", "mean / log-std")
    card(d, (570, 420, 710, 495), "EV actions", C["ev"], "#6B91C9")
    card(d, (570, 570, 710, 645), "BES action", C["bes"], "#5EA875")
    arrow(d, [(470, 500), (570, 455)], C["blue"], w=3)
    arrow(d, [(470, 540), (570, 610)], C["blue"], w=3)

    rounded(d, (860, 270, 2180, 800), C["white"], "#6BB89B", r=22, w=2)
    label(d, 900, 325, "Role-preserving twin reward/cost critics", 27, bold=True)
    for i, role in enumerate(["CS", "BES", "PV", "ES", "EV"]):
        card(d, (910 + i * 100, 390, 985 + i * 100, 445), role, "#F8FAFC", "#8796AA", size=15)
    card(d, (1495, 388, 1625, 448), "concat", "#E8F6EE", "#4E9F69", size=16)
    for i in range(5):
        arrow(d, [(985 + i * 100, 418), (1495, 418)], C["green"], w=2, head=False)
    network(d, 930, 635, [3, 4, 2], "#BFDBFE", "Q-r1")
    network(d, 1160, 635, [3, 4, 2], "#BFDBFE", "Q-r2")
    network(d, 1460, 635, [3, 4, 2], "#FDE68A", "Q-c1")
    network(d, 1690, 635, [3, 4, 2], "#FDE68A", "Q-c2")
    label(d, 1120, 760, "economic value", 18, fill=C["muted"], anchor="mm")
    label(d, 1620, 760, "constraint-risk value", 18, fill=C["muted"], anchor="mm")

    rounded(d, (220, 900, 2080, 1230), C["white"], "#6BB89B", r=22, w=2)
    label(d, 260, 955, "Local constrained SAC update loop", 27, bold=True)
    stages = [
        ("transition", "obs/action\nreward/cost"),
        ("replay\nbuffer", "local only"),
        ("critic\nupdate", "reward and cost"),
        ("actor\nupdate", "policy improvement"),
        ("temperature /\nmultiplier", "exploration and\nconstraint pressure"),
        ("policy\nrefresh", "next decision"),
    ]
    for i, (a, b) in enumerate(stages):
        x = 310 + i * 285
        card(d, (x, 1035, x + 190, 1135), a, "#E8F6EE", "#6BB89B", size=18)
        label(d, x + 95, 1180, b, 15, fill=C["muted"], anchor="mm")
        if i < len(stages) - 1:
            arrow(d, [(x + 190, 1085), (x + 285, 1085)], C["green"], w=4, dashed=True)
    return img.resize((w, h), Image.Resampling.LANCZOS)


def draw_safe_execution() -> Image.Image:
    w, h = 2600, 900
    img, d = canvas(w, h, "Module D. Safe Execution and Regional TR Coordination", "Execution-time constraints are handled separately from federated parameter sharing.", C["exec"])
    stages = [
        ("Actor\nactions", "policy output", C["white"]),
        ("Action\nmapping", "requested EV/BES energy", C["white"]),
        ("Device\nbounds", "SoC / power / V2G", C["bes"]),
        ("CS capacity-safe\nadjustment", "local station limit", C["pv"]),
        ("Secure\naggregation", "aggregate exchange only", C["tr"]),
        ("TR\ncoordination", "shared transformer pressure", "#FFE4C7"),
        ("Final\nexecution", "EV/BES/PV flows", C["ev"]),
        ("Feedback\nsignals", "reward / cost / TR", C["white"]),
    ]
    x0, y0 = 115, 340
    for i, (title, sub, fill) in enumerate(stages):
        x = x0 + i * 305
        card(d, (x, y0, x + 220, y0 + 135), title, fill, "#C58A43", size=18)
        label(d, x + 110, y0 + 190, sub, 15, fill=C["muted"], anchor="mm")
        if i < len(stages) - 1:
            arrow(d, [(x + 220, y0 + 68), (x0 + (i + 1) * 305, y0 + 68)], C["orange"], w=5)
    label(d, 1300, 775, "Outputs: final executable energy flows, local rewards/costs, and TR feedback for next-step observation.", 21, fill=C["muted"], anchor="mm")
    return img.resize((w, h), Image.Resampling.LANCZOS)


def draw_federation() -> Image.Image:
    w, h = 2100, 1100
    img, d = canvas(w, h, "Module E. Personalized Relation Federation", "Only selected actor-side shared relation transforms participate in the slow time-scale branch.", C["fed"])
    for i, park in enumerate(["Residential", "Office", "Commercial"]):
        y = 300 + i * 145
        card(d, (145, y, 360, y + 78), f"{park}\nshared blocks", "#DFF3E7", "#4E9F69", size=17)
        lock(d, 445, y + 20)
        label(d, 490, y + 52, "private adapters / critics / replay buffer blocked", 18, fill=C["red"], anchor="lm")
        arrow(d, [(360, y + 39), (920, 520)], C["purple"], w=4, dashed=True)
    card(d, (920, 420, 1210, 635), "personalized\naggregation\nserver", C["white"], "#9A7EDB", size=23)
    label(d, 1415, 395, "cross-park compatibility", 18, bold=True, anchor="mm")
    heat = [["#6D28D9", "#C4B5FD", "#DDD6FE"], ["#C4B5FD", "#6D28D9", "#BCA8FA"], ["#DDD6FE", "#BCA8FA", "#6D28D9"]]
    for r in range(3):
        for c in range(3):
            rect(d, (1335 + c * 56, 430 + r * 56, 1385 + c * 56, 480 + r * 56), heat[r][c], C["white"], w=1)
    card(d, (1450, 620, 1920, 740), "park-specific shared relation references", C["white"], "#9A7EDB", size=22)
    arrow(d, [(1210, 525), (1335, 515)], C["purple"], w=4, dashed=True)
    arrow(d, [(1500, 596), (1600, 620)], C["purple"], w=4, dashed=True)
    arrow(d, [(1450, 680), (360, 825)], C["purple"], w=4, dashed=True)
    label(d, 1050, 945, "Federation improves transferable relation knowledge without centralizing trajectories.", 21, fill=C["muted"], anchor="mm")
    return img.resize((w, h), Image.Resampling.LANCZOS)


def draw_combined(modules: dict[str, Image.Image]) -> Image.Image:
    w, h = 4200, 3200
    img = Image.new("RGB", (w, h), C["bg"])
    placements = [
        ("local", 70, 120, 1320, 822),
        ("encoder", 1450, 120, 1700, 1005),
        ("actor", 70, 1040, 1700, 1035),
        ("execution", 1870, 1580, 2100, 727),
        ("federation", 2130, 1040, 1500, 786),
    ]
    for key, x, y, ww, hh in placements:
        im = modules[key].resize((ww, hh), Image.Resampling.LANCZOS)
        img.paste(im, (x, y))
    d = ImageDraw.Draw(img)
    label(d, 90, 65, "SP-RGNN-CSAC architecture modules - no inter-module links yet", 38, bold=True)
    label(d, 90, 3045, "Next step: add only port-to-port links after each module is approved.", 24, fill=C["muted"])
    return img


def save(name: str, img: Image.Image) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    img.save(path)
    return path


def main() -> None:
    modules = {
        "local": draw_local_observation(),
        "encoder": draw_relational_encoder(),
        "actor": draw_actor_critic_learning(),
        "execution": draw_safe_execution(),
        "federation": draw_federation(),
    }
    outputs = [
        save("01_local_relational_observation.png", modules["local"]),
        save("02_shared_private_relational_encoder.png", modules["encoder"]),
        save("03_actor_critic_constrained_learning.png", modules["actor"]),
        save("04_safe_execution_tr_coordination.png", modules["execution"]),
        save("05_personalized_relation_federation.png", modules["federation"]),
        save("00_combined_modules_no_links.png", draw_combined(modules)),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
