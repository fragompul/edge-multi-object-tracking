"""Builds the README banner and the GitHub social-preview image from real tracked frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from make_gifs import load, render_panel
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from emot.viz.draw import TrailBuffer
from emot.viz.palette import AMBER, CYAN, INK, MAGENTA, MUTED, NAVY, TRACKER_COLORS, VIOLET

OUT = Path("docs/assets")
FONTS = [
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for f in FONTS:
        if Path(f).exists():
            try:
                return ImageFont.truetype(
                    f, size, index=(8 if bold else 7) if f.endswith("Avenir Next.ttc") else 0
                )
            except OSError:
                continue
    return ImageFont.load_default()


def hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def background(w: int, h: int) -> Image.Image:
    img = np.zeros((h, w, 3), np.float32)
    img[:] = hex_rgb(NAVY)
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy, r, col, a in [
        (0.85, 0.0, 0.6, CYAN, 0.18),
        (0.05, 0.9, 0.5, MAGENTA, 0.14),
        (0.5, 0.5, 0.7, VIOLET, 0.06),
    ]:
        d = np.hypot((xx - cx * w) / (r * w), (yy - cy * h) / (r * w))
        img += (np.clip(1 - d, 0, 1)[..., None] ** 2) * a * np.array(hex_rgb(col), np.float32)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def tracked_frame(seq: str, tracker: str, idx: int) -> Image.Image:
    frames, t = load(seq)
    trails = TrailBuffer(30)
    img = None
    for i in range(max(0, idx - 30), idx + 1):
        img = render_panel(
            frames[i], t["trackers"][tracker][i], trails, "", TRACKER_COLORS[tracker], [], [], i
        )
    assert img is not None
    return Image.fromarray(img[:, :, ::-1])


def rounded(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *img.size), radius, fill=255)
    out = img.convert("RGBA")
    out.putalpha(mask)
    return out


def paste_card(base: Image.Image, card: Image.Image, xy: tuple[int, int], glow: str) -> None:
    glow_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow_layer).rounded_rectangle(
        (xy[0] - 4, xy[1] - 4, xy[0] + card.width + 4, xy[1] + card.height + 4),
        24,
        fill=(*hex_rgb(glow), 90),
    )
    base.alpha_composite(glow_layer.filter(ImageFilter.GaussianBlur(18)))
    base.alpha_composite(rounded(card, 20), xy)
    ImageDraw.Draw(base).rounded_rectangle(
        (xy[0], xy[1], xy[0] + card.width, xy[1] + card.height),
        20,
        outline=(255, 255, 255, 40),
        width=2,
    )


def draw_mark(d: ImageDraw.ImageDraw, x: int, y: int, s: float) -> None:
    def p(px: float, py: float) -> tuple[float, float]:
        return x + px * s, y + py * s

    w = max(3, int(4 * s))
    for pts, col in [
        ([(6, 20), (6, 10), (10, 6), (20, 6)], CYAN),
        ([(44, 6), (54, 6), (58, 10), (58, 20)], VIOLET),
        ([(58, 44), (58, 54), (54, 58), (44, 58)], MAGENTA),
        ([(20, 58), (10, 58), (6, 54), (6, 44)], VIOLET),
    ]:
        d.line([p(*q) for q in pts], fill=hex_rgb(col), width=w, joint="curve")
    curve = [(16, 44), (20, 43), (24, 36), (28, 31), (32, 30), (37, 28), (42, 23), (48, 20)]
    d.line([p(*q) for q in curve], fill=hex_rgb(VIOLET), width=max(2, int(3.2 * s)), joint="curve")
    for (cx, cy), r, col in [
        ((16, 44), 2.6, CYAN),
        ((32, 30), 3.2, VIOLET),
        ((48, 20), 4.8, MAGENTA),
    ]:
        d.ellipse([*p(cx - r, cy - r), *p(cx + r, cy + r)], fill=hex_rgb(col))


def gradient_text(
    img: Image.Image, xy: tuple[int, int], text: str, fnt, c1: str, c2: str, c3: str
) -> None:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).text(xy, text, font=fnt, fill=255)
    bbox = mask.getbbox()
    if not bbox:
        return
    gx = np.linspace(0, 1, bbox[2] - bbox[0])
    cols = np.array([hex_rgb(c1), hex_rgb(c2), hex_rgb(c3)], np.float32)
    row = np.stack([np.interp(gx, [0, 0.5, 1], cols[:, k]) for k in range(3)], axis=1)
    arr = np.tile(row[None], (img.height, 1, 1)).astype(np.uint8)
    canvas = np.zeros((img.height, img.width, 3), np.uint8)
    canvas[:, bbox[0] : bbox[2]] = arr
    img.paste(Image.fromarray(canvas), (0, 0), mask)


def chips(base: Image.Image, x: int, y: int, labels: list[str], fnt) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for lab in labels:
        w = d.textlength(lab, font=fnt) + 28
        d.rounded_rectangle(
            (x, y, x + w, y + 38),
            19,
            outline=(255, 255, 255, 70),
            fill=(255, 255, 255, 14),
            width=2,
        )
        d.text((x + 14, y + 7), lab, font=fnt, fill=(*hex_rgb(INK), 255))
        x += int(w) + 10
    base.alpha_composite(layer)


def banner(w: int, h: int, name: str, cards_scale: float) -> None:
    base = background(w, h).convert("RGBA")
    shots = [
        tracked_frame("MOT17-04", "botsort", 40),
        tracked_frame("MOT17-10", "botsort_cmc", 40),
        tracked_frame("MOT17-02", "deepsort_osnet_cascade", 60),
    ]
    cw = int(w * 0.30 * cards_scale)
    for k, (shot, glow, dx, dy) in enumerate(
        zip(shots, [CYAN, MAGENTA, VIOLET], [0.50, 0.64, 0.56], [0.10, 0.34, 0.60], strict=True)
    ):
        card = shot.resize((cw, int(cw * shot.height / shot.width)), Image.LANCZOS)
        paste_card(base, card, (int(w * dx), int(h * dy * (1 if k < 2 else 0.9))), glow)
    d = ImageDraw.Draw(base)
    pad = int(w * 0.05)
    draw_mark(d, pad, int(h * 0.12), h * 0.2 / 64)
    d.text(
        (pad + int(h * 0.26), int(h * 0.13)),
        "EMOT",
        font=font(int(h * 0.15), True),
        fill=hex_rgb(INK),
    )
    d.text(
        (pad + 2, int(h * 0.40)),
        "See every identity.",
        font=font(int(h * 0.105), True),
        fill=hex_rgb(INK),
    )
    gradient_text(
        base,
        (pad + 2, int(h * 0.40) + int(h * 0.125)),
        "On the edge.",
        font(int(h * 0.105), True),
        CYAN,
        VIOLET,
        MAGENTA,
    )
    d = ImageDraw.Draw(base)
    d.text(
        (pad + 2, int(h * 0.70)),
        "Multi-object tracking, built from scratch,\nbenchmarked on MOT17, deployed to CPU,\nCoreML and the browser.",
        font=font(int(h * 0.04)),
        fill=hex_rgb(MUTED),
        spacing=8,
    )
    chips(
        base,
        pad,
        int(h * 0.86),
        ["ByteTrack", "BoT-SORT", "DeepSORT", "OSNet ReID", "YOLO11", "ONNX", "WebGPU"],
        font(int(h * 0.032)),
    )
    base.convert("RGB").save(OUT / name, optimize=True)
    print(OUT / name, base.size)
    _ = AMBER


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    banner(1600, 560, "banner.png", 1.0)
    banner(1280, 640, "social-preview.png", 0.92)
    (OUT / "brand").mkdir(exist_ok=True)
    Image.open(OUT / "social-preview.png").save(OUT / "brand" / "social-preview.png", optimize=True)


if __name__ == "__main__":
    main()
