from functools import lru_cache
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Anchor = Literal["lt", "mm", "mt"]

FONT_CANDIDATES = (
    "/System/Library/Fonts/Avenir Next.ttc",
    "/System/Library/Fonts/SFNSRounded.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Avenir.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:\\Windows\\Fonts\\segoeui.ttf",
)


@lru_cache(maxsize=1)
def _font_path() -> str:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError("No suitable Unicode font was found on this system.")


@lru_cache(maxsize=128)
def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_font_path(), max(int(size), 8))


@lru_cache(maxsize=512)
def measure_text(text: str, font_size: int) -> Tuple[int, int]:
    font = _font(font_size)
    left, top, right, bottom = font.getbbox(text)
    return max(int(right - left), 1), max(int(bottom - top), 1)


@lru_cache(maxsize=512)
def _render_text_rgba(
    text: str,
    font_size: int,
    color_bgr: Tuple[int, int, int],
) -> np.ndarray:
    font = _font(font_size)
    left, top, right, bottom = font.getbbox(text)
    text_w = max(int(right - left), 1)
    text_h = max(int(bottom - top), 1)
    pad = 2
    overlay = Image.new("RGBA", (text_w + pad * 2, text_h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]), 255)
    draw.text((pad - left, pad - top), text, font=font, fill=color_rgb)
    return np.asarray(overlay, dtype=np.uint8)


def draw_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_size: int,
    color_bgr: Tuple[int, int, int],
    anchor: Anchor = "lt",
) -> None:
    text_w, text_h = measure_text(text, font_size)

    x, y = int(position[0]), int(position[1])
    if anchor == "mm":
        x -= text_w // 2
        y -= text_h // 2
    elif anchor == "mt":
        x -= text_w // 2

    x = max(x, 0)
    y = max(y, 0)
    if x >= frame.shape[1] or y >= frame.shape[0]:
        return

    overlay_np = _render_text_rgba(text, font_size, color_bgr)
    h, w = overlay_np.shape[:2]
    roi_x2 = min(x + w, frame.shape[1])
    roi_y2 = min(y + h, frame.shape[0])
    overlay_np = overlay_np[: roi_y2 - y, : roi_x2 - x]
    if overlay_np.size == 0:
        return

    roi = frame[y:roi_y2, x:roi_x2].astype(np.float32)
    alpha = overlay_np[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = overlay_np[:, :, :3][:, :, ::-1].astype(np.float32)
    blended = overlay_bgr * alpha + roi * (1.0 - alpha)
    frame[y:roi_y2, x:roi_x2] = blended.astype(np.uint8)
