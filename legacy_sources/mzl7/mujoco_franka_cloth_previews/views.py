"""Shared helpers for splitting composite previews into per-view videos.

The composite MP4 written by ``generate_previews`` is a horizontal stack of
camera panels in the order ``["front", "top", <wrist>]``. These helpers map
each camera panel to a human-facing view label (``main`` / ``top`` / ``side``)
and burn a small caption onto each frame so the split-out clips are
self-identifying.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Camera name (as recorded in metadata) -> view label. The panel order in the
# composite is front | top | wrist, exposed to consumers as main | top | wrist.
_INDEX_FALLBACK = {0: "main", 1: "top", 2: "wrist"}


def view_label(cam_name, index):
    """Human-facing view label for a camera panel."""
    n = str(cam_name).lower()
    if n == "front":
        return "main"
    if n == "top":
        return "top"
    if "wrist" in n:
        return "wrist"
    return _INDEX_FALLBACK.get(index, f"view{index}")


def view_names(cameras):
    """List of view labels aligned with the composite panel order."""
    return [view_label(c, i) for i, c in enumerate(cameras)]


def caption_text(label):
    return f"{label.upper()} VIEW"


def caption_frame(img, text):
    """Return a copy of ``img`` (HxWx3 uint8) with ``text`` burned top-left.

    Overlays the caption on the existing pixels, so the frame dimensions are
    unchanged (keeps the width/height even, as libx264/yuv420p requires when
    the source panels already are).
    """
    im = Image.fromarray(np.asarray(img)).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
    except AttributeError:  # very old Pillow
        tw, th = draw.textsize(text, font=font)
    draw.rectangle([0, 0, tw + 8, th + 8], fill=(18, 18, 22))
    draw.text((4, 4), text, fill=(240, 240, 240), font=font)
    return np.asarray(im)
