"""Coach text / numbered markers on video frames (BGR OpenCV images).

Used by ``swing_coach_annotation_app`` and by ``SkeletonProcessor`` export paths
when ``overlay_options["static_labels"]`` is set. Coordinates are **full-frame
pixels** (same space as ``static_lines``).

Label dict schema (``style`` determines rendering):

- ``numbered_circle``: ``{"style": "numbered_circle", "x": int, "y": int, "main": "1", "sub": ""}``
- ``caption_box``: ``{"style": "caption_box", "x", "y", "main", "sub", "color": (B,G,R)}``
- ``headline``: ``{"style": "headline", "x", "y", "text": "ALL CAPS PHRASE"}``
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

__all__ = ["draw_static_labels_on_frame"]

# BGR — vivid ring for numbered markers (tutorial-style)
_NUMBER_RING_GREEN = (55, 235, 95)
_DEFAULT_CAPTION_BGR = (255, 255, 255)


def _put_text_stroke(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_face: int,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int,
    stroke: int = 2,
    stroke_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    x, y = int(org[0]), int(org[1])
    for ox in range(-stroke, stroke + 1):
        for oy in range(-stroke, stroke + 1):
            if ox == 0 and oy == 0:
                continue
            cv2.putText(
                image,
                text,
                (x + ox, y + oy),
                font_face,
                font_scale,
                stroke_color,
                thickness,
                cv2.LINE_AA,
            )
    cv2.putText(image, text, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)


def _draw_numbered_circle_marker(
    image: np.ndarray,
    cx: int,
    cy: int,
    digit: str,
    sub: str,
    w: int,
    h: int,
) -> None:
    ref = float(max(w, h))
    font = cv2.FONT_HERSHEY_DUPLEX
    r0 = int(np.clip(ref / 28.0, 20, 62))
    scale = float(np.clip(r0 / 26.0, 0.75, 2.2))
    th = max(2, int(round(scale * 1.8)))
    (tw, th_px), baseline = cv2.getTextSize(digit, font, scale, th)
    r = int(max(r0, tw * 0.55 + 10, (th_px + baseline) * 0.55 + 8))
    r = int(np.clip(r, 18, min(w, h) // 4))
    ring = tuple(int(c) for c in _NUMBER_RING_GREEN)
    cx = int(np.clip(cx, r + 1, w - r - 2))
    cy = int(np.clip(cy, r + 1, h - r - 2))

    cv2.circle(image, (cx, cy), r, (32, 36, 32), -1, cv2.LINE_AA)
    ring_th = max(2, min(4, r // 14))
    cv2.circle(image, (cx, cy), r, ring, ring_th, cv2.LINE_AA)

    bl_x = cx - tw // 2
    bl_y = int(cy + (th_px + baseline) // 2)
    _put_text_stroke(
        image, digit, (bl_x, bl_y), font, scale, (255, 255, 255), th, stroke=max(2, th // 2)
    )

    sub = sub.strip()
    if not sub:
        return
    sub_scale = float(np.clip(scale * 0.42, 0.35, 1.0))
    sub_th = max(1, int(round(sub_scale)))
    (sw, sh), sbl = cv2.getTextSize(sub, font, sub_scale, sub_th)
    gap = max(6, r // 5)
    sub_cy = cy + r + gap + sh
    if sub_cy >= h - 2:
        sub_cy = cy - r - gap - sbl
    sx = int(np.clip(cx - sw // 2, 2, w - sw - 2))
    sub_bl_y = min(sub_cy, h - 2)
    _put_text_stroke(
        image,
        sub,
        (sx, sub_bl_y),
        font,
        sub_scale,
        (255, 255, 255),
        sub_th,
        stroke=max(1, sub_th + 1),
    )


def _draw_headline_text(
    image: np.ndarray,
    cx: int,
    cy: int,
    text: str,
    w: int,
    h: int,
) -> None:
    text = text.strip().upper()
    if not text:
        return
    ref = float(max(w, h))
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = float(np.clip(ref / 140.0, 1.0, 3.8))
    th = max(2, int(round(scale * 2)))
    (tw, th_px), baseline = cv2.getTextSize(text, font, scale, th)
    while tw > int(w * 0.88) and scale > 0.55:
        scale -= 0.06
        th = max(2, int(round(scale * 2)))
        (tw, th_px), baseline = cv2.getTextSize(text, font, scale, th)

    cx = int(np.clip(cx, tw // 2 + 2, w - tw // 2 - 2))
    cy = int(np.clip(cy, (th_px + baseline) // 2 + 2, h - (th_px + baseline) // 2 - 2))
    bl_x = cx - tw // 2
    bl_y = int(cy + (th_px + baseline) // 2)
    _put_text_stroke(
        image,
        text,
        (bl_x, bl_y),
        font,
        scale,
        (255, 255, 255),
        th,
        stroke=max(3, th + 1),
        stroke_color=(0, 0, 0),
    )


def _draw_caption_box_label(
    image: np.ndarray,
    cx: int,
    cy: int,
    main: str,
    sub: str,
    color: Tuple[int, int, int],
    w: int,
    h: int,
) -> None:
    ref = float(max(h, w))
    main_scale = float(np.clip(ref / 420.0, 0.65, 2.4))
    sub_scale = float(np.clip(main_scale * 0.52, 0.38, 1.25))
    font = cv2.FONT_HERSHEY_DUPLEX
    main_th = max(1, int(round(main_scale)))
    sub_th = max(1, int(round(sub_scale)))

    lines: List[Tuple[str, float, int]] = []
    if main:
        lines.append((main, main_scale, main_th))
    if sub:
        lines.append((sub, sub_scale, sub_th))

    total_h = 0
    max_tw = 0
    sizes: List[Tuple[int, int]] = []
    for txt, sc, thk in lines:
        (tw, th_px), _ = cv2.getTextSize(txt, font, sc, thk)
        sizes.append((tw, th_px))
        total_h += th_px + 4
        max_tw = max(max_tw, tw)
    if not sizes:
        return
    total_h -= 4
    pad_x, pad_y = 10, 6
    box_w = min(max_tw + pad_x * 2, w)
    box_h = min(total_h + pad_y * 2, h)
    x0 = max(0, min(cx - box_w // 2, max(0, w - box_w)))
    y0 = max(0, min(cy - box_h // 2, max(0, h - box_h)))
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1, cv2.LINE_AA)
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), color, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)

    ty = y0 + pad_y
    for (txt, sc, thk), (tw, th_px) in zip(lines, sizes):
        tx = x0 + (box_w - tw) // 2
        baseline_y = ty + th_px
        _put_text_stroke(image, txt, (tx, baseline_y), font, sc, color, thk)
        ty = baseline_y + 4


def draw_static_labels_on_frame(image: np.ndarray, labels: List[Dict[str, Any]]) -> None:
    """Draw coach labels in place on a BGR frame (full-frame pixel coordinates)."""
    if not labels:
        return
    h, w = image.shape[:2]

    for lb in labels:
        cx = int(np.clip(int(lb["x"]), 0, w - 1))
        cy = int(np.clip(int(lb["y"]), 0, h - 1))
        style = str(lb.get("style", "caption_box"))

        if style == "headline":
            txt = str(lb.get("text", "") or "")
            _draw_headline_text(image, cx, cy, txt, w, h)
            continue

        if style == "numbered_circle":
            main = str(lb.get("main", "")).strip()
            sub = str(lb.get("sub", "") or "")
            if main:
                _draw_numbered_circle_marker(image, cx, cy, main, sub, w, h)
            continue

        raw_c = lb.get("color", _DEFAULT_CAPTION_BGR)
        try:
            color = tuple(int(c) for c in raw_c)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            color = _DEFAULT_CAPTION_BGR
        if len(color) != 3:
            color = _DEFAULT_CAPTION_BGR
        main = str(lb.get("main", "")).strip()
        sub = str(lb.get("sub", "") or "").strip()
        if not main and not sub:
            continue
        _draw_caption_box_label(image, cx, cy, main, sub, color, w, h)
