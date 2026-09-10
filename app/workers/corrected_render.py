"""
Rendering for the corrected swing video.

What the golfer paid for is a side-by-side: their own swing with the AI
skeleton on the left, and the instructor's corrected pose on the right,
moving from where they were to where they should be.

Two things make this honest rather than a guess:

  * The corrected pose arrives in avatar world metres, not image
    coordinates, and no camera parameters exist for the original footage.
    So the right panel is drawn as a plain orthographic projection of the
    world joints - front is (x, y), back is (-x, y), left and right are
    (-/+z, y), top is (x, z). The instructor posed it in 3D, so every one
    of those views is real. Nothing is fitted, nothing is invented.

  * Only the joints the instructor actually moved are moved. Anything the
    correction does not name is left exactly where the AI put it, and an
    unknown joint id is ignored rather than guessed at.

Nothing is written on the frame except the two panel labels. No angles,
no scores, no verdicts - the same honesty rule the web pages follow.
"""
import logging
import math
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.integrations.ml_cv_engine.coach_corrections import CorrectionHandler
from app.integrations.ml_cv_engine.coach_overlay_labels import draw_static_labels_on_frame
from app.integrations.ml_cv_engine.overlay_2d import (
    COACH_MOVING_LINE_COLOR,
    COACH_STATIC_LINE_COLOR,
    POSE_CONNECTIONS,
)

logger = logging.getLogger(__name__)

# Timeline, in seconds. Kept as constants because the acceptance criteria
# name them: at least 6 seconds, freeze at 2.5, blend across one second.
PLAY_UNTIL = 2.5
BLEND_UNTIL = 3.5
TOTAL_SECONDS = 6.0
FPS = 30

PANEL_BG = (24, 24, 24)          # dark, so the gold skeleton reads clearly
LABEL_LEFT = "Your swing"
LABEL_RIGHT = "Corrected by your instructor"

# Core joints used to find the address frame when a correction predates the
# Unity change that started sending frame_num: both shoulders, both hips.
CORE_JOINTS = (11, 12, 23, 24)
SHOULDER_IDS = (11, 12)
HIP_IDS = (23, 24)


# ---------------------------------------------------------------------------
# Skeleton helpers
# ---------------------------------------------------------------------------

def _joint_id(joint: Dict[str, Any], name_to_id: Dict[str, int]) -> Optional[int]:
    """
    Prefer the explicit id the Unity export now sends. Fall back to the
    joint name for corrections saved before that change, using the AI
    skeleton's own name/id pairs. Anything still unrecognised is treated
    as "not corrected" rather than mapped to a guess.
    """
    jid = joint.get("id")
    if jid is None:
        jid = joint.get("joint_id")
    if jid is not None:
        try:
            return int(jid)
        except (TypeError, ValueError):
            return None
    name = str(joint.get("joint_name") or joint.get("name") or "").strip().lower()
    return name_to_id.get(name)


def _build_name_to_id(frames: List[Dict[str, Any]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for frame in frames:
        for j in frame.get("joints", []):
            name = str(j.get("name") or j.get("joint_name") or "").strip().lower()
            if name and j.get("id") is not None:
                try:
                    mapping[name] = int(j["id"])
                except (TypeError, ValueError):
                    continue
        if mapping:
            break
    return mapping


def _pick_corrected_frame(
    skeleton_frames: List[Dict[str, Any]], correction: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Returns (frame, how_it_was_chosen). The instructor's export names the
    frame; only older corrections need the address-frame fallback.
    """
    wanted = correction.get("frame_num")
    if wanted is None:
        wanted = correction.get("frame_index")
    if wanted is not None:
        try:
            wanted_i = int(wanted)
            for fr in skeleton_frames:
                if int(fr.get("frame_num", -1)) == wanted_i:
                    return fr, f"frame_num={wanted_i}"
            if 0 <= wanted_i < len(skeleton_frames):
                return skeleton_frames[wanted_i], f"frame_index={wanted_i}"
        except (TypeError, ValueError):
            pass

    # Address frame: first frame where every core joint is confidently seen.
    for fr in skeleton_frames:
        by_id = {}
        for j in fr.get("joints", []):
            try:
                by_id[int(j.get("id", -1))] = j
            except (TypeError, ValueError):
                continue
        if all(
            jid in by_id and float(by_id[jid].get("confidence", 0.0)) >= 0.5
            for jid in CORE_JOINTS
        ):
            return fr, f"address_frame={fr.get('frame_num')}"

    if skeleton_frames:
        return skeleton_frames[0], "first_frame_fallback"
    return None, "no_frames"


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _project(world: Tuple[float, float, float], angle: str) -> Tuple[float, float]:
    """
    Orthographic view of a world joint for one camera angle. Screen y grows
    downwards, world y grows upwards, so y is negated.
    """
    x, y, z = world
    if angle == "front":
        return x, -y
    if angle == "back":
        return -x, -y
    if angle == "left":
        return -z, -y
    if angle == "right":
        return z, -y
    if angle == "top":
        return x, z
    return x, -y


def _world_joints(frame: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
    """AI world coordinates, keyed by joint id, in apply_correction's shape."""
    out: Dict[int, Dict[str, float]] = {}
    for j in frame.get("joints", []):
        try:
            jid = int(j.get("id", -1))
        except (TypeError, ValueError):
            continue
        if jid < 0:
            continue
        out[jid] = {
            "id": jid,
            "x": float(j.get("wx", 0.0)),
            "y": float(j.get("wy", 0.0)),
            "z": float(j.get("wz", 0.0)),
            "confidence": float(j.get("confidence", 1.0)),
        }
    return out


def _corrected_joints(
    correction: Dict[str, Any], name_to_id: Dict[str, int]
) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for j in correction.get("joints", []):
        jid = _joint_id(j, name_to_id)
        if jid is None:
            continue
        out[jid] = {
            "id": jid,
            "x": float(j.get("x", 0.0)),
            "y": float(j.get("y", 0.0)),
            "z": float(j.get("z", 0.0)),
            "confidence": float(j.get("confidence", 1.0)),
        }
    return out


def _torso_height(points: Dict[int, Tuple[float, float]]) -> float:
    """Shoulder midpoint to hip midpoint - the scale reference for a panel."""
    try:
        sh = [points[i] for i in SHOULDER_IDS if i in points]
        hp = [points[i] for i in HIP_IDS if i in points]
        if not sh or not hp:
            return 0.0
        sy = sum(p[1] for p in sh) / len(sh)
        hy = sum(p[1] for p in hp) / len(hp)
        return abs(hy - sy)
    except Exception:
        return 0.0


def _fit_to_panel(
    joints: Dict[int, Dict[str, float]],
    angle: str,
    panel_w: int,
    panel_h: int,
    target_torso_px: float,
) -> Dict[int, Tuple[int, int]]:
    """
    Project, scale so the torso matches the left panel, and centre on the
    hip midpoint. Same treatment for original and corrected, so the two
    share a frame of reference and the movement between them is the only
    thing that changes.
    """
    projected = {jid: _project((j["x"], j["y"], j["z"]), angle) for jid, j in joints.items()}
    if not projected:
        return {}

    torso = _torso_height(projected)
    if torso <= 1e-6:
        xs = [p[0] for p in projected.values()]
        ys = [p[1] for p in projected.values()]
        spread = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
        scale = (panel_h * 0.6) / spread
    else:
        scale = (target_torso_px or panel_h * 0.25) / torso

    hips = [projected[i] for i in HIP_IDS if i in projected]
    if hips:
        cx = sum(p[0] for p in hips) / len(hips)
        cy = sum(p[1] for p in hips) / len(hips)
    else:
        cx = sum(p[0] for p in projected.values()) / len(projected)
        cy = sum(p[1] for p in projected.values()) / len(projected)

    out: Dict[int, Tuple[int, int]] = {}
    for jid, (px, py) in projected.items():
        sx = int(round(panel_w / 2 + (px - cx) * scale))
        sy = int(round(panel_h * 0.55 + (py - cy) * scale))
        out[jid] = (sx, sy)
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_skeleton(
    canvas: np.ndarray,
    points: Dict[int, Tuple[int, int]],
    colour: Tuple[int, int, int],
    thickness: int = 3,
) -> None:
    h, w = canvas.shape[:2]

    def visible(p: Tuple[int, int]) -> bool:
        return -w <= p[0] <= 2 * w and -h <= p[1] <= 2 * h

    for a, b in POSE_CONNECTIONS:
        if a in points and b in points and visible(points[a]) and visible(points[b]):
            cv2.line(canvas, points[a], points[b], colour, thickness, cv2.LINE_AA)
    for p in points.values():
        if visible(p):
            cv2.circle(canvas, p, max(2, thickness), colour, -1, cv2.LINE_AA)


def _image_points(
    frame: Dict[str, Any], panel_w: int, panel_h: int
) -> Dict[int, Tuple[int, int]]:
    """AI skeleton in image coordinates (x, y are normalised 0-1)."""
    out: Dict[int, Tuple[int, int]] = {}
    for j in frame.get("joints", []):
        try:
            jid = int(j.get("id", -1))
        except (TypeError, ValueError):
            continue
        if jid < 0 or j.get("x") is None or j.get("y") is None:
            continue
        if float(j.get("confidence", 1.0)) < 0.2:
            continue
        out[jid] = (
            int(round(float(j["x"]) * panel_w)),
            int(round(float(j["y"]) * panel_h)),
        )
    return out


def _fit_image(img: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
    """Letterbox into the panel, preserving aspect."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((panel_h, panel_w, 3), PANEL_BG, np.uint8)
    scale = min(panel_w / w, panel_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((panel_h, panel_w, 3), PANEL_BG, np.uint8)
    y0, x0 = (panel_h - nh) // 2, (panel_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _labels(width: int, height: int) -> List[Dict[str, Any]]:
    return [
        {"x": int(width * 0.25), "y": int(height * 0.07),
         "text": LABEL_LEFT, "style": "headline"},
        {"x": int(width * 0.75), "y": int(height * 0.07),
         "text": LABEL_RIGHT, "style": "headline"},
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_angle(
    angle: str,
    out_path: str,
    width: int,
    height: int,
    skeleton_frames: List[Dict[str, Any]],
    correction: Dict[str, Any],
    still_path: Optional[str],
    video_path: Optional[str],
    ffmpeg_path: str,
) -> Dict[str, Any]:
    """
    Render one angle. Returns a small dict describing what was used, so the
    caller can audit-log which frame the render was built from.
    """
    panel_w, panel_h = width // 2, height
    handler = CorrectionHandler()
    name_to_id = _build_name_to_id(skeleton_frames)

    corrected_frame, how = _pick_corrected_frame(skeleton_frames, correction)
    if corrected_frame is None:
        raise ValueError("no AI skeleton frames to render against")

    original_world = _world_joints(corrected_frame)
    corrected_world = _corrected_joints(correction, name_to_id)
    matched = [jid for jid in corrected_world if jid in original_world]

    # Left panel scale reference, so the right panel matches it.
    left_pts_at_corrected = _image_points(corrected_frame, panel_w, panel_h)
    target_torso = _torso_height(
        {k: (float(v[0]), float(v[1])) for k, v in left_pts_at_corrected.items()}
    ) or panel_h * 0.25

    cap = None
    video_frames: List[np.ndarray] = []
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            video_frames.append(fr)
        cap.release()

    still = None
    if still_path and os.path.exists(still_path):
        still = cv2.imread(still_path)

    raw_path = out_path + ".raw.mp4"
    writer = cv2.VideoWriter(
        raw_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {raw_path}")

    total = int(TOTAL_SECONDS * FPS)
    labels = _labels(width, height)

    try:
        for i in range(total):
            t = i / FPS
            canvas = np.full((height, width, 3), PANEL_BG, np.uint8)

            # ---------------- left panel: the golfer's own swing ----------
            if t < PLAY_UNTIL and video_frames:
                idx = min(int((t / PLAY_UNTIL) * len(video_frames)), len(video_frames) - 1)
                base = _fit_image(video_frames[idx], panel_w, panel_h)
                sk_frame = (
                    skeleton_frames[min(idx, len(skeleton_frames) - 1)]
                    if skeleton_frames else None
                )
            else:
                # after the freeze, and for every still-only angle
                src = still if still is not None else (
                    video_frames[0] if video_frames else None
                )
                base = (
                    _fit_image(src, panel_w, panel_h) if src is not None
                    else np.full((panel_h, panel_w, 3), PANEL_BG, np.uint8)
                )
                sk_frame = corrected_frame if video_frames or still is None else None

            left = base
            # Only the angle that actually has the swing video carries the
            # per-frame skeleton. A still from another angle never gets the
            # video angle's skeleton drawn on it.
            if sk_frame is not None and (video_frames or still is None):
                pts = _image_points(sk_frame, panel_w, panel_h)
                if pts:
                    _draw_skeleton(left, pts, COACH_STATIC_LINE_COLOR, 3)

            # ---------------- right panel: the corrected pose -------------
            right = np.full((panel_h, panel_w, 3), PANEL_BG, np.uint8)
            if t <= PLAY_UNTIL:
                blend = 0.0
            elif t >= BLEND_UNTIL:
                blend = 1.0
            else:
                blend = (t - PLAY_UNTIL) / (BLEND_UNTIL - PLAY_UNTIL)

            merged = handler.apply_correction(
                list(original_world.values()),
                list(corrected_world.values()),
                blend_factor=blend,
            )
            merged_map = {}
            for j in merged:
                try:
                    merged_map[int(j.get("id", -1))] = j
                except (TypeError, ValueError):
                    continue
            merged_map.pop(-1, None)

            pts_r = _fit_to_panel(merged_map, angle, panel_w, panel_h, target_torso)
            if pts_r:
                _draw_skeleton(right, pts_r, COACH_MOVING_LINE_COLOR, 3)

            canvas[:, :panel_w] = left
            canvas[:, panel_w:] = right
            cv2.line(canvas, (panel_w, 0), (panel_w, height), (60, 60, 60), 2)

            draw_static_labels_on_frame(canvas, labels)
            writer.write(canvas)
    finally:
        writer.release()

    # mp4v is what cv2 can write; H.264 is what browsers play.
    cmd = [
        ffmpeg_path, "-y", "-i", raw_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg transcode failed for angle {angle}: {proc.stderr.decode()[:200]}"
        )
    try:
        os.remove(raw_path)
    except OSError:
        pass

    return {
        "angle": angle,
        "frame_used": how,
        "joints_corrected": len(matched),
        "joints_in_correction": len(corrected_world),
        "had_video": bool(video_frames),
        "seconds": TOTAL_SECONDS,
    }
