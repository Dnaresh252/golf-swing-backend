"""2D skeleton overlay rendering using OpenCV.

Body-only skeleton (no head bones): green bones,
left-side RED / right-side BLUE dots, no text labels.
Skeleton ends at the shoulders; head uses a static reference line.
Sections (Legs, Waistline, Shoulders) can be toggled independently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Standard MediaPipe Pose landmark connections (33-landmark model).
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]

CORE_RENDER_IDS = {
    0,
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28,
}
CORE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

# ---------------------------------------------------------------------------
# Skeleton section definitions — each section maps to its own set of
# connections.  Head connections are ALWAYS excluded from the moving skeleton.
# ---------------------------------------------------------------------------

_HEAD_CONNECTIONS: frozenset = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
])

SECTION_SHOULDERS_CONNECTIONS: frozenset = frozenset([
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
])

SECTION_WAISTLINE_CONNECTIONS: frozenset = frozenset([
    (23, 24), (11, 23), (12, 24),
])

SECTION_LEGS_CONNECTIONS: frozenset = frozenset([
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32),
])

# Body-only connections (everything minus head).
_BODY_CONNECTIONS: frozenset = (
    SECTION_SHOULDERS_CONNECTIONS | SECTION_WAISTLINE_CONNECTIONS | SECTION_LEGS_CONNECTIONS
)

_LEFT_IDS = {1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
_RIGHT_IDS = {4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
_HEAD_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

# BGR — skeleton connection lines (green)
_BONE_COLOR = (0, 255, 0)
_BONE_COLOR_WHITE = _BONE_COLOR  # legacy alias (same bone line color)
_DOT_LEFT = (0, 0, 220)       # red  (BGR)
_DOT_RIGHT = (220, 60, 0)     # blue (BGR)
_DOT_HEAD = (0, 200, 255)     # yellow/amber (BGR)
_DOT_CENTER = (240, 240, 240) # white for torso anchors

# Body-part groups kept for backward compat imports
_TORSO_IDS = {11, 12, 23, 24}
_LEFT_ARM_IDS = {11, 13, 15, 17, 19, 21}
_RIGHT_ARM_IDS = {12, 14, 16, 18, 20, 22}
_LEFT_LEG_IDS = {23, 25, 27, 29, 31}
_RIGHT_LEG_IDS = {24, 26, 28, 30, 32}
_FACE_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
_BONE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "torso": _BONE_COLOR,
    "left_arm": _BONE_COLOR,
    "right_arm": _BONE_COLOR,
    "left_leg": _BONE_COLOR,
    "right_leg": _BONE_COLOR,
    "face": _BONE_COLOR,
    "default": _BONE_COLOR,
}

# Colors for coach annotations (BGR)
COACH_STATIC_LINE_COLOR = (255, 255, 255)  # white default
COACH_MOVING_LINE_COLOR = (0, 255, 255)    # yellow default
HEAD_REFERENCE_COLOR = (0, 200, 255)        # amber/yellow
SPINE_DASH_COLOR = (0, 220, 0)              # green dashed spine line


def _bone_color(_a_id: int, _b_id: int) -> Tuple[int, int, int]:
    return _BONE_COLOR


def _dot_color(jid: int) -> Tuple[int, int, int]:
    if jid in _LEFT_IDS:
        return _DOT_LEFT
    if jid in _RIGHT_IDS:
        return _DOT_RIGHT
    return _DOT_CENTER


def _enabled_connections(
    show_shoulders: bool = True,
    show_waistline: bool = True,
    show_legs: bool = True,
) -> frozenset:
    """Return the set of bone connections to draw based on enabled sections."""
    conns: set = set()
    if show_shoulders:
        conns |= SECTION_SHOULDERS_CONNECTIONS
    if show_waistline:
        conns |= SECTION_WAISTLINE_CONNECTIONS
    if show_legs:
        conns |= SECTION_LEGS_CONNECTIONS
    return frozenset(conns)


def _joints_for_connections(connections: frozenset) -> frozenset:
    """Return the set of joint IDs referenced by the given connections."""
    ids: set = set()
    for a, b in connections:
        ids.add(a)
        ids.add(b)
    return frozenset(ids)


class SkeletonOverlay:
    """Draw 2D stick-figure overlays from pose joints — all 33 landmarks."""

    RENDER_MIN_CONFIDENCE = 0.20
    BONE_SKIP_CONFIDENCE = 0.20
    RENDER_MIN_CONFIDENCE_AUX = 0.40
    VERY_LOW_CONFIDENCE_THRESHOLD = 0.35
    LOW_CONFIDENCE_THRESHOLD = 0.6
    DOT_COLOR_CORRECT = (0, 255, 70)
    DOT_COLOR_CLOSE = (0, 255, 255)
    DOT_COLOR_OFF = (0, 0, 255)
    LABELED_JOINT_IDS = ()

    def __init__(self) -> None:
        # Normalized x,y EMA for arm landmarks when ``smooth_arm_display`` is used.
        self._arm_display_ema: Dict[int, Tuple[float, float]] = {}

    def reset_arm_display_smooth(self) -> None:
        """Reset arm EMA when starting a new exported clip (avoids bleed across videos)."""
        self._arm_display_ema.clear()

    def _smooth_arms_for_display(self, joints: list, alpha: float = 0.18) -> list:
        """EMA-smooth normalized arm joint positions (11–22) for steadier arm bones."""
        if not joints:
            return joints
        ema = self._arm_display_ema
        out: List[dict] = []
        for j in joints:
            jid = int(j.get("id", -1))
            if jid not in range(11, 23):
                out.append(j)
                continue
            x = float(np.clip(float(j.get("x", 0.0)), 0.0, 1.0))
            y = float(np.clip(float(j.get("y", 0.0)), 0.0, 1.0))
            if jid in ema:
                px, py = ema[jid]
                x = alpha * x + (1.0 - alpha) * px
                y = alpha * y + (1.0 - alpha) * py
            ema[jid] = (x, y)
            nj = dict(j)
            nj["x"], nj["y"] = x, y
            out.append(nj)
        return out

    @staticmethod
    def _repair_arm_bone_pixels(
        points_by_id: Dict[int, Tuple[int, int]],
        width: int,
        height: int,
        min_forearm_px: float = 9.0,
    ) -> None:
        """Ensure elbow–wrist segments draw: fill missing wrist or extend degenerate forearms."""
        def _clip(x: float, y: float) -> Tuple[int, int]:
            return (
                int(np.clip(round(x), 0, width - 1)),
                int(np.clip(round(y), 0, height - 1)),
            )

        for shoulder_id, elbow_id, wrist_id in ((11, 13, 15), (12, 14, 16)):
            if elbow_id not in points_by_id:
                continue
            ex, ey = (float(points_by_id[elbow_id][0]), float(points_by_id[elbow_id][1]))
            if shoulder_id in points_by_id:
                sx, sy = (float(points_by_id[shoulder_id][0]), float(points_by_id[shoulder_id][1]))
                ulen = float(np.hypot(ex - sx, ey - sy)) + 1e-6
                ux, uy = (ex - sx) / ulen, (ey - sy) / ulen
            else:
                ux, uy = 0.0, -1.0
                ulen = 1.0
            if wrist_id not in points_by_id:
                flen = max(min_forearm_px, ulen * 0.9)
                points_by_id[wrist_id] = _clip(ex + ux * flen, ey + uy * flen)
                continue
            wx, wy = (float(points_by_id[wrist_id][0]), float(points_by_id[wrist_id][1]))
            dx, dy = wx - ex, wy - ey
            fl = float(np.hypot(dx, dy))
            if fl < 1.0:
                dx, dy = ux * min_forearm_px, uy * min_forearm_px
                fl = float(np.hypot(dx, dy))
            if fl < min_forearm_px:
                s = min_forearm_px / max(fl, 1e-6)
                wx, wy = ex + dx * s, ey + dy * s
                points_by_id[wrist_id] = _clip(wx, wy)

    @staticmethod
    def _scale_sizes(width: int, height: int) -> Tuple[int, int]:
        diag = (width ** 2 + height ** 2) ** 0.5
        radius = max(3, int(round(diag * 0.006)))
        thickness = max(2, int(round(diag * 0.0038)))
        return radius, thickness

    def joint_dot_color(self, confidence: float) -> Tuple[int, int, int]:
        c = float(confidence)
        if c < self.VERY_LOW_CONFIDENCE_THRESHOLD:
            return self.DOT_COLOR_OFF
        if c < self.LOW_CONFIDENCE_THRESHOLD:
            return self.DOT_COLOR_CLOSE
        return self.DOT_COLOR_CORRECT

    def draw_overlay(self, image_path: str, joints: list, output_path: str) -> str:
        """Draw pose overlay on an image file and save as PNG."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        rendered = self.draw_overlay_frame(image, joints)
        if not output_path.lower().endswith(".png"):
            raise ValueError("Output must be a PNG path ending with '.png'.")
        ok = cv2.imwrite(output_path, rendered)
        if not ok:
            raise ValueError(f"Unable to save output image: {output_path}")
        return output_path

    def draw_overlay_frame(
        self,
        frame: np.ndarray,
        joints: list,
        smooth_arm_display: bool = False,
        *,
        show_shoulders: bool = True,
        show_waistline: bool = True,
        show_legs: bool = True,
        show_spine: bool = True,
        head_reference_y: Optional[int] = None,
        head_reference_line_color: Optional[Tuple[int, int, int]] = None,
        use_orange_yellow_joint_dots: bool = False,
        static_lines: Optional[List[Dict[str, Any]]] = None,
        tracking_lines: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """Draw pose overlay on a raw OpenCV frame.

        Body-only skeleton (head joints 0-10 are never drawn as moving bones).
        Sections can be toggled:
          - shoulders: shoulder bar + arms + hands  (joints 11-22)
          - waistline: torso pillars + hip bar      (joints 11,12,23,24)
          - legs:      hips down to feet            (joints 23-32)
          - spine:     dashed green line from shoulder midpoint to hip midpoint

        Additional overlay layers:
          - ``head_reference_y``: pixel Y for a static horizontal head line (fixed for the clip)
          - ``static_lines``:  coach-drawn reference lines (non-moving)
          - ``tracking_lines``: coach lines that follow specific joints each frame
          - Coach text/markers (``static_labels``) are composited in
            ``SkeletonProcessor.export_*_video`` / ``coach_overlay_labels``, not here.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame provided.")

        output = frame.copy()
        height, width = output.shape[:2]
        joint_radius, bone_thickness = self._scale_sizes(width, height)
        bone_thickness = max(bone_thickness, 3)

        joints_draw = joints
        if smooth_arm_display and joints:
            joints_draw = self._smooth_arms_for_display(joints, alpha=0.18)

        joints_by_id: Dict[int, dict] = {}
        points_by_id: Dict[int, Tuple[int, int]] = {}
        for joint in joints_draw:
            try:
                joint_id = int(joint["id"])
                x_norm = float(joint["x"])
                y_norm = float(joint["y"])
            except (KeyError, TypeError, ValueError):
                continue
            x_px = int(round(np.clip(x_norm, 0.0, 1.0) * (width - 1)))
            y_px = int(round(np.clip(y_norm, 0.0, 1.0) * (height - 1)))
            joints_by_id[joint_id] = joint
            points_by_id[joint_id] = (x_px, y_px)

        self._repair_arm_bone_pixels(points_by_id, width, height)

        # --- Static head reference line (non-moving) ---
        if head_reference_y is not None:
            y_ref = int(np.clip(head_reference_y, 0, height - 1))
            hc = head_reference_line_color if head_reference_line_color is not None else HEAD_REFERENCE_COLOR
            cv2.line(output, (0, y_ref), (width - 1, y_ref), hc, 2, cv2.LINE_AA)

        # --- Coach static reference lines (non-moving) ---
        if static_lines:
            self._draw_coach_static_lines(output, static_lines, width, height)

        # --- Coach tracking / moving lines ---
        if tracking_lines:
            self._draw_coach_tracking_lines(output, tracking_lines, points_by_id,
                                            width, height)

        # --- Skeleton bones (body only, section-filtered) ---
        active_conns = _enabled_connections(show_shoulders, show_waistline, show_legs)
        drawable_joints = _joints_for_connections(active_conns)

        for a_id, b_id in POSE_CONNECTIONS:
            if (a_id, b_id) not in active_conns:
                continue
            if a_id not in points_by_id or b_id not in points_by_id:
                continue
            cv2.line(
                output,
                points_by_id[a_id],
                points_by_id[b_id],
                _BONE_COLOR,
                bone_thickness,
                cv2.LINE_AA,
            )

        # --- Dashed spine line (shoulder midpoint → hip midpoint) ---
        if show_spine:
            self._draw_spine_dashed(output, points_by_id, bone_thickness)

        for joint_id, point in points_by_id.items():
            if joint_id in _HEAD_IDS:
                continue
            if joint_id not in drawable_joints:
                continue
            if use_orange_yellow_joint_dots:
                if joint_id in _LEFT_IDS:
                    color = (0, 165, 255)  # orange (BGR)
                elif joint_id in _RIGHT_IDS:
                    color = (0, 255, 255)  # yellow (BGR)
                else:
                    color = (0, 200, 255)
            else:
                color = _dot_color(joint_id)
            cv2.circle(output, point, joint_radius + 1, (10, 10, 10), thickness=-1)
            cv2.circle(output, point, joint_radius, color, thickness=-1)

        return output

    # ------------------------------------------------------------------
    # Spine dashed line
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_spine_dashed(
        image: np.ndarray,
        points_by_id: Dict[int, Tuple[int, int]],
        thickness: int,
    ) -> None:
        """Draw a dashed green line from shoulder midpoint to hip midpoint."""
        ls = points_by_id.get(11)
        rs = points_by_id.get(12)
        lh = points_by_id.get(23)
        rh = points_by_id.get(24)
        if ls is None or rs is None or lh is None or rh is None:
            return
        shoulder_mid = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        hip_mid = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
        SkeletonOverlay._draw_dashed_line(
            image, shoulder_mid, hip_mid, SPINE_DASH_COLOR, thickness, 12,
        )

    # ------------------------------------------------------------------
    # Coach line helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_coach_static_lines(
        image: np.ndarray,
        lines: List[Dict[str, Any]],
        width: int,
        height: int,
    ) -> None:
        """Draw coach-placed reference lines that do NOT move between frames.

        Each entry in *lines*:
          {"start": (x, y), "end": (x, y), "color": (B,G,R), "thickness": int}
        Coordinates are pixel values.  ``color`` and ``thickness`` are optional.
        """
        for line_def in lines:
            start = line_def.get("start")
            end = line_def.get("end")
            if start is None or end is None:
                continue
            color = tuple(line_def.get("color", COACH_STATIC_LINE_COLOR))
            thickness = int(line_def.get("thickness", 2))
            sx = int(np.clip(start[0], 0, width - 1))
            sy = int(np.clip(start[1], 0, height - 1))
            ex = int(np.clip(end[0], 0, width - 1))
            ey = int(np.clip(end[1], 0, height - 1))
            cv2.line(image, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _draw_coach_tracking_lines(
        image: np.ndarray,
        lines: List[Dict[str, Any]],
        points_by_id: Dict[int, Tuple[int, int]],
        width: int,
        height: int,
    ) -> None:
        """Draw coach lines that follow (track) specific joint positions.

        Each entry:
          {"joint_id": int, "direction": "horizontal"|"vertical",
           "color": (B,G,R), "thickness": int}
        or for a line between two joints:
          {"joint_a": int, "joint_b": int, "color": (B,G,R), "thickness": int}
        """
        for line_def in lines:
            color = tuple(line_def.get("color", COACH_MOVING_LINE_COLOR))
            thickness = int(line_def.get("thickness", 2))

            if "joint_a" in line_def and "joint_b" in line_def:
                pa = points_by_id.get(int(line_def["joint_a"]))
                pb = points_by_id.get(int(line_def["joint_b"]))
                if pa is not None and pb is not None:
                    cv2.line(image, pa, pb, color, thickness, cv2.LINE_AA)
                continue

            jid = line_def.get("joint_id")
            if jid is None:
                continue
            pt = points_by_id.get(int(jid))
            if pt is None:
                continue
            direction = str(line_def.get("direction", "horizontal"))
            if direction == "horizontal":
                cv2.line(image, (0, pt[1]), (width - 1, pt[1]),
                         color, thickness, cv2.LINE_AA)
            elif direction == "vertical":
                cv2.line(image, (pt[0], 0), (pt[0], height - 1),
                         color, thickness, cv2.LINE_AA)

    def draw_motion_trail(
        self, video_path: str, skeleton_json: dict, joint_id: int, output_path: str
    ) -> str:
        """Draw a gradient motion trail for one joint on the first frame."""
        cap = cv2.VideoCapture(video_path)
        ok, first_frame = cap.read()
        cap.release()
        if not ok or first_frame is None:
            raise ValueError(f"Unable to read first frame from video: {video_path}")

        out = first_frame.copy()
        h, w = out.shape[:2]
        frames = skeleton_json.get("frames", [])

        points: List[Tuple[int, int]] = []
        for frame in frames:
            for joint in frame.get("joints", []):
                if int(joint.get("id", -1)) != int(joint_id):
                    continue
                x_px = int(np.clip(float(joint["x"]), 0.0, 1.0) * (w - 1))
                y_px = int(np.clip(float(joint["y"]), 0.0, 1.0) * (h - 1))
                points.append((x_px, y_px))
                break

        if len(points) < 2:
            raise ValueError("Insufficient points to draw motion trail.")

        overlay = out.copy()
        for i in range(1, len(points)):
            alpha = i / float(len(points) - 1)
            color = (0, int(255 * alpha), int(255 * (1.0 - alpha)))
            cv2.line(overlay, points[i - 1], points[i], color, 3, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.75, out, 0.25, 0.0)

        cv2.circle(out, points[0], 8, (0, 255, 0), -1)
        cv2.circle(out, points[-1], 8, (0, 0, 255), -1)

        if not output_path.lower().endswith(".png"):
            raise ValueError("Output must be a PNG path ending with '.png'.")
        if not cv2.imwrite(output_path, out):
            raise ValueError(f"Unable to save motion trail image: {output_path}")
        return output_path

    def draw_angle_arcs(
        self, image: np.ndarray, joints: list, angles_dict: dict, ideal_ranges: dict
    ) -> np.ndarray:
        """Draw protractor-style arcs and angle labels on an image."""
        output = image.copy()
        h, w = output.shape[:2]
        joints_by_id = {int(j["id"]): j for j in joints if "id" in j}

        angle_joint_map = {
            "spine_angle": 23,
            "hip_rotation_angle": 24,
            "knee_flex_left": 25,
            "knee_flex_right": 26,
            "arm_extension_left": 13,
            "arm_extension_right": 14,
        }

        for angle_name, value in angles_dict.items():
            if value is None or angle_name not in angle_joint_map:
                continue
            pivot_id = angle_joint_map[angle_name]
            pivot_joint = joints_by_id.get(pivot_id)
            if pivot_joint is None:
                continue

            x_px = int(np.clip(float(pivot_joint["x"]), 0.0, 1.0) * (w - 1))
            y_px = int(np.clip(float(pivot_joint["y"]), 0.0, 1.0) * (h - 1))
            center = (x_px, y_px)

            low, high = ideal_ranges.get(angle_name, (-1e9, 1e9))
            color = (0, 255, 0) if low <= float(value) <= high else (0, 0, 255)

            start_angle = 0
            end_angle = int(np.clip(float(value), 0.0, 180.0))
            cv2.ellipse(output, center, (28, 28), 0, start_angle, end_angle, color, 2)
            cv2.putText(
                output,
                f"{int(round(float(value)))} deg",
                (x_px + 8, y_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        return output

    def draw_confidence_heatmap(
        self, image_path: str, joints: list, output_path: str
    ) -> str:
        """Draw confidence-based joint heatmap and confidence-aware bones."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        out = image.copy()
        h, w = out.shape[:2]
        joints_by_id = {int(j["id"]): j for j in joints if "id" in j}
        points = {}
        for jid, j in joints_by_id.items():
            points[jid] = (
                int(np.clip(float(j["x"]), 0.0, 1.0) * (w - 1)),
                int(np.clip(float(j["y"]), 0.0, 1.0) * (h - 1)),
            )

        for a_id, b_id in POSE_CONNECTIONS:
            if a_id not in points or b_id not in points:
                continue
            conf = min(
                float(joints_by_id[a_id].get("confidence", 0.0)),
                float(joints_by_id[b_id].get("confidence", 0.0)),
            )
            if conf < 0.5:
                self._draw_dashed_line(out, points[a_id], points[b_id], _BONE_COLOR_WHITE, 2, 10)
            else:
                cv2.line(out, points[a_id], points[b_id], _BONE_COLOR_WHITE, 2, cv2.LINE_AA)

        for jid, point in points.items():
            conf = float(joints_by_id[jid].get("confidence", 0.5))
            color = self.joint_dot_color(conf)
            cv2.circle(out, point, 6, color, -1)
            cv2.putText(
                out,
                f"{int(round(conf * 100))}%",
                (point[0] + 6, point[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.rectangle(out, (10, 10), (170, 70), (20, 20, 20), -1)
        cv2.putText(out, "Low", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(out, "Medium", (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(out, "High", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if not cv2.imwrite(output_path, out):
            raise ValueError(f"Unable to save confidence heatmap image: {output_path}")
        return output_path

    @staticmethod
    def _draw_dashed_line(
        image: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        dash_len: int,
    ) -> None:
        p1_arr = np.array(p1, dtype=np.float64)
        p2_arr = np.array(p2, dtype=np.float64)
        length = float(np.linalg.norm(p2_arr - p1_arr))
        if length < 1.0:
            return
        direction = (p2_arr - p1_arr) / length
        on = True
        dist = 0.0
        while dist < length:
            start = p1_arr + direction * dist
            end = p1_arr + direction * min(dist + dash_len, length)
            if on:
                cv2.line(
                    image,
                    (int(start[0]), int(start[1])),
                    (int(end[0]), int(end[1])),
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
            on = not on
            dist += dash_len
