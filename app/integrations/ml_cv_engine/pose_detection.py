"""MediaPipe pose detection utilities for images and frames.

Example usage:
    # from ml_cv_engine.pose_detection import PoseDetector
    # detector = PoseDetector(min_confidence=0.3)
    # joints = detector.detect_from_image("swing.jpg")
    # if joints is not None:
    #     print(joints["joints"][11])  # left_shoulder landmark data
"""

from __future__ import annotations

import atexit
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

LOGGER = logging.getLogger(__name__)


def load_image_bgr(path: str) -> Optional[np.ndarray]:
    """Load an image as BGR uint8 for OpenCV / MediaPipe.

    Tries ``cv2.imdecode`` (Unicode paths on Windows), ``cv2.imread``, then Pillow
    for formats OpenCV may not register (HEIC, GIF frames, TIFF, etc.).
    """
    p = str(path)
    try:
        data = np.fromfile(p, dtype=np.uint8)
        if data.size > 0:
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
    except OSError:
        pass
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image

        with Image.open(p) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# --- Arm alignment post-processing (fixes A–E) --------------------------------


class ArmSwapDetector:
    """Detect left/right arm label swap from shoulder–elbow x proximity."""

    @staticmethod
    def detect_arm_swap(joints_dict: Dict[int, dict]) -> bool:
        need = (11, 12, 13)
        if not all(k in joints_dict for k in need):
            return False
        left_shoulder_x = float(joints_dict[11]["x"])
        right_shoulder_x = float(joints_dict[12]["x"])
        left_elbow_x = float(joints_dict[13]["x"])

        left_correct = abs(left_elbow_x - left_shoulder_x)
        left_swapped = abs(left_elbow_x - right_shoulder_x)
        if left_swapped < left_correct * 0.6:
            return True
        return False

    @staticmethod
    def apply_swap(joints: List[dict]) -> None:
        """Swap spatial data for elbows, wrists, and finger pairs (not shoulders)."""
        pairs = [
            (13, 14),
            (15, 16),
            (17, 18),
            (19, 20),
            (21, 22),
        ]
        by_id = {int(j["id"]): j for j in joints if "id" in j}
        for a, b in pairs:
            ja, jb = by_id.get(a), by_id.get(b)
            if ja is None or jb is None:
                continue
            for key in ("x", "y", "z", "confidence"):
                if key in ja and key in jb:
                    ja[key], jb[key] = jb[key], ja[key]
            if "interpolated" in ja or "interpolated" in jb:
                ia, ib = bool(ja.get("interpolated")), bool(jb.get("interpolated"))
                ja["interpolated"], jb["interpolated"] = ib, ia


class AnatomicalValidator:
    """Elbow angle must stay within a plausible range; otherwise interpolate."""

    @staticmethod
    def validate_elbow_angle(shoulder: dict, elbow: dict, wrist: dict) -> bool:
        v1 = (
            float(elbow["x"]) - float(shoulder["x"]),
            float(elbow["y"]) - float(shoulder["y"]),
        )
        v2 = (
            float(wrist["x"]) - float(elbow["x"]),
            float(wrist["y"]) - float(elbow["y"]),
        )
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        mag2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return False
        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))
        return 25.0 <= angle_deg <= 185.0

    @staticmethod
    def interpolate_elbow(shoulder: dict, elbow: dict, wrist: dict) -> None:
        mx = (float(shoulder["x"]) + float(wrist["x"])) / 2.0
        my = (float(shoulder["y"]) + float(wrist["y"])) / 2.0
        elbow["x"] = float(np.clip(mx, 0.0, 1.0))
        elbow["y"] = float(np.clip(my, 0.0, 1.0))
        elbow["interpolated"] = True


def _clamp_forearm_to_upper_arm_ratio(
    shoulder: dict, elbow: dict, wrist: dict
) -> None:
    """Forearm / upper-arm length ratio — slightly wider band for full extension."""
    sx, sy = float(shoulder["x"]), float(shoulder["y"])
    ex, ey = float(elbow["x"]), float(elbow["y"])
    wx, wy = float(wrist["x"]), float(wrist["y"])
    ux, uy = ex - sx, ey - sy
    upper = math.hypot(ux, uy)
    if upper < 1e-9:
        return
    fx, fy = wx - ex, wy - ey
    forearm = math.hypot(fx, fy)
    if forearm < 1e-9:
        return
    ratio = forearm / upper
    if 0.58 <= ratio <= 1.58:
        return
    inv_f = 1.0 / forearm
    dx, dy = fx * inv_f, fy * inv_f
    lo, hi = upper * 0.58, upper * 1.42
    new_len = min(max(forearm, lo), hi)
    wrist["x"] = float(np.clip(ex + dx * new_len, 0.0, 1.0))
    wrist["y"] = float(np.clip(ey + dy * new_len, 0.0, 1.0))


class ArmSmoother:
    """EWMA smoothing with stronger low-pass on fast-moving arm joints."""

    _ARM_IDS = frozenset({13, 14, 15, 16})
    _BODY_IDS = frozenset({11, 12, 23, 24})
    _LEG_IDS = frozenset({25, 26, 27, 28})

    def __init__(self) -> None:
        self._prev: Dict[int, Tuple[float, float, float]] = {}

    def reset(self) -> None:
        self._prev.clear()

    @classmethod
    def _alpha_for(cls, joint_id: int) -> Optional[float]:
        jid = int(joint_id)
        if jid in cls._ARM_IDS:
            return None  # arms: no per-frame EWMA, trust MediaPipe directly
        if jid in cls._BODY_IDS:
            return 0.5
        if jid in cls._LEG_IDS:
            return 0.6
        return None

    def smooth_joint(self, joint_id: int, x: float, y: float, z: float) -> Tuple[float, float, float]:
        alpha = self._alpha_for(joint_id)
        if alpha is None:
            return x, y, z
        jid = int(joint_id)
        if jid not in self._prev:
            self._prev[jid] = (x, y, z)
            return x, y, z
        px, py, pz = self._prev[jid]
        nx = alpha * x + (1.0 - alpha) * px
        ny = alpha * y + (1.0 - alpha) * py
        nz = alpha * z + (1.0 - alpha) * pz
        self._prev[jid] = (nx, ny, nz)
        return nx, ny, nz


class OcclusionHandler:
    """Heuristic: elbow x between hips suggests arm behind torso; trust shoulder + wrist."""

    @staticmethod
    def is_arm_occluded(elbow: dict, joints_dict: Dict[int, dict]) -> bool:
        need = (23, 24)
        if not all(k in joints_dict for k in need):
            return False
        left_hip_x = float(joints_dict[23]["x"])
        right_hip_x = float(joints_dict[24]["x"])
        body_left = min(left_hip_x, right_hip_x)
        body_right = max(left_hip_x, right_hip_x)
        elbow_x = float(elbow["x"])
        return bool(body_left < elbow_x < body_right)

    @staticmethod
    def fix_occluded_elbow(shoulder: dict, elbow: dict, wrist: dict) -> None:
        AnatomicalValidator.interpolate_elbow(shoulder, elbow, wrist)


def _apply_arm_alignment_pipeline(
    joints: List[dict],
    frame_idx: int,
    arm_smoother: ArmSmoother,
) -> None:
    """Post-process one frame's joints: swap correction, forearm clamp, light EWMA.

    Occlusion replacement and elbow-angle replacement have been removed — they
    created wrong arm positions during backswing/downswing by replacing valid
    MediaPipe detections with computed midpoints that don't match the real limb.
    """
    by_id: Dict[int, dict] = {int(j["id"]): j for j in joints if "id" in j}

    # A: swap correction — left elbow closer to right shoulder than left shoulder
    if ArmSwapDetector.detect_arm_swap(by_id):
        ArmSwapDetector.apply_swap(joints)
        by_id = {int(j["id"]): j for j in joints if "id" in j}
        LOGGER.info("Frame %d: Arm swap corrected", frame_idx)

    # C: forearm length ratio — keep wrist anatomically connected to elbow
    for shoulder_id, elbow_id, wrist_id in (
        (11, 13, 15),
        (12, 14, 16),
    ):
        sh, el, wr = by_id.get(shoulder_id), by_id.get(elbow_id), by_id.get(wrist_id)
        if sh and el and wr:
            _clamp_forearm_to_upper_arm_ratio(sh, el, wr)

    # D: light EWMA — only body/leg joints (arms disabled to prevent swing lag)
    for j in joints:
        if "id" not in j:
            continue
        jid = int(j["id"])
        if jid in ArmSmoother._ARM_IDS:
            continue  # arms: trust MediaPipe directly, no per-frame lag
        sx, sy, sz = arm_smoother.smooth_joint(
            jid, float(j["x"]), float(j["y"]), float(j["z"])
        )
        if ArmSmoother._alpha_for(jid) is not None:
            j["x"] = float(np.clip(sx, 0.0, 1.0))
            j["y"] = float(np.clip(sy, 0.0, 1.0))
            j["z"] = float(sz)


class _EmptyPoseBackend:
    """Fallback backend when MediaPipe Pose Solutions API is unavailable."""

    @staticmethod
    def process(_frame_rgb):
        class _Result:
            pose_landmarks = None

        return _Result()


class _TaskPoseAdapter:
    """Adapter exposing `.process()` like the classic Solutions API."""

    def __init__(self, landmarker) -> None:
        self._landmarker = landmarker

    def process(self, frame_rgb):
        if self._landmarker is None:
            class _Result:
                pose_landmarks = None

            return _Result()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        task_result = self._landmarker.detect(mp_image)

        class _PoseLandmarks:
            def __init__(self, landmarks):
                self.landmark = landmarks

        class _Result:
            def __init__(self, pose_landmarks):
                self.pose_landmarks = pose_landmarks

        if not task_result.pose_landmarks:
            return _Result(None)
        return _Result(_PoseLandmarks(task_result.pose_landmarks[0]))


class _TaskPoseVideoAdapter:
    """Tasks API in VIDEO mode — required for temporal tracking between frames."""

    def __init__(self, landmarker) -> None:
        self._landmarker = landmarker
        self._timestamp_ms = 0

    def reset_stream(self) -> None:
        """Keep _timestamp_ms monotonic across clips for the same PoseLandmarker."""
        pass

    def process(self, frame_rgb, frame_dt_ms: int = 33):
        if self._landmarker is None:
            class _Result:
                pose_landmarks = None

            return _Result()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        ts = int(self._timestamp_ms)
        self._timestamp_ms += max(1, int(frame_dt_ms))
        task_result = self._landmarker.detect_for_video(mp_image, ts)

        class _PoseLandmarks:
            def __init__(self, landmarks):
                self.landmark = landmarks

        class _Result:
            def __init__(self, pose_landmarks):
                self.pose_landmarks = pose_landmarks

        if not task_result.pose_landmarks:
            return _Result(None)
        return _Result(_PoseLandmarks(task_result.pose_landmarks[0]))


class PoseDetector:
    """Detect and format MediaPipe Pose landmarks from images or frames.

    This detector uses CPU MediaPipe Pose and can process:
    - static image files via `detect_from_image`
    - OpenCV BGR frames via `detect_from_frame`

    For video pipelines, set ``static_image_mode=False`` so MediaPipe uses
    temporal tracking between frames (reduces jitter and dropout).
    """

    JOINT_NAMES: List[str] = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]

    def __init__(
        self,
        min_confidence: float = 0.05,
        static_image_mode: bool = True,
        model_complexity: int = 2,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize the pose detector.

        Args:
            min_confidence: Minimum landmark visibility score to include.
                Set very low (default 0.05) so that partially-occluded
                limbs (e.g. the trailing arm during a golf follow-through)
                are still emitted with their real confidence value.
                The overlay renderer uses confidence for visual styling
                (dashed lines, colored dots) rather than dropping joints.
            static_image_mode: True for single images, False for video
                sequences (enables temporal tracking for better accuracy
                during fast motion).
            model_complexity: 0 (lite), 1 (full), or 2 (heavy). Higher
                values give better accuracy at the cost of speed.
            model_path: Optional path to a MediaPipe Tasks model file.
        """
        self.min_confidence = min_confidence
        self._mp_pose = None
        self._pose = None
        self._task_landmarker = None
        self._task_is_video = False
        self._frame_dt_ms = 33
        self._arm_smoother = ArmSmoother()
        self._arm_alignment_frame_idx = 0

        det_conf = max(0.10, min_confidence * 0.5)
        if static_image_mode:
            track_conf = max(0.10, min_confidence * 0.4)
        else:
            track_conf = max(0.38, min_confidence * 0.75)

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf,
            )
        elif hasattr(mp, "tasks"):
            if model_path is None:
                model_path = os.path.join(
                    os.path.dirname(__file__), "models", "pose_landmarker_full.task"
                )
            if not os.path.exists(model_path):
                LOGGER.warning(
                    "MediaPipe Tasks detected but model file not found at: %s. "
                    "Falling back to empty pose backend.",
                    model_path,
                )
                self._pose = _EmptyPoseBackend()
                return
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                run_mode = (
                    vision.RunningMode.VIDEO
                    if not static_image_mode
                    else vision.RunningMode.IMAGE
                )
                options = vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=model_path),
                    running_mode=run_mode,
                    min_pose_detection_confidence=det_conf,
                    min_pose_presence_confidence=det_conf,
                    min_tracking_confidence=track_conf,
                    num_poses=1,
                )
                self._task_landmarker = vision.PoseLandmarker.create_from_options(options)
                if static_image_mode:
                    self._pose = _TaskPoseAdapter(self._task_landmarker)
                    self._task_is_video = False
                else:
                    self._pose = _TaskPoseVideoAdapter(self._task_landmarker)
                    self._task_is_video = True
                atexit.register(self.close)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to initialize MediaPipe Tasks PoseLandmarker (%s). "
                    "Falling back to empty pose backend.",
                    exc,
                )
                self._pose = _EmptyPoseBackend()
        else:
            LOGGER.warning(
                "MediaPipe Pose Solutions API is unavailable in this environment. "
                "Falling back to an empty pose backend."
            )
            self._pose = _EmptyPoseBackend()

    def close(self) -> None:
        """Release MediaPipe resources and break references eagerly."""
        if self._task_landmarker is not None:
            try:
                self._task_landmarker.close()
            except Exception:
                pass
        self._task_landmarker = None

        if isinstance(self._pose, (_TaskPoseAdapter, _TaskPoseVideoAdapter)):
            self._pose._landmarker = None
            self._pose = _EmptyPoseBackend()
        self._task_is_video = False
        try:
            self._arm_smoother.reset()
        except Exception:
            pass

        if self._mp_pose is not None and self._pose is not None:
            try:
                close_fn = getattr(self._pose, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass

    def __del__(self) -> None:
        """Best-effort cleanup to avoid shutdown-time MediaPipe errors."""
        try:
            self.close()
        except Exception:
            pass

    def set_video_fps(self, fps: float) -> None:
        """Set per-frame timestamp step for Tasks VIDEO mode (call before decoding)."""
        if fps > 1e-6:
            self._frame_dt_ms = max(1, int(round(1000.0 / float(fps))))

    def reset_video_stream(self) -> None:
        """Reset Tasks VIDEO stream timestamps at the start of each new clip."""
        if isinstance(self._pose, _TaskPoseVideoAdapter):
            self._pose.reset_stream()
        self._arm_smoother.reset()
        self._arm_alignment_frame_idx = 0

    def detect_from_image(self, image_path: str) -> Optional[Dict[str, List[Dict[str, float]]]]:
        """Detect pose landmarks from an image file (many formats via ``load_image_bgr``)."""
        self._arm_smoother.reset()
        self._arm_alignment_frame_idx = 0
        image_bgr = load_image_bgr(image_path)
        if image_bgr is None:
            return None
        return self.detect_from_frame(image_bgr)

    def detect_from_frame(self, frame: np.ndarray) -> Optional[Dict[str, List[Dict[str, float]]]]:
        """Detect pose landmarks from an OpenCV BGR frame."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(gray.mean()) < 70.0:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if isinstance(self._pose, _TaskPoseVideoAdapter):
            results = self._pose.process(frame_rgb, self._frame_dt_ms)
        else:
            results = self._pose.process(frame_rgb)
        if not results.pose_landmarks:
            return None

        joints: List[Dict[str, float]] = []
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            confidence = float(landmark.visibility)
            joints.append(
                {
                    "id": idx,
                    "name": self.JOINT_NAMES[idx],
                    "x": float(np.clip(landmark.x, 0.0, 1.0)),
                    "y": float(np.clip(landmark.y, 0.0, 1.0)),
                    "z": float(landmark.z),
                    "confidence": confidence,
                }
            )

        fidx = self._arm_alignment_frame_idx
        _apply_arm_alignment_pipeline(joints, fidx, self._arm_smoother)
        self._arm_alignment_frame_idx = fidx + 1
        return {"joints": joints}
