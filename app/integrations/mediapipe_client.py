"""
MediaPipe integration layer.

Specialist 3 will drop their module (specialist3_ml) into the Python path.
This client wraps their API so the rest of the codebase stays stable
regardless of whether the real ML module is available.

Expected interface from Specialist 3:
    specialist3_ml.detect_pose_from_image(image_path: str, angle: str) -> list[dict]
    specialist3_ml.detect_pose_from_video(video_path: str) -> list[dict]
    specialist3_ml.generate_3d_avatar(skeleton_data: dict) -> dict
    specialist3_ml.render_avatar_angles(model_path: str) -> dict
"""

import logging
import math
import random
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 33 MediaPipe Pose landmark names (matches BlazePose topology)
_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Realistic golf-address-position approximations (normalised 0–1 space)
_GOLF_ADDRESS_JOINTS = [
    # nose
    (0.50, 0.10, 0.00, 0.99),
    # left_eye_inner … right_eye_outer (eyes)
    (0.51, 0.09, 0.02, 0.99), (0.52, 0.09, 0.03, 0.99), (0.53, 0.09, 0.03, 0.98),
    (0.49, 0.09, 0.02, 0.99), (0.48, 0.09, 0.03, 0.99), (0.47, 0.09, 0.03, 0.98),
    # ears
    (0.54, 0.10, -0.02, 0.97), (0.46, 0.10, -0.02, 0.97),
    # mouth
    (0.51, 0.11, 0.03, 0.99), (0.49, 0.11, 0.03, 0.99),
    # left_shoulder, right_shoulder
    (0.57, 0.25, 0.00, 0.99), (0.43, 0.25, 0.00, 0.99),
    # left_elbow, right_elbow
    (0.62, 0.42, 0.02, 0.98), (0.38, 0.42, 0.02, 0.98),
    # left_wrist, right_wrist
    (0.60, 0.58, 0.05, 0.97), (0.40, 0.58, 0.05, 0.97),
    # left_pinky, right_pinky
    (0.61, 0.61, 0.06, 0.95), (0.39, 0.61, 0.06, 0.95),
    # left_index, right_index
    (0.60, 0.62, 0.07, 0.95), (0.40, 0.62, 0.07, 0.95),
    # left_thumb, right_thumb
    (0.59, 0.60, 0.06, 0.96), (0.41, 0.60, 0.06, 0.96),
    # left_hip, right_hip
    (0.54, 0.55, 0.00, 0.99), (0.46, 0.55, 0.00, 0.99),
    # left_knee, right_knee
    (0.55, 0.72, -0.02, 0.98), (0.45, 0.72, -0.02, 0.98),
    # left_ankle, right_ankle
    (0.56, 0.88, -0.01, 0.97), (0.44, 0.88, -0.01, 0.97),
    # left_heel, right_heel
    (0.56, 0.90, -0.03, 0.96), (0.44, 0.90, -0.03, 0.96),
    # left_foot_index, right_foot_index
    (0.57, 0.92, 0.02, 0.95), (0.43, 0.92, 0.02, 0.95),
]


class MediaPipeClient:
    """
    Integration layer between the platform and Specialist 3's ML module.
    Falls back to realistic mock data when the real module is unavailable.
    """

    def __init__(self) -> None:
        self._mock_mode = False
        self._specialist3 = None

        try:
            import specialist3_ml  # type: ignore
            self._specialist3 = specialist3_ml
            logger.info("specialist3_ml loaded — using real MediaPipe pipeline.")
        except ImportError:
            self._mock_mode = True
            logger.warning(
                "specialist3_ml not found — MediaPipeClient running in MOCK MODE. "
                "Real skeleton detection is disabled."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_skeleton_from_image(self, image_path: str, angle: str) -> Dict[str, Any]:
        """
        Run pose detection on a single image.

        Returns a dict with key 'joints': list of 33 joint dicts, each containing:
            {id, name, x, y, z, confidence}

        Raises ValueError on failure.
        """
        if self._mock_mode:
            logger.debug("MOCK detect_skeleton_from_image: angle=%s path=%s", angle, image_path)
            return self._mock_single_frame(angle=angle)

        try:
            raw = self._specialist3.detect_pose_from_image(image_path, angle)
            return self._normalise_joints(raw)
        except Exception as exc:
            logger.error("detect_skeleton_from_image failed (%s): %s", image_path, exc)
            raise ValueError(f"Skeleton detection failed for angle '{angle}': {exc}") from exc

    def detect_skeleton_from_video(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Run pose detection on every frame of a video.

        Returns a list of frame dicts, each containing:
            {frame_num, timestamp, joints: [...]}

        Raises ValueError on failure.
        """
        if self._mock_mode:
            logger.debug("MOCK detect_skeleton_from_video: path=%s", video_path)
            return self._mock_video_frames(num_frames=30)

        try:
            raw_frames = self._specialist3.detect_pose_from_video(video_path)
            return [
                {
                    "frame_num": i,
                    "timestamp": frame.get("timestamp", i / 30.0),
                    "joints": self._normalise_joints(frame.get("joints", []))["joints"],
                }
                for i, frame in enumerate(raw_frames)
            ]
        except Exception as exc:
            logger.error("detect_skeleton_from_video failed (%s): %s", video_path, exc)
            raise ValueError(f"Video skeleton detection failed: {exc}") from exc

    def generate_avatar_model(self, skeleton_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a 3-D avatar from combined skeleton data.

        Returns:
            {"obj_path": str, "fbx_path": str, "glb_path": str}

        Raises ValueError on failure.
        """
        if self._mock_mode:
            logger.debug("MOCK generate_avatar_model")
            return {
                "obj_path": "/tmp/mock_avatar.obj",
                "fbx_path": "/tmp/mock_avatar.fbx",
                "glb_path": "/tmp/mock_avatar.glb",
            }

        try:
            result = self._specialist3.generate_3d_avatar(skeleton_data)
            if not all(k in result for k in ("obj_path", "fbx_path", "glb_path")):
                raise ValueError("generate_3d_avatar returned incomplete data.")
            return result
        except Exception as exc:
            logger.error("generate_avatar_model failed: %s", exc)
            raise ValueError(f"Avatar model generation failed: {exc}") from exc

    def generate_angle_views(self, avatar_model_path: str) -> Dict[str, str]:
        """
        Render 5 PNG screenshots of the avatar from each cardinal angle.

        Returns:
            {"top": path, "front": path, "left": path, "right": path, "back": path}

        Raises ValueError on failure.
        """
        if self._mock_mode:
            logger.debug("MOCK generate_angle_views: model=%s", avatar_model_path)
            return {
                "top":   "/tmp/mock_view_top.png",
                "front": "/tmp/mock_view_front.png",
                "left":  "/tmp/mock_view_left.png",
                "right": "/tmp/mock_view_right.png",
                "back":  "/tmp/mock_view_back.png",
            }

        try:
            result = self._specialist3.render_avatar_angles(avatar_model_path)
            required = {"top", "front", "left", "right", "back"}
            missing = required - set(result.keys())
            if missing:
                raise ValueError(f"render_avatar_angles missing angles: {missing}")
            return result
        except Exception as exc:
            logger.error("generate_angle_views failed: %s", exc)
            raise ValueError(f"Angle view generation failed: {exc}") from exc

    def get_mock_skeleton_data(self, submission_id: str) -> Dict[str, Any]:
        """
        Return a realistic full skeleton dataset for testing without Specialist 3.
        Format mirrors what detect_skeleton_from_video would return for a 30-frame swing.
        """
        return {
            "submission_id": submission_id,
            "source": "mock",
            "image_frames": [
                {
                    "angle": angle,
                    **self._mock_single_frame(angle=angle),
                }
                for angle in ("front", "left", "right", "back")
            ],
            "video_frames": self._mock_video_frames(num_frames=30),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_joints(raw: Any) -> Dict[str, Any]:
        """
        Coerce whatever Specialist 3 returns into our canonical joint format.
        Handles both list-of-dicts and dict-with-landmarks styles.
        """
        if isinstance(raw, dict) and "joints" in raw:
            joints = raw["joints"]
        elif isinstance(raw, list):
            joints = raw
        else:
            joints = []

        normalised = []
        for idx, j in enumerate(joints):
            normalised.append({
                "id": j.get("id", idx),
                "name": j.get("name", _LANDMARK_NAMES[idx] if idx < 33 else f"joint_{idx}"),
                "x": float(j.get("x", 0.0)),
                "y": float(j.get("y", 0.0)),
                "z": float(j.get("z", 0.0)),
                "confidence": float(j.get("confidence", j.get("visibility", 0.0))),
            })
        return {"joints": normalised}

    def _mock_single_frame(self, angle: str = "front") -> Dict[str, Any]:
        """Return one frame of mock joint data in golf address position."""
        angle_offset = {
            "front": 0.0, "back": 0.0, "left": 0.02, "right": -0.02, "top": 0.0
        }.get(angle, 0.0)

        joints = []
        for idx, (x, y, z, conf) in enumerate(_GOLF_ADDRESS_JOINTS):
            # Add tiny random jitter so data looks measured not copied
            joints.append({
                "id": idx,
                "name": _LANDMARK_NAMES[idx],
                "x": round(x + angle_offset + random.uniform(-0.002, 0.002), 5),
                "y": round(y + random.uniform(-0.002, 0.002), 5),
                "z": round(z + random.uniform(-0.001, 0.001), 5),
                "confidence": round(conf - random.uniform(0.0, 0.03), 4),
            })
        return {"joints": joints}

    def _mock_video_frames(self, num_frames: int = 30) -> List[Dict[str, Any]]:
        """Return mock video frames simulating a golf swing motion."""
        frames = []
        for i in range(num_frames):
            t = i / 30.0
            # Simulate slight swing rotation over the clip
            swing_phase = math.sin(math.pi * t / (num_frames / 30.0))
            frame_data = self._mock_single_frame(angle="front")
            # Apply small phase-based offsets to wrists to simulate swing
            for j in frame_data["joints"]:
                if "wrist" in j["name"]:
                    j["x"] = round(j["x"] + swing_phase * 0.05, 5)
                    j["y"] = round(j["y"] - abs(swing_phase) * 0.03, 5)
            frames.append({
                "frame_num": i,
                "timestamp": round(t, 4),
                "joints": frame_data["joints"],
            })
        return frames


# Singleton used throughout the application
mediapipe_client = MediaPipeClient()
