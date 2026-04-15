"""
MediaPipe integration layer.

Wraps Specialist 3's ml_cv_engine module so the rest of the codebase
stays stable regardless of whether the real ML module is available.

Real API (app.integrations.ml_cv_engine.api_functions):
    celery_task_analyze(submission_id, image_paths, video_path) -> dict
    celery_task_correction(submission_id, original_joints, corrected_joints) -> dict

Both return dicts that are mapped to our canonical structure in run_full_pipeline()
and apply_correction() below.
"""

import logging
import math
import random
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
            from app.integrations import ml_cv_engine  # type: ignore  # noqa: F401
            import app.integrations.ml_cv_engine.api_functions as specialist3_ml  # type: ignore
            self._specialist3 = specialist3_ml
            logger.info("specialist3_ml loaded — using real MediaPipe pipeline.")
        except Exception as exc:
            self._mock_mode = True
            logger.warning(
                "specialist3_ml import failed — MediaPipeClient running in MOCK MODE. "
                "Real skeleton detection is disabled. Error: %s: %s",
                type(exc).__name__, exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        submission_id: str,
        image_paths: Dict[str, str],
        video_path: str,
    ) -> Dict[str, Any]:
        """
        Run the complete ML pipeline: skeleton detection, avatar generation,
        and angle-view renders in a single call.

        Args:
            submission_id: Unique submission identifier.
            image_paths:   {"front": path, "left": path, "right": path, "back": path}
            video_path:    Local path to the swing video file.

        Returns:
            {
                "skeleton_json":      {"submission_id": str, "frames": [...]},
                "avatar_paths":       {"obj": str, "glb": str, "fbx": str},
                "render_paths":       {"top": str, "front": str, "left": str,
                                       "right": str, "back": str},
                "key_frames":         dict,
                "swing_angles":       dict,
                "analysis_warnings":  list[str],
            }

        Raises ValueError on pipeline failure.
        """
        if self._mock_mode:
            logger.debug("MOCK run_full_pipeline: submission=%s", submission_id)
            return self._mock_full_pipeline(submission_id)

        try:
            result = self._specialist3.celery_task_analyze(
                submission_id, image_paths, video_path
            )
            return {
                "skeleton_json":     result["skeleton_json"],
                "avatar_paths":      result["avatar_paths"],
                "render_paths":      result["render_paths"],
                "key_frames":        result.get("key_frames", {}),
                "swing_angles":      result.get("swing_angles", {}),
                "analysis_warnings": result.get("analysis_warnings", []),
            }
        except Exception as exc:
            logger.error("run_full_pipeline failed (%s): %s", submission_id, exc)
            raise ValueError(
                f"ML pipeline failed for submission '{submission_id}': {exc}"
            ) from exc

    def apply_correction(
        self,
        submission_id: str,
        original_joints: List[Dict[str, Any]],
        corrected_joints: List[Dict[str, Any]],
        blend: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Apply a coach correction and regenerate corrected avatar renders.

        Args:
            submission_id:    Submission identifier for output naming.
            original_joints:  Baseline frame joints from the detector.
            corrected_joints: Coach-adjusted joints.
            blend:            Blend factor in [0, 1]. Defaults to 1.0 (full correction).

        Returns:
            {
                "submission_id":  str,
                "updated_joints": list,
                "render_paths":   dict,
                "avatar_paths":   {"obj": str, "glb": str, "fbx": str},
            }

        Raises ValueError on failure.
        """
        if self._mock_mode:
            logger.debug("MOCK apply_correction: submission=%s", submission_id)
            return {
                "submission_id":  submission_id,
                "updated_joints": corrected_joints,
                "render_paths": {
                    "top":   "/tmp/mock_corrected_top.png",
                    "front": "/tmp/mock_corrected_front.png",
                    "left":  "/tmp/mock_corrected_left.png",
                    "right": "/tmp/mock_corrected_right.png",
                    "back":  "/tmp/mock_corrected_back.png",
                },
                "avatar_paths": {
                    "obj": "/tmp/mock_corrected_avatar.obj",
                    "glb": "/tmp/mock_corrected_avatar.glb",
                    "fbx": "/tmp/mock_corrected_avatar.fbx",
                },
            }

        try:
            result = self._specialist3.celery_task_correction(
                submission_id, original_joints, corrected_joints
            )
            return {
                "submission_id":  result.get("submission_id", submission_id),
                "updated_joints": result.get("updated_joints", corrected_joints),
                "render_paths":   result.get("render_paths", {}),
                "avatar_paths":   result.get("avatar_paths", {}),
            }
        except Exception as exc:
            logger.error("apply_correction failed (%s): %s", submission_id, exc)
            raise ValueError(
                f"Coach correction failed for submission '{submission_id}': {exc}"
            ) from exc

    def get_mock_skeleton_data(self, submission_id: str) -> Dict[str, Any]:
        """
        Return a realistic full skeleton dataset for testing without Specialist 3.
        Format mirrors the skeleton_json returned by run_full_pipeline().
        """
        return {
            "submission_id": submission_id,
            "source": "mock",
            "frames": self._mock_video_frames(num_frames=30),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mock_full_pipeline(self, submission_id: str) -> Dict[str, Any]:
        """Return mock pipeline output in the same structure as the real pipeline."""
        return {
            "skeleton_json": {
                "submission_id": submission_id,
                "frames": self._mock_video_frames(num_frames=30),
            },
            "avatar_paths": {
                "obj": "/tmp/mock_avatar.obj",
                "glb": "/tmp/mock_avatar.glb",
                "fbx": "/tmp/mock_avatar.fbx",
            },
            "render_paths": {
                "top":   "/tmp/mock_view_top.png",
                "front": "/tmp/mock_view_front.png",
                "left":  "/tmp/mock_view_left.png",
                "right": "/tmp/mock_view_right.png",
                "back":  "/tmp/mock_view_back.png",
            },
            "key_frames":        {},
            "swing_angles":      {},
            "analysis_warnings": [],
        }

    def _mock_single_frame(self, angle: str = "front") -> Dict[str, Any]:
        """Return one frame of mock joint data in golf address position."""
        angle_offset = {
            "front": 0.0, "back": 0.0, "left": 0.02, "right": -0.02, "top": 0.0
        }.get(angle, 0.0)

        joints = []
        for idx, (x, y, z, conf) in enumerate(_GOLF_ADDRESS_JOINTS):
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
            swing_phase = math.sin(math.pi * t / (num_frames / 30.0))
            frame_data = self._mock_single_frame(angle="front")
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
