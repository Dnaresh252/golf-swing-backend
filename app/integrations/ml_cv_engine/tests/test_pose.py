from __future__ import annotations

import cv2
import numpy as np
import pytest

from ml_cv_engine.pose_detection import PoseDetector


@pytest.fixture
def dummy_image_path(tmp_path):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    path = tmp_path / "dummy.png"
    ok = cv2.imwrite(str(path), image)
    assert ok
    return str(path)


def test_detect_from_image_returns_none_on_black_image(dummy_image_path):
    detector = PoseDetector()
    result = detector.detect_from_image(dummy_image_path)
    assert result is None or "joints" in result


def test_detect_from_image_has_joints_key_with_mocked_result(dummy_image_path, monkeypatch):
    detector = PoseDetector()

    def fake_detect_from_frame(_frame):
        return {"joints": [{"id": 11, "name": "left_shoulder", "x": 0.4, "y": 0.5, "z": 0.0, "confidence": 0.9}]}

    monkeypatch.setattr(detector, "detect_from_frame", fake_detect_from_frame)
    result = detector.detect_from_image(dummy_image_path)
    assert result is not None
    assert "joints" in result


def test_all_landmarks_emitted_with_confidence(monkeypatch):
    """All 33 landmarks are emitted regardless of visibility; confidence is preserved."""
    detector = PoseDetector(min_confidence=0.5)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    class FakeLandmark:
        def __init__(self, x, y, z, visibility):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility

    class FakePoseLandmarks:
        def __init__(self, landmarks):
            self.landmark = landmarks

    class FakeResult:
        def __init__(self, landmarks):
            self.pose_landmarks = FakePoseLandmarks(landmarks)

    def fake_process(_frame_rgb):
        landmarks = [
            FakeLandmark(0.2, 0.3, 0.0, 0.49),
            FakeLandmark(0.5, 0.4, 0.1, 0.95),
        ]
        landmarks.extend(FakeLandmark(0.0, 0.0, 0.0, 0.0) for _ in range(31))
        return FakeResult(landmarks)

    monkeypatch.setattr(detector._pose, "process", fake_process)
    result = detector.detect_from_frame(frame)
    assert result is not None
    assert "joints" in result
    assert len(result["joints"]) == 33
    by_id = {j["id"]: j for j in result["joints"]}
    assert by_id[0]["confidence"] < 0.5
    assert by_id[1]["confidence"] > 0.9
