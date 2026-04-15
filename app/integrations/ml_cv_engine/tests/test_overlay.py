from __future__ import annotations

import cv2
import numpy as np
import pytest

from ml_cv_engine.overlay_2d import SkeletonOverlay


@pytest.fixture
def dummy_image_file(tmp_path):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    input_path = tmp_path / "input.png"
    ok = cv2.imwrite(str(input_path), image)
    assert ok
    return str(input_path)


@pytest.fixture
def mock_joints():
    return [
        {"id": 11, "name": "left_shoulder", "x": 0.35, "y": 0.20, "z": 0.10, "confidence": 0.98},
        {"id": 12, "name": "right_shoulder", "x": 0.65, "y": 0.20, "z": 0.10, "confidence": 0.98},
        {"id": 23, "name": "left_hip", "x": 0.42, "y": 0.50, "z": 0.10, "confidence": 0.98},
        {"id": 24, "name": "right_hip", "x": 0.58, "y": 0.50, "z": 0.10, "confidence": 0.98},
    ]


def test_draw_overlay_creates_output_png_and_preserves_dimensions(
    dummy_image_file, mock_joints, tmp_path
):
    overlay = SkeletonOverlay()
    output_path = tmp_path / "overlay.png"

    returned = overlay.draw_overlay(dummy_image_file, mock_joints, str(output_path))
    assert returned == str(output_path)
    assert output_path.exists()

    original = cv2.imread(dummy_image_file)
    rendered = cv2.imread(str(output_path))
    assert original is not None
    assert rendered is not None
    assert rendered.shape[:2] == original.shape[:2]
