"""Smoke tests for coach label drawing on BGR frames."""

from __future__ import annotations

import numpy as np

from ml_cv_engine.coach_overlay_labels import draw_static_labels_on_frame


def test_draw_numbered_circle_smoke() -> None:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    draw_static_labels_on_frame(
        img,
        [{"style": "numbered_circle", "x": 200, "y": 180, "main": "1", "sub": "test"}],
    )
    assert img.mean() > 0.1


def test_draw_headline_smoke() -> None:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    draw_static_labels_on_frame(
        img,
        [{"style": "headline", "x": 320, "y": 80, "text": "demo headline"}],
    )
    assert img.mean() > 0.1


def test_empty_labels_noop() -> None:
    img = np.ones((100, 100, 3), dtype=np.uint8) * 40
    before = img.copy()
    draw_static_labels_on_frame(img, [])
    assert np.array_equal(img, before)
