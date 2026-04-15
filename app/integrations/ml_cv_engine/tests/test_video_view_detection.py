"""Tests for overhead / head-down view heuristics."""

from __future__ import annotations

from ml_cv_engine import video_view_detection as vvd


def _j(
    jid: int, x: float, y: float, conf: float = 0.95
) -> dict:
    return {"id": jid, "name": str(jid), "x": x, "y": y, "z": 0.0, "confidence": conf}


def test_overhead_heuristic_true_when_torso_compressed() -> None:
    """Wide shoulders vs small shoulder–hip vertical gap → overhead-like."""
    joints = [
        _j(11, 0.35, 0.42),
        _j(12, 0.65, 0.42),
        _j(23, 0.38, 0.48),
        _j(24, 0.62, 0.48),
    ]
    assert vvd._joints_indicate_head_overhead(joints) is True


def test_overhead_heuristic_false_for_typical_frontal_torso() -> None:
    joints = [
        _j(11, 0.40, 0.28),
        _j(12, 0.60, 0.28),
        _j(23, 0.42, 0.62),
        _j(24, 0.58, 0.62),
    ]
    assert vvd._joints_indicate_head_overhead(joints) is False


def test_overhead_heuristic_false_when_shoulders_missing() -> None:
    assert vvd._joints_indicate_head_overhead([]) is False
