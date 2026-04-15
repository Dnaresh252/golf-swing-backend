"""Integration tests: full skeleton pipeline on bundled sample MP4s."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_cv_engine.skeleton_processing import SkeletonProcessor
from ml_cv_engine.utils import prepare_video_for_skeleton_pipeline

_SAMPLE_DIR = Path(__file__).resolve().parent / "sample_videos"
_SAMPLE_MP4S = ("Q.mp4", "QQ.mp4", "Z.mp4", "11.mp4", "111.mp4", "1111.mp4", "22.mp4")


@pytest.mark.parametrize("filename", _SAMPLE_MP4S)
def test_process_video_sample_mp4(filename: str) -> None:
    path = _SAMPLE_DIR / filename
    assert path.is_file(), f"Missing sample video: {path}"

    work, is_temp = prepare_video_for_skeleton_pipeline(path)
    processor = SkeletonProcessor()
    try:
        out = processor.process_video(str(work), submission_id=f"test_{filename}")
    finally:
        processor.close()
        if is_temp:
            work.unlink(missing_ok=True)

    assert out.get("submission_id") == f"test_{filename}"
    frames = out.get("frames") or []
    assert len(frames) > 0
    # Full pose appears at least once; individual frames may have fewer joints
    # after arm outlier rejection when gaps are too long to interpolate.
    joint_counts = [len(frame.get("joints") or []) for frame in frames]
    assert max(joint_counts) == 33
    for frame in frames:
        joints = frame.get("joints") or []
        assert 0 < len(joints) <= 33
        seen: set[int] = set()
        for j in joints:
            jid = int(j["id"])
            assert 0 <= jid <= 32
            assert jid not in seen
            seen.add(jid)
