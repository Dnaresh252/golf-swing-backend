"""Integration tests: multi-angle image pipeline on bundled sample JPEGs."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_cv_engine.avatar_3d import AvatarGenerator
from ml_cv_engine.skeleton_processing import SkeletonProcessor

_SAMPLE_DIR = Path(__file__).resolve().parent / "sample_images"
_REQUIRED = ("front", "left", "right", "back")


def _image_paths() -> dict[str, str]:
    paths: dict[str, str] = {}
    for angle in _REQUIRED:
        for ext in (".jpg", ".jpeg", ".png"):
            p = _SAMPLE_DIR / f"{angle}{ext}"
            if p.is_file():
                paths[angle] = str(p)
                break
    return paths


def test_sample_images_bundle_present() -> None:
    paths = _image_paths()
    missing = [a for a in _REQUIRED if a not in paths]
    assert not missing, (
        f"Missing sample_images for: {', '.join(missing)} "
        f"(expected {_SAMPLE_DIR}/<angle>.jpg)"
    )


@pytest.mark.skipif(
    len(_image_paths()) < len(_REQUIRED),
    reason="sample_images not complete; run scripts/build_handoff_sample_outputs.py",
)
def test_process_images_sample_jpegs() -> None:
    paths = _image_paths()
    processor = SkeletonProcessor()
    try:
        out = processor.process_images(paths, submission_id="test_sample_images")
    finally:
        processor.close()

    assert out.get("submission_id") == "test_sample_images"
    angles = out.get("angles") or {}
    assert set(angles.keys()) == set(_REQUIRED)
    # At least one angle should detect a golfer (same source frame used for all in demo bundle).
    joint_counts = []
    for angle in _REQUIRED:
        block = angles.get(angle)
        if block is None:
            joint_counts.append(0)
            continue
        joints = block.get("joints") or []
        joint_counts.append(len(joints))
    assert max(joint_counts) > 0, "No pose detected on any sample image angle"
    for angle in _REQUIRED:
        block = angles[angle]
        if block is None:
            continue
        joints = block.get("joints") or []
        if not joints:
            continue
        assert len(joints) <= 33
        seen: set[int] = set()
        for j in joints:
            jid = int(j["id"])
            assert 0 <= jid <= 32
            assert jid not in seen
            seen.add(jid)


def _front_path() -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = _SAMPLE_DIR / f"front{ext}"
        if p.is_file():
            return p
    return None


@pytest.mark.skipif(
    _front_path() is None,
    reason="sample front image missing; run scripts/build_handoff_sample_outputs.py",
)
def test_process_images_single_front_builds_avatar() -> None:
    """Subset of angles: one still is enough for mesh export (views mirrored internally)."""
    p = _front_path()
    processor = SkeletonProcessor()
    try:
        out = processor.process_images({"front": str(p)}, submission_id="one_angle")
    finally:
        processor.close()
    angles = out.get("angles") or {}
    assert set(angles.keys()) == {"front"}
    mesh = AvatarGenerator().generate_from_multi_angle(out)
    assert not mesh.is_empty
