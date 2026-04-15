"""Shared helpers for the ML/CV pipeline."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .skeleton_processing import SkeletonProcessor

# Extra wall-clock seconds added to the decoded timeline before skeleton / exports
# (uniform resample; output FPS stays within processor limits).
USER_UPLOAD_ALIGNMENT_EXTRA_SECONDS = 10.0


def write_alignment_stretched_copy(
    src: Path,
    add_seconds: float,
    *,
    temp_prefix: str = "align_stretch_",
    smooth_temporal_blend: bool = False,
) -> Path:
    """Decode ``src``, resample frames so duration increases by ``add_seconds``, write temp MP4.

    Output container FPS is ``src`` FPS clamped to
    ``[SkeletonProcessor.MIN_ALLOWED_FPS, SkeletonProcessor.MAX_ALLOWED_FPS]``.
    Frame content is nearest-neighbour–sampled along the original timeline unless
    ``smooth_temporal_blend`` is True (linear blend between neighbour frames), used
    for overhead / head-down clips to reduce stutter from duplicated timeline steps.

    Returns:
        Path to a new temporary ``.mp4`` (caller must delete).

    Raises:
        ValueError: If ``add_seconds`` is not positive, the file cannot be opened,
        no frames are decoded, or the writer cannot be created.
    """
    if add_seconds <= 1e-9:
        raise ValueError("add_seconds must be positive")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    max_sec = SkeletonProcessor.MAX_DURATION_SECONDS
    max_frames = int(max_sec * max(fps, 60.0)) + 120

    frames: list = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from {src}")

    h, w = frames[0].shape[:2]
    duration = len(frames) / max(fps, 1e-6)

    min_f = SkeletonProcessor.MIN_ALLOWED_FPS
    max_f = SkeletonProcessor.MAX_ALLOWED_FPS
    out_fps = max(min_f, min(max_f, fps))

    new_duration = duration + add_seconds
    n_out = max(len(frames), int(round(new_duration * out_fps)))

    fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix=temp_prefix)
    os.close(fd)
    out_p = Path(tmp)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, out_fps, (w, h))
    if not writer.isOpened():
        out_p.unlink(missing_ok=True)
        raise ValueError(f"Cannot create alignment-stretched video for {src}")

    n_src = len(frames)
    if not smooth_temporal_blend:
        for k in range(n_out):
            if n_out <= 1 or n_src <= 1:
                idx = 0
            else:
                idx = int(round(k * (n_src - 1) / (n_out - 1)))
            idx = max(0, min(n_src - 1, idx))
            writer.write(frames[idx])
    else:
        denom = max(n_out - 1, 1)
        for k in range(n_out):
            if n_src <= 1:
                writer.write(frames[0])
                continue
            t = k * (n_src - 1) / denom
            i0 = int(np.floor(t))
            i1 = min(i0 + 1, n_src - 1)
            alpha = float(t - i0)
            if i0 == i1 or alpha < 1e-6:
                writer.write(frames[i0])
            else:
                blended = cv2.addWeighted(
                    frames[i0], 1.0 - alpha, frames[i1], alpha, 0.0
                )
                writer.write(blended)

    writer.release()
    return out_p


def _prepare_video_transcode_if_needed(src: Path, target_fps: float) -> tuple[Path, bool]:
    """Return ``(path, is_temp)`` for skeleton pipeline (FPS / duration normalization only)."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (nframes / fps) if fps > 1e-6 and nframes > 0 else 0.0

    min_f = SkeletonProcessor.MIN_ALLOWED_FPS
    max_f = SkeletonProcessor.MAX_ALLOWED_FPS
    max_sec = SkeletonProcessor.MAX_DURATION_SECONDS

    fps_ok = min_f <= fps <= max_f
    short_enough = duration <= max_sec + 1e-3

    if fps_ok and short_enough:
        cap.release()
        return src, False

    max_read = max(1, int(max_sec * fps))
    if fps_ok:
        out_fps = fps
        step = 1
    else:
        out_fps = target_fps
        step = max(1, int(round(fps / out_fps)))

    fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="pipe_prep_")
    os.close(fd)
    out_p = Path(tmp)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, out_fps, (w, h))
    if not writer.isOpened():
        cap.release()
        out_p.unlink(missing_ok=True)
        raise ValueError(f"Cannot create temp video for {src}")

    i = 0
    while i < max_read:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            writer.write(frame)
        i += 1
    cap.release()
    writer.release()
    return out_p, True


def prepare_video_for_skeleton_pipeline(
    src: Path,
    target_fps: float = 30.0,
    *,
    alignment_add_seconds: float = USER_UPLOAD_ALIGNMENT_EXTRA_SECONDS,
    head_overhead_view: bool = False,
) -> tuple[Path, bool]:
    """Return ``(path, is_temp)`` for use with ``SkeletonProcessor.process_video``.

    By default, stretches the decoded timeline by ``USER_UPLOAD_ALIGNMENT_EXTRA_SECONDS``
    (uniform resample) before FPS/duration normalization so skeleton / exports see
    the slower clip. Pass ``alignment_add_seconds=0`` to skip (e.g. fast A/B checks).

    When ``head_overhead_view`` is True (overhead / head-down camera), the stretch
    step uses temporal blending between frames for smoother motion on the lengthened
    timeline. Pass the same flag into ``SkeletonProcessor.process_video`` for
    matching arm smoothing.

    When stretching applies, the returned path is usually a temp file: delete it when
    ``is_temp`` is true. With ``alignment_add_seconds=0``, if the source already has
    allowed FPS and duration, returns ``(src, False)``; otherwise returns a transcode
    temp as before.
    """
    stretch_path: Path | None = None
    try:
        if alignment_add_seconds > 1e-9:
            stretch_path = write_alignment_stretched_copy(
                src,
                alignment_add_seconds,
                temp_prefix="align_stretch_",
                smooth_temporal_blend=bool(head_overhead_view),
            )
            work_src = stretch_path
        else:
            work_src = src

        work, needs_cleanup = _prepare_video_transcode_if_needed(work_src, target_fps)

        if stretch_path is None:
            return work, needs_cleanup

        if needs_cleanup and work != stretch_path:
            stretch_path.unlink(missing_ok=True)
            return work, True

        # Stretched file passed FPS/duration checks (or was the only artifact).
        return stretch_path, True
    except Exception:
        if stretch_path is not None:
            stretch_path.unlink(missing_ok=True)
        raise
