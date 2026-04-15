"""Heuristic detection of overhead / head-down (top) camera golf footage.

Uses sparse MediaPipe pose samples on the source clip. When ``True``, callers
should prefer smooth temporal resampling during alignment stretch and slightly
wider arm smoothing in ``SkeletonProcessor.process_video``.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


def _joints_indicate_head_overhead(joints: List[dict]) -> bool:
    """Return True when shoulder–hip foreshortening matches a top-down view."""
    if not joints:
        return False
    by_id = {int(j["id"]): j for j in joints if "id" in j}
    need = (11, 12, 23, 24)
    if not all(i in by_id for i in need):
        return False
    for i in need:
        if float(by_id[i].get("confidence", 0.0)) < 0.22:
            return False

    ls, rs, lh, rh = by_id[11], by_id[12], by_id[23], by_id[24]
    sy = (float(ls["y"]) + float(rs["y"])) * 0.5
    hy = (float(lh["y"]) + float(rh["y"])) * 0.5
    shoulder_w = abs(float(ls["x"]) - float(rs["x"])) + 1e-6
    vertical_sep = abs(sy - hy)
    ratio = vertical_sep / shoulder_w
    # Top-down: shoulders read wide; shoulder–hip stack is compressed in image y.
    if shoulder_w < 0.065:
        return False
    return bool(ratio < 0.52 and vertical_sep < 0.32)


def detect_head_overhead_view_video(video_path: str, *, max_samples: int = 9) -> bool:
    """Sample frames from ``video_path`` and vote on overhead geometry.

    Returns:
        ``True`` if a majority of confidently-detected samples look overhead.
        ``False`` on open failure, very short clips, or ambiguous pose.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if n < 10:
            return False
        idxs = np.linspace(0, n - 1, num=min(max_samples, n), dtype=int)
        from .pose_detection import PoseDetector

        det = PoseDetector(min_confidence=0.12, static_image_mode=True, model_complexity=1)
        try:
            yes = 0
            no = 0
            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                pose = det.detect_from_frame(frame)
                joints = pose["joints"] if pose is not None else []
                if not joints:
                    continue
                if _joints_indicate_head_overhead(joints):
                    yes += 1
                else:
                    no += 1
        finally:
            det.close()
    finally:
        cap.release()

    total = yes + no
    if total < 3:
        return False
    return yes >= no and yes >= max(2, int(round(0.45 * total)))
