"""Skeleton processing utilities for video and multi-angle images."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import savgol_filter

from .coach_overlay_labels import draw_static_labels_on_frame
from .golf_analysis import GolfSwingAnalyzer
from .overlay_2d import SkeletonOverlay
from .pose_detection import PoseDetector


class _OneEuroFilter1D:
    """One Euro filter for a single scalar signal.

    Adaptive low-pass: smooth when the signal is slow, responsive when
    it moves fast.  This is the ONLY temporal filter in the pipeline.
    """

    __slots__ = ("_min_cutoff", "_beta", "_d_cutoff", "_x_prev", "_dx_prev", "_initialized")

    def __init__(self, min_cutoff: float = 0.7, beta: float = 0.5, d_cutoff: float = 1.0) -> None:
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_prev = 0.0
        self._dx_prev = 0.0
        self._initialized = False

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(1e-9, cutoff))
        return max(0.0, min(1.0, dt / (dt + tau)))

    def step(self, x: float, dt: float) -> float:
        if not self._initialized:
            self._x_prev = x
            self._dx_prev = 0.0
            self._initialized = True
            return x
        if dt < 1e-9:
            return self._x_prev
        dx = (x - self._x_prev) / dt
        ad = self._alpha(self._d_cutoff, dt)
        dx_hat = ad * dx + (1.0 - ad) * self._dx_prev
        cutoff = self._min_cutoff + self._beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


def _one_euro_params(joint_id: int) -> Tuple[float, float, float]:
    """Per-joint One Euro tuning  (min_cutoff, beta, d_cutoff).

    Uniform smoothing across ALL joints — no special arm treatment.
    Low min_cutoff = heavy smoothing at rest.
    High beta = cutoff rises fast with speed (no lag during swings).
    """
    jid = int(joint_id)
    if jid in range(0, 11):
        return (0.55, 1.2, 1.0)
    if jid in (11, 12):
        return (0.45, 0.9, 1.0)
    if jid in (13, 14):
        return (0.40, 2.5, 1.5)
    if jid in (15, 16):
        return (0.40, 3.0, 1.5)
    if jid in range(17, 23):
        return (0.40, 2.5, 1.5)
    if jid in (23, 24):
        return (0.40, 0.8, 1.0)
    if jid in (25, 26):
        return (0.42, 0.9, 1.0)
    if jid in (27, 28):
        return (0.45, 1.0, 1.0)
    return (0.50, 1.2, 1.0)


# Bone chains: (parent, child) for anatomical length enforcement
_BONE_CHAINS: List[Tuple[int, int]] = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
    (11, 23), (12, 24),
]


class SkeletonProcessor:
    """Process golf media inputs into skeleton JSON structures."""

    MAX_DURATION_SECONDS = 20.0
    MIN_ALLOWED_FPS = 25.0
    MAX_ALLOWED_FPS = 50.0

    def __init__(self, min_confidence: float = 0.05) -> None:
        """Initialize processor with a configured pose detector."""
        self._min_confidence = float(min_confidence)
        self.pose_detector: Optional[PoseDetector] = None
        self._pose_detector_static_mode: Optional[bool] = None
        self.overlay = SkeletonOverlay()
        self.swing_analyzer = GolfSwingAnalyzer()

    def _get_pose_detector(self, static_image_mode: bool = True) -> PoseDetector:
        if (
            self.pose_detector is None
            or self._pose_detector_static_mode != static_image_mode
        ):
            if self.pose_detector is not None:
                self.pose_detector.close()
            self.pose_detector = PoseDetector(
                min_confidence=self._min_confidence,
                static_image_mode=static_image_mode,
                model_complexity=2,
            )
            self._pose_detector_static_mode = static_image_mode
        return self.pose_detector

    def _get_video_detector(self) -> PoseDetector:
        """Get or create a detector configured for video (temporal tracking)."""
        return self._get_pose_detector(static_image_mode=False)

    def close(self) -> None:
        """Release detector resources explicitly."""
        if self.pose_detector is not None:
            self.pose_detector.close()
            self.pose_detector = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Video processing
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_swing_motion_frame_range(
        video_path: str, fps: float, max_seconds: float
    ) -> Tuple[int, int]:
        """Pick an inclusive [start, end] frame index window with peak RGB motion.

        Used to temporally oversample only the swing-like segment so MediaPipe
        video mode sees more steps per real frame. Returns ``(-1, -2)`` when
        no clear motion peak exists (caller should treat as disabled).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (-1, -2)
        motions: List[float] = []
        prev: Optional[np.ndarray] = None
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if (idx / fps) > max_seconds + 1e-6:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev is not None:
                motions.append(float(np.mean(np.abs(small - prev))))
            else:
                motions.append(0.0)
            prev = small
            idx += 1
        cap.release()
        n = len(motions)
        if n < 16:
            return (-1, -2)
        m = np.asarray(motions, dtype=np.float64)
        win = max(3, min(7, (n // 20) | 1))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float64) / float(win)
        ms = np.convolve(m, kernel, mode="same")
        peak = int(np.argmax(ms))
        peak_val = float(ms[peak])
        median_val = float(np.median(ms))
        if peak_val < 1.5 or peak_val < median_val * 1.35:
            return (-1, -2)
        thresh = max(median_val * 1.15, float(np.percentile(ms, 55)))
        left, right = peak, peak
        while left > 0 and ms[left] >= thresh * 0.4:
            left -= 1
        while right < n - 1 and ms[right] >= thresh * 0.4:
            right += 1
        min_span = max(10, int(fps * 0.35))
        if right - left + 1 < min_span:
            pad = (min_span - (right - left + 1)) // 2
            left = max(0, left - pad)
            right = min(n - 1, right + pad)
        max_span = min(int(fps * 2.8), max(min_span, int(n * 0.5)))
        if right - left + 1 > max_span:
            c = (left + right) // 2
            half = max_span // 2
            left = max(0, c - half)
            right = min(n - 1, left + max_span - 1)
        return (int(left), int(right))

    def process_video(
        self,
        video_path: str,
        submission_id: str,
        *,
        swing_motion_slowdown: bool = False,
        head_overhead_view: bool = False,
    ) -> dict:
        """Process an MP4 video into frame-wise skeleton data.

        Args:
            video_path: Path to MP4 video input.
            submission_id: Submission identifier to embed in output.
            swing_motion_slowdown: If True, run pose on a temporally expanded
                window around the RGB motion peak (duplicate frames there) so
                MediaPipe can better track fast arms, then collapse back to one
                joint set per original frame for exports and analysis.
            head_overhead_view: If True (overhead / head-down camera), apply slightly
                wider arm Savitzky–Golay smoothing after detection for steadier joints.

        Returns:
            Skeleton data in format:
            {
              "submission_id": "<id>",
              "frames": [
                {
                  "frame_num": 0,
                  "timestamp": 0.0,
                  "joints": [...]
                }
              ]
            }

        Raises:
            ValueError: For invalid file type, unsupported fps, unreadable video,
                or duration greater than ``MAX_DURATION_SECONDS``.
        """
        if not video_path.lower().endswith(".mp4"):
            raise ValueError("Invalid video format: only MP4 files are supported.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            cap.release()
            raise ValueError("Invalid video FPS: unable to read frame rate.")

        is_allowed_fps = self.MIN_ALLOWED_FPS <= fps <= self.MAX_ALLOWED_FPS
        if not is_allowed_fps:
            cap.release()
            raise ValueError(
                "Unsupported FPS: expected range "
                f"{self.MIN_ALLOWED_FPS:.2f}-{self.MAX_ALLOWED_FPS:.2f}, got {fps:.2f}."
            )

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if frame_count > 0 else 0.0
        if duration > self.MAX_DURATION_SECONDS:
            cap.release()
            raise ValueError(
                f"Video too long: maximum {self.MAX_DURATION_SECONDS:.0f} seconds allowed."
            )

        slow_lo, slow_hi = (-1, -2)
        if swing_motion_slowdown:
            slow_lo, slow_hi = self._estimate_swing_motion_frame_range(
                video_path, fps, self.MAX_DURATION_SECONDS
            )

        detector = self._get_video_detector()
        detector.set_video_fps(fps)
        detector.reset_video_stream()
        output: Dict[str, object] = {
            "submission_id": submission_id,
            "frames": [],
            "video_view": {"head_overhead": bool(head_overhead_view)},
        }

        use_slow = swing_motion_slowdown and slow_hi >= slow_lo

        if not use_slow:
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp = frame_num / fps
                if timestamp > self.MAX_DURATION_SECONDS:
                    break

                pose = detector.detect_from_frame(frame)
                joints = pose["joints"] if pose is not None else []
                output["frames"].append(
                    {
                        "frame_num": frame_num,
                        "timestamp": float(round(timestamp, 6)),
                        "joints": joints,
                    }
                )
                frame_num += 1
            cap.release()
        else:
            cap.release()
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Unable to open video file: {video_path}")

            repeat = 3 if fps >= 48.0 else 2
            raw_by_source: Dict[int, List[dict]] = {}
            source_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if (source_idx / fps) > self.MAX_DURATION_SECONDS:
                    break

                reps = repeat if slow_lo <= source_idx <= slow_hi else 1
                for _ in range(reps):
                    pose = detector.detect_from_frame(frame)
                    joints = pose["joints"] if pose is not None else []
                    raw_by_source.setdefault(source_idx, []).append(joints)
                source_idx += 1

            cap.release()

            def _pick_joints(candidates: List[List[dict]]) -> List[dict]:
                for joints in reversed(candidates):
                    if joints:
                        return joints
                return []

            for i in range(source_idx):
                joints = _pick_joints(raw_by_source.get(i, [[]]))
                output["frames"].append(
                    {
                        "frame_num": i,
                        "timestamp": float(round(i / fps, 6)),
                        "joints": joints,
                    }
                )

        output["frames"] = self._interpolate_missing_joints(output["frames"], max_gap_frames=22)
        output["frames"] = SkeletonProcessor._reject_arm_outliers(output["frames"], fps)
        arm_sg_scale = 1.42 if head_overhead_view else 1.0
        output["frames"] = self._smooth_one_euro(
            output["frames"], fps, arm_sg_scale=arm_sg_scale
        )
        output["frames"] = SkeletonProcessor._enforce_anatomy(output["frames"], fps)
        output["frames"] = SkeletonProcessor._snap_arms_two_link(output["frames"], fps)
        return output

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------

    def process_images(self, image_paths: dict, submission_id: str) -> dict:
        """Process multi-angle images into skeleton data.

        Args:
            image_paths: Dict with any subset of angle keys:
                ``front``, ``left``, ``right``, ``back``, ``top`` (each a path string).
                At least **one** path is required. Missing views are inferred when
                building a 3D mesh (see ``AvatarGenerator.generate_from_multi_angle``).
            submission_id: Submission identifier to embed in output.

        Returns:
            {
              "submission_id": "<id>",
              "angles": {
                "front": {"joints": [...]} | null,
                ...
              }
            }
            Only requested angles appear in ``angles``. Any angle with no detected
            pose returns ``None`` for that key.
        """
        allowed = ("front", "left", "right", "back", "top")
        chosen = [
            a
            for a in allowed
            if isinstance(image_paths.get(a), str) and str(image_paths[a]).strip()
        ]
        if not chosen:
            raise ValueError(
                "Provide at least one image path under keys: "
                "front, left, right, back, or top."
            )

        result = {"submission_id": submission_id, "angles": {}}
        det = self._get_pose_detector(static_image_mode=True)
        for angle in chosen:
            result["angles"][angle] = det.detect_from_image(str(image_paths[angle]).strip())
        return result

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_skeleton_json(self, skeleton_data: dict, output_path: str) -> str:
        """Export skeleton data to a JSON file and return saved path."""
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(skeleton_data, f, indent=2)

        return output_path

    # ------------------------------------------------------------------
    # Smoothing — Savitzky-Golay for arms, One Euro for all other joints
    # ------------------------------------------------------------------

    # Arm joint IDs that get batch SG smoothing (no causal lag)
    _ARM_JOINT_IDS_SET = frozenset(range(11, 23))

    @staticmethod
    def _smooth_one_euro(
        frames: List[dict], fps: float, *, arm_sg_scale: float = 1.0
    ) -> List[dict]:
        """Smooth all 33 joints with the best filter for each group.

        Arms (11-22): Savitzky-Golay polynomial batch smoother.
            - Non-causal: uses future frames too → zero phase lag.
            - Preserves fast swing transients exactly.
            - Window adapts to clip length (min 5, max 11 frames, odd).
        All others: One Euro adaptive causal filter as before.

        ``arm_sg_scale`` widens the arm SG window (e.g. 1.42 for overhead head view).
        """
        if not frames or fps <= 0:
            return frames

        n = len(frames)
        dt = 1.0 / fps

        frame_maps: List[Dict[int, dict]] = [
            {int(j["id"]): j for j in f.get("joints", []) if "id" in j}
            for f in frames
        ]
        all_ids: set = set()
        for m in frame_maps:
            all_ids.update(m.keys())

        # ── Savitzky-Golay on arm joints ──────────────────────────────
        _ARM = SkeletonProcessor._ARM_JOINT_IDS_SET
        # ~0.18s window damps arm jitter on export; capped for short clips.
        scale = max(0.5, min(2.5, float(arm_sg_scale)))
        sg_win_raw = max(5, int(round(fps * 0.18 * scale)))
        sg_win = sg_win_raw if sg_win_raw % 2 == 1 else sg_win_raw + 1
        sg_win = min(sg_win, n if n % 2 == 1 else n - 1)
        sg_win = min(sg_win, 17)  # odd cap; wider than before for steadier bones
        if sg_win >= 3 and sg_win % 2 == 0:
            sg_win -= 1
        sg_poly = min(2, max(1, sg_win - 1))

        for jid in sorted(all_ids):
            if jid not in _ARM:
                continue
            # Gather full x/y/z trajectory (None where joint missing)
            xs = [None] * n
            ys = [None] * n
            zs = [None] * n
            for i in range(n):
                j = frame_maps[i].get(jid)
                if j is not None:
                    xs[i] = float(j["x"])
                    ys[i] = float(j["y"])
                    zs[i] = float(j.get("z", 0.0))

            # Linear-interpolate any None gaps before filtering
            def _fill(arr: List) -> np.ndarray:
                a = np.array([v if v is not None else float("nan") for v in arr])
                nans = np.isnan(a)
                if nans.all():
                    return a
                idx = np.arange(n)
                a[nans] = np.interp(idx[nans], idx[~nans], a[~nans])
                return a

            ax = _fill(xs)
            ay = _fill(ys)
            az = _fill(zs)

            if sg_win >= 3 and n >= sg_win:
                ax = savgol_filter(ax, sg_win, sg_poly, mode="nearest")
                ay = savgol_filter(ay, sg_win, sg_poly, mode="nearest")
                # z: use same window/poly as x/y (simpler and avoids edge cases)
                az = savgol_filter(az, sg_win, sg_poly, mode="nearest")

            for i in range(n):
                j = frame_maps[i].get(jid)
                if j is not None and xs[i] is not None:
                    j["x"] = float(np.clip(ax[i], 0.0, 1.0))
                    j["y"] = float(np.clip(ay[i], 0.0, 1.0))
                    j["z"] = float(az[i])

        # ── One Euro for all non-arm joints ───────────────────────────
        for jid in sorted(all_ids):
            if jid in _ARM:
                continue
            mc, beta, dc = _one_euro_params(jid)
            fx = _OneEuroFilter1D(min_cutoff=mc, beta=beta, d_cutoff=dc)
            fy = _OneEuroFilter1D(min_cutoff=mc, beta=beta, d_cutoff=dc)
            fz = _OneEuroFilter1D(min_cutoff=mc, beta=beta, d_cutoff=dc)
            for i in range(n):
                j = frame_maps[i].get(jid)
                if j is None:
                    continue
                conf = float(j.get("confidence", 0.0))
                rx = float(j.get("x", 0.0))
                ry = float(j.get("y", 0.0))
                rz = float(j.get("z", 0.0))
                if conf >= 0.12:
                    sx = fx.step(rx, dt)
                    sy = fy.step(ry, dt)
                    sz = fz.step(rz, dt)
                elif fx._initialized:
                    sx = fx._x_prev
                    sy = fy._x_prev
                    sz = fz._x_prev
                else:
                    sx, sy, sz = rx, ry, rz
                j["x"] = max(0.0, min(1.0, sx))
                j["y"] = max(0.0, min(1.0, sy))
                j["z"] = sz

        out: List[dict] = []
        for i, frame in enumerate(frames):
            nf = dict(frame)
            nf["joints"] = list(frame_maps[i].values())
            out.append(nf)
        return out

    # ------------------------------------------------------------------
    # Arm outlier rejection — the KEY step for fast-swing accuracy
    # ------------------------------------------------------------------

    @staticmethod
    def _reject_arm_outliers(frames: List[dict], fps: float) -> List[dict]:
        """Detect and remove bad arm detections, then interpolate through them.

        MediaPipe sometimes returns wrong arm positions during swings. This
        pass flags extreme outliers only — thresholds stay loose enough that
        legitimate fast golf motion is not stripped (which caused missing arms).
        Bad frames have arm joints removed so interpolation can smooth gaps.
        """
        if len(frames) < 5 or fps <= 0:
            return frames

        ARM_JOINT_IDS = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        maps: List[Dict[int, dict]] = [
            {int(j["id"]): dict(j) for j in f.get("joints", []) if "id" in j}
            for f in frames
        ]
        n = len(frames)
        body_scales = SkeletonProcessor._body_envelope_scales(maps, n)

        for jid in ARM_JOINT_IDS:
            xs: List[Optional[float]] = [None] * n
            ys: List[Optional[float]] = [None] * n
            for i in range(n):
                j = maps[i].get(jid)
                if j is not None:
                    xs[i] = float(j["x"])
                    ys[i] = float(j["y"])

            velocities: List[Optional[float]] = [None] * n
            for i in range(1, n):
                if xs[i] is not None and xs[i - 1] is not None:
                    velocities[i] = math.hypot(
                        float(xs[i]) - float(xs[i - 1]),
                        float(ys[i]) - float(ys[i - 1]),
                    )

            # Running median velocity over a 7-frame window
            half_w = 3
            median_vel: List[float] = [0.0] * n
            for i in range(n):
                window = []
                for k in range(max(0, i - half_w), min(n, i + half_w + 1)):
                    if velocities[k] is not None:
                        window.append(float(velocities[k]))
                if window:
                    window.sort()
                    median_vel[i] = window[len(window) // 2]

            # Shoulder anchor for this joint
            shoulder_id = 11 if jid in (13, 15, 17, 19, 21) else 12

            bad_frames: List[int] = []
            for i in range(1, n):
                if xs[i] is None:
                    continue
                bs = body_scales[i]
                is_bad = False

                # Check 1: velocity spike — only extreme spikes vs median (golf-safe)
                v = velocities[i]
                mv = median_vel[i]
                if v is not None and mv > 0.001:
                    if v > max(mv * 6.0, 0.045 * bs):
                        is_bad = True

                # Check 2: distance from shoulder — allow full extension in swing
                sj = maps[i].get(shoulder_id)
                if sj is not None:
                    sd = math.hypot(
                        float(xs[i]) - float(sj["x"]),
                        float(ys[i]) - float(sj["y"]),
                    )
                    hip_id = 23 if shoulder_id == 11 else 24
                    hj = maps[i].get(hip_id)
                    if hj is not None:
                        torso_len = math.hypot(
                            float(sj["x"]) - float(hj["x"]),
                            float(sj["y"]) - float(hj["y"]),
                        )
                        if torso_len > 0.02 and sd > torso_len * 2.35:
                            is_bad = True

                # Check 3: sharp reversal only (not normal swing direction changes)
                if i >= 2 and xs[i - 1] is not None and xs[i - 2] is not None:
                    dx1 = float(xs[i - 1]) - float(xs[i - 2])
                    dy1 = float(ys[i - 1]) - float(ys[i - 2])
                    dx2 = float(xs[i]) - float(xs[i - 1])
                    dy2 = float(ys[i]) - float(ys[i - 1])
                    d1 = math.hypot(dx1, dy1)
                    d2 = math.hypot(dx2, dy2)
                    if d1 > 0.012 * bs and d2 > 0.012 * bs:
                        dot = dx1 * dx2 + dy1 * dy2
                        cos_angle = dot / (d1 * d2)
                        if cos_angle < -0.82:
                            is_bad = True

                if is_bad:
                    bad_frames.append(i)

            # Remove bad detections — let interpolation fill them
            for i in bad_frames:
                if jid in maps[i]:
                    del maps[i][jid]

        # Rebuild frames with cleaned joint maps
        out: List[dict] = []
        for i, f in enumerate(frames):
            nf = dict(f)
            nf["joints"] = list(maps[i].values())
            out.append(nf)

        # Re-run interpolation to fill the gaps we just created
        out = SkeletonProcessor._interpolate_missing_joints(out, max_gap_frames=36)
        return out

    # ------------------------------------------------------------------
    # Interpolation (fill short detection gaps for ALL body joints)
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_missing_joints(frames: List[dict], max_gap_frames: int = 12) -> List[dict]:
        """Interpolate missing joints across short detector dropouts.

        Covers all 33 MediaPipe landmarks (0-32) so that occluded limbs
        (e.g. the trailing arm during a golf follow-through) are filled
        in rather than vanishing for several frames.
        """
        if not frames:
            return frames

        maps = [{int(j["id"]): j for j in f.get("joints", []) if "id" in j} for f in frames]
        n = len(frames)

        for jid in range(0, 33):
            present = [i for i in range(n) if jid in maps[i]]
            if len(present) < 2:
                continue
            for a, b in zip(present[:-1], present[1:]):
                gap = b - a - 1
                if gap <= 0 or gap > int(max_gap_frames):
                    continue
                ja = maps[a][jid]
                jb = maps[b][jid]
                for k in range(1, gap + 1):
                    idx = a + k
                    if jid in maps[idx]:
                        continue
                    t = k / float(gap + 1)
                    maps[idx][jid] = {
                        "id": jid,
                        "name": ja.get("name", jb.get("name", str(jid))),
                        "x": (1.0 - t) * float(ja.get("x", 0.0)) + t * float(jb.get("x", 0.0)),
                        "y": (1.0 - t) * float(ja.get("y", 0.0)) + t * float(jb.get("y", 0.0)),
                        "z": (1.0 - t) * float(ja.get("z", 0.0)) + t * float(jb.get("z", 0.0)),
                        "confidence": min(
                            float(ja.get("confidence", 0.0)),
                            float(jb.get("confidence", 0.0)),
                        ) * 0.85,
                    }

        out = []
        for i, f in enumerate(frames):
            nf = dict(f)
            nf["joints"] = list(maps[i].values())
            out.append(nf)
        return out

    # ------------------------------------------------------------------
    # Anatomy enforcement — ONE clean pass for ALL joints
    # ------------------------------------------------------------------

    @staticmethod
    def _enforce_anatomy(frames: List[dict], fps: float) -> List[dict]:
        """Enforce bone-length stability and per-frame velocity limits for ALL joints.

        Uses EMA bone lengths (learned over the clip) to clamp segments that
        stretch/shrink implausibly, and caps per-frame velocity so no joint
        can teleport between consecutive frames.
        """
        if len(frames) < 2 or fps <= 0:
            return frames

        maps: List[Dict[int, dict]] = [
            {int(j["id"]): dict(j) for j in f.get("joints", []) if "id" in j}
            for f in frames
        ]
        n = len(frames)
        body_scales = SkeletonProcessor._body_envelope_scales(maps, n)

        # Velocity caps — arm joints (11-22) are intentionally uncapped so
        # fast golf swings are not artificially slowed down by this filter.
        # Only stable joints (face, hips, legs) are capped to prevent jitter.
        _NO_CAP = frozenset(range(11, 23))  # shoulders, elbows, wrists, hands
        max_vel: Dict[int, float] = {}
        for jid in range(0, 11):
            max_vel[jid] = 0.018
        for jid in (23, 24):
            max_vel[jid] = 0.014
        for jid in (25, 26):
            max_vel[jid] = 0.018
        for jid in (27, 28):
            max_vel[jid] = 0.022
        for jid in range(29, 33):
            max_vel[jid] = 0.024

        for i in range(1, n):
            bs = body_scales[i]
            prev_m, curr_m = maps[i - 1], maps[i]
            for jid in range(0, 33):
                if jid in _NO_CAP:
                    continue  # arms/hands: no velocity capping
                cj = curr_m.get(jid)
                pj = prev_m.get(jid)
                if cj is None or pj is None:
                    continue
                mv = max_vel.get(jid, 0.028) * bs
                cx, cy = float(cj["x"]), float(cj["y"])
                px, py = float(pj["x"]), float(pj["y"])
                dx, dy = cx - px, cy - py
                d = math.hypot(dx, dy)
                if d > mv and d > 1e-9:
                    s = mv / d
                    cj["x"] = px + dx * s
                    cj["y"] = py + dy * s
                    cz = float(cj.get("z", 0.0))
                    pz = float(pj.get("z", 0.0))
                    cj["z"] = pz + (cz - pz) * s

        # Arm reach check: clamp elbows/wrists that fly far from torso center
        _ARM_IDS = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22}
        for i in range(n):
            bs = body_scales[i]
            m = maps[i]
            ls = m.get(11)
            rs = m.get(12)
            lh = m.get(23)
            rh = m.get(24)
            if ls is None or rs is None or lh is None or rh is None:
                continue
            tcx = (float(ls["x"]) + float(rs["x"]) + float(lh["x"]) + float(rh["x"])) * 0.25
            tcy = (float(ls["y"]) + float(rs["y"]) + float(lh["y"]) + float(rh["y"])) * 0.25
            torso_h = max(
                math.hypot(float(ls["x"]) - float(lh["x"]), float(ls["y"]) - float(lh["y"])),
                math.hypot(float(rs["x"]) - float(rh["x"]), float(rs["y"]) - float(rh["y"])),
                0.05,
            )
            # Golf swings extend arms well past 2× torso from mid-torso; tighter
            # values pulled limbs inward and looked like “missing” arm alignment.
            max_reach = torso_h * 3.1
            for jid in _ARM_IDS:
                j = m.get(jid)
                if j is None:
                    continue
                jx, jy = float(j["x"]), float(j["y"])
                d = math.hypot(jx - tcx, jy - tcy)
                if d > max_reach and d > 1e-9:
                    s = max_reach / d
                    j["x"] = tcx + (jx - tcx) * s
                    j["y"] = tcy + (jy - tcy) * s

        ema_len: Dict[Tuple[int, int], float] = {}
        alpha_bone = 0.15
        for i in range(n):
            bs = body_scales[i]
            m = maps[i]
            for parent_id, child_id in _BONE_CHAINS:
                pj = m.get(parent_id)
                cj = m.get(child_id)
                if pj is None or cj is None:
                    continue
                px, py = float(pj["x"]), float(pj["y"])
                cx, cy = float(cj["x"]), float(cj["y"])
                L = math.hypot(cx - px, cy - py)
                if L < 0.003 * bs:
                    continue

                key = (parent_id, child_id)
                if key not in ema_len:
                    ema_len[key] = L
                else:
                    prev_L = ema_len[key]
                    if 0.45 * prev_L <= L <= 1.65 * prev_L:
                        ema_len[key] = (1.0 - alpha_bone) * prev_L + alpha_bone * L

                target_L = ema_len[key]
                # Arm chains (shoulder→hand) stretch more in a swing.
                _arm_bone = (
                    parent_id in (11, 12, 13, 14, 15, 16)
                    and child_id in (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
                )
                if _arm_bone:
                    lo, hi = target_L * 0.48, target_L * 1.72
                else:
                    lo, hi = target_L * 0.55, target_L * 1.50
                if lo <= L <= hi:
                    continue
                new_L = min(max(L, lo), hi)
                if L > 1e-9:
                    s = new_L / L
                    cj["x"] = px + (cx - px) * s
                    cj["y"] = py + (cy - py) * s

        for i, f in enumerate(frames):
            f["joints"] = list(maps[i].values())
        return frames

    @staticmethod
    def _snap_arms_two_link(frames: List[dict], fps: float) -> List[dict]:
        """Kinematic two-link arm snap — DISABLED.

        This pass replaced MediaPipe elbow/wrist positions with positions
        computed from EMA-smoothed directions. During fast golf swings the
        EMA direction is several frames stale, causing the skeleton to point
        the wrong way. Returning raw frames so MediaPipe detections are used
        directly after outlier rejection and One Euro.
        """
        return frames
        if not frames or fps <= 0:
            return frames

        maps: List[Dict[int, dict]] = [
            {int(j["id"]): j for j in f.get("joints", []) if "id" in j} for f in frames
        ]
        n = len(frames)
        body_scales = SkeletonProcessor._body_envelope_scales(maps, n)
        alpha_len = 0.25
        alpha_dir = 0.60
        prev_u: Dict[int, Tuple[float, float]] = {}
        prev_v: Dict[int, Tuple[float, float]] = {}
        ema_u: Dict[int, Optional[float]] = {11: None, 12: None}
        ema_f: Dict[int, Optional[float]] = {13: None, 14: None}

        for i in range(n):
            bs = body_scales[i]
            m = maps[i]
            for s_id, e_id, w_id in ((11, 13, 15), (12, 14, 16)):
                sj = m.get(s_id)
                ej = m.get(e_id)
                wj = m.get(w_id)
                if sj is None or ej is None or wj is None:
                    continue
                if float(sj.get("confidence", 0.0)) < 0.10:
                    continue
                sx, sy = float(sj["x"]), float(sj["y"])
                sz = float(sj.get("z", 0.0))
                ex, ey = float(ej["x"]), float(ej["y"])
                ez = float(ej.get("z", 0.0))
                wx, wy = float(wj["x"]), float(wj["y"])
                wz = float(wj.get("z", 0.0))
                du = math.hypot(ex - sx, ey - sy)
                df = math.hypot(wx - ex, wy - ey)
                if du < 0.004 * bs or df < 0.004 * bs:
                    continue
                if ema_u[s_id] is None:
                    ema_u[s_id] = du
                    ema_f[e_id] = df
                else:
                    lu0 = float(ema_u[s_id])
                    lf0 = float(ema_f[e_id])
                    if 0.50 * lu0 <= du <= 1.55 * lu0 and 0.50 * lf0 <= df <= 1.55 * lf0:
                        ema_u[s_id] = (1.0 - alpha_len) * lu0 + alpha_len * du
                        ema_f[e_id] = (1.0 - alpha_len) * lf0 + alpha_len * df
                lu = float(ema_u[s_id] or du)
                lf = float(ema_f[e_id] or df)
                u_raw = ((ex - sx) / du, (ey - sy) / du)
                if e_id in prev_u:
                    pxu, pyu = prev_u[e_id]
                    ux = alpha_dir * u_raw[0] + (1.0 - alpha_dir) * pxu
                    uy = alpha_dir * u_raw[1] + (1.0 - alpha_dir) * pyu
                    nu = math.hypot(ux, uy)
                    u = (ux / nu, uy / nu) if nu > 1e-9 else u_raw
                else:
                    u = u_raw
                prev_u[e_id] = u
                ex2 = sx + u[0] * lu
                ey2 = sy + u[1] * lu
                ez2 = ez * 0.70 + sz * 0.30
                vx_raw, vy_raw = wx - ex2, wy - ey2
                vl = math.hypot(vx_raw, vy_raw)
                v_raw = (vx_raw / vl, vy_raw / vl) if vl > 0.003 * bs else u
                if e_id in prev_v:
                    pxv, pyv = prev_v[e_id]
                    vx = alpha_dir * v_raw[0] + (1.0 - alpha_dir) * pxv
                    vy = alpha_dir * v_raw[1] + (1.0 - alpha_dir) * pyv
                    nv = math.hypot(vx, vy)
                    v = (vx / nv, vy / nv) if nv > 1e-9 else v_raw
                else:
                    v = v_raw
                prev_v[e_id] = v
                wx2 = ex2 + v[0] * lf
                wy2 = ey2 + v[1] * lf
                wz2 = wz * 0.60 + ez2 * 0.40
                ej["x"], ej["y"], ej["z"] = ex2, ey2, ez2
                wj["x"], wj["y"], wj["z"] = wx2, wy2, wz2
                ej["confidence"] = min(0.97, max(float(ej.get("confidence", 0.5)), 0.55))
                wj["confidence"] = min(0.97, max(float(wj.get("confidence", 0.5)), 0.52))

        return frames

    @staticmethod
    def _body_envelope_scales(
        frame_maps: List[Dict[int, dict]], n: int
    ) -> List[float]:
        """Per-frame body-envelope scale factor (1.0 = full-frame body)."""
        _ANCHOR_IDS = (11, 12, 23, 24)
        scales: List[float] = []
        last_good = 1.0
        for i in range(n):
            m = frame_maps[i]
            xs = []
            ys = []
            for aid in _ANCHOR_IDS:
                j = m.get(aid)
                if j is None or float(j.get("confidence", 0)) < 0.15:
                    continue
                xs.append(float(j.get("x", 0.0)))
                ys.append(float(j.get("y", 0.0)))
            if len(xs) >= 3:
                span = max(max(xs) - min(xs), max(ys) - min(ys), 0.01)
                s = span / 0.30
                last_good = max(0.5, min(3.0, s))
            scales.append(last_good)
        return scales

    # ------------------------------------------------------------------
    # Render-time stabilization (simple EMA blend for ALL joints)
    # ------------------------------------------------------------------

    @staticmethod
    def _swing_phase_label(frame_num: int, key_frames: dict) -> str:
        """Map frame index to a human-readable swing phase (continuous timeline)."""
        order = (
            ("address", "Address"),
            ("backswing", "Backswing"),
            ("top", "Top"),
            ("downswing", "Downswing"),
            ("impact", "Impact"),
            ("follow_through", "Follow-through"),
        )
        anchors: List[Tuple[int, str]] = []
        for key, label in order:
            fn = int(key_frames.get(key, -1))
            if fn >= 0:
                anchors.append((fn, label))
        if not anchors:
            return "Setup"
        anchors.sort(key=lambda t: t[0])
        if frame_num < anchors[0][0]:
            return "Setup"
        current = anchors[0][1]
        for i, (fn, lab) in enumerate(anchors):
            if frame_num >= fn:
                current = lab
            else:
                break
        return current

    @staticmethod
    def _export_overlay_ui_layout(width: int, height: int) -> Dict[str, object]:
        """Compact HUD + chart panel sizes for annotated / wrist-elbow export videos."""
        # Keep chart width inside frame (narrow clips / portrait).
        side_w = max(96, min(168, int(width * 0.095), width // 4))
        return {
            "hud_w": max(180, min(280, int(width * 0.13))),
            "hud_h": 72,
            "hud_lines_y": (20, 40, 58),
            "hud_font": (0.52, 0.48, 0.48),
            "side_w": side_w,
            "side_h": max(110, min(178, int(height * 0.20))),
            "chart_gap": 5,
            "phase_flash_scale": 0.95,
        }

    @staticmethod
    def _stabilize_joints_for_render(
        frame_num: int,
        current_joints: List[dict],
        previous_joints: List[dict],
        last_joint_frame: int,
        blend_alpha: float = 0.72,
        max_hold_frames: int = 8,
        body_scale: float = 1.0,
    ) -> List[dict]:
        """Temporal blend for render stability; arms get a lighter mix to reduce jitter."""
        bs = max(0.5, float(body_scale))
        if current_joints:
            if not previous_joints:
                return current_joints
            prev_map = {int(j["id"]): j for j in previous_joints if "id" in j}
            out: List[dict] = []
            seen_ids: set = set()
            a = max(0.0, min(1.0, float(blend_alpha)))
            jitter_db = 0.003 * bs
            for joint in current_joints:
                jid = int(joint.get("id", -1))
                seen_ids.add(jid)
                prev = prev_map.get(jid)
                if prev is None:
                    out.append(joint)
                    continue
                cx = float(joint.get("x", 0.0))
                cy = float(joint.get("y", 0.0))
                cz = float(joint.get("z", 0.0))
                px = float(prev.get("x", 0.0))
                py = float(prev.get("y", 0.0))
                pz = float(prev.get("z", 0.0))
                conf = float(joint.get("confidence", 1.0))
                dist = math.hypot(cx - px, cy - py)
                if jid in range(11, 23):
                    # Arms: light blend with previous frame (not dead-band snap) to
                    # reduce overlay shake; follow fast motion when step is large.
                    arm_a = 0.50 if dist > (0.014 * bs) else 0.34
                    merged = dict(joint)
                    merged["x"] = arm_a * cx + (1.0 - arm_a) * px
                    merged["y"] = arm_a * cy + (1.0 - arm_a) * py
                    merged["z"] = arm_a * cz + (1.0 - arm_a) * pz
                else:
                    if dist < jitter_db:
                        cx, cy, cz = px, py, pz
                    elif conf < 0.20:
                        cx, cy, cz = px, py, pz
                    merged = dict(joint)
                    merged["x"] = a * cx + (1.0 - a) * px
                    merged["y"] = a * cy + (1.0 - a) * py
                    merged["z"] = a * cz + (1.0 - a) * pz
                out.append(merged)
            if (frame_num - last_joint_frame) <= int(max_hold_frames):
                for jid, prev in prev_map.items():
                    if jid in seen_ids:
                        continue
                    if float(prev.get("confidence", 0.0)) < 0.05:
                        continue
                    carried = dict(prev)
                    carried["confidence"] = float(prev.get("confidence", 1.0)) * 0.82
                    out.append(carried)
            return out

        if previous_joints and (frame_num - last_joint_frame) <= int(max_hold_frames):
            return previous_joints
        return []

    # ------------------------------------------------------------------
    # Annotated video export (reference-style: clean overlay)
    # ------------------------------------------------------------------

    def export_annotated_video(
        self,
        video_path: str,
        skeleton_json: dict,
        output_path: str,
        *,
        show_shoulders: bool = True,
        show_waistline: bool = True,
        show_legs: bool = True,
        show_spine: bool = True,
        show_head_reference: bool = True,
        static_lines: Optional[List[Dict[str, Any]]] = None,
        tracking_lines: Optional[List[Dict[str, Any]]] = None,
        static_labels: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Export MP4 with skeleton overlay, HUD, phase labels, AND analysis charts.

        Includes:
        - Body-only skeleton (no head bones; ends at shoulders)
        - Toggleable sections: shoulders, waistline, legs
        - Dashed spine reference line (shoulder midpoint to hip midpoint)
        - Static head reference line (fixed screen Y from first valid pose; no head bones)
        - Coach static + tracking reference lines
        - Optional ``static_labels`` (same schema as ``coach_overlay_labels.draw_static_labels_on_frame``)
        - Frame/time/phase HUD (top-left)
        - Flashing phase labels on key-frame transitions
        - Elbow-angle chart (top-right panel)
        - Wrist-speed chart (top-right panel, below elbow)
        - Left/Right elbow readout (bottom-left)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ui = self._export_overlay_ui_layout(width, height)
        render_hold_frames = max(40, int(fps * 1.25))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Unable to create output video: {output_path}")

        frames = skeleton_json.get("frames", [])
        frame_lookup = {int(f["frame_num"]): f.get("joints", []) for f in frames if "frame_num" in f}
        key_frames = self.swing_analyzer.detect_key_frames(frames)

        dt = 1.0 / fps if fps > 1e-6 else 1.0 / 30.0
        prev_mid: Optional[Tuple[float, float]] = None
        smoothed_speed: Optional[float] = None
        speed_smooth = 0.30
        speed_history: List[float] = []
        elbow_history: List[float] = []
        elbow_chart_ema: Optional[float] = None
        elbow_chart_alpha = 0.35
        speed_chart_ceiling = 95.0
        history_window = max(60, int((fps if fps > 0 else 30.0) * 6.0))

        frame_maps_for_scale = [
            {int(j["id"]): j for j in f.get("joints", []) if "id" in j}
            for f in frames
        ]
        body_scales = self._body_envelope_scales(frame_maps_for_scale, len(frames))
        scale_lookup = {
            int(f["frame_num"]): body_scales[i]
            for i, f in enumerate(frames) if "frame_num" in f
        }

        # Head reference: capture Y-pixel of head top on the first valid frame
        head_ref_y: Optional[int] = None

        flash_until = -1
        flash_label = ""
        current_phase = "Setup"
        prev_render_joints: List[dict] = []
        last_joint_frame = -(10 ** 9)
        frame_num = 0
        self.overlay.reset_arm_display_smooth()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            raw_joints = frame_lookup.get(frame_num, [])
            bs = scale_lookup.get(frame_num, 1.0)
            joints = self._stabilize_joints_for_render(
                frame_num=frame_num,
                current_joints=raw_joints,
                previous_joints=prev_render_joints,
                last_joint_frame=last_joint_frame,
                body_scale=bs,
                max_hold_frames=render_hold_frames,
            )
            if joints:
                prev_render_joints = joints
            if raw_joints:
                last_joint_frame = frame_num

            # Capture head reference on first valid detection
            if show_head_reference and head_ref_y is None and joints:
                head_ref_y = self._capture_head_reference_y(joints, height)

            rendered = self.overlay.draw_overlay_frame(
                frame,
                joints if joints else [],
                smooth_arm_display=True,
                show_shoulders=show_shoulders,
                show_waistline=show_waistline,
                show_legs=show_legs,
                show_spine=show_spine,
                head_reference_y=head_ref_y if show_head_reference else None,
                static_lines=static_lines,
                tracking_lines=tracking_lines,
            )
            if static_labels:
                draw_static_labels_on_frame(rendered, static_labels)

            phase_now = SkeletonProcessor._swing_phase_label(frame_num, key_frames)
            if phase_now != current_phase:
                flash_label = phase_now
                flash_until = frame_num + 15
                current_phase = phase_now

            if frame_num <= flash_until and ((frame_num - (flash_until - 15)) % 2 == 0):
                cv2.putText(
                    rendered,
                    flash_label,
                    (max(24, width // 3), 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    float(ui["phase_flash_scale"]),
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            # --- HUD: frame / time / phase (top-left) ---
            hud_w = int(ui["hud_w"])
            hud_h = int(ui["hud_h"])
            hud_dy = ui["hud_lines_y"]
            hf0, hf1, hf2 = ui["hud_font"]
            x0, y0 = 14, 14
            cv2.rectangle(rendered, (x0, y0), (x0 + hud_w, y0 + hud_h), (5, 10, 8), -1)
            cv2.rectangle(rendered, (x0, y0), (x0 + hud_w, y0 + hud_h), (80, 220, 140), 1)
            cv2.putText(
                rendered, f"Frame: {frame_num}",
                (x0 + 10, y0 + hud_dy[0]), cv2.FONT_HERSHEY_SIMPLEX, float(hf0),
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                rendered, f"Time: {frame_num / fps:.2f}s",
                (x0 + 10, y0 + hud_dy[1]), cv2.FONT_HERSHEY_SIMPLEX, float(hf1),
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                rendered, f"Phase: {current_phase}",
                (x0 + 10, y0 + hud_dy[2]), cv2.FONT_HERSHEY_SIMPLEX, float(hf2),
                (0, 255, 255), 1, cv2.LINE_AA,
            )

            # --- Analysis: wrist speed + elbow angles ---
            wrist_speed_px_s: Optional[float] = None
            left_elbow: Optional[float] = None
            right_elbow: Optional[float] = None

            if joints:
                mid = self._wrist_midpoint_pixels(joints, width, height)
                if mid is not None and prev_mid is not None:
                    dist = math.hypot(mid[0] - prev_mid[0], mid[1] - prev_mid[1])
                    wrist_speed_px_s = dist / dt
                if mid is not None:
                    prev_mid = mid
                angles = self.swing_analyzer.analyze_swing_angles(joints)
                left_elbow = angles.get("arm_extension_left")
                right_elbow = angles.get("arm_extension_right")

            if wrist_speed_px_s is not None:
                smoothed_speed = (
                    wrist_speed_px_s
                    if smoothed_speed is None
                    else speed_smooth * wrist_speed_px_s + (1.0 - speed_smooth) * smoothed_speed
                )
            if smoothed_speed is not None:
                speed_history.append(smoothed_speed)
                speed_chart_ceiling = max(
                    45.0,
                    speed_chart_ceiling * 0.992,
                    float(smoothed_speed) * 1.28,
                )
                if len(speed_history) > history_window:
                    speed_history = speed_history[-history_window:]
            if left_elbow is not None and right_elbow is not None:
                raw_elb = (float(left_elbow) + float(right_elbow)) * 0.5
                if elbow_chart_ema is None:
                    elbow_chart_ema = raw_elb
                else:
                    elbow_chart_ema = elbow_chart_alpha * raw_elb + (1.0 - elbow_chart_alpha) * elbow_chart_ema
                elbow_history.append(float(elbow_chart_ema))
                if len(elbow_history) > history_window:
                    elbow_history = elbow_history[-history_window:]

            # --- Chart panels (top-right) ---
            side_w = min(int(ui["side_w"]), max(80, width - 20))
            side_x0 = max(6, width - side_w - 8)
            side_y0 = 8
            side_h = int(ui["side_h"])
            chart_gap = int(ui["chart_gap"])
            chart_h = max(52, (side_h - chart_gap) // 2)
            chart_w = side_w

            cv2.rectangle(rendered, (side_x0, side_y0), (side_x0 + side_w, side_y0 + side_h), (18, 22, 20), -1)
            cv2.rectangle(rendered, (side_x0, side_y0), (side_x0 + side_w, side_y0 + side_h), (64, 72, 64), 1)

            avg_elbow = elbow_chart_ema if elbow_chart_ema is not None else (elbow_history[-1] if elbow_history else None)
            self._draw_metric_chart(
                image=rendered,
                rect=(side_x0 + 4, side_y0 + 4, chart_w - 8, chart_h - 6),
                title="Elbow",
                unit="deg",
                current_value=avg_elbow,
                values=elbow_history,
                color=(0, 222, 255),
                y_min=40.0,
                y_max=180.0,
            )
            self._draw_metric_chart(
                image=rendered,
                rect=(side_x0 + 4, side_y0 + chart_h + chart_gap + 4, chart_w - 8, chart_h - 6),
                title="Spd",
                unit="px/s",
                current_value=smoothed_speed,
                values=speed_history,
                color=(72, 235, 102),
                y_min=0.0,
                y_max=max(12.0, speed_chart_ceiling),
            )

            # --- Elbow readout (bottom-left) ---
            info_lines = [
                f"L{left_elbow:.0f}" if left_elbow is not None else "L--",
                f"R{right_elbow:.0f}" if right_elbow is not None else "R--",
            ]
            y_info = min(height - 8, height - 6)
            max_lw = max(40, width // 5)
            for i, line in enumerate(reversed(info_lines)):
                yy = y_info - i * 14
                SkeletonProcessor._put_text_fit_width(
                    rendered,
                    line,
                    (8, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (220, 220, 220),
                    1,
                    max_lw,
                )

            writer.write(rendered)
            frame_num += 1

        cap.release()
        writer.release()
        return output_path

    # ------------------------------------------------------------------
    # Wrist / elbow analysis video
    # ------------------------------------------------------------------

    def export_wrist_elbow_analysis_video(
        self,
        video_path: str,
        skeleton_json: dict,
        output_path: str,
        *,
        show_shoulders: bool = True,
        show_waistline: bool = True,
        show_legs: bool = True,
        show_spine: bool = True,
        show_head_reference: bool = True,
        static_lines: Optional[List[Dict[str, Any]]] = None,
        tracking_lines: Optional[List[Dict[str, Any]]] = None,
        static_labels: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Export MP4 with skeleton overlay plus wrist speed and elbow angles."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ui = self._export_overlay_ui_layout(width, height)
        render_hold_frames = max(40, int(fps * 1.25))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Unable to create output video: {output_path}")

        frames = skeleton_json.get("frames", [])
        frame_lookup = {
            int(f["frame_num"]): f.get("joints", []) for f in frames if "frame_num" in f
        }
        key_frames = self.swing_analyzer.detect_key_frames(frames)

        dt = 1.0 / fps if fps > 1e-6 else 1.0 / 30.0
        prev_mid: Optional[Tuple[float, float]] = None
        smoothed_speed: Optional[float] = None
        speed_smooth = 0.30
        speed_chart_ceiling2 = 95.0
        speed_history: List[float] = []
        elbow_history: List[float] = []
        elbow_chart_ema2: Optional[float] = None
        elbow_chart_alpha2 = 0.35
        history_window = max(60, int((fps if fps > 0 else 30.0) * 6.0))
        frame_maps_for_scale2 = [
            {int(j["id"]): j for j in f.get("joints", []) if "id" in j}
            for f in frames
        ]
        body_scales2 = self._body_envelope_scales(frame_maps_for_scale2, len(frames))
        scale_lookup2 = {
            int(f["frame_num"]): body_scales2[i]
            for i, f in enumerate(frames) if "frame_num" in f
        }

        flash_until = -1
        flash_label = ""
        current_phase = "Setup"
        prev_render_joints: List[dict] = []
        last_joint_frame = -(10 ** 9)
        head_ref_y: Optional[int] = None

        frame_num = 0
        self.overlay.reset_arm_display_smooth()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            raw_joints = frame_lookup.get(frame_num, [])
            bs2 = scale_lookup2.get(frame_num, 1.0)
            joints = self._stabilize_joints_for_render(
                frame_num=frame_num,
                current_joints=raw_joints,
                previous_joints=prev_render_joints,
                last_joint_frame=last_joint_frame,
                body_scale=bs2,
                max_hold_frames=render_hold_frames,
            )
            if joints:
                prev_render_joints = joints
            if raw_joints:
                last_joint_frame = frame_num

            if show_head_reference and head_ref_y is None and joints:
                head_ref_y = self._capture_head_reference_y(joints, height)

            rendered = (
                self.overlay.draw_overlay_frame(
                    frame, joints, smooth_arm_display=True,
                    show_shoulders=show_shoulders,
                    show_waistline=show_waistline,
                    show_legs=show_legs,
                    show_spine=show_spine,
                    head_reference_y=head_ref_y if show_head_reference else None,
                    static_lines=static_lines,
                    tracking_lines=tracking_lines,
                )
                if joints
                else frame.copy()
            )
            if static_labels:
                draw_static_labels_on_frame(rendered, static_labels)

            phase_now = SkeletonProcessor._swing_phase_label(frame_num, key_frames)
            if phase_now != current_phase:
                flash_label = phase_now
                flash_until = frame_num + 15
                current_phase = phase_now

            if frame_num <= flash_until and ((frame_num - (flash_until - 15)) % 2 == 0):
                cv2.putText(
                    rendered,
                    flash_label,
                    (max(24, width // 3), 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    float(ui["phase_flash_scale"]),
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            hud_w = int(ui["hud_w"])
            hud_h = int(ui["hud_h"])
            hud_dy = ui["hud_lines_y"]
            hf0, hf1, hf2 = ui["hud_font"]
            x0, y0 = 14, 14
            cv2.rectangle(rendered, (x0, y0), (x0 + hud_w, y0 + hud_h), (5, 10, 8), -1)
            cv2.rectangle(rendered, (x0, y0), (x0 + hud_w, y0 + hud_h), (80, 220, 140), 1)
            cv2.putText(
                rendered, f"Frame: {frame_num}",
                (x0 + 10, y0 + hud_dy[0]), cv2.FONT_HERSHEY_SIMPLEX, float(hf0),
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                rendered, f"Time: {frame_num / fps:.2f}s",
                (x0 + 10, y0 + hud_dy[1]), cv2.FONT_HERSHEY_SIMPLEX, float(hf1),
                (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                rendered, f"Phase: {current_phase}",
                (x0 + 10, y0 + hud_dy[2]), cv2.FONT_HERSHEY_SIMPLEX, float(hf2),
                (0, 255, 255), 1, cv2.LINE_AA,
            )

            wrist_speed_px_s: Optional[float] = None
            left_elbow: Optional[float] = None
            right_elbow: Optional[float] = None

            if joints:
                mid = self._wrist_midpoint_pixels(joints, width, height)
                if mid is not None and prev_mid is not None:
                    dist = math.hypot(mid[0] - prev_mid[0], mid[1] - prev_mid[1])
                    wrist_speed_px_s = dist / dt
                if mid is not None:
                    prev_mid = mid

                angles = self.swing_analyzer.analyze_swing_angles(joints)
                left_elbow = angles.get("arm_extension_left")
                right_elbow = angles.get("arm_extension_right")

            if wrist_speed_px_s is not None:
                smoothed_speed = (
                    wrist_speed_px_s
                    if smoothed_speed is None
                    else speed_smooth * wrist_speed_px_s + (1.0 - speed_smooth) * smoothed_speed
                )
            if smoothed_speed is not None:
                speed_history.append(smoothed_speed)
                speed_chart_ceiling2 = max(
                    45.0,
                    speed_chart_ceiling2 * 0.992,
                    float(smoothed_speed) * 1.28,
                )
                if len(speed_history) > history_window:
                    speed_history = speed_history[-history_window:]
            if left_elbow is not None and right_elbow is not None:
                raw_elb = (float(left_elbow) + float(right_elbow)) * 0.5
                if elbow_chart_ema2 is None:
                    elbow_chart_ema2 = raw_elb
                else:
                    elbow_chart_ema2 = elbow_chart_alpha2 * raw_elb + (1.0 - elbow_chart_alpha2) * elbow_chart_ema2
                elbow_history.append(float(elbow_chart_ema2))
                if len(elbow_history) > history_window:
                    elbow_history = elbow_history[-history_window:]
            side_w = min(int(ui["side_w"]), max(80, width - 20))
            side_x0 = max(6, width - side_w - 8)
            side_y0 = 8
            side_h = int(ui["side_h"])
            chart_gap = int(ui["chart_gap"])
            chart_h = max(52, (side_h - chart_gap) // 2)
            chart_w = side_w

            cv2.rectangle(rendered, (side_x0, side_y0), (side_x0 + side_w, side_y0 + side_h), (18, 22, 20), -1)
            cv2.rectangle(rendered, (side_x0, side_y0), (side_x0 + side_w, side_y0 + side_h), (64, 72, 64), 1)

            avg_elbow = (
                elbow_chart_ema2 if elbow_chart_ema2 is not None else (elbow_history[-1] if elbow_history else None)
            )
            self._draw_metric_chart(
                image=rendered,
                rect=(side_x0 + 4, side_y0 + 4, chart_w - 8, chart_h - 6),
                title="Elbow",
                unit="deg",
                current_value=avg_elbow,
                values=elbow_history,
                color=(0, 222, 255),
                y_min=40.0,
                y_max=180.0,
            )
            self._draw_metric_chart(
                image=rendered,
                rect=(side_x0 + 4, side_y0 + chart_h + chart_gap + 4, chart_w - 8, chart_h - 6),
                title="Spd",
                unit="px/s",
                current_value=smoothed_speed,
                values=speed_history,
                color=(72, 235, 102),
                y_min=0.0,
                y_max=max(12.0, speed_chart_ceiling2),
            )

            info_lines = [
                f"L{left_elbow:.0f}" if left_elbow is not None else "L--",
                f"R{right_elbow:.0f}" if right_elbow is not None else "R--",
            ]
            y_info = min(height - 8, height - 6)
            max_lw = max(40, width // 5)
            for i, line in enumerate(reversed(info_lines)):
                yy = y_info - i * 14
                SkeletonProcessor._put_text_fit_width(
                    rendered,
                    line,
                    (8, yy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (220, 220, 220),
                    1,
                    max_lw,
                )

            writer.write(rendered)
            frame_num += 1

        cap.release()
        writer.release()
        return output_path

    # ------------------------------------------------------------------
    # Swing GIF export
    # ------------------------------------------------------------------

    def export_swing_gif(
        self,
        video_path: str,
        skeleton_json: dict,
        key_frames: dict,
        output_path: str,
    ) -> dict:
        """Export Address->Follow-through loop as GIF and short MP4 (max 2s, 480x480)."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required for GIF export. Install `pillow`.") from exc

        start_frame = int(key_frames.get("address", 0))
        end_frame = int(key_frames.get("follow_through", start_frame))
        if end_frame < start_frame:
            end_frame = start_frame

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        frame_lookup = {
            int(f["frame_num"]): f.get("joints", [])
            for f in skeleton_json.get("frames", [])
            if "frame_num" in f
        }

        max_frames = max(1, int(min(2.0 * fps, (end_frame - start_frame + 1))))
        collected_rgb = []
        clip_mp4_path = os.path.splitext(output_path)[0] + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(clip_mp4_path, fourcc, fps, (480, 480))
        if not writer.isOpened():
            cap.release()
            raise ValueError(f"Unable to create short clip: {clip_mp4_path}")

        frame_num = 0
        used = 0
        self.overlay.reset_arm_display_smooth()
        while used < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_num < start_frame:
                frame_num += 1
                continue
            if frame_num > end_frame:
                break

            joints = frame_lookup.get(frame_num, [])
            rendered = (
                self.overlay.draw_overlay_frame(frame, joints, smooth_arm_display=True)
                if joints
                else frame
            )
            rendered = cv2.resize(rendered, (480, 480))
            writer.write(rendered)
            collected_rgb.append(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
            used += 1
            frame_num += 1

        cap.release()
        writer.release()

        if not collected_rgb:
            raise ValueError("No frames available for GIF export in selected swing range.")
        pil_frames = [Image.fromarray(frame) for frame in collected_rgb]
        duration_ms = int(max(1, round(1000.0 / fps)))
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            format="GIF",
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        return {"gif_path": output_path, "mp4_path": clip_mp4_path}

    # ------------------------------------------------------------------
    # Head reference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_head_reference_y(joints: List[dict], frame_height: int) -> Optional[int]:
        """Return pixel-Y for a fixed horizontal head reference line on screen.

        Captured once per export from the first valid frame; it does not track
        the head. Uses the nose (id 0), else ear midpoint (7, 8).
        """
        jmap = {int(j["id"]): j for j in joints if "id" in j}
        nose = jmap.get(0)
        if nose is not None:
            y_norm = float(nose.get("y", 0.5))
            return int(round(max(0.0, min(1.0, y_norm)) * (frame_height - 1)))
        ear_l = jmap.get(7)
        ear_r = jmap.get(8)
        if ear_l is not None and ear_r is not None:
            y_norm = (float(ear_l["y"]) + float(ear_r["y"])) * 0.5
            return int(round(max(0.0, min(1.0, y_norm)) * (frame_height - 1)))
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _put_text_fit_width(
        image,
        text: str,
        org: Tuple[int, int],
        font_face: int,
        base_scale: float,
        color: Tuple[int, int, int],
        thickness: int,
        max_width: int,
    ) -> int:
        """Draw ``text`` scaled down until it fits ``max_width``; returns text height + baseline."""
        sc = float(base_scale)
        while sc >= 0.22:
            (tw, th), bl = cv2.getTextSize(text, font_face, sc, thickness)
            if tw <= max_width:
                cv2.putText(image, text, org, font_face, sc, color, thickness, cv2.LINE_AA)
                return int(th + bl)
            sc *= 0.86
        sc = 0.22
        cv2.putText(image, text, org, font_face, sc, color, thickness, cv2.LINE_AA)
        (_tw, th), bl = cv2.getTextSize(text, font_face, sc, thickness)
        return int(th + bl)

    @staticmethod
    def _draw_metric_chart(
        image,
        rect: Tuple[int, int, int, int],
        title: str,
        unit: str,
        current_value: Optional[float],
        values: List[float],
        color: Tuple[int, int, int],
        y_min: float,
        y_max: float,
    ) -> None:
        """Draw a compact scrolling line chart panel for one metric."""
        x, y, w, h = rect
        w = max(72, int(w))
        h = max(50, int(h))
        ih, iw = int(image.shape[0]), int(image.shape[1])
        x = max(2, min(x, iw - w - 2))
        y = max(2, min(y, ih - h - 2))
        x2 = x + w
        y2 = y + h

        cv2.rectangle(image, (x, y), (x2, y2), (28, 34, 30), -1)
        cv2.rectangle(image, (x, y), (x2, y2), (70, 74, 70), 1)

        tw_max = max(40, w - 10)
        if unit.lower() in ("px/s", "pxs", "px"):
            val_txt = (
                f"{int(round(float(current_value)))}"
                if current_value is not None
                else "--"
            )
        else:
            val_txt = (
                f"{float(current_value):.0f}"
                if current_value is not None
                else "--"
            )
        title_txt = title[:18]
        unit_s = unit[:6]
        if val_txt != "--" and unit_s:
            val_txt = f"{val_txt}{unit_s}"

        ts = max(0.30, min(0.48, h / 175.0))
        y_cursor = y + int(12 + ts * 2)
        h_title = SkeletonProcessor._put_text_fit_width(
            image, title_txt, (x + 4, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, ts, (235, 235, 235), 1, tw_max,
        )
        y_cursor += h_title + 2
        vs = max(0.30, min(0.50, h / 160.0))
        h_val = SkeletonProcessor._put_text_fit_width(
            image, val_txt, (x + 4, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, vs, color, 1, tw_max,
        )
        y_cursor += h_val + 3

        plot_x = x + 4
        plot_y = max(y_cursor, y + int(h * 0.36))
        plot_y = min(plot_y, y + h - 22)
        plot_w = max(20, w - 8)
        plot_h = max(22, y2 - plot_y - 4)
        cv2.rectangle(image, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (40, 46, 42), -1)
        cv2.rectangle(image, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (74, 80, 74), 1)

        for t in (0.25, 0.5, 0.75):
            gy = int(plot_y + plot_h * t)
            cv2.line(image, (plot_x + 1, gy), (plot_x + plot_w - 1, gy), (55, 60, 56), 1)

        if len(values) < 2 or y_max <= y_min:
            return

        recent = values[-plot_w:]
        span = max(1e-6, y_max - y_min)
        points = []
        for idx, v in enumerate(recent):
            px = plot_x + idx
            ratio = (float(v) - y_min) / span
            ratio = max(0.0, min(1.0, ratio))
            py = plot_y + int((1.0 - ratio) * (plot_h - 1))
            points.append((px, py))

        if len(points) >= 2:
            for i in range(1, len(points)):
                cv2.line(image, points[i - 1], points[i], color, 2)

        last_x, last_y = points[-1]
        cv2.circle(image, (last_x, last_y), 2, color, -1)

    @staticmethod
    def _wrist_midpoint_pixels(
        joints: list, width: int, height: int
    ) -> Optional[Tuple[float, float]]:
        """Pixel (x, y) of midpoint between left (15) and right (16) wrists."""

        def _c01(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        jmap = {int(j["id"]): j for j in joints if "id" in j}
        l_w = jmap.get(15)
        r_w = jmap.get(16)
        if l_w is None or r_w is None:
            return None
        x = (_c01(l_w["x"]) + _c01(r_w["x"])) * 0.5 * (width - 1)
        y = (_c01(l_w["y"]) + _c01(r_w["y"])) * 0.5 * (height - 1)
        return (x, y)
