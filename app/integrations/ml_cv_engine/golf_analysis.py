"""Golf-specific swing phase and angle analysis utilities."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class GolfSwingAnalyzer:
    """Analyze golf swing key frames and joint-based biomechanics."""

    # MediaPipe Pose IDs used:
    # shoulders: left=11, right=12
    # elbows: left=13, right=14
    # wrists: left=15, right=16
    # hips: left=23, right=24
    # knees: left=25, right=26
    # ankles: left=27, right=28

    @staticmethod
    def _interp_nan_series(values: np.ndarray) -> Optional[np.ndarray]:
        """Linear interpolation over NaNs; returns None if no samples."""
        idx = np.arange(len(values), dtype=np.float64)
        mask = ~np.isnan(values)
        if not mask.any():
            return None
        return np.interp(idx, idx[mask], values[mask])

    def detect_key_frames(self, skeleton_frames: list) -> dict:
        """Detect six golf swing phases from frame-wise skeletons.

        Uses every frame (missing wrists are filled via interpolation) so phase
        detection survives short dropout during fast swings. Wrist height is
        smoothed before locating top; impact is placed near peak downswing speed
        (max positive dy), which tracks ball contact better than wrist–hip distance alone.

        Args:
            skeleton_frames: List of frame dicts from skeleton JSON, each containing
                at least {"frame_num": int, "joints": [...]}

        Returns:
            {
              "address": int,
              "backswing": int,
              "top": int,
              "downswing": int,
              "impact": int,
              "follow_through": int
            }
            Values are frame numbers from input (or -1 if unavailable).
        """
        empty = {
            "address": -1,
            "backswing": -1,
            "top": -1,
            "downswing": -1,
            "impact": -1,
            "follow_through": -1,
        }
        if not skeleton_frames:
            return dict(empty)

        n = len(skeleton_frames)
        frame_nums = np.array(
            [int(f.get("frame_num", i)) for i, f in enumerate(skeleton_frames)],
            dtype=np.int64,
        )
        wy = np.full(n, np.nan, dtype=np.float64)
        shoulder_y = np.full(n, np.nan, dtype=np.float64)

        for i, frame in enumerate(skeleton_frames):
            joint_map = {int(j["id"]): j for j in frame.get("joints", []) if "id" in j}
            w = self._midpoint(joint_map, 15, 16)
            if w is not None:
                wy[i] = float(w[1])
            s = self._midpoint(joint_map, 11, 12)
            if s is not None:
                shoulder_y[i] = float(s[1])

        wy_f = self._interp_nan_series(wy)
        if wy_f is None:
            return dict(empty)
        shoulder_f = self._interp_nan_series(shoulder_y)
        if shoulder_f is None:
            sm = shoulder_y[~np.isnan(shoulder_y)]
            fb = float(np.median(sm)) if sm.size > 0 else 0.45
            shoulder_f = np.full(n, fb, dtype=np.float64)

        win = max(3, min(9, (n // 25) | 1))
        if win % 2 == 0:
            win += 1
        kernel = np.ones(win, dtype=np.float64) / float(win)
        wy_s = np.convolve(wy_f, kernel, mode="same")

        motion = np.abs(np.diff(wy_s, prepend=wy_s[0]))
        low_th = float(np.percentile(motion, 20)) if len(motion) > 2 else 0.0
        steady = np.where(motion <= low_th)[0]
        address_idx = int(steady[0]) if steady.size else 0

        search_cap = max(3, int(n * 0.96))
        top_idx = int(np.argmin(wy_s[:search_cap]))

        if top_idx <= address_idx:
            top_idx = min(address_idx + 1, n - 1)

        dw = np.diff(wy_s, prepend=wy_s[0])
        impact_idx = top_idx
        if top_idx + 1 < n:
            down = dw[top_idx + 1 :]
            if down.size > 0:
                pos = np.maximum(down, 0.0)
                if float(np.max(pos)) > 1e-7:
                    impact_rel = int(np.argmax(pos))
                else:
                    impact_rel = int(np.argmax(np.abs(down)))
                impact_idx = top_idx + 1 + impact_rel
        impact_idx = int(np.clip(impact_idx, min(top_idx + 1, n - 1), n - 1))

        follow_idx = impact_idx
        for i in range(impact_idx + 1, n):
            sy = float(shoulder_f[i])
            if wy_s[i] < sy - 0.008:
                follow_idx = i
                break

        backswing_idx = int(round((address_idx + top_idx) * 0.5))
        downswing_idx = int(round((top_idx + impact_idx) * 0.5))
        backswing_idx = max(address_idx, min(top_idx, backswing_idx))
        downswing_idx = max(top_idx, min(impact_idx, downswing_idx))

        return {
            "address": int(frame_nums[address_idx]),
            "backswing": int(frame_nums[backswing_idx]),
            "top": int(frame_nums[top_idx]),
            "downswing": int(frame_nums[downswing_idx]),
            "impact": int(frame_nums[impact_idx]),
            "follow_through": int(frame_nums[follow_idx]),
        }

    def analyze_swing_angles(self, joints: list) -> dict:
        """Calculate golf-relevant swing angles for one frame.

        Joint IDs used:
        - Spine angle: shoulders (11,12), hips (23,24)
        - Hip rotation: hips (23,24)
        - Knee flex: hip-knee-ankle (left 23-25-27, right 24-26-28)
        - Arm extension: shoulder-elbow-wrist
          (left 11-13-15, right 12-14-16)
        """
        joint_map = {int(j["id"]): j for j in joints if "id" in j}

        shoulder_mid = self._midpoint(joint_map, 11, 12)
        hip_mid = self._midpoint(joint_map, 23, 24)

        spine_angle = None
        if shoulder_mid is not None and hip_mid is not None:
            spine_vec = shoulder_mid - hip_mid
            vertical = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            spine_angle = self._angle_between(spine_vec, vertical)

        hip_rotation = self._hip_rotation_proxy(joint_map)

        left_knee_flex = self._joint_angle(joint_map, 23, 25, 27)
        right_knee_flex = self._joint_angle(joint_map, 24, 26, 28)

        left_arm_extension = self._joint_angle(joint_map, 11, 13, 15)
        right_arm_extension = self._joint_angle(joint_map, 12, 14, 16)

        return {
            "spine_angle": spine_angle,
            "hip_rotation_angle": hip_rotation,
            "knee_flex_left": left_knee_flex,
            "knee_flex_right": right_knee_flex,
            "arm_extension_left": left_arm_extension,
            "arm_extension_right": right_arm_extension,
        }

    def calculate_tempo(
        self,
        skeleton_json: dict,
        key_frames: dict,
        *,
        chart_output_path: Optional[str] = None,
    ) -> dict:
        """Calculate tempo metrics and export a phase-duration bar chart PNG.

        Args:
            skeleton_json: Per-frame skeleton with ``submission_id`` and ``frames``.
            key_frames: Phase frame numbers from ``detect_key_frames``.
            chart_output_path: If set, write the chart here; otherwise
                ``outputs/<submission_id>/tempo_chart.png``.
        """
        frame_entries = skeleton_json.get("frames", [])
        frame_to_time = {
            int(f.get("frame_num", -1)): float(f.get("timestamp", 0.0))
            for f in frame_entries
            if "frame_num" in f
        }

        address = int(key_frames.get("address", -1))
        top = int(key_frames.get("top", -1))
        impact = int(key_frames.get("impact", -1))
        follow = int(key_frames.get("follow_through", -1))

        def ms_between(a: int, b: int) -> Optional[float]:
            if a not in frame_to_time or b not in frame_to_time:
                return None
            return max(0.0, (frame_to_time[b] - frame_to_time[a]) * 1000.0)

        durations = {
            "address_to_top_ms": ms_between(address, top),
            "top_to_impact_ms": ms_between(top, impact),
            "impact_to_follow_through_ms": ms_between(impact, follow),
            "address_to_impact_ms": ms_between(address, impact),
        }

        backswing = durations["address_to_top_ms"]
        downswing = durations["top_to_impact_ms"]
        ratio = None
        ratio_in_ideal_range = None
        if backswing is not None and downswing is not None and downswing > 1e-6:
            ratio = backswing / downswing
            ratio_in_ideal_range = 2.5 <= ratio <= 3.5

        chart_path = self._export_tempo_chart(
            durations=durations,
            ratio=ratio,
            ratio_in_ideal_range=ratio_in_ideal_range,
            skeleton_json=skeleton_json,
            output_path=chart_output_path,
        )

        return {
            "durations_ms": durations,
            "backswing_downswing_ratio": ratio,
            "ratio_in_ideal_range": ratio_in_ideal_range,
            "ideal_ratio_range": (2.5, 3.5),
            "chart_path": chart_path,
        }

    def export_score_card(
        self, swing_angles: dict, ideal_ranges: dict, output_path: str
    ) -> str:
        """Export a radar/spider score-card chart as PNG.

        Axes:
        - Spine Angle
        - Hip Rotation
        - Arm Extension
        - Balance
        - Tempo
        """
        labels = [
            "Spine Angle",
            "Hip Rotation",
            "Arm Extension",
            "Balance",
            "Tempo",
        ]
        metric_values = [
            swing_angles.get("spine_angle"),
            swing_angles.get("hip_rotation_angle"),
            swing_angles.get("arm_extension_left"),
            swing_angles.get("balance"),
            swing_angles.get("tempo"),
        ]
        metric_keys = [
            "spine_angle",
            "hip_rotation_angle",
            "arm_extension",
            "balance",
            "tempo",
        ]

        scores = []
        for key, value in zip(metric_keys, metric_values):
            if value is None:
                scores.append(50.0)
                continue
            ideal = ideal_ranges.get(key)
            if not ideal:
                scores.append(50.0)
                continue
            low, high = float(ideal[0]), float(ideal[1])
            scores.append(self._score_from_range(float(value), low, high))

        overall = float(np.mean(scores)) if scores else 0.0
        color = "#ef4444" if overall < 50 else "#f59e0b" if overall < 75 else "#22c55e"

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        values = np.concatenate([np.array(scores, dtype=np.float64), [scores[0]]])

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.set_ylim(0, 100)
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["25", "50", "75", "100"])
        ax.grid(True, alpha=0.35)
        ax.text(0.5, 0.5, f"{int(round(overall))}", transform=ax.transAxes, ha="center", va="center", fontsize=22, fontweight="bold")

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def _export_tempo_chart(
        self,
        durations: dict,
        ratio: Optional[float],
        ratio_in_ideal_range: Optional[bool],
        skeleton_json: dict,
        output_path: Optional[str] = None,
    ) -> str:
        if output_path is None:
            submission_id = skeleton_json.get("submission_id", "unknown_submission")
            output_path = os.path.join("outputs", submission_id, "tempo_chart.png")
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        labels = [
            "Address->Top",
            "Top->Impact",
            "Impact->Follow",
        ]
        values = [
            float(durations.get("address_to_top_ms") or 0.0),
            float(durations.get("top_to_impact_ms") or 0.0),
            float(durations.get("impact_to_follow_through_ms") or 0.0),
        ]
        colors = ["#60a5fa", "#f59e0b", "#34d399"]
        if ratio_in_ideal_range is False:
            colors[1] = "#ef4444"

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, values, color=colors)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() * 0.5, bar.get_height() + 5, f"{int(round(value))} ms", ha="center", va="bottom", fontsize=9)

        # Ideal ratio reference line in duration space:
        # use backswing duration and mark expected downswing duration at 3:1 center.
        backswing = values[0]
        if backswing > 0:
            ideal_downswing = backswing / 3.0
            ax.axhline(
                y=ideal_downswing,
                color="#9ca3af",
                linestyle="--",
                linewidth=1.5,
                label="Ideal 3:1 reference (downswing ms)",
            )

        ratio_text = "N/A" if ratio is None else f"{ratio:.2f}:1"
        ax.set_title(f"Swing Tempo Durations (Ratio {ratio_text})")
        ax.set_ylabel("Duration (ms)")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    @staticmethod
    def _score_from_range(value: float, low: float, high: float) -> float:
        if low > high:
            low, high = high, low
        if low <= value <= high:
            return 100.0
        spread = max(high - low, 1e-6)
        if value < low:
            return float(max(0.0, 100.0 - ((low - value) / spread) * 100.0))
        return float(max(0.0, 100.0 - ((value - high) / spread) * 100.0))

    def _midpoint(
        self, joint_map: Dict[int, dict], a_id: int, b_id: int
    ) -> Optional[np.ndarray]:
        a = joint_map.get(a_id)
        b = joint_map.get(b_id)
        if a is None or b is None:
            return None
        return np.array(
            [
                (float(a["x"]) + float(b["x"])) * 0.5,
                (float(a["y"]) + float(b["y"])) * 0.5,
                (float(a.get("z", 0.0)) + float(b.get("z", 0.0))) * 0.5,
            ],
            dtype=np.float64,
        )

    def _joint_angle(
        self, joint_map: Dict[int, dict], a_id: int, b_id: int, c_id: int
    ) -> Optional[float]:
        a = self._joint_vec(joint_map, a_id)
        b = self._joint_vec(joint_map, b_id)
        c = self._joint_vec(joint_map, c_id)
        if a is None or b is None or c is None:
            return None
        ba = a - b
        bc = c - b
        return self._angle_between(ba, bc)

    def _hip_rotation_proxy(self, joint_map: Dict[int, dict]) -> Optional[float]:
        left_hip = self._joint_vec(joint_map, 23)
        right_hip = self._joint_vec(joint_map, 24)
        if left_hip is None or right_hip is None:
            return None
        hip_vec = right_hip - left_hip
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return self._angle_between(hip_vec, x_axis)

    def _joint_vec(self, joint_map: Dict[int, dict], joint_id: int) -> Optional[np.ndarray]:
        joint = joint_map.get(joint_id)
        if joint is None:
            return None
        return np.array(
            [
                float(joint["x"]),
                float(joint["y"]),
                float(joint.get("z", 0.0)),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 1e-12 or n2 <= 1e-12:
            return None
        cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))
