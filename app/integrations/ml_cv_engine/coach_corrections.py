"""Coach correction blending, validation, and diff utilities."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# Subset of key bone connections used for plausibility checks.
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (11, 12),  # shoulders
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (23, 24),  # hips
    (11, 23),
    (12, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


class CorrectionHandler:
    """Apply, inspect, and validate coach-provided joint corrections."""

    def apply_correction(
        self,
        original_joints: list,
        adjusted_joints: list,
        blend_factor: float = 1.0,
    ) -> list:
        """Blend original joints with coach-adjusted joints.

        Args:
            original_joints: Full joint list from detector.
            adjusted_joints: Partial or full list containing only corrected joints.
            blend_factor: 0.0 keeps original, 1.0 uses corrected, 0.5 midpoint.

        Returns:
            Merged list of joints preserving original joints not touched by coach.
        """
        blend = float(np.clip(blend_factor, 0.0, 1.0))
        original_map = self._to_joint_map(original_joints)
        adjusted_map = self._to_joint_map(adjusted_joints)

        merged: List[dict] = []
        for joint_id in sorted(original_map.keys()):
            base = dict(original_map[joint_id])
            corrected = adjusted_map.get(joint_id)
            if corrected is None:
                merged.append(base)
                continue

            out_joint = dict(base)
            for key in ("x", "y", "z"):
                base_val = float(base.get(key, 0.0))
                corr_val = float(corrected.get(key, base_val))
                out_joint[key] = float((1.0 - blend) * base_val + blend * corr_val)

            if "confidence" in corrected:
                base_conf = float(base.get("confidence", 1.0))
                corr_conf = float(corrected["confidence"])
                out_joint["confidence"] = float(
                    (1.0 - blend) * base_conf + blend * corr_conf
                )
            merged.append(out_joint)

        return merged

    def diff_skeletons(self, original_joints: list, corrected_joints: list) -> list:
        """Return joints that differ between original and corrected skeletons."""
        original_map = self._to_joint_map(original_joints)
        corrected_map = self._to_joint_map(corrected_joints)
        diffs: List[dict] = []

        for joint_id, corrected in corrected_map.items():
            original = original_map.get(joint_id)
            if original is None:
                diffs.append(corrected)
                continue

            changed = any(
                not np.isclose(
                    float(original.get(key, 0.0)),
                    float(corrected.get(key, original.get(key, 0.0))),
                    atol=1e-6,
                )
                for key in ("x", "y", "z", "confidence")
            )
            if changed:
                diffs.append(corrected)
        return diffs

    def validate_correction(self, joints: list) -> bool:
        """Validate normalized ranges and basic bone-length plausibility."""
        joint_map = self._to_joint_map(joints)

        for joint in joint_map.values():
            x = float(joint.get("x", -1.0))
            y = float(joint.get("y", -1.0))
            if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                LOGGER.warning(
                    "Invalid correction: joint id=%s has out-of-range x/y values.",
                    joint.get("id"),
                )
                return False

        # Simple anatomical plausibility:
        # bone lengths should be neither collapsed nor unrealistically large in normalized space.
        lengths: List[float] = []
        for a_id, b_id in POSE_CONNECTIONS:
            a = joint_map.get(a_id)
            b = joint_map.get(b_id)
            if a is None or b is None:
                continue
            av = np.array(
                [float(a.get("x", 0.0)), float(a.get("y", 0.0)), float(a.get("z", 0.0))]
            )
            bv = np.array(
                [float(b.get("x", 0.0)), float(b.get("y", 0.0)), float(b.get("z", 0.0))]
            )
            lengths.append(float(np.linalg.norm(av - bv)))

        for value in lengths:
            if value < 0.005 or value > 1.2:
                LOGGER.warning(
                    "Invalid correction: implausible bone length detected (%.4f).", value
                )
                return False

        if lengths:
            median = float(np.median(lengths))
            for value in lengths:
                if value > median * 3.0 or value < max(median * 0.15, 1e-4):
                    LOGGER.warning(
                        "Invalid correction: abnormal relative bone length (%.4f).",
                        value,
                    )
                    return False

        return True

    @staticmethod
    def _to_joint_map(joints: list) -> Dict[int, dict]:
        """Convert list of joints to dict keyed by id."""
        result: Dict[int, dict] = {}
        for joint in joints or []:
            if "id" not in joint:
                continue
            result[int(joint["id"])] = joint
        return result
