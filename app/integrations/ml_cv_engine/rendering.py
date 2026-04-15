"""Avatar rendering utilities for multi-angle PNG generation.

Renders each view with a golf-course background (sky gradient + green grass)
composited behind the avatar so the final PNG looks like a real scene.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import cv2
import numpy as np
import trimesh


def _golf_background(h: int, w: int) -> np.ndarray:
    """Generate a golf-course background: azure sky above, green grass below (BGR)."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    horizon = int(h * 0.58)   # horizon line at 58 % from top

    for y in range(h):
        if y < horizon:
            t = y / max(horizon, 1)           # 0 = top, 1 = horizon
            # Sky: deep azure → pale hazy white at horizon
            r = int(120 + 125 * t)            # 120 → 245
            g = int(185 + 58  * t)            # 185 → 243
            b = int(230 + 15  * t)            # 230 → 245
            bg[y, :] = [b, g, r]              # BGR
        else:
            t = (y - horizon) / max(h - horizon, 1)   # 0 = horizon, 1 = bottom
            # Grass: light green near horizon → darker at bottom
            r = int(78  - 28 * t)             # 78  → 50
            g = int(148 - 38 * t)             # 148 → 110
            b = int(58  - 18 * t)             # 58  → 40
            bg[y, :] = [b, g, r]              # BGR

    return bg


def _composite_over_background(
    rendered: np.ndarray,
    bg: np.ndarray,
) -> np.ndarray:
    """Alpha-composite a rendered BGRA (or BGR) image over a background."""
    if rendered.ndim == 3 and rendered.shape[2] == 4:
        # BGRA render — use the alpha channel
        alpha = rendered[:, :, 3:4].astype(np.float32) / 255.0
        fg    = rendered[:, :, :3].astype(np.float32)
        result = (fg * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    else:
        # BGR render — treat near-white pixels as background
        gray = rendered[:, :, :3].astype(np.uint16).sum(axis=2)
        mask = (gray > 740).astype(np.float32)[:, :, np.newaxis]   # white ≈ (255,255,255)
        result = (
            rendered[:, :, :3].astype(np.float32) * (1.0 - mask)
            + bg.astype(np.float32) * mask
        ).astype(np.uint8)
    return result


class AvatarRenderer:
    """Render 3D avatar meshes from predefined camera angles."""

    IMAGE_SIZE: Tuple[int, int] = (1024, 1024)
    _INTERNAL_RES_SCALE: int = 2

    def render_5_angles(self, mesh: trimesh.Trimesh) -> dict:
        """Render top / front / left / right / back views.

        Returns:
            Dict keyed by angle name; each value is a BGR ``np.ndarray``.
        """
        if mesh is None or mesh.is_empty:
            raise ValueError("Input mesh is empty or invalid.")

        # (azimuth_deg, elevation_deg, show_floor)
        views: Dict[str, Tuple[float, float, bool]] = {
            "front": (  0.0, 10.0, True),
            "back":  (180.0, 10.0, True),
            "left":  ( 90.0, 10.0, True),
            "right": (270.0, 10.0, True),
            "top":   (  0.0, 88.0, False),
        }

        renders: Dict[str, np.ndarray] = {}
        for name, (az, el, floor) in views.items():
            renders[name] = self._render_single(mesh, az, el, show_floor=floor)
        return renders

    def save_renders(self, renders: dict, output_dir: str) -> dict:
        """Save rendered images as ``avatar_<angle>.png`` under *output_dir*."""
        os.makedirs(output_dir, exist_ok=True)
        required = ["top", "front", "left", "right", "back"]
        saved: Dict[str, str] = {}
        for angle in required:
            if angle not in renders:
                raise ValueError(f"Missing render for angle '{angle}'.")
            file_path = os.path.join(output_dir, f"avatar_{angle}.png")
            ok = cv2.imwrite(file_path, renders[angle])
            if not ok:
                raise ValueError(f"Failed to save render: {file_path}")
            saved[angle] = file_path
        return saved

    # ------------------------------------------------------------------ #

    def _render_single(
        self,
        mesh: trimesh.Trimesh,
        azimuth_deg: float,
        elevation_deg: float,
        *,
        show_floor: bool = True,
    ) -> np.ndarray:
        scene = trimesh.Scene()

        bounds  = mesh.bounds
        center  = (bounds[0] + bounds[1]) * 0.5
        extents = bounds[1] - bounds[0]
        span    = float(max(extents) + 1e-6)
        radius  = float(np.linalg.norm(extents) * 1.30 + 1e-6)

        if show_floor:
            min_y      = float(bounds[0][1])
            floor_ext  = max(span * 1.8, 0.6)
            floor_thick = max(span * 0.012, 0.015)
            floor = trimesh.creation.box(extents=(floor_ext, floor_thick, floor_ext))
            floor.apply_translation(
                np.array([center[0], min_y - 0.025 - floor_thick * 0.5, center[2]],
                         dtype=np.float64)
            )
            # Grass-green floor
            floor_rgba = np.array([[88, 148, 68, 255]], dtype=np.uint8)
            floor.visual = trimesh.visual.ColorVisuals(
                mesh=floor,
                vertex_colors=np.repeat(floor_rgba, len(floor.vertices), axis=0),
            )
            scene.add_geometry(floor, geom_name="ground")

        scene.add_geometry(mesh, geom_name="avatar")

        azimuth   = np.radians(azimuth_deg)
        elevation = np.radians(elevation_deg)

        fov = (42.0, 42.0)
        scene.camera.fov = fov
        scene.set_camera(
            angles=(elevation, azimuth, 0.0),
            distance=radius,
            center=center,
            fov=fov,
        )

        iw, ih = self.IMAGE_SIZE
        scale = max(1, int(self._INTERNAL_RES_SCALE))
        png_bytes = scene.save_image(
            resolution=(iw * scale, ih * scale),
            visible=True,
        )
        if png_bytes is None:
            raise ValueError("Failed to render image with trimesh scene.")

        buf   = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Rendered image decode failed.")

        # Composite over golf-course background
        bg = _golf_background(image.shape[0], image.shape[1])
        image = _composite_over_background(image, bg)

        if scale > 1:
            image = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_AREA)

        return image
