"""3D avatar generation from multi-angle skeleton observations.

Produces a realistic, fully-connected volumetric human mesh using:
- Frustum (tapered cylinder) geometry for arms and legs
- Octagonal torso frustum (wider at shoulders, narrower at hips)
- Head ellipsoid + white cap (brim + crown)
- Neck, hands, belt, and shoe geometry
- Vertex colours sampled region-by-region from source photographs
- Absolute-scale Z depth priors (same units as MediaPipe X/Y)
"""

from __future__ import annotations

import os
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import cv2
import numpy as np
import trimesh

POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

# ---------------------------------------------------------------------------
# Depth priors — absolute normalized units (same scale as MediaPipe X, Y).
# Positive Z = toward the front camera.  Shoulder plane = 0.
# ---------------------------------------------------------------------------
_Z_PRIOR: Dict[int, float] = {
    0:  0.10,                              # nose
    1:  0.08,  2:  0.06,  3:  0.04,
    4:  0.08,  5:  0.06,  6:  0.04,
    7: -0.02,  8: -0.02,
    9:  0.09, 10:  0.09,
    11: 0.00, 12:  0.00,
    13: 0.06, 14:  0.06,
    15: 0.12, 16:  0.12,
    17: 0.14, 18:  0.14,
    19: 0.14, 20:  0.14,
    21: 0.13, 22:  0.13,
    23:-0.04, 24: -0.04,
    25: 0.02, 26:  0.02,
    27: 0.04, 28:  0.04,
    29: 0.03, 30:  0.03,
    31: 0.07, 32:  0.07,
}

# ---------------------------------------------------------------------------
# Limb segments as frustums: (joint_a, joint_b, r_at_a, r_at_b, colour_region)
# Radii are fractions of body_h.  Tapered for natural human proportions.
# ---------------------------------------------------------------------------
_SEGS_FRUSTUM: List[Tuple[int, int, float, float, str]] = [
    # ── Left arm ──────────────────────────────────────────────────────────
    (11, 13, 0.050, 0.040, "l_upper_arm"),   # shoulder → elbow
    (13, 15, 0.038, 0.026, "l_forearm"),     # elbow → wrist
    # ── Right arm ─────────────────────────────────────────────────────────
    (12, 14, 0.050, 0.040, "r_upper_arm"),
    (14, 16, 0.038, 0.026, "r_forearm"),
    # ── Left leg ──────────────────────────────────────────────────────────
    (23, 25, 0.064, 0.052, "l_thigh"),       # hip → knee
    (25, 27, 0.050, 0.034, "l_shin"),        # knee → ankle
    # ── Right leg ─────────────────────────────────────────────────────────
    (24, 26, 0.064, 0.052, "r_thigh"),
    (26, 28, 0.050, 0.034, "r_shin"),
]

# ---------------------------------------------------------------------------
# Realistic golf-player colour palette (fallback when no source image).
# Order: RGBA  (R, G, B, 255).
# ---------------------------------------------------------------------------
_FALLBACK: Dict[str, np.ndarray] = {
    "head":        np.array([200, 155, 120, 255], dtype=np.uint8),  # warm skin
    "hat":         np.array([245, 245, 245, 255], dtype=np.uint8),  # white cap
    "neck":        np.array([195, 150, 115, 255], dtype=np.uint8),  # skin
    "torso":       np.array([ 70, 130, 180, 255], dtype=np.uint8),  # blue polo
    "l_upper_arm": np.array([ 70, 130, 180, 255], dtype=np.uint8),  # blue sleeve
    "r_upper_arm": np.array([ 70, 130, 180, 255], dtype=np.uint8),  # blue sleeve
    "l_forearm":   np.array([195, 150, 115, 255], dtype=np.uint8),  # skin
    "r_forearm":   np.array([195, 150, 115, 255], dtype=np.uint8),  # skin
    "l_hand":      np.array([225, 222, 215, 255], dtype=np.uint8),  # white glove
    "r_hand":      np.array([195, 150, 115, 255], dtype=np.uint8),  # skin
    "belt":        np.array([ 30,  22,  14, 255], dtype=np.uint8),  # dark belt
    "l_thigh":     np.array([ 58,  62,  70, 255], dtype=np.uint8),  # charcoal pants
    "r_thigh":     np.array([ 58,  62,  70, 255], dtype=np.uint8),
    "l_shin":      np.array([ 52,  56,  64, 255], dtype=np.uint8),
    "r_shin":      np.array([ 52,  56,  64, 255], dtype=np.uint8),
    "l_shoe":      np.array([235, 232, 225, 255], dtype=np.uint8),  # white shoes
    "r_shoe":      np.array([235, 232, 225, 255], dtype=np.uint8),
}

# Which 2D landmarks bound each colour region (for image sampling)
_COLOUR_REGIONS: Dict[str, List[int]] = {
    "head":        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "torso":       [11, 12, 23, 24],
    "l_upper_arm": [11, 13],
    "l_forearm":   [13, 15],
    "r_upper_arm": [12, 14],
    "r_forearm":   [14, 16],
    "l_hand":      [15, 17, 19, 21],
    "r_hand":      [16, 18, 20, 22],
    "l_thigh":     [23, 25],
    "l_shin":      [25, 27],
    "r_thigh":     [24, 26],
    "r_shin":      [26, 28],
    "l_shoe":      [27, 29, 31],
    "r_shoe":      [28, 30, 32],
}

_LIMB_SEGS = 22   # radial segments for limbs (smooth but not heavy)
_HEAD_COUNT = [26, 26]


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _paint(mesh: trimesh.Trimesh, rgba: np.ndarray) -> None:
    rgba = np.asarray(rgba, dtype=np.uint8).reshape(4)
    colors = np.broadcast_to(rgba, (len(mesh.vertices), 4)).copy()
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)


def _frustum(
    p0: np.ndarray,
    p1: np.ndarray,
    r0: float,
    r1: float,
    segs: int = _LIMB_SEGS,
) -> Optional[trimesh.Trimesh]:
    """Tapered cylinder (frustum) from p0 (radius r0) to p1 (radius r1).

    Uses direct vertex/face construction so trimesh's cylinder alignment
    quirks cannot affect the result.
    """
    vec = np.asarray(p1, float) - np.asarray(p0, float)
    L = float(np.linalg.norm(vec))
    if L < 1e-8:
        return None

    theta = np.linspace(0.0, 2.0 * np.pi, segs, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)

    # Vertices in Z-up local frame
    vbot = np.column_stack([r0 * c, r0 * s, np.zeros(segs)])
    vtop = np.column_stack([r1 * c, r1 * s, np.full(segs, L)])
    cbi, cti = 2 * segs, 2 * segs + 1
    verts = np.vstack([vbot, vtop,
                       np.array([[0.0, 0.0, 0.0]]),
                       np.array([[0.0, 0.0, L]])])

    n = segs
    tri: List[List[int]] = []
    for i in range(n):
        j = (i + 1) % n
        tri += [[i, j, n + i], [j, n + j, n + i]]   # side quads
        tri.append([cbi, j, i])                       # bottom cap
        tri.append([cti, n + i, n + j])               # top cap

    mesh = trimesh.Trimesh(
        vertices=verts, faces=np.array(tri, dtype=np.int64), process=False
    )

    # Rotate local Z-axis to align with vec
    z_ax = np.array([0.0, 0.0, 1.0])
    axis = vec / L
    dot = float(np.clip(np.dot(z_ax, axis), -1.0, 1.0))
    if dot > 0.9999:
        rot4 = np.eye(4)
    elif dot < -0.9999:
        rot4 = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        cross = np.cross(z_ax, axis)
        rot4 = trimesh.transformations.rotation_matrix(float(np.arccos(dot)), cross)

    rot4[:3, 3] = np.asarray(p0, float)
    mesh.apply_transform(rot4)
    return mesh


def _sphere(center: np.ndarray, radius: float) -> trimesh.Trimesh:
    s = trimesh.creation.uv_sphere(radius=float(radius), count=[18, 18])
    s.apply_translation(center)
    return s


# ---------------------------------------------------------------------------
# Colour sampling from source photograph
# ---------------------------------------------------------------------------

def _sample_region_colours(
    img: np.ndarray,
    joints_2d: List[dict],
) -> Dict[str, np.ndarray]:
    """Median BGR→RGB colour from the bounding box of each region's landmarks."""
    ih, iw = img.shape[:2]
    jmap = {int(j["id"]): j for j in joints_2d if "id" in j}

    def _px(jid: int) -> Optional[Tuple[int, int]]:
        j = jmap.get(jid)
        if j is None:
            return None
        return (
            int(np.clip(float(j["x"]) * iw, 0, iw - 1)),
            int(np.clip(float(j["y"]) * ih, 0, ih - 1)),
        )

    def _region_colour(joint_ids: List[int], pad: int = 20) -> Optional[np.ndarray]:
        pxs = [_px(jid) for jid in joint_ids]
        pxs = [p for p in pxs if p is not None]
        if not pxs:
            return None
        x0 = max(0, min(p[0] for p in pxs) - pad)
        y0 = max(0, min(p[1] for p in pxs) - pad)
        x1 = min(iw, max(p[0] for p in pxs) + pad)
        y1 = min(ih, max(p[1] for p in pxs) + pad)
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        b = int(np.median(patch[:, :, 0]))
        g = int(np.median(patch[:, :, 1]))
        r = int(np.median(patch[:, :, 2]))
        return np.array([r, g, b, 255], dtype=np.uint8)

    result: Dict[str, np.ndarray] = {}
    for region, jids in _COLOUR_REGIONS.items():
        c = _region_colour(jids)
        result[region] = c if c is not None else _FALLBACK.get(region, _FALLBACK["torso"])

    # Hat: sample from region directly above the nose (above face landmarks)
    nose = jmap.get(0)
    if nose is not None:
        nx = int(float(nose["x"]) * iw)
        ny = int(float(nose["y"]) * ih)
        pad_h = max(14, int(ih * 0.06))
        y_top = max(0, ny - pad_h * 3)
        y_bot = max(0, ny - pad_h)
        x_l = max(0, nx - pad_h)
        x_r = min(iw, nx + pad_h)
        patch = img[y_top:y_bot, x_l:x_r]
        if patch.size > 0:
            b = int(np.median(patch[:, :, 0]))
            g = int(np.median(patch[:, :, 1]))
            r = int(np.median(patch[:, :, 2]))
            result["hat"] = np.array([r, g, b, 255], dtype=np.uint8)

    # Belt: sample narrow horizontal band near the top of the hip region
    for hip_id in (23, 24):
        hp = _px(hip_id)
        if hp is not None:
            px, py = hp
            pad_b = max(10, int(ih * 0.025))
            patch = img[max(0, py - pad_b):min(ih, py + pad_b),
                        max(0, px - pad_b * 2):min(iw, px + pad_b * 2)]
            if patch.size > 0:
                b = int(np.median(patch[:, :, 0]))
                g = int(np.median(patch[:, :, 1]))
                r = int(np.median(patch[:, :, 2]))
                result["belt"] = np.array([r, g, b, 255], dtype=np.uint8)
            break

    # Fill any still-missing regions with fallback
    for k, v in _FALLBACK.items():
        result.setdefault(k, v)

    return result


def _load_image(path: str) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(path, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AvatarGenerator:
    """Build a realistic, colour-sampled 3-D human avatar from skeleton data."""

    def __init__(self) -> None:
        self._source_images: Dict[str, str] = {}
        self._source_joints_2d: Dict[str, List[dict]] = {}

    def set_source_images(
        self,
        image_paths: Dict[str, str],
        angle_data: Optional[dict] = None,
    ) -> None:
        self._source_images = dict(image_paths)
        self._source_joints_2d.clear()
        if angle_data:
            for key in ("front", "left", "right", "back", "top"):
                block = angle_data.get(key)
                if isinstance(block, dict) and block.get("joints"):
                    self._source_joints_2d[key] = list(block["joints"])

    # ------------------------------------------------------------------ #

    def generate_from_multi_angle(self, skeleton_data: dict) -> trimesh.Trimesh:
        raw = dict(skeleton_data.get("angles", skeleton_data))
        angle_data, real_views = self._normalize_angles(raw)

        for key in ("front", "left", "right", "back", "top"):
            block = angle_data.get(key)
            if key not in self._source_joints_2d and isinstance(block, dict) and block.get("joints"):
                self._source_joints_2d[key] = list(block["joints"])

        joints_3d = self._triangulate(angle_data, real_views)
        if not joints_3d:
            return trimesh.creation.box(extents=(0.25, 0.75, 0.20))

        body_h = self._body_height(joints_3d)
        colours = self._get_colours(real_views)
        return self._build_mesh(joints_3d, body_h, colours)

    def export_obj(self, mesh: trimesh.Trimesh, output_path: str) -> str:
        self._mkdir(output_path)
        mesh.export(output_path, file_type="obj")
        return output_path

    def export_glb(self, mesh: trimesh.Trimesh, output_path: str) -> str:
        self._mkdir(output_path)
        mesh.export(output_path, file_type="glb")
        return output_path

    def export_fbx(self, mesh: trimesh.Trimesh, output_path: str) -> str:
        self._mkdir(output_path)
        # Always use our own ASCII writer — trimesh's native FBX export is
        # unreliable (missing normals, vertex colours, root connection).
        try:
            self._write_fbx_ascii(mesh, output_path)
            return output_path
        except BaseException:
            # Ultimate fallback: GLB (full fidelity, rename if needed)
            fb = os.path.splitext(output_path)[0] + ".glb"
            mesh.export(fb, file_type="glb")
            return fb

    # ------------------------------------------------------------------ #
    # Colour resolution
    # ------------------------------------------------------------------ #

    def _get_colours(self, real_views: FrozenSet[str]) -> Dict[str, np.ndarray]:
        for view in ("front", "left", "right", "back", "top"):
            if view not in real_views:
                continue
            img_path = self._source_images.get(view)
            joints_2d = self._source_joints_2d.get(view)
            if not img_path or not joints_2d:
                continue
            img = _load_image(img_path)
            if img is None:
                continue
            return _sample_region_colours(img, joints_2d)
        return dict(_FALLBACK)

    # ------------------------------------------------------------------ #
    # Mesh construction
    # ------------------------------------------------------------------ #

    def _build_mesh(
        self,
        pts: Dict[int, np.ndarray],
        body_h: float,
        colours: Dict[str, np.ndarray],
    ) -> trimesh.Trimesh:
        parts: List[trimesh.Trimesh] = []

        # Compute up-direction (shoulder-mid → nose)
        up = np.array([0.0, 1.0, 0.0])
        if 0 in pts and 11 in pts and 12 in pts:
            smid = (pts[11] + pts[12]) * 0.5
            up_raw = pts[0] - smid
            n = np.linalg.norm(up_raw)
            if n > 1e-6:
                up = up_raw / n

        # ── HEAD ──────────────────────────────────────────────────────────
        head_r = body_h * 0.112
        if 0 in pts:
            head = trimesh.creation.uv_sphere(radius=1.0, count=_HEAD_COUNT)
            # Slightly egg-shaped: taller than wide, shallower front-back
            head.apply_transform(np.diag([head_r, head_r * 1.20, head_r * 0.88, 1.0]))
            head_center = pts[0] + up * head_r * 0.28
            T = np.eye(4); T[:3, 3] = head_center
            head.apply_transform(T)
            _paint(head, colours.get("head", _FALLBACK["head"]))
            parts.append(head)

            hat_col = colours.get("hat", _FALLBACK["hat"])

            # Hat crown (slightly tapered cylinder)
            crown_base = head_center + up * head_r * 0.52
            crown_tip  = head_center + up * head_r * 1.42
            crown = _frustum(crown_base, crown_tip,
                             head_r * 0.90, head_r * 0.80, segs=28)
            if crown is not None:
                _paint(crown, hat_col)
                parts.append(crown)

            # Hat brim (thin wide disk)
            brim_bot = head_center + up * head_r * 0.50
            brim_top = head_center + up * head_r * 0.57
            brim = _frustum(brim_bot, brim_top,
                            head_r * 1.45, head_r * 1.45, segs=36)
            if brim is not None:
                _paint(brim, hat_col)
                parts.append(brim)

        # ── NECK ──────────────────────────────────────────────────────────
        if 0 in pts and 11 in pts and 12 in pts:
            smid = (pts[11] + pts[12]) * 0.5
            neck_bot = smid
            neck_top = pts[0] - up * head_r * 0.62
            neck = _frustum(neck_bot, neck_top,
                            body_h * 0.032, body_h * 0.026, segs=16)
            if neck is not None:
                _paint(neck, colours.get("neck", _FALLBACK["neck"]))
                parts.append(neck)

        # ── TORSO ─────────────────────────────────────────────────────────
        if all(j in pts for j in (11, 12, 23, 24)):
            smid = (pts[11] + pts[12]) * 0.5
            hmid = (pts[23] + pts[24]) * 0.5
            sw   = float(np.linalg.norm(pts[11] - pts[12]))
            hw   = float(np.linalg.norm(pts[23] - pts[24]))
            torso_col = colours.get("torso", _FALLBACK["torso"])

            # Main trunk: octagonal prism gives body-like silhouette
            r_top = sw * 0.50
            r_bot = max(hw * 0.47, sw * 0.38)
            trunk = _frustum(smid, hmid, r_top, r_bot, segs=10)
            if trunk is not None:
                _paint(trunk, torso_col)
                parts.append(trunk)

            # Shoulder cap spheres to blend arms into torso
            for sid in (11, 12):
                s = _sphere(pts[sid], body_h * 0.050)
                _paint(s, torso_col)
                parts.append(s)

            # Hip cap spheres
            for hid in (23, 24):
                s = _sphere(pts[hid], body_h * 0.046)
                _paint(s, torso_col)
                parts.append(s)

            # Belt: thin wide ring just above hips
            belt_col  = colours.get("belt", _FALLBACK["belt"])
            spine_vec = smid - hmid
            belt_ctr  = hmid + spine_vec * 0.10
            belt_bot  = belt_ctr - spine_vec * 0.018
            belt_top  = belt_ctr + spine_vec * 0.018
            belt_r    = r_bot * 1.04
            belt = _frustum(belt_bot, belt_top, belt_r, belt_r, segs=10)
            if belt is not None:
                _paint(belt, belt_col)
                parts.append(belt)

        # ── ARMS (tapered frustum) ─────────────────────────────────────────
        for a_id, b_id, ra, rb, region in _SEGS_FRUSTUM[:4]:
            if a_id not in pts or b_id not in pts:
                continue
            seg = _frustum(pts[a_id], pts[b_id], ra * body_h, rb * body_h)
            if seg is not None:
                _paint(seg, colours.get(region, _FALLBACK.get(region, _FALLBACK["torso"])))
                parts.append(seg)
            # Joint sphere at elbow
            if b_id in (13, 14):
                s = _sphere(pts[b_id], rb * body_h * 1.10)
                _paint(s, colours.get(region, _FALLBACK.get(region, _FALLBACK["torso"])))
                parts.append(s)

        # ── HANDS ─────────────────────────────────────────────────────────
        for wrist_id, region in [(15, "l_hand"), (16, "r_hand")]:
            if wrist_id not in pts:
                continue
            hand = _sphere(pts[wrist_id], body_h * 0.030)
            _paint(hand, colours.get(region, _FALLBACK[region]))
            parts.append(hand)

        # ── LEGS (tapered frustum) ─────────────────────────────────────────
        for a_id, b_id, ra, rb, region in _SEGS_FRUSTUM[4:]:
            if a_id not in pts or b_id not in pts:
                continue
            seg = _frustum(pts[a_id], pts[b_id], ra * body_h, rb * body_h)
            if seg is not None:
                _paint(seg, colours.get(region, _FALLBACK.get(region, _FALLBACK["torso"])))
                parts.append(seg)
            # Knee joint sphere
            if b_id in (25, 26):
                s = _sphere(pts[b_id], rb * body_h * 1.08)
                _paint(s, colours.get(region, _FALLBACK.get(region, _FALLBACK["torso"])))
                parts.append(s)

        # ── SHOES ─────────────────────────────────────────────────────────
        shoe_pairs = [(27, 29, 31, "l_shoe"), (28, 30, 32, "r_shoe")]
        for ankle_id, heel_id, toe_id, region in shoe_pairs:
            shoe_col = colours.get(region, _FALLBACK[region])
            if ankle_id in pts:
                s = _sphere(pts[ankle_id], body_h * 0.036)
                _paint(s, shoe_col)
                parts.append(s)
            for j1, j2, ra, rb in [
                (ankle_id, heel_id, 0.028, 0.024),
                (ankle_id, toe_id,  0.028, 0.020),
                (heel_id,  toe_id,  0.022, 0.020),
            ]:
                if j1 in pts and j2 in pts:
                    seg = _frustum(pts[j1], pts[j2],
                                   ra * body_h, rb * body_h, segs=14)
                    if seg is not None:
                        _paint(seg, shoe_col)
                        parts.append(seg)

        if not parts:
            return trimesh.creation.box(extents=(0.25, 0.75, 0.20))
        return trimesh.util.concatenate(parts)

    # ------------------------------------------------------------------ #
    # 3-D joint triangulation
    # ------------------------------------------------------------------ #

    def _triangulate(
        self,
        angle_data: dict,
        real_views: FrozenSet[str],
    ) -> Dict[int, np.ndarray]:
        by_angle: Dict[str, Dict[int, dict]] = {}
        for angle in ("front", "left", "right", "back", "top"):
            block = angle_data.get(angle)
            if not block or "joints" not in block:
                by_angle[angle] = {}
            else:
                by_angle[angle] = {
                    int(j["id"]): j
                    for j in block["joints"]
                    if "id" in j and "x" in j and "y" in j
                }

        all_ids: set = set()
        for d in by_angle.values():
            all_ids.update(d.keys())

        output: Dict[int, np.ndarray] = {}
        for jid in sorted(all_ids):
            fj = by_angle["front"].get(jid)
            bj = by_angle["back"].get(jid)
            lj = by_angle["left"].get(jid)
            rj = by_angle["right"].get(jid)
            tj = by_angle.get("top", {}).get(jid)

            x_v: List[float] = []
            y_v: List[float] = []
            z_v: List[float] = []

            if fj and "front" in real_views:
                x_v.append(float(fj["x"]) - 0.5)
                y_v.append(0.5 - float(fj["y"]))
            if bj and "back" in real_views:
                x_v.append(0.5 - float(bj["x"]))
                y_v.append(0.5 - float(bj["y"]))
            if lj and "left" in real_views:
                z_v.append(float(lj["x"]) - 0.5)
                y_v.append(0.5 - float(lj["y"]))
            if rj and "right" in real_views:
                z_v.append(0.5 - float(rj["x"]))
                y_v.append(0.5 - float(rj["y"]))
            if tj and "top" in real_views:
                x_v.append(float(tj["x"]) - 0.5)
                z_v.append(0.5 - float(tj["y"]))

            if not z_v:
                prior = _Z_PRIOR.get(jid, 0.0)
                mp_z = 0.0
                if fj:
                    mp_z = -float(fj.get("z", 0.0))
                elif bj:
                    mp_z = float(bj.get("z", 0.0))
                z_v.append(prior * 0.65 + mp_z * 0.35)

            if not x_v:
                if lj:
                    x_v.append(-float(lj.get("z", 0.0)))
                elif rj:
                    x_v.append(float(rj.get("z", 0.0)))
            if not y_v:
                for jt in (fj, bj, lj, rj):
                    if jt:
                        y_v.append(0.5 - float(jt["y"]))
                        break

            if not x_v or not y_v or not z_v:
                continue

            output[jid] = np.array(
                [float(np.mean(x_v)), float(np.mean(y_v)), float(np.mean(z_v))],
                dtype=np.float64,
            )

        self._clamp_bone_lengths(output)
        return output

    @staticmethod
    def _clamp_bone_lengths(pts: Dict[int, np.ndarray]) -> None:
        if 11 not in pts or 12 not in pts:
            return
        sw = float(np.linalg.norm(pts[11] - pts[12]))
        if sw < 1e-6:
            return
        limits = {
            (11, 13): 1.4, (13, 15): 1.4,
            (12, 14): 1.4, (14, 16): 1.4,
            (23, 25): 2.0, (25, 27): 2.0,
            (24, 26): 2.0, (26, 28): 2.0,
        }
        for (a, b), max_r in limits.items():
            if a not in pts or b not in pts:
                continue
            vec = pts[b] - pts[a]
            L = float(np.linalg.norm(vec))
            cap = sw * max_r
            if L > cap > 1e-8:
                pts[b] = pts[a] + vec * (cap / L)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_angles(angle_data: dict) -> Tuple[dict, FrozenSet[str]]:
        picked: Dict[str, List[dict]] = {}
        ref: Optional[List[dict]] = None
        for key in ("front", "left", "right", "back", "top"):
            block = angle_data.get(key)
            if not isinstance(block, dict):
                continue
            joints = block.get("joints")
            if joints:
                picked[key] = [dict(j) for j in joints]
                if ref is None:
                    ref = picked[key]

        if not ref:
            raise ValueError("No detected pose in any angle; cannot build 3D avatar.")

        real_views: Set[str] = set(picked.keys())
        out: Dict[str, object] = {}
        for key in ("front", "left", "right", "back"):
            jl = picked.get(key, ref)
            out[key] = {"joints": [dict(j) for j in jl]}
        if "top" in picked:
            out["top"] = {"joints": [dict(j) for j in picked["top"]]}
        return out, frozenset(real_views)

    @staticmethod
    def _body_height(pts: Dict[int, np.ndarray]) -> float:
        cands: List[float] = []
        for head_id in (0, 9, 10):
            for foot_id in (27, 28):
                if head_id in pts and foot_id in pts:
                    cands.append(float(np.linalg.norm(pts[head_id] - pts[foot_id])))
        if 11 in pts and 27 in pts:
            cands.append(float(np.linalg.norm(pts[11] - pts[27])) * 1.22)
        if 12 in pts and 28 in pts:
            cands.append(float(np.linalg.norm(pts[12] - pts[28])) * 1.22)
        if cands:
            return float(max(cands))
        all_y = [p[1] for p in pts.values()]
        return max(all_y) - min(all_y) + 0.1

    @staticmethod
    def _mkdir(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    @staticmethod
    def _write_fbx_ascii(mesh: trimesh.Trimesh, path: str) -> None:
        """Write a valid FBX 7.4 ASCII file with normals, vertex colours and
        correct scene-root connection so Unity/Blender display the mesh.

        Vertices are scaled by 200 so that a normalized body_h ≈ 0.8 becomes
        ≈ 160 cm — a realistic human height in cm-unit FBX files.
        """
        mc = mesh.copy()
        mc.fix_normals()

        # Scale to centimetre units so the figure is ~160 cm tall in Unity
        _FBX_SCALE = 200.0
        verts   = np.asarray(mc.vertices,      dtype=np.float64) * _FBX_SCALE
        faces   = np.asarray(mc.faces,         dtype=np.int64)
        normals = np.asarray(mc.vertex_normals, dtype=np.float64)

        if verts.size == 0 or faces.size == 0:
            raise ValueError("Empty mesh; cannot write FBX.")

        nv = len(verts)
        nf = len(faces)

        v_str = ",".join(f"{v:.6f}" for v in verts.reshape(-1))
        n_str = ",".join(f"{v:.6f}" for v in normals.reshape(-1))

        poly_ix: List[int] = []
        for f in faces:
            poly_ix.extend([int(f[0]), int(f[1]), -int(f[2]) - 1])
        p_str = ",".join(str(i) for i in poly_ix)

        # ── Vertex colour block (RGBA in [0,1]) ──────────────────────────
        color_block = ""
        color_layer_entry = ""
        if (
            hasattr(mc.visual, "vertex_colors")
            and mc.visual.vertex_colors is not None
            and len(mc.visual.vertex_colors) == nv
        ):
            cols = np.asarray(mc.visual.vertex_colors, dtype=np.float32)
            c_vals: List[float] = []
            for c in cols:
                c_vals += [
                    float(c[0]) / 255.0,
                    float(c[1]) / 255.0,
                    float(c[2]) / 255.0,
                    float(c[3]) / 255.0 if len(c) > 3 else 1.0,
                ]
            c_str  = ",".join(f"{v:.6f}" for v in c_vals)
            ci_str = ",".join(str(i) for i in range(nv))
            color_block = (
                "\n\t\tLayerElementVertexColor: 0  {\n"
                "\t\t\tVersion: 101\n"
                '\t\t\tName: "Col"\n'
                '\t\t\tMappingInformationType: "ByVertice"\n'
                '\t\t\tReferenceInformationType: "IndexToDirect"\n'
                f"\t\t\tColors: *{len(c_vals)}  {{\n\t\t\t\ta: {c_str}\n\t\t\t}}\n"
                f"\t\t\tColorIndex: *{nv}  {{\n\t\t\t\ta: {ci_str}\n\t\t\t}}\n"
                "\t\t}"
            )
            color_layer_entry = (
                "\n\t\t\tLayerElement:  {\n"
                '\t\t\t\tType: "LayerElementVertexColor"\n'
                "\t\t\t\tTypedIndex: 0\n"
                "\t\t\t}"
            )

        # ── Assemble FBX sections using plain string concatenation ────────
        # (No f-strings for the structural parts so {{ }} escaping is not
        #  needed for the many literal braces in FBX syntax.)
        header = (
            "; FBX 7.4.0 project file\n"
            "; Generated by ml_cv_engine AvatarGenerator\n\n"
            "FBXHeaderExtension:  {\n"
            "\tFBXHeaderVersion: 1003\n"
            "\tFBXVersion: 7400\n"
            '\tCreator: "ml_cv_engine"\n'
            "}\n\n"
            "GlobalSettings:  {\n"
            "\tVersion: 1000\n"
            "\tProperties70:  {\n"
            '\t\tP: "UpAxis", "int", "Integer", "", 1\n'
            '\t\tP: "UpAxisSign", "int", "Integer", "", 1\n'
            '\t\tP: "FrontAxis", "int", "Integer", "", 2\n'
            '\t\tP: "FrontAxisSign", "int", "Integer", "", 1\n'
            '\t\tP: "CoordAxis", "int", "Integer", "", 0\n'
            '\t\tP: "CoordAxisSign", "int", "Integer", "", 1\n'
            '\t\tP: "UnitScaleFactor", "double", "Number", "", 1\n'
            "\t}\n"
            "}\n\n"
            "Definitions:  {\n"
            "\tVersion: 100\n"
            "\tCount: 2\n"
            '\tObjectType: "Geometry"  {\n'
            "\t\tCount: 1\n"
            "\t}\n"
            '\tObjectType: "Model"  {\n'
            "\t\tCount: 1\n"
            "\t}\n"
            "}\n\n"
        )

        objects_open  = "Objects:  {\n"
        geom_open     = '\tGeometry: 100000, "Geometry::AvatarMesh", "Mesh"  {\n'
        geom_verts    = f"\t\tVertices: *{nv * 3}  {{\n\t\t\ta: {v_str}\n\t\t}}\n"
        geom_polys    = f"\t\tPolygonVertexIndex: *{nf * 3}  {{\n\t\t\ta: {p_str}\n\t\t}}\n"
        geom_version  = "\t\tGeometryVersion: 124\n"
        normal_block  = (
            "\t\tLayerElementNormal: 0  {\n"
            "\t\t\tVersion: 101\n"
            '\t\t\tName: ""\n'
            '\t\t\tMappingInformationType: "ByVertice"\n'
            '\t\t\tReferenceInformationType: "Direct"\n'
            f"\t\t\tNormals: *{nv * 3}  {{\n\t\t\t\ta: {n_str}\n\t\t\t}}\n"
            "\t\t}"
        )
        layer_block   = (
            "\n\t\tLayer: 0  {\n"
            "\t\t\tVersion: 100\n"
            "\t\t\tLayerElement:  {\n"
            '\t\t\t\tType: "LayerElementNormal"\n'
            "\t\t\t\tTypedIndex: 0\n"
            "\t\t\t}"
            + color_layer_entry
            + "\n\t\t}\n"
        )
        geom_close    = "\t}\n"
        model_block   = (
            '\tModel: 100001, "Model::AvatarModel", "Mesh"  {\n'
            "\t\tVersion: 232\n"
            "\t\tProperties70:  {\n"
            '\t\t\tP: "DefaultAttributeIndex", "int", "Integer", "", 0\n'
            "\t\t}\n"
            "\t\tShading: T\n"
            '\t\tCulling: "CullingOff"\n'
            "\t}\n"
        )
        objects_close = "}\n\n"
        connections   = (
            "Connections:  {\n"
            "\tC: \"OO\",100000,100001\n"   # geometry → model
            "\tC: \"OO\",100001,0\n"        # model → scene root
            "}\n"
        )

        content = (
            header
            + objects_open
            + geom_open
            + geom_verts
            + geom_polys
            + geom_version
            + normal_block
            + color_block
            + layer_block
            + geom_close
            + model_block
            + objects_close
            + connections
        )

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
