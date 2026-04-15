from __future__ import annotations

import trimesh

from ml_cv_engine.avatar_3d import AvatarGenerator


def _mock_angle_joints(offset_x: float = 0.0, offset_z: float = 0.0):
    return {
        "joints": [
            {"id": 11, "name": "left_shoulder", "x": 0.40 + offset_x, "y": 0.30, "z": 0.05 + offset_z, "confidence": 0.99},
            {"id": 12, "name": "right_shoulder", "x": 0.60 + offset_x, "y": 0.30, "z": 0.05 + offset_z, "confidence": 0.99},
            {"id": 13, "name": "left_elbow", "x": 0.35 + offset_x, "y": 0.42, "z": 0.03 + offset_z, "confidence": 0.99},
            {"id": 14, "name": "right_elbow", "x": 0.65 + offset_x, "y": 0.42, "z": 0.03 + offset_z, "confidence": 0.99},
            {"id": 15, "name": "left_wrist", "x": 0.30 + offset_x, "y": 0.54, "z": 0.02 + offset_z, "confidence": 0.99},
            {"id": 16, "name": "right_wrist", "x": 0.70 + offset_x, "y": 0.54, "z": 0.02 + offset_z, "confidence": 0.99},
            {"id": 23, "name": "left_hip", "x": 0.45 + offset_x, "y": 0.55, "z": 0.06 + offset_z, "confidence": 0.99},
            {"id": 24, "name": "right_hip", "x": 0.55 + offset_x, "y": 0.55, "z": 0.06 + offset_z, "confidence": 0.99},
            {"id": 25, "name": "left_knee", "x": 0.46 + offset_x, "y": 0.72, "z": 0.08 + offset_z, "confidence": 0.99},
            {"id": 26, "name": "right_knee", "x": 0.54 + offset_x, "y": 0.72, "z": 0.08 + offset_z, "confidence": 0.99},
            {"id": 27, "name": "left_ankle", "x": 0.47 + offset_x, "y": 0.90, "z": 0.10 + offset_z, "confidence": 0.99},
            {"id": 28, "name": "right_ankle", "x": 0.53 + offset_x, "y": 0.90, "z": 0.10 + offset_z, "confidence": 0.99},
        ]
    }


def test_generate_from_multi_angle_returns_trimesh():
    generator = AvatarGenerator()
    skeleton_data = {
        "angles": {
            "front": _mock_angle_joints(0.00, 0.00),
            "left": _mock_angle_joints(-0.02, 0.01),
            "right": _mock_angle_joints(0.02, -0.01),
            "back": _mock_angle_joints(0.00, 0.00),
        }
    }
    mesh = generator.generate_from_multi_angle(skeleton_data)
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty


def test_generate_from_multi_angle_single_view_filled():
    """Only ``front`` supplied — internal normalization duplicates for triangulation."""
    generator = AvatarGenerator()
    skeleton_data = {
        "angles": {
            "front": _mock_angle_joints(0.00, 0.00),
        }
    }
    mesh = generator.generate_from_multi_angle(skeleton_data)
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty


def test_generate_from_multi_angle_with_optional_top_view():
    generator = AvatarGenerator()
    skeleton_data = {
        "angles": {
            "front": _mock_angle_joints(0.00, 0.00),
            "left": _mock_angle_joints(-0.02, 0.01),
            "right": _mock_angle_joints(0.02, -0.01),
            "back": _mock_angle_joints(0.00, 0.00),
            "top": _mock_angle_joints(0.01, -0.01),
        }
    }
    mesh = generator.generate_from_multi_angle(skeleton_data)
    assert isinstance(mesh, trimesh.Trimesh)
    assert not mesh.is_empty


def test_export_glb_creates_file(tmp_path):
    generator = AvatarGenerator()
    skeleton_data = {
        "angles": {
            "front": _mock_angle_joints(0.00, 0.00),
            "left": _mock_angle_joints(-0.02, 0.01),
            "right": _mock_angle_joints(0.02, -0.01),
            "back": _mock_angle_joints(0.00, 0.00),
        }
    }
    mesh = generator.generate_from_multi_angle(skeleton_data)
    output_path = tmp_path / "avatar.glb"
    returned = generator.export_glb(mesh, str(output_path))
    assert returned == str(output_path)
    assert output_path.exists()
