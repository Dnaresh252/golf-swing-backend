"""FastAPI-ready orchestration and Celery task wrappers for ML/CV engine.

Example usage:
    # FastAPI endpoint style:
    # result = await analyze_submission("sub_12345", image_paths, "swing.mp4")
    #
    # Celery task style:
    # result = celery_task_analyze("sub_12345", image_paths, "swing.mp4")
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from .avatar_3d import AvatarGenerator
from .backend_client import BackendApiError, BackendClient
from .coach_corrections import CorrectionHandler
from .golf_analysis import GolfSwingAnalyzer
from .overlay_2d import SkeletonOverlay
from .rendering import AvatarRenderer
from .skeleton_processing import SkeletonProcessor
from .utils import prepare_video_for_skeleton_pipeline
from .video_view_detection import detect_head_overhead_view_video

# Default overlay configuration for exported videos.
# Callers can override any key when invoking the pipeline.
# ``static_labels``: list of dicts for coach markers/headlines (see ``coach_overlay_labels``).
DEFAULT_OVERLAY_OPTIONS: Dict[str, Any] = {
    "show_shoulders": True,
    "show_waistline": True,
    "show_legs": True,
    "show_spine": True,
    "show_head_reference": True,
    "static_lines": None,
    "static_labels": None,
    "tracking_lines": None,
}

# Coaching score-card axes (see ``GolfSwingAnalyzer.export_score_card``).
_DEFAULT_SCORECARD_IDEAL_RANGES: Dict[str, tuple] = {
    "spine_angle": (20.0, 48.0),
    "hip_rotation_angle": (25.0, 55.0),
    "arm_extension": (150.0, 180.0),
    "balance": (42.0, 58.0),
    "tempo": (2.5, 3.5),
}

# Protractor overlays at impact (see ``SkeletonOverlay.draw_angle_arcs``).
_ARC_IDEAL_RANGES: Dict[str, tuple] = {
    "spine_angle": (20.0, 48.0),
    "hip_rotation_angle": (25.0, 55.0),
    "knee_flex_left": (150.0, 175.0),
    "knee_flex_right": (150.0, 175.0),
    "arm_extension_left": (150.0, 180.0),
    "arm_extension_right": (150.0, 180.0),
}


def _read_video_frame_bgr(video_path: str, frame_index: int) -> Optional[Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _json_safe_tempo_metrics(tempo_info: dict) -> dict:
    out: dict = {}
    for key, val in tempo_info.items():
        if key == "chart_path":
            continue
        if isinstance(val, tuple):
            out[key] = list(val)
        else:
            out[key] = val
    return out


def _pipeline_output_dir(submission_id: str) -> str:
    """Build an output directory for generated artifacts."""
    return os.path.join("outputs", submission_id)


def _to_backend_skeleton_schema(submission_id: str, video_skeleton: dict) -> dict:
    """Convert video skeleton to backend-required DB schema."""
    frames = []
    for frame in video_skeleton.get("frames", []):
        if "frame_num" not in frame:
            continue
        frame_num = int(frame["frame_num"])
        timestamp = frame.get("timestamp")
        if timestamp is None:
            timestamp = 0.0
        frames.append(
            {
                "frame_num": frame_num,
                "timestamp": float(timestamp),
                "joints": frame.get("joints", []),
            }
        )
    return {"submission_id": submission_id, "frames": frames}


def _write_multi_angle_artifacts(
    output_dir: str,
    image_paths: dict,
    image_skeleton: dict,
    overlay: SkeletonOverlay,
    analysis_warnings: List[str],
) -> tuple[Optional[str], Dict[str, str]]:
    """Write ``multi_angle_skeleton.json`` and ``skeleton_overlay_on_samples/<angle>.png``.

    Returns ``(multi_angle_json_path, overlay_paths_by_angle)``. Missing angles are
    omitted from the overlay dict when pose detection failed or draw raised.
    """
    multi_path = os.path.join(output_dir, "multi_angle_skeleton.json")
    try:
        with open(multi_path, "w", encoding="utf-8") as f:
            json.dump(image_skeleton, f, indent=2)
    except OSError as exc:
        analysis_warnings.append(f"multi_angle_skeleton_json: {exc}")
        multi_path = None

    overlay_dir = os.path.join(output_dir, "skeleton_overlay_on_samples")
    os.makedirs(overlay_dir, exist_ok=True)
    overlay_paths: Dict[str, str] = {}
    for angle, img_path in image_paths.items():
        block = (image_skeleton.get("angles") or {}).get(angle)
        if not isinstance(block, dict):
            continue
        joints = block.get("joints") or []
        if not joints:
            continue
        out_png = os.path.join(overlay_dir, f"{angle}.png")
        try:
            overlay_paths[angle] = overlay.draw_overlay(str(img_path), joints, out_png)
        except Exception as exc:
            analysis_warnings.append(f"skeleton_overlay_on_samples.{angle}: {exc}")
    return multi_path, overlay_paths


def _analyze_submission_sync(
    submission_id: str,
    image_paths: dict,
    video_path: str,
    overlay_options: Optional[Dict[str, Any]] = None,
) -> dict:
    """Synchronous pipeline used by both async API and Celery wrappers."""
    skeleton_processor = SkeletonProcessor()
    avatar_generator = AvatarGenerator()
    renderer = AvatarRenderer()
    analyzer = GolfSwingAnalyzer()

    output_dir = _pipeline_output_dir(submission_id)
    os.makedirs(output_dir, exist_ok=True)

    video_work: Path | None = None
    video_is_temp = False
    try:
        analysis_warnings: List[str] = []

        image_skeleton = skeleton_processor.process_images(image_paths, submission_id)

        overlay = SkeletonOverlay()
        multi_angle_skeleton_json_path, multi_angle_overlay_paths = _write_multi_angle_artifacts(
            output_dir, image_paths, image_skeleton, overlay, analysis_warnings
        )

        head_overhead = detect_head_overhead_view_video(video_path)
        video_work, video_is_temp = prepare_video_for_skeleton_pipeline(
            Path(video_path), head_overhead_view=head_overhead
        )
        video_skeleton = skeleton_processor.process_video(
            str(video_work),
            submission_id,
            swing_motion_slowdown=True,
            head_overhead_view=head_overhead,
        )

        skeleton_json = _to_backend_skeleton_schema(submission_id, video_skeleton)
        skeleton_path = skeleton_processor.export_skeleton_json(
            skeleton_json, os.path.join(output_dir, "skeleton.json")
        )

        avatar_generator.set_source_images(image_paths, image_skeleton.get("angles"))
        mesh = avatar_generator.generate_from_multi_angle(image_skeleton)
        # write skeleton JSON used to build the mesh
        import json as _json
        _skel_json_path = os.path.join(output_dir, "avatar_skeleton.json")
        with open(_skel_json_path, "w", encoding="utf-8") as _sf:
            _json.dump(image_skeleton, _sf, indent=2)
        avatar_paths = {
            "obj": avatar_generator.export_obj(mesh, os.path.join(output_dir, "avatar.obj")),
            "glb": avatar_generator.export_glb(mesh, os.path.join(output_dir, "avatar.glb")),
            "fbx": avatar_generator.export_fbx(mesh, os.path.join(output_dir, "avatar.fbx")),
        }

        renders = renderer.render_5_angles(mesh)
        render_paths = renderer.save_renders(renders, os.path.join(output_dir, "renders"))

        key_frames = analyzer.detect_key_frames(video_skeleton.get("frames", []))
        swing_angles = {}
        impact_frame_num = key_frames.get("impact", -1)
        if impact_frame_num >= 0:
            impact_frame = next(
                (
                    frame
                    for frame in video_skeleton.get("frames", [])
                    if int(frame.get("frame_num", -1)) == int(impact_frame_num)
                ),
                None,
            )
            if impact_frame is not None:
                swing_angles = analyzer.analyze_swing_angles(impact_frame.get("joints", []))

        ov = {**DEFAULT_OVERLAY_OPTIONS, **(overlay_options or {})}
        wrist_elbow_video: Optional[str] = None

        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        analysis_outputs: Dict[str, Optional[str]] = {
            "annotated_swing_video": None,
            "wrist_elbow_analysis_video": None,
            "tempo_chart": None,
            "swing_score_card": None,
            "angle_arcs_impact": None,
            "confidence_heatmap_impact": None,
            "motion_trail_left_wrist": None,
        }

        annotated_path = os.path.join(output_dir, "annotated_swing.mp4")
        try:
            analysis_outputs["annotated_swing_video"] = skeleton_processor.export_annotated_video(
                str(video_work),
                video_skeleton,
                annotated_path,
                show_shoulders=ov["show_shoulders"],
                show_waistline=ov["show_waistline"],
                show_legs=ov["show_legs"],
                show_spine=ov["show_spine"],
                show_head_reference=ov["show_head_reference"],
                static_lines=ov["static_lines"],
                tracking_lines=ov["tracking_lines"],
                static_labels=ov.get("static_labels"),
            )
        except Exception as exc:
            analysis_warnings.append(f"annotated_swing: {exc}")

        wrist_elbow_path = os.path.join(output_dir, "wrist_elbow_analysis.mp4")
        try:
            wrist_elbow_video = skeleton_processor.export_wrist_elbow_analysis_video(
                str(video_work),
                skeleton_json,
                wrist_elbow_path,
                show_shoulders=ov["show_shoulders"],
                show_waistline=ov["show_waistline"],
                show_legs=ov["show_legs"],
                show_spine=ov["show_spine"],
                show_head_reference=ov["show_head_reference"],
                static_lines=ov["static_lines"],
                tracking_lines=ov["tracking_lines"],
                static_labels=ov.get("static_labels"),
            )
            analysis_outputs["wrist_elbow_analysis_video"] = wrist_elbow_video
        except Exception as exc:
            analysis_warnings.append(f"wrist_elbow_analysis: {exc}")

        tempo_chart_path = os.path.join(analysis_dir, "tempo_chart.png")
        try:
            tempo_info = analyzer.calculate_tempo(
                skeleton_json,
                key_frames,
                chart_output_path=tempo_chart_path,
            )
            analysis_outputs["tempo_chart"] = tempo_info.get("chart_path")
        except Exception as exc:
            tempo_info = {}
            analysis_warnings.append(f"tempo_chart: {exc}")

        swing_for_scorecard = dict(swing_angles)
        ratio = tempo_info.get("backswing_downswing_ratio") if tempo_info else None
        if ratio is not None:
            swing_for_scorecard["tempo"] = float(ratio)

        score_card_path = os.path.join(analysis_dir, "swing_score_card.png")
        try:
            analysis_outputs["swing_score_card"] = analyzer.export_score_card(
                swing_for_scorecard,
                _DEFAULT_SCORECARD_IDEAL_RANGES,
                score_card_path,
            )
        except Exception as exc:
            analysis_warnings.append(f"swing_score_card: {exc}")

        impact_fn = int(key_frames.get("impact", -1))
        impact_joints: list = []
        if impact_fn >= 0:
            for fr in video_skeleton.get("frames", []):
                if int(fr.get("frame_num", -1)) == impact_fn:
                    impact_joints = list(fr.get("joints") or [])
                    break

        if impact_fn >= 0 and impact_joints:
            impact_bgr = _read_video_frame_bgr(str(video_work), impact_fn)
            if impact_bgr is not None:
                fd, tmp_impact = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                try:
                    if cv2.imwrite(tmp_impact, impact_bgr):
                        heatmap_path = os.path.join(analysis_dir, "confidence_heatmap_impact.png")
                        try:
                            analysis_outputs["confidence_heatmap_impact"] = overlay.draw_confidence_heatmap(
                                tmp_impact,
                                impact_joints,
                                heatmap_path,
                            )
                        except Exception as exc:
                            analysis_warnings.append(f"confidence_heatmap_impact: {exc}")

                        arcs_path = os.path.join(analysis_dir, "angle_arcs_impact.png")
                        try:
                            arc_img = overlay.draw_angle_arcs(
                                impact_bgr,
                                impact_joints,
                                swing_angles,
                                _ARC_IDEAL_RANGES,
                            )
                            if cv2.imwrite(arcs_path, arc_img):
                                analysis_outputs["angle_arcs_impact"] = arcs_path
                            else:
                                analysis_warnings.append("angle_arcs_impact: imwrite failed")
                        except Exception as exc:
                            analysis_warnings.append(f"angle_arcs_impact: {exc}")
                finally:
                    try:
                        os.unlink(tmp_impact)
                    except OSError:
                        pass
            else:
                analysis_warnings.append("impact_still: could not decode video frame")

        trail_path = os.path.join(analysis_dir, "motion_trail_left_wrist.png")
        try:
            analysis_outputs["motion_trail_left_wrist"] = overlay.draw_motion_trail(
                str(video_work),
                video_skeleton,
                15,
                trail_path,
            )
        except Exception as exc:
            analysis_warnings.append(f"motion_trail_left_wrist: {exc}")

        analysis_path = os.path.join(output_dir, "swing_analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as af:
            json.dump(
                {
                    "key_frames": key_frames,
                    "swing_angles": swing_angles,
                    "tempo_metrics": _json_safe_tempo_metrics(tempo_info) if tempo_info else {},
                    "analysis_outputs": {k: v for k, v in analysis_outputs.items() if v},
                    "analysis_warnings": analysis_warnings,
                    "pipeline_metadata": {
                        "head_overhead_view_detected": bool(head_overhead),
                    },
                },
                af,
                indent=2,
            )

        return {
            "skeleton_json": skeleton_json,
            "image_skeleton": image_skeleton,
            "video_skeleton": video_skeleton,
            "head_overhead_view_detected": bool(head_overhead),
            "avatar_paths": avatar_paths,
            "render_paths": render_paths,
            "key_frames": key_frames,
            "swing_angles": swing_angles,
            "skeleton_json_path": skeleton_path,
            "swing_analysis_json_path": analysis_path,
            "wrist_elbow_analysis_video": wrist_elbow_video,
            "annotated_swing_video": analysis_outputs.get("annotated_swing_video"),
            "analysis_outputs": analysis_outputs,
            "analysis_warnings": analysis_warnings,
            "multi_angle_skeleton_json_path": multi_angle_skeleton_json_path,
            "multi_angle_overlay_paths": multi_angle_overlay_paths,
        }
    finally:
        skeleton_processor.close()
        if video_is_temp and video_work is not None:
            video_work.unlink(missing_ok=True)


def _extract_backend_status(status_payload: dict) -> str:
    """Best-effort extraction of status string from backend payload."""
    for key in ("status", "submission_status", "state"):
        val = status_payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    data = status_payload.get("data")
    if isinstance(data, dict):
        for key in ("status", "submission_status", "state"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return "UNKNOWN"


async def analyze_submission(
    submission_id: str,
    image_paths: dict,
    video_path: str,
    overlay_options: Optional[Dict[str, Any]] = None,
) -> dict:
    """Run the complete AI golf submission pipeline asynchronously.

    Pipeline:
    1) detect skeletons from four static images
    2) detect skeleton frames from video (RGB-motion peak window is temporally
       oversampled so MediaPipe video mode can track fast arms/shoulders better)
    3) generate 3D avatar mesh from multi-angle skeleton data
    4) render 5 standard avatar views
    5) detect key swing frames and compute swing angles

    Args:
        submission_id: Unique submission identifier.
        image_paths: Dict with keys: front, left, right, back.
        video_path: Input swing video path (MP4).
        overlay_options: Optional dict overriding DEFAULT_OVERLAY_OPTIONS
            keys (show_shoulders, show_waistline, show_legs, show_spine,
            show_head_reference, static_lines, static_labels, tracking_lines).

    Returns:
        Dict with ``skeleton_json``, ``avatar_paths``, ``render_paths``,
        ``key_frames``, ``swing_angles``, ``swing_analysis_json_path``,
        ``annotated_swing_video`` (path to the single overlay MP4, or None if
        export failed), ``wrist_elbow_analysis_video`` (path to wrist/elbow
        metrics MP4, or None if export failed), ``analysis_outputs``,
        ``analysis_warnings``, ``head_overhead_view_detected`` (overhead / head-down
        camera heuristic for alignment + smoothing).
    """
    result = await asyncio.to_thread(
        _analyze_submission_sync, submission_id, image_paths, video_path, overlay_options,
    )
    return {
        "skeleton_json": result["skeleton_json"],
        "avatar_paths": result["avatar_paths"],
        "render_paths": result["render_paths"],
        "key_frames": result["key_frames"],
        "swing_angles": result["swing_angles"],
        "swing_analysis_json_path": result["swing_analysis_json_path"],
        "wrist_elbow_analysis_video": result["wrist_elbow_analysis_video"],
        "annotated_swing_video": result.get("annotated_swing_video"),
        "analysis_outputs": result.get("analysis_outputs", {}),
        "analysis_warnings": result.get("analysis_warnings", []),
        "multi_angle_skeleton_json_path": result.get("multi_angle_skeleton_json_path"),
        "multi_angle_overlay_paths": result.get("multi_angle_overlay_paths", {}),
        "head_overhead_view_detected": bool(result.get("head_overhead_view_detected", False)),
    }


def analyze_and_sync_with_backend(
    submission_id: str,
    image_paths: dict,
    video_path: str,
    *,
    backend_token: str,
    backend_base_url: str = "https://golf-swing-backend-production.up.railway.app",
    poll_interval_seconds: float = 3.0,
    poll_timeout_seconds: float = 600.0,
    overlay_options: Optional[Dict[str, Any]] = None,
) -> dict:
    """Run local ML pipeline, submit media to backend, and poll analysis status.

    This helper is intended for worker integration where ML service coordinates
    with the production backend in one call.
    """
    local_result = _analyze_submission_sync(submission_id, image_paths, video_path, overlay_options)
    client = BackendClient(base_url=backend_base_url, token=backend_token)

    client.upload_and_submit(submission_id=submission_id, image_paths=image_paths, video_path=video_path)

    terminal_statuses = {"COMPLETED", "APPROVED", "REJECTED", "FAILED", "ERROR"}
    started = time.time()
    status_history: List[dict] = []
    final_status_payload: dict = {}
    final_status = "UNKNOWN"

    while (time.time() - started) <= float(poll_timeout_seconds):
        status_payload = client.get_submission_status(submission_id)
        status_history.append(status_payload)
        final_status_payload = status_payload
        final_status = _extract_backend_status(status_payload).upper()
        if final_status in terminal_statuses:
            break
        time.sleep(max(0.25, float(poll_interval_seconds)))

    backend_summary = {
        "submission_id": submission_id,
        "final_status": final_status,
        "status_payload": final_status_payload,
        "status_history_count": len(status_history),
    }
    try:
        backend_summary["submission"] = client.get_submission(submission_id)
    except BackendApiError as exc:
        backend_summary["submission_error"] = str(exc)
    try:
        backend_summary["avatar"] = client.get_avatar(submission_id)
    except BackendApiError as exc:
        backend_summary["avatar_error"] = str(exc)
    try:
        backend_summary["skeleton"] = client.get_skeleton(submission_id)
    except BackendApiError as exc:
        backend_summary["skeleton_error"] = str(exc)

    return {
        "local_analysis": local_result,
        "backend": backend_summary,
    }


async def analyze_and_sync_with_backend_async(
    submission_id: str,
    image_paths: dict,
    video_path: str,
    *,
    backend_token: str,
    backend_base_url: str = "https://golf-swing-backend-production.up.railway.app",
    poll_interval_seconds: float = 3.0,
    poll_timeout_seconds: float = 600.0,
    overlay_options: Optional[Dict[str, Any]] = None,
) -> dict:
    """Async wrapper for end-to-end local + backend sync workflow."""
    return await asyncio.to_thread(
        analyze_and_sync_with_backend,
        submission_id,
        image_paths,
        video_path,
        backend_token=backend_token,
        backend_base_url=backend_base_url,
        poll_interval_seconds=poll_interval_seconds,
        poll_timeout_seconds=poll_timeout_seconds,
        overlay_options=overlay_options,
    )


async def apply_coach_correction(
    submission_id: str,
    original_joints: list,
    corrected_joints: list,
    blend: float,
) -> dict:
    """Apply coach correction and regenerate corrected avatar renders.

    Args:
        submission_id: Submission identifier for output naming.
        original_joints: Baseline frame joints from detector.
        corrected_joints: Coach-adjusted (possibly partial) joints.
        blend: Blend factor in [0, 1].

    Returns:
        {
          "submission_id": str,
          "updated_joints": list,
          "render_paths": dict,
          "avatar_paths": dict with ``obj``, ``glb``, ``fbx`` mesh exports
        }
    """
    handler = CorrectionHandler()
    avatar_generator = AvatarGenerator()
    renderer = AvatarRenderer()

    updated_joints = handler.apply_correction(
        original_joints=original_joints,
        adjusted_joints=corrected_joints,
        blend_factor=blend,
    )
    if not handler.validate_correction(updated_joints):
        raise ValueError("Corrected joints failed validation checks.")

    # Reuse corrected joints for all angles when only one corrected pose is provided.
    corrected_multi_angle = {
        "angles": {
            "front": {"joints": updated_joints},
            "left": {"joints": updated_joints},
            "right": {"joints": updated_joints},
            "back": {"joints": updated_joints},
        }
    }

    mesh = avatar_generator.generate_from_multi_angle(corrected_multi_angle)
    renders = await asyncio.to_thread(renderer.render_5_angles, mesh)

    out_dir = os.path.join(_pipeline_output_dir(submission_id), "corrected_renders")
    render_paths = await asyncio.to_thread(renderer.save_renders, renders, out_dir)

    mesh_dir = os.path.join(_pipeline_output_dir(submission_id), "corrected_avatar")
    os.makedirs(mesh_dir, exist_ok=True)
    avatar_paths = {
        "obj": avatar_generator.export_obj(mesh, os.path.join(mesh_dir, "avatar.obj")),
        "glb": avatar_generator.export_glb(mesh, os.path.join(mesh_dir, "avatar.glb")),
        "fbx": avatar_generator.export_fbx(mesh, os.path.join(mesh_dir, "avatar.fbx")),
    }

    return {
        "submission_id": submission_id,
        "updated_joints": updated_joints,
        "render_paths": render_paths,
        "avatar_paths": avatar_paths,
    }


def celery_task_analyze(
    submission_id: str,
    image_paths: dict,
    video_path: str,
    overlay_options: Optional[Dict[str, Any]] = None,
) -> dict:
    """Synchronous analyze wrapper intended for Celery worker execution."""
    result = _analyze_submission_sync(submission_id, image_paths, video_path, overlay_options)
    return {
        "skeleton_json": result["skeleton_json"],
        "avatar_paths": result["avatar_paths"],
        "render_paths": result["render_paths"],
        "key_frames": result["key_frames"],
        "swing_angles": result["swing_angles"],
        "swing_analysis_json_path": result["swing_analysis_json_path"],
        "wrist_elbow_analysis_video": result["wrist_elbow_analysis_video"],
        "annotated_swing_video": result.get("annotated_swing_video"),
        "analysis_outputs": result.get("analysis_outputs", {}),
        "analysis_warnings": result.get("analysis_warnings", []),
        "multi_angle_skeleton_json_path": result.get("multi_angle_skeleton_json_path"),
        "multi_angle_overlay_paths": result.get("multi_angle_overlay_paths", {}),
        "head_overhead_view_detected": bool(result.get("head_overhead_view_detected", False)),
    }


def celery_task_correction(
    submission_id: str, original_joints: list, corrected_joints: list
) -> dict:
    """Synchronous correction wrapper intended for Celery worker execution."""
    handler = CorrectionHandler()
    avatar_generator = AvatarGenerator()
    renderer = AvatarRenderer()

    updated_joints = handler.apply_correction(original_joints, corrected_joints, blend_factor=1.0)
    if not handler.validate_correction(updated_joints):
        raise ValueError("Corrected joints failed validation checks.")

    corrected_multi_angle = {
        "angles": {
            "front": {"joints": updated_joints},
            "left": {"joints": updated_joints},
            "right": {"joints": updated_joints},
            "back": {"joints": updated_joints},
        }
    }
    mesh = avatar_generator.generate_from_multi_angle(corrected_multi_angle)
    renders = renderer.render_5_angles(mesh)
    out_dir = os.path.join(_pipeline_output_dir(submission_id), "corrected_renders")
    render_paths = renderer.save_renders(renders, out_dir)

    mesh_dir = os.path.join(_pipeline_output_dir(submission_id), "corrected_avatar")
    os.makedirs(mesh_dir, exist_ok=True)
    avatar_paths = {
        "obj": avatar_generator.export_obj(mesh, os.path.join(mesh_dir, "avatar.obj")),
        "glb": avatar_generator.export_glb(mesh, os.path.join(mesh_dir, "avatar.glb")),
        "fbx": avatar_generator.export_fbx(mesh, os.path.join(mesh_dir, "avatar.fbx")),
    }

    return {
        "submission_id": submission_id,
        "updated_joints": updated_joints,
        "render_paths": render_paths,
        "avatar_paths": avatar_paths,
    }
