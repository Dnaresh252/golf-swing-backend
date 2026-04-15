"""Golf Swing Coach Annotation Tool — Tkinter + OpenCV + MediaPipe.

Coach workflow:
  1) Open video from disk
  2) Clip is temporally stretched (+10s alignment; smooth blended stretch if overhead/head-down is detected) then MediaPipe runs on every frame
  3) Video auto-plays with skeleton overlay, head reference, coach lines (green/white/red) + numbered green-ring markers, captions, or large headlines
  4) Export annotated MP4 with all overlays baked in
  5) **Multi-angle 3D avatars:** pick any subset of Front / Left / Top / Back / Right
     (one image is enough), then export ``avatar.obj``, ``avatar.fbx``, and ``avatar.glb``.
     Image formats: common raster types plus anything Pillow can open.

Run from repo root::

    py -3 -m ml_cv_engine.swing_coach_annotation_app
"""

from __future__ import annotations

import math
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Pillow is required for the coach UI. Install with: py -3 -m pip install pillow"
    ) from e

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from .avatar_3d import AvatarGenerator
from .coach_overlay_labels import draw_static_labels_on_frame
from .overlay_2d import SkeletonOverlay
from .pose_detection import PoseDetector
from .rendering import AvatarRenderer
from .skeleton_processing import SkeletonProcessor
from .utils import prepare_video_for_skeleton_pipeline
from .video_view_detection import detect_head_overhead_view_video

_FIVE_ANGLES_ORDER = ("front", "left", "top", "back", "right")
_FIVE_ANGLE_LABELS = {
    "front": "Front",
    "left": "Left side",
    "top": "Top (overhead)",
    "back": "Back",
    "right": "Right side",
}

_IMAGE_DIALOG_TYPES = [
    (
        "Images",
        "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.gif "
        "*.heic *.heif *.jp2 *.jpf *.jpx *.ppm *.pgm *.pbm",
    ),
    ("Any file", "*.*"),
]

# BGR colours
_HEAD_REF_BLUE = (255, 0, 0)
_PURPLE_IDEAL = (255, 0, 255)
_YELLOW_SWING_PATH = (0, 255, 255)
_COACH_WHITE = (255, 255, 255)
_COACH_RED = (0, 0, 255)
# Bright “coach / swing plane” green (BGR), similar to common golf tutorial overlays
_COACH_GREEN = (60, 220, 80)

_MAX_DISPLAY_W = 860
_MAX_DISPLAY_H = 480


def _joint_norm_xy(joints: list, jid: int) -> Optional[Tuple[float, float]]:
    for j in joints:
        if int(j.get("id", -1)) == jid:
            return float(j["x"]), float(j["y"])
    return None


def _wrist_mid_norm(joints: list) -> Optional[Tuple[float, float]]:
    a = _joint_norm_xy(joints, 15)
    b = _joint_norm_xy(joints, 16)
    if a and b:
        return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5
    return a or b


def _shoulder_mid_norm(joints: list) -> Optional[Tuple[float, float]]:
    a = _joint_norm_xy(joints, 11)
    b = _joint_norm_xy(joints, 12)
    if a and b:
        return (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5
    return a or b


def _norm_to_px(xy: Tuple[float, float], w: int, h: int) -> Tuple[int, int]:
    return (
        int(round(np.clip(xy[0], 0.0, 1.0) * (w - 1))),
        int(round(np.clip(xy[1], 0.0, 1.0) * (h - 1))),
    )


def _ideal_purple_endpoints(
    w: int, h: int, shoulder_mid_px: Tuple[int, int], angle_deg: float = -22.0
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    smx, smy = shoulder_mid_px
    th = math.radians(angle_deg)
    dx, dy = math.cos(th), math.sin(th)
    span = float(max(w, h)) * 3.0
    return (
        (int(round(smx - dx * span)), int(round(smy - dy * span))),
        (int(round(smx + dx * span)), int(round(smy + dy * span))),
    )


class SwingCoachAnnotationApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Golf Swing Coach — Annotation")
        self.minsize(920, 680)
        self.geometry("960x720")

        self._overlay = SkeletonOverlay()

        self._video_path: Optional[str] = None
        self._prep_temp = False
        self._total = 0
        self._fps = 30.0
        self._full_w = 0
        self._full_h = 0
        self._disp_scale = 1.0
        self._disp_w = 1
        self._disp_h = 1

        self._raw_frames: List[np.ndarray] = []
        self.joints_per_frame: List[list] = []
        self.head_ref_y: Optional[int] = None
        self._ideal_segment: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.wrist_trail_px: List[Optional[Tuple[int, int]]] = []

        self.static_lines: List[Dict[str, Any]] = []
        self.static_labels: List[Dict[str, Any]] = []
        self._label_counter = 0
        self._drag_start: Optional[Tuple[int, int]] = None
        self._drag_preview: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None

        self.var_shoulders = tk.BooleanVar(value=True)
        self.var_waist = tk.BooleanVar(value=True)
        self.var_legs = tk.BooleanVar(value=True)
        self.var_spine = tk.BooleanVar(value=True)
        self.var_swing_paths = tk.BooleanVar(value=False)
        self.var_coach_ink = tk.StringVar(value="green")
        self.var_playing = tk.BooleanVar(value=False)
        self.var_draw_tool = tk.StringVar(value="line")
        self.var_label_numbered = tk.BooleanVar(value=True)

        self._frame_index = 0
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_progress = ""
        self._play_after_id: Optional[str] = None

        self._five_paths: Dict[str, Optional[str]] = {k: None for k in _FIVE_ANGLES_ORDER}
        self._five_lbl: Dict[str, ttk.Label] = {}
        self._btn_export_avatars: Optional[ttk.Button] = None
        self._avatar_export_status: Optional[Tuple[str, str]] = None

        self._main_scroll: Optional[tk.Canvas] = None
        self._scroll_inner: Optional[ttk.Frame] = None
        self._scroll_inner_win: Optional[int] = None

        self._build_ui()

    def _is_under_scroll_inner(self, widget: Any) -> bool:
        if self._main_scroll is not None and widget is self._main_scroll:
            return True
        inner = self._scroll_inner
        if inner is None:
            return False
        w: Any = widget
        while w is not None:
            if w is inner:
                return True
            if w is self:
                return False
            w = getattr(w, "master", None)
        return False

    def _on_scroll_mousewheel(self, event: tk.Event) -> Optional[str]:
        if self._main_scroll is None or not self._is_under_scroll_inner(event.widget):
            return None
        d = getattr(event, "delta", 0) or 0
        if d:
            self._main_scroll.yview_scroll(int(-1 * (d / 120)), "units")
        elif getattr(event, "num", 0) == 4:
            self._main_scroll.yview_scroll(-1, "units")
        elif getattr(event, "num", 0) == 5:
            self._main_scroll.yview_scroll(1, "units")
        return "break"

    def _bind_scroll_mousewheel_recursive(self, widget: Any) -> None:
        widget.bind("<MouseWheel>", self._on_scroll_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_scroll_mousewheel, add="+")
        widget.bind("<Button-5>", self._on_scroll_mousewheel, add="+")
        for ch in widget.winfo_children():
            self._bind_scroll_mousewheel_recursive(ch)

    def _sync_scroll_inner_width(self, event: Optional[tk.Event] = None) -> None:
        if self._main_scroll is None or self._scroll_inner_win is None:
            return
        try:
            cw = self._main_scroll.winfo_width()
        except tk.TclError:
            return
        if cw > 1:
            self._main_scroll.itemconfigure(self._scroll_inner_win, width=cw)

    def _sync_scroll_region(self, event: Optional[tk.Event] = None) -> None:
        if self._main_scroll is None:
            return
        self._main_scroll.configure(scrollregion=self._main_scroll.bbox("all"))

    # ------------------------------------------------------------------
    # UI layout — bottom bar fixed; main column scrolls vertically
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # --- bottom bar (fixed; pack first so it stays visible) ---
        bot = ttk.Frame(self, padding=6)
        bot.pack(fill=tk.X, side=tk.BOTTOM)
        self.btn_play = ttk.Button(bot, text="Play", width=7, command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=4)
        self.scale = tk.Scale(bot, from_=0, to=0, orient=tk.HORIZONTAL, showvalue=0, command=self._on_scale)
        self.scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.meta = ttk.Label(bot, text="Frame — / —   t = —", width=28)
        self.meta.pack(side=tk.LEFT, padx=4)
        ttk.Button(bot, text="Save annotated MP4…", command=self._on_export).pack(side=tk.RIGHT, padx=4)

        # --- scrollable main column (controls + video) ---
        scroll_wrap = ttk.Frame(self)
        scroll_wrap.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scroll_wrap.columnconfigure(0, weight=1)
        scroll_wrap.rowconfigure(0, weight=1)

        try:
            _canvas_bg = self.cget("bg") or "#ececec"
        except tk.TclError:
            _canvas_bg = "#ececec"
        self._main_scroll = tk.Canvas(scroll_wrap, highlightthickness=0, bg=_canvas_bg)
        vsb = ttk.Scrollbar(scroll_wrap, orient=tk.VERTICAL, command=self._main_scroll.yview)
        self._main_scroll.configure(yscrollcommand=vsb.set)
        self._main_scroll.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self._scroll_inner = ttk.Frame(self._main_scroll, padding=0)
        self._scroll_inner_win = self._main_scroll.create_window((0, 0), window=self._scroll_inner, anchor="nw")
        self._scroll_inner.bind("<Configure>", lambda e: self.after_idle(self._sync_scroll_region))
        self._main_scroll.bind("<Configure>", self._sync_scroll_inner_width)

        inner = self._scroll_inner

        # --- top bar ---
        top = ttk.Frame(inner, padding=6)
        top.pack(fill=tk.X, side=tk.TOP)
        ttk.Button(top, text="Open video…", command=self._on_open_video).pack(side=tk.LEFT, padx=4)
        self.status = ttk.Label(top, text="Open a golf swing video to begin.")
        self.status.pack(side=tk.LEFT, padx=12)

        # --- multi-angle 3D avatar export (OBJ / FBX / GLB); one image is enough ---
        av = ttk.LabelFrame(
            inner,
            text="3D avatar from stills (1+ angles — unused views copied from first pose)",
            padding=6,
        )
        av.pack(fill=tk.X, padx=8, pady=2, side=tk.TOP)
        for col, angle in enumerate(_FIVE_ANGLES_ORDER):
            fr = ttk.Frame(av)
            fr.grid(row=0, column=col, padx=4, pady=2, sticky=tk.N)
            ttk.Button(
                fr,
                text=_FIVE_ANGLE_LABELS[angle],
                width=14,
                command=lambda a=angle: self._on_pick_five_angle(a),
            ).pack()
            lb = ttk.Label(fr, text="—", width=18, anchor=tk.CENTER)
            lb.pack(pady=2)
            self._five_lbl[angle] = lb
        self._btn_export_avatars = ttk.Button(
            av,
            text="Export avatar OBJ / FBX / GLB…",
            command=self._on_export_avatars,
            state=tk.DISABLED,
        )
        self._btn_export_avatars.grid(row=0, column=5, padx=12, sticky=tk.N)

        # --- skeleton toggles ---
        ctrl = ttk.LabelFrame(inner, text="Skeleton (stops at shoulders — no head/neck)", padding=6)
        ctrl.pack(fill=tk.X, padx=8, pady=2, side=tk.TOP)
        ttk.Checkbutton(ctrl, text="Shoulders", variable=self.var_shoulders, command=self._refresh).grid(row=0, column=0, padx=6, sticky=tk.W)
        ttk.Checkbutton(ctrl, text="Waistline", variable=self.var_waist, command=self._refresh).grid(row=0, column=1, padx=6, sticky=tk.W)
        ttk.Checkbutton(ctrl, text="Legs", variable=self.var_legs, command=self._refresh).grid(row=0, column=2, padx=6, sticky=tk.W)
        ttk.Checkbutton(ctrl, text="Spine (green dashed)", variable=self.var_spine, command=self._refresh).grid(row=0, column=3, padx=6, sticky=tk.W)
        ttk.Checkbutton(ctrl, text="Swing paths (purple ideal + yellow club)", variable=self.var_swing_paths, command=self._refresh).grid(row=0, column=4, padx=6, sticky=tk.W)

        # --- coach tools ---
        tools = ttk.Frame(inner, padding=4)
        tools.pack(fill=tk.X, padx=8, side=tk.TOP)
        ttk.Label(tools, text="Coach ink:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Radiobutton(tools, text="Green", variable=self.var_coach_ink, value="green", command=self._refresh).pack(side=tk.LEFT, padx=1)
        ttk.Radiobutton(tools, text="White", variable=self.var_coach_ink, value="white", command=self._refresh).pack(side=tk.LEFT, padx=1)
        ttk.Radiobutton(tools, text="Red", variable=self.var_coach_ink, value="red", command=self._refresh).pack(side=tk.LEFT, padx=1)
        ttk.Button(tools, text="Clear coach drawings", command=self._clear_lines).pack(side=tk.LEFT, padx=8)
        ttk.Separator(tools, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=2)
        ttk.Label(tools, text="Tool:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Radiobutton(tools, text="Line", variable=self.var_draw_tool, value="line", command=self._on_draw_tool_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(tools, text="Label", variable=self.var_draw_tool, value="label", command=self._on_draw_tool_change).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(tools, text="Headline", variable=self.var_draw_tool, value="headline", command=self._on_draw_tool_change).pack(side=tk.LEFT, padx=2)
        ttk.Label(tools, text="Label text:").pack(side=tk.LEFT, padx=(8, 2))
        self.entry_label_text = ttk.Entry(tools, width=22)
        self.entry_label_text.pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(
            tools,
            text="Numbered (1, 2, 3…)",
            variable=self.var_label_numbered,
            command=self._refresh,
        ).pack(side=tk.LEFT, padx=6)

        # --- canvas (inside scroll area; min height so preview stays usable) ---
        vid = ttk.Frame(inner, padding=4)
        vid.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        self.canvas = tk.Canvas(vid, bg="#222", highlightthickness=0, height=_MAX_DISPLAY_H + 12)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        self._bind_scroll_mousewheel_recursive(inner)
        self.after_idle(self._sync_scroll_inner_width)
        self.after_idle(self._sync_scroll_region)

    # ------------------------------------------------------------------
    # coordinate helpers
    # ------------------------------------------------------------------
    def _disp_to_full(self, x: int, y: int) -> Tuple[int, int]:
        if self._disp_scale <= 1e-9:
            return x, y
        return (
            int(np.clip(round(x / self._disp_scale), 0, self._full_w - 1)),
            int(np.clip(round(y / self._disp_scale), 0, self._full_h - 1)),
        )

    # ------------------------------------------------------------------
    # open / prepare / infer  (all heavy work in one background thread)
    # ------------------------------------------------------------------
    def _on_open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select golf swing video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All files", "*.*")],
        )
        if not path:
            return
        self._stop_play()
        self._release_all()
        self._frame_index = 0
        self.scale.config(to=0)
        self.scale.set(0)
        self.canvas.delete("all")
        self.status.config(text="Step 1/3 — Preparing video (+10s slow)…")
        self.update_idletasks()
        self._worker_thread = threading.Thread(target=self._load_worker, args=(path,), daemon=True)
        self._worker_thread.start()
        self.after(150, self._poll_worker)

    def _load_worker(self, src_path: str) -> None:
        try:
            head_ov = detect_head_overhead_view_video(src_path)
            self._worker_progress = (
                "Step 1/3 — Preparing video (+10s, smooth overhead alignment)…"
                if head_ov
                else "Step 1/3 — Preparing video (+10s slow)…"
            )
            work, is_temp = prepare_video_for_skeleton_pipeline(
                Path(src_path), head_overhead_view=head_ov
            )
        except Exception as exc:
            self._worker_progress = f"ERROR: {exc}"
            return
        self._video_path = str(work)
        self._prep_temp = is_temp

        cap = cv2.VideoCapture(self._video_path)
        if not cap.isOpened():
            self._worker_progress = f"ERROR: Cannot open {self._video_path}"
            return
        self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self._fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._full_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._worker_progress = f"Step 2/3 — Reading {self._total} frames…"
        frames: List[np.ndarray] = []
        for _ in range(self._total):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
        cap.release()
        self._raw_frames = frames
        self._total = len(frames)

        self._worker_progress = f"Step 3/3 — MediaPipe skeleton (0/{self._total})…"
        det = PoseDetector(min_confidence=0.25, static_image_mode=False, model_complexity=1)
        det.set_video_fps(self._fps)
        det.reset_video_stream()
        joints: List[list] = []
        try:
            for i, frame in enumerate(frames):
                pose = det.detect_from_frame(frame)
                joints.append(pose["joints"] if pose is not None else [])
                if i % 20 == 0:
                    self._worker_progress = f"Step 3/3 — MediaPipe skeleton ({i}/{self._total})…"
        finally:
            det.close()
        self.joints_per_frame = joints
        self._worker_progress = "DONE"

    def _poll_worker(self) -> None:
        prog = self._worker_progress
        if prog.startswith("ERROR:"):
            self._worker_thread = None
            messagebox.showerror("Load failed", prog[6:].strip())
            self.status.config(text="Open a golf swing video to begin.")
            return
        if prog != "DONE":
            self.status.config(text=prog)
            self.after(200, self._poll_worker)
            return
        self._worker_thread = None
        self._finish_load()

    def _finish_load(self) -> None:
        if self._total < 1:
            self.status.config(text="No frames decoded.")
            return
        sx = _MAX_DISPLAY_W / max(1, self._full_w)
        sy = _MAX_DISPLAY_H / max(1, self._full_h)
        self._disp_scale = min(1.0, sx, sy)
        self._disp_w = max(1, int(round(self._full_w * self._disp_scale)))
        self._disp_h = max(1, int(round(self._full_h * self._disp_scale)))
        self.canvas.config(width=self._disp_w, height=self._disp_h)
        self.scale.config(to=max(0, self._total - 1))
        self.scale.set(0)
        self._frame_index = 0
        self._overlay.reset_arm_display_smooth()
        self._build_head_and_paths()
        dur = self._total / max(self._fps, 1e-6)
        self.status.config(
            text=f"Ready — {self._total} frames, {self._fps:.0f} fps, {dur:.1f}s. "
            f"Draw lines or switch to Label and click on the video."
        )
        self._refresh()
        self.after_idle(self._sync_scroll_region)
        self.after_idle(self._sync_scroll_inner_width)
        self._auto_play()

    def _build_head_and_paths(self) -> None:
        self.head_ref_y = None
        self._ideal_segment = None
        self.wrist_trail_px = []
        if not self.joints_per_frame:
            return
        for joints in self.joints_per_frame[:30]:
            nose = _joint_norm_xy(joints, 0)
            if nose:
                self.head_ref_y = _norm_to_px(nose, self._full_w, self._full_h)[1]
                break
        for joints in self.joints_per_frame[:45]:
            sm = _shoulder_mid_norm(joints)
            if sm:
                smx, smy = _norm_to_px(sm, self._full_w, self._full_h)
                self._ideal_segment = _ideal_purple_endpoints(self._full_w, self._full_h, (smx, smy))
                break
        for joints in self.joints_per_frame:
            wm = _wrist_mid_norm(joints)
            self.wrist_trail_px.append(_norm_to_px(wm, self._full_w, self._full_h) if wm else None)

    # ------------------------------------------------------------------
    # five-angle 3D avatar export (OBJ / FBX / GLB)
    # ------------------------------------------------------------------
    def _on_pick_five_angle(self, angle: str) -> None:
        path = filedialog.askopenfilename(
            title=f"Select {_FIVE_ANGLE_LABELS[angle]} image",
            filetypes=_IMAGE_DIALOG_TYPES,
        )
        if not path:
            return
        self._five_paths[angle] = path
        short = Path(path).name
        if len(short) > 22:
            short = short[:19] + "…"
        self._five_lbl[angle].config(text=short)
        self._refresh_five_export_button()

    def _refresh_five_export_button(self) -> None:
        if self._btn_export_avatars is None:
            return
        ready = any(self._five_paths.get(k) for k in _FIVE_ANGLES_ORDER)
        self._btn_export_avatars.config(state=tk.NORMAL if ready else tk.DISABLED)

    def _on_export_avatars(self) -> None:
        paths = {
            k: str(Path(self._five_paths[k]).resolve())
            for k in _FIVE_ANGLES_ORDER
            if self._five_paths.get(k)
        }
        if not paths:
            messagebox.showinfo(
                "3D avatar",
                "Pick at least one still (Front, Left, Top, Back, or Right).",
            )
            return
        out_dir = filedialog.askdirectory(
            title="Folder for avatar.obj, avatar.fbx, avatar.glb",
            initialdir=str(_ROOT / "outputs"),
        )
        if not out_dir:
            return
        self._avatar_export_status = None

        def run() -> None:
            try:
                proc = SkeletonProcessor()
                try:
                    skel = proc.process_images(paths, submission_id="swing_coach_ui")
                finally:
                    proc.close()
                gen = AvatarGenerator()
                gen.set_source_images(paths, skel.get("angles"))
                mesh = gen.generate_from_multi_angle(skel)
                base = Path(out_dir)
                obj_p = gen.export_obj(mesh, str(base / "avatar.obj"))
                glb_p = gen.export_glb(mesh, str(base / "avatar.glb"))
                fbx_p = gen.export_fbx(mesh, str(base / "avatar.fbx"))
                # export skeleton JSON alongside the mesh files
                import json as _json
                skel_json_p = base / "avatar_skeleton.json"
                with open(skel_json_p, "w", encoding="utf-8") as _f:
                    _json.dump(skel, _f, indent=2)
                saved_lines = f"{obj_p}\n{fbx_p}\n{glb_p}\n{skel_json_p}"
                try:
                    renderer = AvatarRenderer()
                    renders = renderer.render_5_angles(mesh)
                    png_paths = renderer.save_renders(renders, str(base))
                    saved_lines += "\n" + "\n".join(png_paths.values())
                except Exception:
                    saved_lines += "\n(5-angle PNGs skipped — renderer unavailable)"
                self._avatar_export_status = ("ok", saved_lines)
            except Exception as exc:
                self._avatar_export_status = ("err", str(exc))

        threading.Thread(target=run, daemon=True).start()
        self.status.config(text="Exporting 3D avatars (pose + mesh)…")
        self.after(200, self._poll_avatar_export)

    def _poll_avatar_export(self) -> None:
        st = self._avatar_export_status
        if st is None:
            self.after(200, self._poll_avatar_export)
            return
        self._avatar_export_status = None
        kind, payload = st
        if kind == "ok":
            self.status.config(text="Avatar export complete.")
            messagebox.showinfo("Avatar export", f"Wrote:\n{payload}")
        else:
            self.status.config(text="Avatar export failed.")
            messagebox.showerror("Avatar export", payload)

    # ------------------------------------------------------------------
    # cleanup
    # ------------------------------------------------------------------
    def _release_all(self) -> None:
        self._raw_frames = []
        self.joints_per_frame = []
        self.head_ref_y = None
        self._ideal_segment = None
        self.wrist_trail_px = []
        self.static_lines = []
        self.static_labels = []
        self._label_counter = 0
        self._drag_start = None
        self._drag_preview = None
        if self._prep_temp and self._video_path:
            try:
                os.unlink(self._video_path)
            except OSError:
                pass
        self._prep_temp = False
        self._video_path = None

    # ------------------------------------------------------------------
    # playback
    # ------------------------------------------------------------------
    def _auto_play(self) -> None:
        if self._total > 1:
            self.var_playing.set(True)
            self.btn_play.config(text="Pause")
            self._schedule_tick()

    def _toggle_play(self) -> None:
        if self._total < 1 or not self.joints_per_frame:
            return
        self.var_playing.set(not self.var_playing.get())
        self.btn_play.config(text="Pause" if self.var_playing.get() else "Play")
        if self.var_playing.get():
            self._schedule_tick()
        else:
            self._cancel_tick()

    def _stop_play(self) -> None:
        self.var_playing.set(False)
        self.btn_play.config(text="Play")
        self._cancel_tick()

    def _cancel_tick(self) -> None:
        if self._play_after_id is not None:
            try:
                self.after_cancel(self._play_after_id)
            except (tk.TclError, ValueError):
                pass
            self._play_after_id = None

    def _schedule_tick(self) -> None:
        if not self.var_playing.get():
            return
        delay = max(1, int(round(1000.0 / max(1.0, self._fps))))
        self._play_after_id = self.after(delay, self._play_tick)

    def _play_tick(self) -> None:
        if not self.var_playing.get():
            return
        if self._frame_index >= self._total - 1:
            self._frame_index = 0
        else:
            self._frame_index += 1
        self.scale.set(self._frame_index)
        self._refresh()
        self._schedule_tick()

    def _on_scale(self, value: str) -> None:
        try:
            self._frame_index = int(float(value))
        except ValueError:
            return
        self._refresh()

    # ------------------------------------------------------------------
    # coach drawing
    # ------------------------------------------------------------------
    def _clear_lines(self) -> None:
        self.static_lines = []
        self.static_labels = []
        self._label_counter = 0
        self._drag_preview = None
        self._refresh()

    def _coach_color(self) -> Tuple[int, int, int]:
        ink = self.var_coach_ink.get()
        if ink == "red":
            return _COACH_RED
        if ink == "green":
            return _COACH_GREEN
        return _COACH_WHITE

    def _on_draw_tool_change(self) -> None:
        self._drag_start = None
        self._drag_preview = None
        self._refresh()

    def _on_mouse_down(self, ev: tk.Event) -> None:
        if not self.joints_per_frame:
            return
        self._drag_start = (int(ev.x), int(ev.y))

    def _on_mouse_move(self, ev: tk.Event) -> None:
        if self._drag_start is None:
            return
        if self.var_draw_tool.get() in ("label", "headline"):
            return
        self._drag_preview = (self._drag_start, (int(ev.x), int(ev.y)))
        self._refresh()

    def _on_mouse_up(self, ev: tk.Event) -> None:
        if self._drag_start is None:
            return
        x1, y1 = self._drag_start
        x2, y2 = int(ev.x), int(ev.y)
        self._drag_start = None
        self._drag_preview = None

        if self.var_draw_tool.get() in ("label", "headline"):
            ax, ay = x1, y1
            fx, fy = self._disp_to_full(ax, ay)
            caption = self.entry_label_text.get().strip()

            if self.var_draw_tool.get() == "headline":
                if not caption:
                    self.status.config(
                        text="Headline: type your phrase in “Label text”, then click to place (center)."
                    )
                    self._refresh()
                    return
                self.static_labels.append(
                    {
                        "style": "headline",
                        "x": fx,
                        "y": fy,
                        "text": caption.upper(),
                    }
                )
                self._refresh()
                return

            numbered = self.var_label_numbered.get()
            if numbered:
                self._label_counter += 1
                main = str(self._label_counter)
                sub = caption
                self.static_labels.append(
                    {
                        "style": "numbered_circle",
                        "x": fx,
                        "y": fy,
                        "main": main,
                        "sub": sub,
                    }
                )
            else:
                if not caption:
                    self.status.config(
                        text="Label: enter text in “Label text”, or enable “Numbered (1, 2, 3…)”."
                    )
                    self._refresh()
                    return
                self.static_labels.append(
                    {
                        "style": "caption_box",
                        "x": fx,
                        "y": fy,
                        "main": caption,
                        "sub": "",
                        "color": self._coach_color(),
                    }
                )
            self._refresh()
            return

        if max(abs(x2 - x1), abs(y2 - y1)) < 4:
            self._refresh()
            return
        self.static_lines.append({
            "start": self._disp_to_full(x1, y1),
            "end": self._disp_to_full(x2, y2),
            "color": self._coach_color(),
            "thickness": 2,
        })
        self._refresh()

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def _compose_frame(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self._raw_frames):
            return np.zeros((max(2, self._full_h), max(2, self._full_w), 3), dtype=np.uint8)
        frame = self._raw_frames[idx]
        joints = self.joints_per_frame[idx] if idx < len(self.joints_per_frame) else []
        static: List[Dict[str, Any]] = list(self.static_lines)
        if self.var_swing_paths.get() and self._ideal_segment is not None:
            static = static + [{"start": self._ideal_segment[0], "end": self._ideal_segment[1], "color": _PURPLE_IDEAL, "thickness": 2}]
        out = self._overlay.draw_overlay_frame(
            frame,
            joints,
            smooth_arm_display=True,
            show_shoulders=self.var_shoulders.get(),
            show_waistline=self.var_waist.get(),
            show_legs=self.var_legs.get(),
            show_spine=self.var_spine.get(),
            head_reference_y=self.head_ref_y,
            head_reference_line_color=_HEAD_REF_BLUE,
            use_orange_yellow_joint_dots=True,
            static_lines=static,
        )
        if self.static_labels:
            draw_static_labels_on_frame(out, self.static_labels)
        if self.var_swing_paths.get():
            pts = [p for p in self.wrist_trail_px[:idx + 1] if p is not None]
            if len(pts) >= 2:
                cv2.polylines(out, [np.array(pts, dtype=np.int32)], False, _YELLOW_SWING_PATH, 2, cv2.LINE_AA)
            elif pts:
                cv2.circle(out, pts[0], 4, _YELLOW_SWING_PATH, -1, cv2.LINE_AA)
        if self._drag_preview:
            (dx1, dy1), (dx2, dy2) = self._drag_preview
            cv2.line(out, self._disp_to_full(dx1, dy1), self._disp_to_full(dx2, dy2), self._coach_color(), 2, cv2.LINE_AA)
        return out

    def _refresh(self) -> None:
        if not self._raw_frames or not self.joints_per_frame:
            return
        full = self._compose_frame(self._frame_index)
        if self._disp_scale < 0.99:
            disp = cv2.resize(full, (self._disp_w, self._disp_h), interpolation=cv2.INTER_AREA)
        else:
            disp = full
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor=tk.NW)
        t = self._frame_index / max(self._fps, 1e-6)
        self.meta.config(text=f"Frame {self._frame_index + 1}/{self._total}  t={t:.2f}s")

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def _on_export(self) -> None:
        if not self._raw_frames or not self.joints_per_frame:
            messagebox.showinfo("Export", "Load a video and wait for processing first.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save annotated MP4", defaultextension=".mp4", filetypes=[("MP4", "*.mp4")],
        )
        if not out_path:
            return
        self._stop_play()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, self._fps, (self._full_w, self._full_h))
        if not writer.isOpened():
            messagebox.showerror("Export", f"Cannot create video writer:\n{out_path}")
            return
        saved_idx = self._frame_index
        try:
            for i in range(self._total):
                writer.write(self._compose_frame(i))
                if i % 15 == 0:
                    self.status.config(text=f"Exporting… {i + 1}/{self._total}")
                    self.update_idletasks()
        finally:
            writer.release()
        self._frame_index = saved_idx
        self.scale.set(saved_idx)
        self._refresh()
        self.status.config(text=f"Saved: {out_path}")
        messagebox.showinfo("Export", f"Wrote:\n{out_path}")

    # ------------------------------------------------------------------
    def destroy(self) -> None:  # type: ignore[override]
        self._stop_play()
        self._release_all()
        super().destroy()


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
    app = SwingCoachAnnotationApp()
    app.mainloop()


if __name__ == "__main__":
    main()
