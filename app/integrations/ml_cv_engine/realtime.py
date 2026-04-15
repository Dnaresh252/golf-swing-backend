"""Real-time webcam pose detection with overlay rendering.

Keyboard (live webcam — OpenCV window must be focused):
  q — quit · v — pick video file → editor mode
  s / w / l / d / h — skeleton + head reference toggles
  x / u — clear / undo coach lines
  With ``--browser``, use the web control panel at the same URL for virtual keys (no browser upload).

Recorded video mode (``--video``): SPACE pause/play, a/f prev/next frame, r restart,
o export coach lines JSON, same s w l d h x u and coach drag as live.

Coach reference lines: click and drag with the left mouse button on the **OpenCV**
window to draw a straight segment (same format as ``static_lines`` in the
export pipeline). Lines stay fixed on screen while the athlete moves.

If the OpenCV window never appears (common from IDE terminals), run with
``--browser`` to view the same feed in **Chrome / Edge** at http://127.0.0.1:8765
(MJPEG). **Keyboard letters only work when the OpenCV window is focused**; in
browser-only mode use the **on-page control panel** (same URL) for toggles and
pause/step. Use ``/stop`` or **Ctrl+C** to release the webcam. For file-based
coaching with upload + export, use ``python -m ml_cv_engine.swing_coach_annotation_app``.

**Recorded video review** (``--video``): load an MP4 (path or file picker), run pose
on each frame, pause/step with keyboard, draw the same coach static lines on
top of the clip. Press **o** to save coach lines + frame size to JSON for the
pipeline / backend.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .overlay_2d import COACH_STATIC_LINE_COLOR, SkeletonOverlay
from .pose_detection import PoseDetector

_WINDOW = "AI Golf Swing - Realtime Pose"

_BROWSER_PRESS_ALLOWED = frozenset(
    ("q", "s", "w", "l", "d", "h", "x", "u", "space", "sp", "a", "f", "r", "o", "p")
)


def _browser_token_to_key_ord(token: str) -> int:
    """Map control-panel token to OpenCV-style key code (lowercase letter or space)."""
    t = (token or "").strip().lower()
    if t in ("space", "sp", "p"):
        return ord(" ")
    if len(t) == 1 and t in "qswldhxuafor":
        return ord(t)
    return 0


def _mjpeg_control_page_html(port: int) -> str:
    """Single-page UI: MJPEG stream + virtual keys (browser-only workflows)."""
    p = int(port)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Golf realtime — controls</title>
<style>
body{{margin:0;background:#111;color:#ddd;font:14px system-ui,sans-serif}}
.bar{{padding:10px 12px;background:#222;border-bottom:1px solid #333;display:flex;flex-wrap:wrap;gap:8px;align-items:center}}
a.stop{{color:#9cf;font-weight:700}}
.panel{{padding:12px;background:#1a1a1a;border-bottom:1px solid #333}}
.panel h3{{margin:0 0 8px;font-size:13px;color:#9cf}}
.keys{{display:flex;flex-wrap:wrap;gap:6px}}
.keys button{{padding:6px 10px;border-radius:6px;border:1px solid #444;background:#333;color:#eee;cursor:pointer;font-size:12px}}
.keys button:hover{{background:#444}}
.hint{{opacity:.75;font-size:12px;margin-top:8px;line-height:1.45}}
img.stream{{width:100%;height:auto;display:block}}
</style></head><body>
<div style="background:#422;color:#fda;padding:8px 12px;font-size:13px;border-bottom:1px solid #633">
  <b>Important:</b> URL must be <code>http://127.0.0.1:{p}/</code> &mdash; <b>not https</b> (https will error).
  If the video box stays gray, wait for the camera; check <a href="/health" style="color:#9ef">/health</a> (should say ok).
</div>
<div class="bar">
  <b>Live / editor</b>
  <a class="stop" href="/stop"><b>Close cam · stop</b></a>
  <span style="opacity:.7">Port {p}</span>
  <span style="color:#fa6;font-size:11px;margin-left:8px">http:// only (not https)</span>
</div>
<div class="panel">
  <h3>Virtual keys (use when there is no OpenCV window)</h3>
  <div class="keys" id="keys"></div>
  <div class="hint">
    <b>Live webcam:</b> S/W/L/D/H = body sections · X clear coach lines · U undo line · Q quit<br/>
    <b>Video editor:</b> Space/P pause · A/F step frame · R restart · O export JSON · same S–U as above<br/>
    <span style="opacity:.85">Clip review from disk: use <code>python -m ml_cv_engine.realtime --video</code> or press <b>V</b> in the OpenCV window. For Tk upload + export use <code>python -m ml_cv_engine.swing_coach_annotation_app</code>.</span>
  </div>
</div>
<img class="stream" src="/stream" alt="stream"/>
<script>
const keys = [
  ['s','S shoulders'],['w','W waist'],['l','L legs'],['d','D spine'],['h','H head line'],
  ['x','X clear lines'],['u','U undo line'],['space','Space pause'],
  ['a','A prev frame'],['f','F next frame'],['r','R restart'],['o','O export JSON'],['q','Q quit']
];
const el = document.getElementById('keys');
for (const [k,label] of keys) {{
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = label;
  b.onclick = () => fetch('/press?k='+encodeURIComponent(k)).catch(()=>{{}});
  el.appendChild(b);
}}
</script>
</body></html>"""


class _JpegFeed:
    """Latest frame as JPEG bytes for MJPEG clients (thread-safe)."""

    __slots__ = ("_lock", "_jpeg", "stop", "_key_q")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stop = threading.Event()
        self._key_q: "queue.Queue[str]" = queue.Queue(maxsize=64)
        self._jpeg: bytes = self._bootstrap_jpeg_bytes()

    @staticmethod
    def _bootstrap_jpeg_bytes() -> bytes:
        """Non-empty JPEG so /stream always has a first part (avoids stuck/broken img)."""
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "Starting...",
            (6, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        return enc.tobytes() if ok else b""

    def publish(self, bgr: Any) -> None:
        ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if ok:
            raw = enc.tobytes()
            with self._lock:
                self._jpeg = raw

    def peek(self) -> bytes:
        with self._lock:
            return self._jpeg

    def enqueue_browser_key(self, token: str) -> None:
        try:
            self._key_q.put_nowait(token.strip().lower())
        except queue.Full:
            pass

    def drain_browser_keys(self) -> List[str]:
        out: List[str] = []
        while True:
            try:
                out.append(self._key_q.get_nowait())
            except queue.Empty:
                break
        return out


def _make_mjpeg_handler(feed: _JpegFeed, listen_port: int) -> type:
    class _MJPEGHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def do_GET(self) -> None:
            up = urlparse(self.path)
            path = up.path
            if path == "/health":
                self.send_response(200)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b"ok\n")
                return
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if path == "/press":
                q = parse_qs(up.query)
                raw_tok = (q.get("k") or [""])[0].strip().lower()
                if raw_tok == "sp":
                    raw_tok = "space"
                if raw_tok in _BROWSER_PRESS_ALLOWED:
                    feed.enqueue_browser_key(raw_tok)
                self.send_response(204)
                self.end_headers()
                return
            if path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(_mjpeg_control_page_html(listen_port).encode("utf-8"))
                return
            if path == "/stop":
                feed.stop.set()
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    "<!DOCTYPE html><html><body style='font:16px sans-serif;padding:24px'>"
                    "<p>Stopping — releasing the webcam. You can close this tab.</p>"
                    "<p>If the stream tab is still open, close it too.</p>"
                    "</body></html>".encode("utf-8")
                )
                return
            if path == "/stream":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                while True:
                    if feed.stop.is_set():
                        return
                    chunk = feed.peek()
                    if chunk:
                        try:
                            self.wfile.write(boundary + chunk + b"\r\n")
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            return
                    time.sleep(0.04)
            self.send_error(404)

    return _MJPEGHandler


def _mjpeg_remote_hint() -> None:
    if os.environ.get("SSH_CONNECTION") or os.environ.get("REMOTE_CONTAINERS"):
        print(
            "Remote session: open the URL on this same machine, or use SSH port forwarding "
            "(e.g. ssh -L 8765:127.0.0.1:<port> ...) for the port printed above.",
            flush=True,
        )


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """One client (e.g. ``/stream``) must not block ``/press`` or ``/stop``."""

    daemon_threads = True
    allow_reuse_address = True


def _print_mjpeg_open_banner(listen_port: int) -> None:
    """Console hint; server is plain HTTP only (HTTPS would show ERR_SSL_PROTOCOL_ERROR)."""
    print(
        f"\n*** Open in your browser: http://127.0.0.1:{listen_port}/ ***\n",
        flush=True,
    )
    print(
        "Use http:// only (not https://) - this preview has no TLS/SSL.",
        flush=True,
    )
    print(
        "Virtual keys and /stop run in parallel with the stream (threading server).",
        flush=True,
    )
    print(
        "If the page is blank or buttons do nothing: use http:// (not https://). "
        "Test: http://127.0.0.1:%d/health should print ok."
        % (listen_port,),
        flush=True,
    )


def _start_mjpeg_server(
    feed: _JpegFeed, preferred_port: int
) -> Tuple[HTTPServer, threading.Thread, int]:
    """Bind MJPEG server; try ``preferred_port`` then the next ports if busy."""
    last_exc: Optional[BaseException] = None
    server: Optional[HTTPServer] = None
    bound_port = preferred_port
    for delta in range(20):
        port = preferred_port + delta
        try:
            handler_cls = _make_mjpeg_handler(feed, port)
            server = _ThreadingHTTPServer(("0.0.0.0", port), handler_cls)
            bound_port = port
            if delta:
                print(
                    f"Note: port {preferred_port} was in use; MJPEG bound to {bound_port}.",
                    flush=True,
                )
            break
        except OSError as e:
            last_exc = e
    else:
        raise RuntimeError(
            f"Could not bind MJPEG server on ports {preferred_port}-{preferred_port + 19}: {last_exc}"
        ) from last_exc
    assert server is not None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, bound_port


def _publish_mjpeg_placeholder(feed: _JpegFeed, message: str, w: int = 640, h: int = 360) -> None:
    """So the browser shows a frame while MediaPipe loads (avoids an empty stream)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (45, 45, 45)
    cv2.putText(
        img,
        message,
        (20, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    feed.publish(img)


def _open_video_capture(device_index: int) -> cv2.VideoCapture:
    """Prefer DirectShow on Windows so the device opens reliably for preview."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(device_index)


def _pick_video_path() -> Optional[str]:
    """Open a native file dialog; return path or None if cancelled / unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("Install tkinter or pass an explicit path: --video path/to/file.mp4", flush=True)
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    path = filedialog.askopenfilename(
        parent=root,
        title="Choose a swing video (MP4, MOV, …)",
        filetypes=[
            ("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"),
            ("MP4", "*.mp4"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path if path else None


def _coach_mouse_callback(event: int, x: int, y: int, _flags: int, param: Dict[str, Any]) -> None:
    """Left-drag adds a static line on release; move updates rubber-band preview."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param["drag_start"] = (x, y)
        param["preview_end"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and param.get("drag_start") is not None:
        param["preview_end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        st = param.get("drag_start")
        if st is not None:
            end = (x, y)
            if (st[0] - end[0]) ** 2 + (st[1] - end[1]) ** 2 >= 9:
                param.setdefault("static_lines", []).append({
                    "start": st,
                    "end": end,
                    "color": COACH_STATIC_LINE_COLOR,
                    "thickness": 2,
                })
        param["drag_start"] = None
        param["preview_end"] = None


class RealTimeDetector:
    """Live webcam pose and optional recorded-video review with coach overlays."""

    def __init__(self, min_confidence: float = 0.05) -> None:
        self._min_confidence = min_confidence
        self.pose_detector: Optional[PoseDetector] = None
        self.overlay = SkeletonOverlay()

    def _ensure_pose_detector(self) -> None:
        """Create MediaPipe on first use so the MJPEG server can bind first (avoids refused)."""
        if self.pose_detector is not None:
            return
        print(
            "Loading pose model (MediaPipe; first run can take 20-40s on some PCs)...",
            flush=True,
        )
        self.pose_detector = PoseDetector(
            min_confidence=self._min_confidence,
            static_image_mode=False,
            model_complexity=1,
        )

    def _close_pose_detector(self) -> None:
        if self.pose_detector is None:
            return
        try:
            self.pose_detector.close()
        finally:
            self.pose_detector = None

    def start_webcam_session(
        self,
        callback: Optional[Callable[[list], None]] = None,
        *,
        device_index: int = 0,
        http_port: Optional[int] = None,
        show_window: bool = True,
    ) -> Optional[str]:
        """Start a webcam session and render live skeleton overlays.

        Returns:
            If the coach presses **V** (file picker), returns the chosen path so
            ``main()`` can continue with :meth:`start_video_review_session`;
            otherwise ``None``.

        Keyboard (OpenCV window must be focused — keys do not go to the terminal):
          q — quit
          v — pick a video file from this PC → switch to video editor mode
          s / w / l / d / h — skeleton & head-ref toggles
          x / u — clear / undo coach lines

        Browser-only (``--browser``): open the printed URL and use the **control panel**
        buttons (same host); keys are not delivered to Python without that window.

        Coach drawing: left-click drag on the OpenCV window to place a static line.

        Args:
            callback: Optional function called as callback(joints) every frame.
            device_index: Webcam index (0 = default). Try 1 if the wrong camera opens.
            http_port: If set, serve the same frames at ``http://127.0.0.1:<port>/`` (MJPEG).
            show_window: If False, skip ``imshow`` (use with ``http_port``; stop with Ctrl+C).

        Note:
            IDE-embedded terminals often never show the OpenCV window. Use
            ``--browser`` (HTTP preview) or run from **cmd.exe** / **PowerShell** on the PC.
        """
        jpeg_feed: Optional[_JpegFeed] = None
        http_server: Optional[HTTPServer] = None
        mjpeg_listen_port: Optional[int] = None
        if http_port is not None:
            jpeg_feed = _JpegFeed()
            http_server, _http_thread, mjpeg_listen_port = _start_mjpeg_server(
                jpeg_feed, int(http_port)
            )
            _print_mjpeg_open_banner(mjpeg_listen_port)
            _mjpeg_remote_hint()

        cap = _open_video_capture(device_index)
        if not cap.isOpened():
            err = (
                f"Unable to open webcam (index {device_index}). "
                "Close other apps, check Windows Privacy > Camera for Python, or try --device 1."
            )
            print(err, flush=True)
            try:
                cap.release()
            except (cv2.error, AttributeError):
                pass
            if jpeg_feed is not None and http_server is not None:
                _publish_mjpeg_placeholder(
                    jpeg_feed,
                    "WEBCAM FAILED - see terminal. Try --device 1 or fix Camera privacy.",
                )
                print(
                    f"Open http://127.0.0.1:{mjpeg_listen_port}/ to read this in the browser. "
                    "Use /stop or Ctrl+C when done.",
                    flush=True,
                )
                try:
                    while not jpeg_feed.stop.is_set():
                        time.sleep(0.25)
                except KeyboardInterrupt:
                    print("\nStopped (KeyboardInterrupt).", flush=True)
                finally:
                    try:
                        http_server.shutdown()
                    except (OSError, RuntimeError):
                        pass
                return None
            raise ValueError(err)

        if show_window:
            cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

        print(
            f"Realtime webcam: device={device_index}, "
            f"window={'on' if show_window else 'off'}, "
            f"http={mjpeg_listen_port if mjpeg_listen_port is not None else 'off'}",
            flush=True,
        )
        if http_port is None:
            print(
                "Tip: no HTTP preview. Add --browser or --http for http://127.0.0.1:8765/",
                flush=True,
            )
        if not show_window:
            print("Browser-only mode: press Ctrl+C in this terminal to stop.", flush=True)
            if mjpeg_listen_port is not None:
                print(
                    f"Or open http://127.0.0.1:{mjpeg_listen_port}/stop to close cam and exit.",
                    flush=True,
                )
        print(
            "Keys (webcam): click the OpenCV window first - "
            "Q quit | V pick video -> editor | S/W/L/D/H overlays | X clear | U undo coach line",
            flush=True,
        )
        if mjpeg_listen_port is not None:
            print(
                f"Browser controls: http://127.0.0.1:{mjpeg_listen_port}/ (virtual keys)",
                flush=True,
            )

        if jpeg_feed is not None:
            _publish_mjpeg_placeholder(
                jpeg_feed,
                "Loading MediaPipe - wait for live frames...",
            )
        self._ensure_pose_detector()
        pose_det = self.pose_detector
        assert pose_det is not None

        frame_count = 0
        last_time = time.perf_counter()
        last_joints: list = []
        run_every_other = False

        show_shoulders = True
        show_waistline = True
        show_legs = True
        show_spine = True
        show_head_ref = True
        head_ref_y: Optional[int] = None

        coach_state: Dict[str, Any] = {
            "drag_start": None,
            "preview_end": None,
            "static_lines": [],
        }
        mouse_cb_registered = False
        window_positioned = False
        editor_handoff: Optional[str] = None

        try:
            while True:
                if jpeg_feed is not None:
                    time.sleep(0)
                if jpeg_feed is not None and jpeg_feed.stop.is_set():
                    print("Stopped (browser /stop).", flush=True)
                    break
                ok, frame = cap.read()
                if not ok:
                    if frame_count == 0:
                        raise ValueError(
                            "Webcam opened but first frame read failed. "
                            "Try another --device index or a different USB port."
                        )
                    break

                frame_count += 1
                now = time.perf_counter()
                elapsed = max(now - last_time, 1e-9)
                fps = 1.0 / elapsed
                last_time = now

                run_every_other = fps < 30.0
                should_run_pose = (not run_every_other) or (frame_count % 2 == 0)

                if should_run_pose:
                    pose = pose_det.detect_from_frame(frame)
                    last_joints = pose["joints"] if pose is not None else []

                height, width = frame.shape[:2]

                if show_window and not window_positioned:
                    window_positioned = True
                    cv2.resizeWindow(_WINDOW, max(480, width), max(360, height))
                    cv2.moveWindow(_WINDOW, 80, 80)
                    try:
                        cv2.setWindowProperty(_WINDOW, cv2.WND_PROP_TOPMOST, 1)
                        cv2.setWindowProperty(_WINDOW, cv2.WND_PROP_TOPMOST, 0)
                    except (cv2.error, AttributeError):
                        pass

                # Capture head reference Y from first valid detection
                if show_head_ref and head_ref_y is None and last_joints:
                    jmap = {int(j["id"]): j for j in last_joints if "id" in j}
                    nose = jmap.get(0)
                    if nose is not None:
                        head_ref_y = int(round(
                            max(0.0, min(1.0, float(nose["y"]))) * (height - 1)
                        ))

                rendered = self.overlay.draw_overlay_frame(
                    frame,
                    last_joints,
                    smooth_arm_display=True,
                    show_shoulders=show_shoulders,
                    show_waistline=show_waistline,
                    show_legs=show_legs,
                    show_spine=show_spine,
                    head_reference_y=head_ref_y if show_head_ref else None,
                    static_lines=list(coach_state.get("static_lines") or []),
                )

                ds = coach_state.get("drag_start")
                pe = coach_state.get("preview_end")
                if ds is not None and pe is not None:
                    cv2.line(rendered, ds, pe, COACH_STATIC_LINE_COLOR, 2, cv2.LINE_AA)

                # HUD — FPS and toggle status
                cv2.putText(
                    rendered,
                    f"FPS: {fps:.1f}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                toggle_parts: List[str] = []
                if show_shoulders:
                    toggle_parts.append("S")
                if show_waistline:
                    toggle_parts.append("W")
                if show_legs:
                    toggle_parts.append("L")
                if show_spine:
                    toggle_parts.append("D")
                if show_head_ref:
                    toggle_parts.append("H")
                toggle_txt = "Sections: " + ("+".join(toggle_parts) if toggle_parts else "none")
                cv2.putText(
                    rendered,
                    toggle_txt,
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                n_coach = len(coach_state.get("static_lines") or [])
                cv2.putText(
                    rendered,
                    f"Coach lines: {n_coach} (drag=LMB, x=clear, u=undo)",
                    (12, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )
                if not show_window and mjpeg_listen_port is not None:
                    cv2.putText(
                        rendered,
                        f"Panel: http://127.0.0.1:{mjpeg_listen_port}/  |  /stop  Ctrl+C",
                        (12, height - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (180, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if jpeg_feed is not None:
                    jpeg_feed.publish(rendered)

                if show_window:
                    cv2.imshow(_WINDOW, rendered)
                    if not mouse_cb_registered:
                        cv2.setMouseCallback(_WINDOW, _coach_mouse_callback, coach_state)
                        mouse_cb_registered = True

                if callback is not None:
                    callback(last_joints)

                raw_key = (cv2.waitKey(1) & 0xFF) if show_window else 0
                if not show_window:
                    time.sleep(0.025)
                key_codes: List[int] = []
                if raw_key:
                    key_codes.append(raw_key)
                if jpeg_feed is not None:
                    for t in jpeg_feed.drain_browser_keys():
                        ko = _browser_token_to_key_ord(t)
                        if ko:
                            key_codes.append(ko)
                if not key_codes:
                    key_codes = [0]

                leave_webcam = False
                for key in key_codes:
                    if key == 0:
                        continue
                    if key in (ord("q"), ord("Q")):
                        leave_webcam = True
                        break
                    if key in (ord("v"), ord("V")):
                        if show_window:
                            picked = _pick_video_path()
                            if picked and os.path.isfile(picked):
                                editor_handoff = picked
                                leave_webcam = True
                        break
                    if key == ord("s"):
                        show_shoulders = not show_shoulders
                    elif key == ord("w"):
                        show_waistline = not show_waistline
                    elif key == ord("l"):
                        show_legs = not show_legs
                    elif key == ord("d"):
                        show_spine = not show_spine
                    elif key == ord("h"):
                        show_head_ref = not show_head_ref
                        if not show_head_ref:
                            head_ref_y = None
                    elif key == ord("x"):
                        coach_state["static_lines"] = []
                        coach_state["drag_start"] = None
                        coach_state["preview_end"] = None
                    elif key == ord("u"):
                        lines = coach_state.get("static_lines") or []
                        if lines:
                            lines.pop()
                            coach_state["static_lines"] = lines
                if leave_webcam:
                    break
                if editor_handoff:
                    break
                if jpeg_feed is not None and jpeg_feed.stop.is_set():
                    print("Stopped (browser /stop).", flush=True)
                    break
        except KeyboardInterrupt:
            print("\nStopped (KeyboardInterrupt).", flush=True)
        finally:
            if http_server is not None:
                try:
                    http_server.shutdown()
                except (OSError, RuntimeError):
                    pass
            if show_window:
                try:
                    cv2.setMouseCallback(_WINDOW, lambda *a, **k: None)
                except (cv2.error, AttributeError):
                    pass
                try:
                    cv2.destroyAllWindows()
                except (cv2.error, AttributeError):
                    pass
            cap.release()
            if editor_handoff is None:
                self._close_pose_detector()
        return editor_handoff

    def start_video_review_session(
        self,
        video_path: str,
        *,
        callback: Optional[Callable[[list], None]] = None,
        http_port: Optional[int] = None,
        show_window: bool = True,
    ) -> None:
        """Play a recorded video with pose overlay and coach static-line drawing.

        OpenCV window (when enabled):
          SPACE — pause / resume
          a / f — previous / next frame (when paused, or single-step while playing)
          r — restart from frame 0 (head reference line re-captured on next valid pose)
          o — save coach lines + frame size to ``<video_stem>_coach_lines.json``
          q — quit
          s w l d h x u — same as webcam session

        Coach drawing: left-click drag on the frame (lines stay fixed on screen).

        Args:
            video_path: Path to MP4/MOV/… readable by OpenCV.
            callback: Optional ``callback(joints)`` after each processed frame.
            http_port: Optional MJPEG mirror at ``http://127.0.0.1:<port>/``.
            show_window: If False, use HTTP preview only (Ctrl+C to stop).
        """
        if not os.path.isfile(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_src = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        delay_ms = max(1, min(60, int(1000.0 / max(1.0, fps_src))))

        self.overlay.reset_arm_display_smooth()

        jpeg_feed: Optional[_JpegFeed] = None
        http_server: Optional[HTTPServer] = None
        mjpeg_listen_port: Optional[int] = None
        if http_port is not None:
            jpeg_feed = _JpegFeed()
            http_server, _http_thread, mjpeg_listen_port = _start_mjpeg_server(
                jpeg_feed, int(http_port)
            )
            _print_mjpeg_open_banner(mjpeg_listen_port)
            _mjpeg_remote_hint()

        if show_window:
            cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

        print(
            f"Video review: {video_path}\n"
            f"  frames={total}, fps={fps_src:.2f}, "
            f"window={'on' if show_window else 'off'}, "
            f"http={mjpeg_listen_port if mjpeg_listen_port is not None else 'off'}\n"
            f"  SPACE pause | a/f step | r restart | o export coach JSON | q quit",
            flush=True,
        )
        if http_port is None:
            print(
                "Tip: no HTTP preview. Re-run with --browser (or --http) to open http://127.0.0.1:8765/",
                flush=True,
            )
        if not show_window:
            print("Browser-only: Ctrl+C in this terminal to stop.", flush=True)
            if mjpeg_listen_port is not None:
                print(
                    f"Or open http://127.0.0.1:{mjpeg_listen_port}/stop to close cam and exit.",
                    flush=True,
                )
        print(
            "Keys (video editor): focus OpenCV window - "
            "Space pause/play | A/F step | R restart | O export JSON | Q quit | "
            "S/W/L/D/H | X clear | U undo",
            flush=True,
        )
        if mjpeg_listen_port is not None:
            print(
                f"Browser panel (virtual keys): http://127.0.0.1:{mjpeg_listen_port}/",
                flush=True,
            )

        if jpeg_feed is not None:
            _publish_mjpeg_placeholder(
                jpeg_feed,
                "Loading MediaPipe - wait for live frames...",
            )
        self._ensure_pose_detector()
        pose_det = self.pose_detector
        assert pose_det is not None

        frame_idx = 0
        paused = False
        last_joints: list = []
        show_shoulders = True
        show_waistline = True
        show_legs = True
        show_spine = True
        show_head_ref = True
        head_ref_y: Optional[int] = None
        head_ref_locked = False

        coach_state: Dict[str, Any] = {
            "drag_start": None,
            "preview_end": None,
            "static_lines": [],
        }
        mouse_cb_registered = False
        window_positioned = False

        def _export_coach_json(w: int, h: int) -> None:
            out_path = os.path.splitext(video_path)[0] + "_coach_lines.json"
            lines_out: List[Dict[str, Any]] = []
            for ln in coach_state.get("static_lines") or []:
                s = ln.get("start")
                e = ln.get("end")
                if s is None or e is None:
                    continue
                lines_out.append({
                    "start": [int(s[0]), int(s[1])],
                    "end": [int(e[0]), int(e[1])],
                    "thickness": int(ln.get("thickness", 2)),
                    "color": [int(c) for c in ln.get("color", COACH_STATIC_LINE_COLOR)],
                })
            payload = {
                "source_video": os.path.abspath(video_path),
                "frame_width": int(w),
                "frame_height": int(h),
                "static_lines": lines_out,
            }
            with open(out_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)
            print(f"Wrote coach overlay: {out_path}", flush=True)

        try:
            while True:
                if jpeg_feed is not None:
                    time.sleep(0)
                if jpeg_feed is not None and jpeg_feed.stop.is_set():
                    print("Stopped (browser /stop).", flush=True)
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, frame_idx), total - 1))
                ok, frame = cap.read()
                if not ok or frame is None:
                    paused = True
                    frame_idx = min(frame_idx, max(0, total - 1))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break

                pose = pose_det.detect_from_frame(frame)
                last_joints = pose["joints"] if pose is not None else []

                height, width = frame.shape[:2]

                if show_window and not window_positioned:
                    window_positioned = True
                    cv2.resizeWindow(_WINDOW, max(480, width), max(360, height))
                    cv2.moveWindow(_WINDOW, 80, 80)

                if show_head_ref and head_ref_y is None and last_joints and not head_ref_locked:
                    jmap = {int(j["id"]): j for j in last_joints if "id" in j}
                    nose = jmap.get(0)
                    if nose is not None:
                        head_ref_y = int(round(
                            max(0.0, min(1.0, float(nose["y"]))) * (height - 1)
                        ))
                        head_ref_locked = True

                rendered = self.overlay.draw_overlay_frame(
                    frame,
                    last_joints,
                    smooth_arm_display=True,
                    show_shoulders=show_shoulders,
                    show_waistline=show_waistline,
                    show_legs=show_legs,
                    show_spine=show_spine,
                    head_reference_y=head_ref_y if show_head_ref else None,
                    static_lines=list(coach_state.get("static_lines") or []),
                )

                ds = coach_state.get("drag_start")
                pe = coach_state.get("preview_end")
                if ds is not None and pe is not None:
                    cv2.line(rendered, ds, pe, COACH_STATIC_LINE_COLOR, 2, cv2.LINE_AA)

                state_txt = "PAUSED" if paused else "PLAY"
                cv2.putText(
                    rendered,
                    f"{state_txt}  frame {frame_idx + 1}/{total}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255) if paused else (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                toggle_parts: List[str] = []
                if show_shoulders:
                    toggle_parts.append("S")
                if show_waistline:
                    toggle_parts.append("W")
                if show_legs:
                    toggle_parts.append("L")
                if show_spine:
                    toggle_parts.append("D")
                if show_head_ref:
                    toggle_parts.append("H")
                toggle_txt = "Sections: " + ("+".join(toggle_parts) if toggle_parts else "none")
                cv2.putText(
                    rendered,
                    toggle_txt,
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                n_coach = len(coach_state.get("static_lines") or [])
                cv2.putText(
                    rendered,
                    f"Coach: {n_coach}  | drag=LMB  x=clr u=undo  o=JSON  SPACE=play/pause  a/f=step",
                    (12, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )
                if not show_window and mjpeg_listen_port is not None:
                    cv2.putText(
                        rendered,
                        f"Panel: http://127.0.0.1:{mjpeg_listen_port}/  |  /stop  Ctrl+C",
                        (12, height - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (180, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if jpeg_feed is not None:
                    jpeg_feed.publish(rendered)

                if show_window:
                    cv2.imshow(_WINDOW, rendered)
                    if not mouse_cb_registered:
                        cv2.setMouseCallback(_WINDOW, _coach_mouse_callback, coach_state)
                        mouse_cb_registered = True

                if callback is not None:
                    callback(last_joints)

                wait = 80 if paused else delay_ms
                raw_key = (cv2.waitKey(wait) & 0xFF) if show_window else 0
                if not show_window:
                    time.sleep(wait / 1000.0)
                key_codes: List[int] = []
                if raw_key:
                    key_codes.append(raw_key)
                if jpeg_feed is not None:
                    for t in jpeg_feed.drain_browser_keys():
                        ko = _browser_token_to_key_ord(t)
                        if ko:
                            key_codes.append(ko)
                if not key_codes:
                    key_codes = [0]

                leave_video = False
                for key in key_codes:
                    if key == 0:
                        continue
                    if key in (ord("q"), ord("Q")):
                        leave_video = True
                        break
                    if key == ord(" "):
                        paused = not paused
                    elif key in (ord("a"), ord("A")):
                        paused = True
                        frame_idx = max(0, frame_idx - 1)
                    elif key in (ord("f"), ord("F")):
                        paused = True
                        frame_idx = min(total - 1, frame_idx + 1)
                    elif key in (ord("r"), ord("R")):
                        frame_idx = 0
                        paused = True
                        head_ref_y = None
                        head_ref_locked = False
                        self.overlay.reset_arm_display_smooth()
                    elif key in (ord("o"), ord("O")):
                        _export_coach_json(width, height)
                    elif key == ord("s"):
                        show_shoulders = not show_shoulders
                    elif key == ord("w"):
                        show_waistline = not show_waistline
                    elif key == ord("l"):
                        show_legs = not show_legs
                    elif key == ord("d"):
                        show_spine = not show_spine
                    elif key == ord("h"):
                        show_head_ref = not show_head_ref
                        if not show_head_ref:
                            head_ref_y = None
                            head_ref_locked = False
                    elif key == ord("x"):
                        coach_state["static_lines"] = []
                        coach_state["drag_start"] = None
                        coach_state["preview_end"] = None
                    elif key == ord("u"):
                        lines = coach_state.get("static_lines") or []
                        if lines:
                            lines.pop()
                            coach_state["static_lines"] = lines
                if leave_video:
                    break

                if jpeg_feed is not None and jpeg_feed.stop.is_set():
                    print("Stopped (browser /stop).", flush=True)
                    break

                if not paused:
                    if frame_idx < total - 1:
                        frame_idx += 1
                    else:
                        paused = True

        except KeyboardInterrupt:
            print("\nStopped (KeyboardInterrupt).", flush=True)
        finally:
            if http_server is not None:
                try:
                    http_server.shutdown()
                except (OSError, RuntimeError):
                    pass
            if show_window:
                try:
                    cv2.setMouseCallback(_WINDOW, lambda *a, **k: None)
                except (cv2.error, AttributeError):
                    pass
                try:
                    cv2.destroyAllWindows()
                except (cv2.error, AttributeError):
                    pass
            cap.release()
            self._close_pose_detector()

    def get_single_frame_joints(self) -> dict:
        """Capture one webcam frame, detect joints, release webcam, and return result."""
        self._ensure_pose_detector()
        assert self.pose_detector is not None
        cap = _open_video_capture(0)
        if not cap.isOpened():
            raise ValueError("Unable to open webcam (index 0).")

        try:
            ok, frame = cap.read()
            if not ok:
                return {"joints": []}
            pose = self.pose_detector.detect_from_frame(frame)
            return pose if pose is not None else {"joints": []}
        finally:
            cap.release()
            self._close_pose_detector()


def main() -> None:
    """Run live webcam or recorded-video review (``python -m ml_cv_engine.realtime``)."""
    import argparse

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    p = argparse.ArgumentParser(
        description="Live webcam or MP4 review with pose overlay + coach static lines.",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        metavar="N",
        help="Webcam index (default 0). Try 1 if the wrong camera opens.",
    )
    p.add_argument(
        "--video",
        nargs="?",
        const="__PICK__",
        default=None,
        metavar="PATH",
        help="Review a file instead of the webcam: optional path, or omit value to pick a file.",
    )
    p.add_argument(
        "--http",
        type=int,
        nargs="?",
        const=8765,
        default=None,
        metavar="PORT",
        help="Serve MJPEG at http://127.0.0.1:PORT/ (default 8765 if flag is given alone).",
    )
    p.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open an OpenCV window (use with --http or --browser).",
    )
    p.add_argument(
        "--browser",
        action="store_true",
        help="View in browser only: same as --http 8765 --no-window.",
    )
    args = p.parse_args()

    http_port: Optional[int] = None
    show_window = True
    if args.browser:
        http_port = 8765
        show_window = False
    elif args.http is not None:
        http_port = int(args.http)
    if args.no_window:
        show_window = False
    if not show_window and http_port is None:
        p.error("--no-window requires --http or --browser")

    if http_port is not None:
        print(
            "After 'Open in your browser' appears, that link is live immediately; "
            "you may see a gray 'Loading MediaPipe' frame for up to ~40s before video.",
            flush=True,
        )
        print(
            "Keep this process running. ERR_CONNECTION_REFUSED means the script is not "
            "running, wrong port, or you used https:// instead of http://.",
            flush=True,
        )

    detector = RealTimeDetector()

    if args.video is not None:
        vpath: str = args.video
        if vpath == "__PICK__":
            picked = _pick_video_path()
            if not picked:
                sys.exit(2)
            vpath = picked
        if not os.path.isfile(vpath):
            p.error(f"Not a file: {vpath}")
        detector.start_video_review_session(
            vpath,
            http_port=http_port,
            show_window=show_window,
        )
    else:
        editor_clip = detector.start_webcam_session(
            device_index=args.device,
            http_port=http_port,
            show_window=show_window,
        )
        if editor_clip:
            if http_port is not None:
                print(
                    "\n>>> VIDEO EDITOR: starting clip. If the browser stream went blank, "
                    "wait a few seconds then refresh: http://127.0.0.1:"
                    f"{int(http_port)}/\n",
                    flush=True,
                )
            else:
                print("\n>>> VIDEO EDITOR: starting clip.\n", flush=True)
            detector.start_video_review_session(
                editor_clip,
                http_port=http_port,
                show_window=show_window,
            )


if __name__ == "__main__":
    main()
