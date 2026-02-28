"""
ui_hub.py  ·  AIILA OS — Neural Interface Terminal
====================================================
Redesigned: Military-grade HUD aesthetic.
Optimised:  Single worker process, frame-drop policy,
            direct numpy→QPixmap path, no intermediate copies.
"""

import sys
import cv2
import numpy as np
import multiprocessing as mp
import time
import math
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QFrame, QSizePolicy,
    QGraphicsDropShadowEffect,
)
from PyQt6.QtGui  import QImage, QPixmap, QFont, QColor, QPainter, QPen, QLinearGradient, QBrush
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect, QSize, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve

from interface.ar_overlay    import assemble_final_os_view
from interface.settings_panel import SettingsPanel


# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS  (one place to change the whole palette)
# ─────────────────────────────────────────────────────────────────────────────
C = {
    'bg':          '#020408',
    'surface':     '#060d14',
    'panel':       '#0a1520',
    'border':      '#0e2030',
    'border_hi':   '#1a3f5c',
    'accent':      '#00c8ff',
    'accent2':     '#ff4d1a',
    'accent3':     '#00ff88',
    'text':        '#8ab0c8',
    'text_bright': '#d4eaf8',
    'text_dim':    '#2a4a60',
    'danger':      '#ff2244',
    'warn':        '#ffaa00',
}

MONO  = "Courier New"
TITLE = "Courier New"

_SS_BASE = f"""
    QWidget {{
        background: {C['bg']};
        color: {C['text_bright']};
        font-family: '{MONO}';
        font-size: 12px;
    }}
    QScrollBar:vertical {{
        background: {C['surface']};
        width: 4px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {C['border_hi']};
        border-radius: 2px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ndarray_to_pixmap(rgb: np.ndarray) -> QPixmap:
    """Zero-copy numpy RGB → QPixmap."""
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


def _glow(widget: QWidget, color: str = C['accent'], radius: int = 18):
    fx = QGraphicsDropShadowEffect(widget)
    fx.setBlurRadius(radius)
    fx.setOffset(0, 0)
    fx.setColor(QColor(color))
    widget.setGraphicsEffect(fx)


# ─────────────────────────────────────────────────────────────────────────────
#  WORKER PROCESS  (separate CPU core for heavy image ops)
# ─────────────────────────────────────────────────────────────────────────────
def _image_worker(in_q: mp.Queue, out_q: mp.Queue):
    while True:
        try:
            ar_canvas, raw_frame, state, proj_active, calib = in_q.get()

            # ── Projector output ──────────────────────────────────────────────
            p_rgb = None
            if proj_active:
                p_rgb = cv2.cvtColor(ar_canvas, cv2.COLOR_BGR2RGB)
                if calib:
                    h, w = p_rgb.shape[:2]
                    for x in range(0, w, 100):
                        cv2.line(p_rgb, (x, 0), (x, h), (0, 255, 0), 1)
                    for y in range(0, h, 100):
                        cv2.line(p_rgb, (0, y), (w, y), (0, 255, 0), 1)

            # ── Main viewport ─────────────────────────────────────────────────
            combined = assemble_final_os_view(ar_canvas, raw_frame)
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            ar_rgb = cv2.resize(combined_rgb, (1100, 620), interpolation=cv2.INTER_LINEAR)

            # ── Camera thumbnail ──────────────────────────────────────────────
            cam_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            cam_rgb = cv2.resize(cam_rgb, (280, 190), interpolation=cv2.INTER_LINEAR)

            # Discard stale output to avoid queue build-up
            while not out_q.empty():
                try:
                    out_q.get_nowait()
                except Exception:
                    pass
            out_q.put((p_rgb, ar_rgb, cam_rgb, state))
        except Exception:
            continue


# ─────────────────────────────────────────────────────────────────────────────
#  REUSABLE WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
class HexButton(QPushButton):
    """Flat button with animated accent-border on hover."""

    _NORMAL = f"""
        QPushButton {{
            background: {C['panel']};
            color: {C['text']};
            border: 1px solid {C['border']};
            border-left: 3px solid {C['border_hi']};
            padding: 0 14px;
            height: 38px;
            font-family: '{MONO}';
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 1px;
            text-align: left;
        }}
        QPushButton:hover {{
            background: #0d2035;
            color: {C['accent']};
            border-left: 3px solid {C['accent']};
            border-top: 1px solid {C['border_hi']};
            border-right: 1px solid {C['border_hi']};
            border-bottom: 1px solid {C['border_hi']};
        }}
        QPushButton:pressed {{
            background: #071520;
            color: {C['accent2']};
            border-left: 3px solid {C['accent2']};
        }}
    """

    _ACTIVE = f"""
        QPushButton {{
            background: #0d2a38;
            color: {C['accent3']};
            border: 1px solid {C['accent3']};
            border-left: 3px solid {C['accent3']};
            padding: 0 14px;
            height: 38px;
            font-family: '{MONO}';
            font-size: 11px;
            font-weight: bold;
            letter-spacing: 1px;
            text-align: left;
        }}
    """

    def __init__(self, text: str, prefix: str = "▸", parent=None):
        super().__init__(f"  {prefix}  {text}", parent)
        self.setStyleSheet(self._NORMAL)
        self._active = False

    def set_active(self, on: bool):
        self._active = on
        self.setStyleSheet(self._ACTIVE if on else self._NORMAL)


class SectionLabel(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.setStyleSheet(f"""
            color: {C['text_dim']};
            font-family: '{MONO}';
            font-size: 9px;
            letter-spacing: 3px;
            padding: 10px 14px 4px 14px;
            border-bottom: 1px solid {C['border']};
            margin-bottom: 4px;
        """)


class Divider(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setStyleSheet(f"color: {C['border']}; margin: 4px 0;")


class StatusBar(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tick = 0
        self._t0 = time.time()
        self.setStyleSheet(f"""
            background: {C['surface']};
            color: {C['text_dim']};
            font-family: '{MONO}';
            font-size: 10px;
            padding: 3px 14px;
            border-top: 1px solid {C['border']};
            letter-spacing: 1px;
        """)
        self.setFixedHeight(22)

    def tick(self, fps: float, gesture: str):
        elapsed = time.time() - self._t0
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        g = gesture or "IDLE"
        self.setText(
            f"  UPTIME {h:02d}:{m:02d}:{s:02d}   ▪   FPS {fps:.0f}   ▪   "
            f"GESTURE {g.upper():<12}   ▪   AIILA OS v2.4"
        )


class TerminalLog(QTextEdit):
    MAX_LINES = 150

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet(f"""
            QTextEdit {{
                background: {C['bg']};
                color: {C['accent3']};
                font-family: '{MONO}';
                font-size: 11px;
                border: 1px solid {C['border']};
                border-top: 2px solid {C['border_hi']};
                padding: 6px;
                line-height: 160%;
            }}
        """)
        self._line_count = 0

    def push(self, msg: str, level: str = "INFO"):
        ts = time.strftime("%H:%M:%S")
        colours = {"INFO": C['accent3'], "WARN": C['warn'],
                   "ERR":  C['danger'],  "SYS":  C['accent']}
        col = colours.get(level, C['text'])
        self.append(
            f'<span style="color:{C["text_dim"]}">[{ts}]</span> '
            f'<span style="color:{col}"><b>{level}</b></span> '
            f'<span style="color:{C["text_bright"]}">{msg}</span>'
        )
        self._line_count += 1
        if self._line_count > self.MAX_LINES:
            cursor = self.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.select(cursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()
            self._line_count -= 1
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class CamPreview(QLabel):
    """Camera thumbnail with corner-bracket overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 190)
        self.setStyleSheet(f"background: {C['bg']}; border-radius: 4px;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap: QPixmap | None = None

    def set_frame(self, px: QPixmap):
        self._pixmap = px
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._pixmap:
            self.setPixmap(
                self._pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.FastTransformation,
                )
            )
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(C['accent']), 2)
        p.setPen(pen)
        L, T, R, B, S = 6, 6, self.width()-6, self.height()-6, 18
        for pts in [
            [(L,L+S),(L,L),(L+S,L)],
            [(R-S,T),(R,T),(R,T+S)],
            [(L,B-S),(L,B),(L+S,B)],
            [(R-S,B),(R,B),(R,B-S)],
        ]:
            for i in range(len(pts)-1):
                p.drawLine(*pts[i], *pts[i+1])
        p.end()


class MetricTile(QFrame):
    """Small stat tile: label + value."""

    def __init__(self, label: str, value: str = "—", accent: str = C['accent'], parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C['surface']};
                border: 1px solid {C['border']};
                border-top: 2px solid {accent};
                border-radius: 2px;
                padding: 6px 8px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)

        self._lbl = QLabel(label)
        self._lbl.setStyleSheet(f"color:{C['text_dim']};font-size:9px;letter-spacing:2px;border:none;background:transparent;")
        self._val = QLabel(value)
        self._val.setStyleSheet(f"color:{accent};font-size:14px;font-weight:bold;border:none;background:transparent;")
        lay.addWidget(self._lbl)
        lay.addWidget(self._val)

    def set_value(self, v: str):
        self._val.setText(v)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
class Sidebar(QFrame):
    def __init__(self, kernel, parent=None):
        super().__init__(parent)
        self.kernel = kernel
        self.setFixedWidth(300)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C['surface']};
                border-right: 1px solid {C['border']};
            }}
        """)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Logo ──────────────────────────────────────────────────────────────
        logo_frame = QFrame()
        logo_frame.setStyleSheet(f"""
            background: {C['bg']};
            border-bottom: 1px solid {C['border_hi']};
            padding: 0;
        """)
        logo_lay = QVBoxLayout(logo_frame)
        logo_lay.setContentsMargins(16, 18, 16, 14)
        logo_lay.setSpacing(2)

        logo = QLabel("AIILA")
        logo.setFont(QFont(TITLE, 28, QFont.Weight.Bold))
        logo.setStyleSheet(f"color: {C['accent']}; letter-spacing: 8px; border:none;background:transparent;")
        _glow(logo, C['accent'], 24)

        sub = QLabel("NEURAL INTERFACE TERMINAL  v2.4")
        sub.setStyleSheet(f"color:{C['text_dim']};font-size:8px;letter-spacing:3px;border:none;background:transparent;")

        logo_lay.addWidget(logo)
        logo_lay.addWidget(sub)
        lay.addWidget(logo_frame)

        # ── Metrics row ───────────────────────────────────────────────────────
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("background:transparent;border:none;")
        m_lay = QHBoxLayout(metrics_frame)
        m_lay.setContentsMargins(10, 10, 10, 6)
        m_lay.setSpacing(6)

        self.tile_fps     = MetricTile("FPS",     "—",   C['accent'])
        self.tile_gesture = MetricTile("GESTURE", "IDLE", C['accent3'])
        self.tile_mode    = MetricTile("AR MODE", "DEF",  C['warn'])
        for t in (self.tile_fps, self.tile_gesture, self.tile_mode):
            m_lay.addWidget(t)
        lay.addWidget(metrics_frame)

        # ── Camera preview ────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("◈  CAMERA FEED"))
        cam_wrap = QFrame()
        cam_wrap.setStyleSheet("background:transparent;border:none;")
        cw_lay = QHBoxLayout(cam_wrap)
        cw_lay.setContentsMargins(10, 0, 10, 6)
        self.cam_preview = CamPreview()
        cw_lay.addWidget(self.cam_preview, alignment=Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(cam_wrap)

        # ── Actions ───────────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("◈  ACTIONS"))
        btn_frame = QFrame()
        btn_frame.setStyleSheet("background:transparent;border:none;")
        btn_lay = QVBoxLayout(btn_frame)
        btn_lay.setContentsMargins(10, 4, 10, 4)
        btn_lay.setSpacing(3)

        self.btn_scan    = HexButton("SCAN UNIT",           "⊕")
        self.btn_voice   = HexButton("ACTIVATE JARVIS",     "◉")
        self.btn_circuit = HexButton("CIRCUIT MODE",        "⎔", )
        self.btn_sim     = HexButton("RUN SIMULATION",      "▶")
        self.btn_project = HexButton("PROJECT TO SCREEN",   "⊞")
        self.btn_calib   = HexButton("CALIBRATION GRID",    "⊟")
        self.btn_save    = HexButton("SAVE CIRCUIT",        "◧")
        self.btn_settings= HexButton("HARDWARE SETTINGS",   "⚙")

        for b in (self.btn_scan, self.btn_voice, self.btn_circuit,
                  self.btn_sim, self.btn_project, self.btn_calib,
                  self.btn_save, self.btn_settings):
            btn_lay.addWidget(b)

        lay.addWidget(btn_frame)

        # ── Terminal ──────────────────────────────────────────────────────────
        lay.addWidget(SectionLabel("◈  KERNEL LOG"))
        self.terminal = TerminalLog()
        lay.addWidget(self.terminal, 1)

    def update_metrics(self, fps: float, state: dict):
        self.tile_fps.set_value(f"{fps:.0f}")
        g = state.get('gesture_active') or 'IDLE'
        self.tile_gesture.set_value(g[:8].upper())
        m = state.get('ar_mode', 'default')
        self.tile_mode.set_value(m[:5].upper())


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN VIEWPORT
# ─────────────────────────────────────────────────────────────────────────────
class Viewport(QLabel):
    """The big AR canvas with corner-bracket overlay + scan-line pulse."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background: {C['bg']}; border: none;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._scan_y  = 0
        self._sim_on  = False
        self._t0      = time.time()

    def set_sim(self, on: bool):
        self._sim_on = on

    def set_frame(self, px: QPixmap):
        self.setPixmap(
            px.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        )
        self._scan_y = (self._scan_y + 3) % max(self.height(), 1)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        S = 22

        # Corner brackets
        col = QColor(C['accent'])
        col.setAlpha(200)
        pen = QPen(col, 2)
        p.setPen(pen)
        for pts in [
            [(S,2),(2,2),(2,S)],
            [(W-S,2),(W-2,2),(W-2,S)],
            [(2,H-S),(2,H-2),(S,H-2)],
            [(W-S,H-2),(W-2,H-2),(W-2,H-S)],
        ]:
            for i in range(len(pts)-1):
                p.drawLine(*pts[i], *pts[i+1])

        # Scan-line (only while sim running)
        if self._sim_on:
            scan_col = QColor(C['accent'])
            scan_col.setAlpha(30)
            p.setPen(QPen(scan_col, 1))
            p.drawLine(0, self._scan_y, W, self._scan_y)

        # Sim indicator badge
        if self._sim_on:
            elapsed = time.time() - self._t0
            pulse = abs(math.sin(elapsed * 3))
            bc = QColor(C['accent3'])
            bc.setAlpha(int(180 + 75 * pulse))
            p.setPen(QPen(bc, 1))
            p.setBrush(QBrush(bc))
            p.drawEllipse(W - 18, 8, 8, 8)
            p.setPen(QPen(QColor(C['text_bright']), 1))
            p.setFont(QFont(MONO, 8))
            p.drawText(W - 80, 18, "SIM LIVE")

        p.end()


# ─────────────────────────────────────────────────────────────────────────────
#  PROJECTOR WINDOW  — Full holographic OS desktop
# ─────────────────────────────────────────────────────────────────────────────
class ProjectorWindow(QWidget):
    """
    Full-screen secondary display rendered entirely in OpenCV onto a QLabel.

    Layout (1920×1080):
    ┌──────────────────────────────────────────────────────────────────────┐
    │  TOPBAR  — logo · clock · sys-stats                                  │
    ├────────────────────────────┬─────────────────────────────────────────┤
    │  AR FEED  (live camera)    │  RIGHT PANEL                            │
    │  960×600                   │    ├─ SYSTEM METRICS  (4 gauges)        │
    │                            │    ├─ KERNEL LOG  (scrolling)           │
    │                            │    └─ APP DOCK  (icon grid)             │
    ├────────────────────────────┴─────────────────────────────────────────┤
    │  TASKBAR — uptime · gesture · active module · status dots            │
    └──────────────────────────────────────────────────────────────────────┘
    """

    W, H = 1920, 1080

    # App dock items: (label, icon_char, accent_bgr)
    APPS = [
        ("Circuit Lab",  "⎔", (255, 200,   0)),
        ("Vision AI",   "◎", (  0, 200, 255)),
        ("Voice CMD",   "◉", (  0, 255, 140)),
        ("Settings",    "⚙", (180, 180, 180)),
        ("Diagnostics", "▦", (255, 120,   0)),
        ("Deploy",      "▶", (100, 255, 100)),
        ("Analytics",   "▪", (200,   0, 255)),
        ("Network",     "⊞", (  0, 160, 255)),
    ]

    def __init__(self, x: int, y: int):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setGeometry(x, y, self.W, self.H)
        self.setStyleSheet("background: black;")

        self._canvas = QLabel(self)
        self._canvas.setGeometry(0, 0, self.W, self.H)

        # State
        self._t0          = time.time()
        self._phase       = "BOOT"      # BOOT → INTRO → DESKTOP
        self._fade        = 0.0
        self._hovered_app = -1          # index of hovered dock item
        self._active_app  = -1
        self._log_lines   = [
            "AIILA KERNEL v2.4 — BOOT SEQUENCE INITIATED",
            "Handlandmark model loaded  [OK]",
            "MediaPipe pipeline active  [OK]",
            "CircuitEngine initialised  [OK]",
            "Voice engine ready         [OK]",
            "AR canvas 1000×700 mapped  [OK]",
            "Projector output detected  [OK]",
            "All subsystems nominal     [READY]",
        ]
        self._log_scroll  = 0
        self._noise_seed  = 0

        # Pre-compute dock rects for hit-testing
        self._dock_rects: list[tuple[int,int,int,int]] = []

        # Rolling metric history (fake but animated)
        self._metric_hist = {k: [0.0]*80 for k in ("CPU","GPU","MEM","NET")}
        self._metric_tick = 0

        # Circuit engine reference (set each frame from UIHub)
        self._circuit_engine = None
        self._circuit_active = False

    # ── rendering entry point ────────────────────────────────────────────────
    def update_display(self, ar_rgb: np.ndarray | None,
                       circuit_engine=None, circuit_active: bool = False):
        elapsed = time.time() - self._t0

        # Phase transitions
        if self._phase == "BOOT" and elapsed > 2.2:
            self._phase = "INTRO"
        if self._phase == "INTRO" and elapsed > 4.5:
            self._phase = "DESKTOP"

        self._circuit_engine = circuit_engine
        self._circuit_active = circuit_active

        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        self._draw_background(frame, elapsed)

        if self._phase == "BOOT":
            self._draw_boot(frame, elapsed)
        elif self._phase == "INTRO":
            self._draw_intro(frame, elapsed)
        else:
            d = min(1.0, (elapsed - 4.5) / 1.0)
            self._draw_desktop(frame, elapsed, ar_rgb, d,
                               self._circuit_engine, self._circuit_active)

        # Always draw topbar + taskbar once past boot
        if self._phase != "BOOT":
            self._draw_topbar(frame, elapsed)
            self._draw_taskbar(frame, elapsed)

        qi = QImage(frame.data, self.W, self.H, self.W * 3, QImage.Format.Format_BGR888)
        self._canvas.setPixmap(QPixmap.fromImage(qi.copy()))

    # ── BACKGROUND ──────────────────────────────────────────────────────────
    def _draw_background(self, f: np.ndarray, t: float):
        """Animated deep-space grid with slow parallax drift."""
        drift_x = int(math.sin(t * 0.08) * 30)
        drift_y = int(math.cos(t * 0.05) * 20)
        gc = int(14 * self._fade)
        ac = int(8  * self._fade)

        # Major grid 120px
        for x in range((drift_x % 120) - 120, self.W + 120, 120):
            cv2.line(f, (x, 0), (x, self.H), (0, gc, gc), 1)
        for y in range((drift_y % 120) - 120, self.H + 120, 120):
            cv2.line(f, (0, y), (self.W, y), (0, gc, gc), 1)

        # Minor grid 40px
        for x in range((drift_x % 40) - 40, self.W + 40, 40):
            cv2.line(f, (x, 0), (x, self.H), (0, ac, ac), 1)
        for y in range((drift_y % 40) - 40, self.H + 40, 40):
            cv2.line(f, (0, y), (self.W, y), (0, ac, ac), 1)

        # Vignette — darken edges
        cx, cy = self.W // 2, self.H // 2
        for r, alpha in [(900, 0.18), (700, 0.10), (500, 0.05)]:
            overlay = f.copy()
            cv2.ellipse(overlay, (cx, cy), (r, int(r*0.56)), 0, 0, 360,
                        (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, f, 1 - alpha, 0, f)

    # ── BOOT SEQUENCE ────────────────────────────────────────────────────────
    def _draw_boot(self, f: np.ndarray, t: float):
        lines = [
            (0.0, "AIILA NEURAL INTERFACE TERMINAL"),
            (0.3, "Initialising hardware abstraction layer..."),
            (0.6, "Loading hand landmarker model..."),
            (0.9, "Calibrating gesture pipeline..."),
            (1.2, "Mounting AR canvas..."),
            (1.5, "Projector handshake..."),
            (1.8, "All systems nominal."),
            (2.0, "BOOT COMPLETE"),
        ]
        cy = 460
        for delay, text in lines:
            if t > delay:
                a = min(1.0, (t - delay) / 0.25)
                col_g = int(255 * a) if text == "BOOT COMPLETE" else int(160 * a)
                col_b = int(255 * a)
                col   = (0, col_g, col_b) if text != "BOOT COMPLETE" else (0, 255, 80)
                th    = 2 if text == "BOOT COMPLETE" else 1
                fs    = 0.8 if text == "BOOT COMPLETE" else 0.55
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                cv2.putText(f, text, ((self.W - tw) // 2, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
            cy += 36

    # ── INTRO SPLASH ─────────────────────────────────────────────────────────
    def _draw_intro(self, f: np.ndarray, t: float):
        a = min(1.0, (t - 2.2) / 0.8)
        c = int(255 * a)

        # Big logo
        logo = "AIILA"
        (lw, lh), _ = cv2.getTextSize(logo, cv2.FONT_HERSHEY_DUPLEX, 9, 14)
        cv2.putText(f, logo, ((self.W - lw) // 2, 560),
                    cv2.FONT_HERSHEY_DUPLEX, 9, (0, int(200*a), c), 14, cv2.LINE_AA)
        cv2.putText(f, logo, ((self.W - lw) // 2, 560),
                    cv2.FONT_HERSHEY_DUPLEX, 9, (0, int(220*a), c), 2, cv2.LINE_AA)

        # Tagline
        tag = "NEURAL INTERFACE TERMINAL  v2.4"
        (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)
        cv2.putText(f, tag, ((self.W - tw) // 2, 620),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, int(130*a), int(160*a)), 1, cv2.LINE_AA)

        # Horizontal rule
        rx = (self.W - 500) // 2
        cv2.line(f, (rx, 640), (rx + 500, 640), (0, int(80*a), int(120*a)), 1)

    # ── TOP BAR ──────────────────────────────────────────────────────────────
    def _draw_topbar(self, f: np.ndarray, t: float):
        a   = min(1.0, (t - 2.2) / 0.6)
        c   = int(255 * a)
        dim = int(80  * a)

        # Bar background
        cv2.rectangle(f, (0, 0), (self.W, 48), (0, int(12*a), int(16*a)), -1)
        cv2.line(f, (0, 48), (self.W, 48), (0, int(50*a), int(70*a)), 1)

        # Logo left
        cv2.putText(f, "AIILA", (16, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, int(200*a), c), 2, cv2.LINE_AA)
        cv2.putText(f, "OS", (102, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, int(160*a), int(200*a)), 1, cv2.LINE_AA)

        # Separator
        cv2.line(f, (155, 10), (155, 38), (0, dim, dim), 1)

        # Module name
        cv2.putText(f, "NEURAL INTERFACE TERMINAL", (168, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, dim, dim), 1, cv2.LINE_AA)

        # Clock right
        ts = time.strftime("%H:%M:%S")
        ds = time.strftime("%Y-%m-%d")
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.putText(f, ts, (self.W - tw - 16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, int(200*a), c), 1, cv2.LINE_AA)
        (dw, _), _ = cv2.getTextSize(ds, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(f, ds, (self.W - dw - 16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, dim, dim), 1, cv2.LINE_AA)

        # Status dots
        for i, (lbl, col) in enumerate([
            ("KERNEL", (0, 255, 120)),
            ("AR",     (0, 200, 255)),
            ("VOICE",  (0, 255, 200)),
            ("CIRCUIT",(255, 200, 0)),
        ]):
            bx = self.W - 350 + i * 80
            pulse = abs(math.sin(t * 2 + i)) * 0.4 + 0.6
            dc = tuple(int(c2 * pulse * a) for c2 in col)
            cv2.circle(f, (bx, 24), 5, dc, -1, cv2.LINE_AA)
            cv2.putText(f, lbl, (bx + 8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, dc, 1, cv2.LINE_AA)

    # ── TASKBAR ──────────────────────────────────────────────────────────────
    def _draw_taskbar(self, f: np.ndarray, t: float):
        a   = min(1.0, (t - 2.2) / 0.6)
        dim = int(80 * a)
        c   = int(255 * a)
        TY  = self.H - 40

        cv2.rectangle(f, (0, TY), (self.W, self.H), (0, int(10*a), int(14*a)), -1)
        cv2.line(f, (0, TY), (self.W, TY), (0, int(45*a), int(60*a)), 1)

        uptime = t
        h_ = int(uptime // 3600)
        m_ = int((uptime % 3600) // 60)
        s_ = int(uptime % 60)
        items = [
            f"UPTIME  {h_:02d}:{m_:02d}:{s_:02d}",
            "GESTURE  READY",
            "AR MODE  DEFAULT",
            "KERNEL  NOMINAL",
            "v2.4.0-RELEASE",
        ]
        spacing = self.W // len(items)
        for i, txt in enumerate(items):
            x = spacing * i + 20
            cv2.putText(f, txt, (x, TY + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, dim, int(c * 0.7)), 1, cv2.LINE_AA)
            if i > 0:
                cv2.line(f, (spacing * i, TY + 6), (spacing * i, self.H - 6),
                         (0, int(30*a), int(40*a)), 1)

    # ── FULL DESKTOP ─────────────────────────────────────────────────────────
    def _draw_desktop(self, f: np.ndarray, t: float,
                      ar_rgb: np.ndarray | None, d: float,
                      circuit_engine=None, circuit_active: bool = False):
        c   = int(255 * d)
        dim = int(120 * d)

        # ── AR FEED panel (left) ─────────────────────────────────────────────
        PAD, TBH, BTH = 12, 48, 40
        feed_x1, feed_y1 = PAD,       TBH + PAD
        feed_x2, feed_y2 = 980,       TBH + PAD + 600

        # Panel border + corner brackets
        cv2.rectangle(f, (feed_x1, feed_y1), (feed_x2, feed_y2),
                      (0, int(40*d), int(60*d)), 1)
        self._corner_brackets(f, feed_x1, feed_y1, feed_x2, feed_y2,
                               (0, int(200*d), c), 20, 2)

        # AR content or placeholder
        if ar_rgb is not None:
            try:
                roi = cv2.resize(ar_rgb, (feed_x2 - feed_x1 - 2, feed_y2 - feed_y1 - 2))
                f[feed_y1+1:feed_y2, feed_x1+1:feed_x2] = roi
            except Exception:
                pass
        else:
            cx_, cy_ = (feed_x1 + feed_x2) // 2, (feed_y1 + feed_y2) // 2
            r = int(60 + 10 * math.sin(t * 2))
            cv2.circle(f, (cx_, cy_), r, (0, dim//2, dim), 1, cv2.LINE_AA)
            cv2.line(f, (cx_ - r - 20, cy_), (cx_ + r + 20, cy_), (0, dim//2, dim), 1)
            cv2.line(f, (cx_, cy_ - r - 20), (cx_, cy_ + r + 20), (0, dim//2, dim), 1)
            cv2.putText(f, "AWAITING AR FEED", (cx_ - 100, cy_ + r + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, dim, dim), 1, cv2.LINE_AA)

        cv2.putText(f, "AR LIVE FEED", (feed_x1 + 4, feed_y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, dim, dim), 1, cv2.LINE_AA)

        sy = feed_y1 + int((t * 80) % (feed_y2 - feed_y1))
        scan_ov = f.copy()
        cv2.line(scan_ov, (feed_x1, sy), (feed_x2, sy), (0, 200, 255), 1)
        cv2.addWeighted(scan_ov, 0.25, f, 0.75, 0, f)

        # ── RIGHT PANEL ──────────────────────────────────────────────────────
        RX = 996
        RW = self.W - RX - PAD

        if circuit_active and circuit_engine is not None:
            # ─ CIRCUIT BOARD (top of right panel) ────────────────────────────
            self._draw_circuit_panel(f, t, d, RX, TBH + PAD, RW, circuit_engine)
        else:
            # ─ METRICS + LOG + DOCK ───────────────────────────────────────────
            self._draw_metrics(f, t, d, RX, TBH + PAD, RW)
            log_y1 = TBH + PAD + 260
            log_y2 = TBH + PAD + 530
            self._draw_kernel_log(f, d, RX, log_y1, RW, log_y2)
            dock_y = TBH + PAD + 545
            self._draw_app_dock(f, t, d, RX, dock_y, RW)

    # ── METRICS PANEL ────────────────────────────────────────────────────────
    def _draw_metrics(self, f: np.ndarray, t: float, d: float,
                      rx: int, ry: int, rw: int):
        c   = int(255 * d)
        dim = int(100 * d)

        # Panel border
        ph = 250
        cv2.rectangle(f, (rx, ry), (rx + rw, ry + ph), (0, int(30*d), int(45*d)), 1)
        self._corner_brackets(f, rx, ry, rx + rw, ry + ph, (0, int(180*d), c), 14, 1)
        cv2.putText(f, "SYSTEM METRICS", (rx + 6, ry - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, dim, dim), 1, cv2.LINE_AA)

        # Animate metric histories
        keys   = ["CPU", "GPU", "MEM", "NET"]
        colors = [(0, 200, 255), (0, 255, 140), (255, 180, 0), (200, 80, 255)]
        bases  = [45.0, 62.0, 71.0, 30.0]
        amps   = [25.0, 18.0, 8.0, 35.0]
        freqs  = [0.7,  0.5,  0.2,  1.1]

        for ki, key in enumerate(keys):
            val = bases[ki] + amps[ki] * (0.5 + 0.5 * math.sin(t * freqs[ki] + ki))
            self._metric_hist[key].append(val)
            self._metric_hist[key] = self._metric_hist[key][-80:]

        # Draw 4 graphs in 2×2 grid
        gw, gh = rw // 2 - 14, 100
        for i, (key, col) in enumerate(zip(keys, colors)):
            gx = rx + 8  + (i % 2) * (gw + 12)
            gy = ry + 14 + (i // 2) * (gh + 18)
            hist = self._metric_hist[key]
            val  = hist[-1]

            # Background
            cv2.rectangle(f, (gx, gy), (gx + gw, gy + gh),
                          (0, int(8*d), int(12*d)), -1)
            cv2.rectangle(f, (gx, gy), (gx + gw, gy + gh),
                          tuple(int(c2 * 0.25 * d) for c2 in col), 1)

            # Waveform
            pts = []
            for j, v in enumerate(hist):
                px_ = gx + int(j * gw / max(len(hist)-1, 1))
                py_ = gy + gh - 4 - int(v / 100 * (gh - 8))
                pts.append((px_, py_))
            if len(pts) > 1:
                for j in range(len(pts) - 1):
                    cv2.line(f, pts[j], pts[j+1],
                             tuple(int(c2 * d) for c2 in col), 1, cv2.LINE_AA)

            # Fill under curve
            fill_pts = [(gx, gy+gh-4)] + pts + [(gx+gw, gy+gh-4)]
            fill_arr = np.array(fill_pts, dtype=np.int32)
            ov = f.copy()
            cv2.fillPoly(ov, [fill_arr], tuple(int(c2 * 0.12 * d) for c2 in col))
            cv2.addWeighted(ov, 0.6, f, 0.4, 0, f)

            # Labels
            cv2.putText(f, key, (gx + 4, gy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        tuple(int(c2 * 0.7 * d) for c2 in col), 1, cv2.LINE_AA)
            pct_str = f"{val:.0f}%"
            (pw, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(f, pct_str, (gx + gw - pw - 4, gy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        tuple(int(c2 * d) for c2 in col), 1, cv2.LINE_AA)

            # Threshold line at 80%
            ty_ = gy + gh - 4 - int(0.80 * (gh - 8))
            cv2.line(f, (gx, ty_), (gx + gw, ty_),
                     tuple(int(c2 * 0.3 * d) for c2 in col), 1)

    # ── KERNEL LOG ───────────────────────────────────────────────────────────
    def _draw_kernel_log(self, f: np.ndarray, d: float,
                          rx: int, y1: int, rw: int, y2: int):
        dim = int(100 * d)
        c   = int(255 * d)
        cv2.rectangle(f, (rx, y1), (rx + rw, y2), (0, int(25*d), int(35*d)), -1)
        cv2.rectangle(f, (rx, y1), (rx + rw, y2), (0, int(40*d), int(55*d)), 1)
        self._corner_brackets(f, rx, y1, rx + rw, y2, (0, int(160*d), c), 12, 1)
        cv2.putText(f, "KERNEL LOG", (rx + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, dim, dim), 1, cv2.LINE_AA)

        line_h = 18
        visible = (y2 - y1 - 10) // line_h
        lines   = self._log_lines[-visible:]
        for i, line in enumerate(lines):
            ly = y1 + 14 + i * line_h
            if "[OK]" in line or "[READY]" in line:
                col = (0, int(200*d), int(80*d))
            elif "[ERR]" in line or "[FAIL]" in line:
                col = (0, int(60*d), int(255*d))
            else:
                col = (0, int(140*d), int(160*d))
            # Truncate to fit panel
            disp = line[:int(rw / 7)]
            cv2.putText(f, disp, (rx + 8, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

        # Blinking cursor on last line
        cy_ = y1 + 14 + len(lines) * line_h
        if int(time.time() * 2) % 2:
            cv2.putText(f, "▌", (rx + 8, cy_),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (0, int(200*d), int(80*d)), 1, cv2.LINE_AA)

    # ── APP DOCK ─────────────────────────────────────────────────────────────
    def _draw_app_dock(self, f: np.ndarray, t: float, d: float,
                        rx: int, ry: int, rw: int):
        c    = int(255 * d)
        cols = 4
        rows = math.ceil(len(self.APPS) / cols)
        cell_w = rw // cols
        cell_h = 56
        total_h = rows * cell_h + 20

        cv2.rectangle(f, (rx, ry), (rx + rw, ry + total_h),
                      (0, int(20*d), int(28*d)), -1)
        cv2.rectangle(f, (rx, ry), (rx + rw, ry + total_h),
                      (0, int(35*d), int(50*d)), 1)
        self._corner_brackets(f, rx, ry, rx + rw, ry + total_h,
                               (0, int(150*d), c), 12, 1)
        cv2.putText(f, "APP DOCK", (rx + 6, ry - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, int(100*d), int(120*d)), 1, cv2.LINE_AA)

        self._dock_rects.clear()
        for i, (name, icon, accent) in enumerate(self.APPS):
            col_i = i % cols
            row_i = i // cols
            cx_   = rx + col_i * cell_w + cell_w // 2
            cy_   = ry + 14 + row_i * cell_h + cell_h // 2

            is_hov = (i == self._hovered_app)
            is_act = (i == self._active_app)

            # Hover / active glow pulse
            if is_hov or is_act:
                pulse = 0.6 + 0.4 * abs(math.sin(t * 4))
                gc_   = tuple(int(v * pulse * d) for v in accent)
                cv2.circle(f, (cx_, cy_), 26, gc_, 1, cv2.LINE_AA)
                cv2.circle(f, (cx_, cy_), 22,
                           tuple(int(v * 0.18 * d) for v in accent), -1)
            else:
                cv2.circle(f, (cx_, cy_), 22,
                           (0, int(18*d), int(24*d)), -1)
                cv2.circle(f, (cx_, cy_), 22,
                           tuple(int(v * 0.35 * d) for v in accent), 1, cv2.LINE_AA)

            # Icon text (use first char as stand-in since cv2 can't render Unicode well)
            icon_ch = name[0]
            (iw, ih), _ = cv2.getTextSize(icon_ch, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.putText(f, icon_ch, (cx_ - iw//2, cy_ + ih//2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7,
                        tuple(int(v * d) for v in accent), 1, cv2.LINE_AA)

            # Label
            (nw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
            cv2.putText(f, name, (cx_ - nw//2, cy_ + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        tuple(int(v * 0.7 * d) for v in accent), 1, cv2.LINE_AA)

            self._dock_rects.append((cx_ - 24, cy_ - 24, cx_ + 24, cy_ + 24))

    # ── UTILITY ──────────────────────────────────────────────────────────────
    # ── CIRCUIT PANEL (right column when circuit mode is active) ─────────────
    def _draw_circuit_panel(self, f: np.ndarray, t: float, d: float,
                             rx: int, ry: int, rw: int, engine):
        """Renders circuit board mini-view + component picker list on projector."""
        c   = int(255 * d)
        dim = int(120 * d)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ── Mini circuit board view (top 380px) ─────────────────────────────
        bh = 380
        cv2.rectangle(f, (rx, ry), (rx + rw, ry + bh), (0, int(18*d), int(28*d)), -1)
        cv2.rectangle(f, (rx, ry), (rx + rw, ry + bh), (0, int(120*d), c), 1)
        self._corner_brackets(f, rx, ry, rx + rw, ry + bh,
                               (0, int(200*d), c), 14, 1)
        cv2.putText(f, "CIRCUIT BOARD", (rx + 6, ry - 6),
                    font, 0.38, (0, dim, dim), 1, cv2.LINE_AA)

        # Render the circuit engine into a temp canvas, then paste
        try:
            tmp = np.zeros((bh - 2, rw - 2, 3), dtype=np.uint8)
            # Save + override engine canvas size temporarily
            orig_w, orig_h = engine.canvas_w, engine.canvas_h
            engine.canvas_w = rw - 2
            engine.canvas_h = bh - 2
            engine.render_board_only(tmp)
            engine.canvas_w, engine.canvas_h = orig_w, orig_h
            f[ry+1:ry+bh-1, rx+1:rx+rw-1] = tmp
        except Exception:
            cv2.putText(f, "CIRCUIT ENGINE", (rx + rw//2 - 60, ry + bh//2),
                        font, 0.5, (0, int(100*d), int(120*d)), 1, cv2.LINE_AA)

        # ── Component picker list (below board) ─────────────────────────────
        py  = ry + bh + 10
        ph  = self.H - py - 50    # remaining space
        cv2.rectangle(f, (rx, py), (rx + rw, py + ph), (0, int(12*d), int(18*d)), -1)
        cv2.rectangle(f, (rx, py), (rx + rw, py + ph), (0, int(60*d), int(80*d)), 1)
        cv2.putText(f, "COMPONENTS", (rx + 6, py - 6),
                    font, 0.38, (0, dim, dim), 1, cv2.LINE_AA)

        # Selected component highlight
        sel = getattr(engine, 'panel_selected', 'resistor')
        sel_comp_name = ""

        # Draw component list (compact rows)
        row_h   = 22
        visible = max(1, (ph - 6) // row_h)
        items   = [it for it in engine._panel_items if it[0] == "comp"]

        # Find selected index for scroll centering
        sel_idx = next((i for i, it in enumerate(items) if it[1] == sel), 0)
        start   = max(0, sel_idx - visible // 2)
        end     = min(len(items), start + visible)

        for i, (kind, tid, lbl) in enumerate(items[start:end]):
            iy = py + 4 + i * row_h
            if iy + row_h > py + ph:
                break
            is_sel = (tid == sel)
            bg = (0, int(35*d), int(55*d)) if is_sel else (0, 0, 0)
            cv2.rectangle(f, (rx + 1, iy), (rx + rw - 1, iy + row_h - 1), bg, -1)

            d_ = engine.CATALOG.get(tid, {}) if hasattr(engine, 'CATALOG') else {}
            from core.circuit_engine import CATALOG as _CAT
            d_ = _CAT.get(tid, {})
            col = d_.get("color", (80, 80, 80))
            # Swatch
            cv2.rectangle(f, (rx + 4, iy + 5), (rx + 14, iy + row_h - 5),
                          tuple(int(cc * d) for cc in col), -1)
            # Name
            text_c = (0, int(200*d), c) if is_sel else (int(160*d), int(200*d), int(220*d))
            cv2.putText(f, lbl[:14], (rx + 18, iy + 15),
                        font, 0.30, text_c, 1, cv2.LINE_AA)
            # Sim state badge if running
            if engine.sim_running and is_sel:
                cv2.putText(f, "►SIM", (rx + rw - 42, iy + 15),
                            font, 0.25, (80, 255, 80), 1, cv2.LINE_AA)

        # Scroll indicator
        if len(items) > visible:
            bar_h = max(8, int(ph * visible / len(items)))
            bar_y = py + int(start / len(items) * ph)
            cv2.rectangle(f, (rx + rw - 4, bar_y), (rx + rw - 2, bar_y + bar_h),
                          (0, int(100*d), c), -1)

        # Sim status badge
        if engine.sim_running:
            elapsed = time.time() - engine._sim_t0
            pulse   = 0.6 + 0.4 * abs(math.sin(elapsed * 3))
            sc      = tuple(int(cc * pulse * d) for cc in (80, 255, 80))
            cv2.circle(f, (rx + rw - 14, ry + 8), 5, sc, -1, cv2.LINE_AA)
            cv2.putText(f, f"SIM {elapsed:.0f}s", (rx + rw - 60, ry + 12),
                        font, 0.28, sc, 1, cv2.LINE_AA)

    @staticmethod
    def _corner_brackets(f, x1, y1, x2, y2, col, size, thick):
        for pts in [
            [(x1, y1+size),(x1,y1),(x1+size,y1)],
            [(x2-size,y1),(x2,y1),(x2,y1+size)],
            [(x1,y2-size),(x1,y2),(x1+size,y2)],
            [(x2-size,y2),(x2,y2),(x2,y2-size)],
        ]:
            for i in range(len(pts)-1):
                cv2.line(f, pts[i], pts[i+1], col, thick, cv2.LINE_AA)

    def push_log(self, line: str):
        self._log_lines.append(line)
        if len(self._log_lines) > 200:
            self._log_lines = self._log_lines[-150:]

    # ── INPUT ────────────────────────────────────────────────────────────────
    def mouseMoveEvent(self, event):
        if self._phase != "DESKTOP":
            return
        x, y = event.pos().x(), event.pos().y()
        self._hovered_app = -1
        for i, (x1, y1, x2, y2) in enumerate(self._dock_rects):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._hovered_app = i
                break
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if self._phase != "DESKTOP":
            return
        x, y = event.pos().x(), event.pos().y()
        for i, (x1, y1, x2, y2) in enumerate(self._dock_rects):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._active_app = i if self._active_app != i else -1
                self.push_log(f"APP LAUNCHED: {self.APPS[i][0]}")
                return

    def mouseDoubleClickEvent(self, e):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN HUB
# ─────────────────────────────────────────────────────────────────────────────
class UIHub(QMainWindow):

    def __init__(self, kernel):
        super().__init__()
        self.kernel           = kernel
        self.projector_window = None
        self.calibration_mode = False
        self._circuit_active  = False
        self._sim_active      = False

        # Worker process
        self._in_q  = mp.Queue(maxsize=2)
        self._out_q = mp.Queue(maxsize=2)
        self._worker = mp.Process(
            target=_image_worker,
            args=(self._in_q, self._out_q),
            daemon=True,
        )
        self._worker.start()

        # FPS counter
        self._fps_times: list[float] = []

        self.setWindowTitle("AIILA OS — NEURAL INTERFACE TERMINAL")
        self.setGeometry(80, 60, 1600, 920)
        self.setStyleSheet(_SS_BASE)

        self._build_ui()

        # High-freq timer (~60 fps)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._loop)
        self._timer.start(16)

    # ── UI assembly ──────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top title bar
        title_bar = QFrame()
        title_bar.setFixedHeight(36)
        title_bar.setStyleSheet(f"""
            background: {C['bg']};
            border-bottom: 1px solid {C['border_hi']};
        """)
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(16, 0, 16, 0)

        lbl = QLabel("◈  AIILA OS  —  NEURAL INTERFACE TERMINAL")
        lbl.setStyleSheet(f"color:{C['text_dim']};font-size:10px;letter-spacing:2px;")
        tb_lay.addWidget(lbl)
        tb_lay.addStretch()

        self._clock = QLabel()
        self._clock.setStyleSheet(f"color:{C['accent']};font-size:10px;letter-spacing:2px;")
        tb_lay.addWidget(self._clock)
        outer.addWidget(title_bar)

        # Body
        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)

        # Sidebar
        self.sidebar = Sidebar(self.kernel)
        body_lay.addWidget(self.sidebar)

        # Viewport
        self.viewport = Viewport()
        body_lay.addWidget(self.viewport, 1)

        outer.addWidget(body, 1)

        # Status bar
        self._status = StatusBar()
        outer.addWidget(self._status)

        # Wire up buttons
        self.sidebar.btn_scan.clicked.connect(self._on_scan)
        self.sidebar.btn_voice.clicked.connect(self._on_voice)
        self.sidebar.btn_circuit.clicked.connect(self._toggle_circuit)
        self.sidebar.btn_sim.clicked.connect(self._toggle_sim)
        self.sidebar.btn_project.clicked.connect(self._toggle_projector)
        self.sidebar.btn_calib.clicked.connect(self._toggle_calib)
        self.sidebar.btn_save.clicked.connect(self._save_circuit)
        self.sidebar.btn_settings.clicked.connect(self._open_settings)

    # ── Main UI loop ─────────────────────────────────────────────────────────
    def _loop(self):
        # Clock
        self._clock.setText(time.strftime("  %Y-%m-%d   %H:%M:%S  "))

        # FPS
        now = time.monotonic()
        self._fps_times.append(now)
        self._fps_times = [t for t in self._fps_times if now - t < 1.0]
        fps = len(self._fps_times)

        try:
            data = None
            while not self._out_q.empty():
                try:
                    data = self._out_q.get_nowait()
                except Exception:
                    break

            if data:
                p_rgb, ar_rgb, cam_rgb, state = data

                # Projector — pass circuit engine so it can render the panel
                if self.projector_window:
                    self.projector_window.update_display(
                        p_rgb,
                        self.kernel.circuit_engine if self._circuit_active else None,
                        self._circuit_active,
                    )

                # Main viewport
                self.viewport.set_frame(_ndarray_to_pixmap(ar_rgb))

                # Camera thumbnail
                self.sidebar.cam_preview.set_frame(_ndarray_to_pixmap(cam_rgb))

                # Terminal
                fb = state.get('voice_feedback', '')
                if fb:
                    self.sidebar.terminal.push(fb)
                    if self.projector_window:
                        self.projector_window.push_log(fb)

                # Metrics
                self.sidebar.update_metrics(fps, state)
                self._status.tick(fps, state.get('gesture_active'))

        except Exception:
            pass

    # ── Kernel → worker bridge ────────────────────────────────────────────────
    def update_display(self, ar_canvas: np.ndarray, raw_frame: np.ndarray, state: dict):
        """Called by kernel on every frame (runs in kernel thread)."""
        if not self._in_q.full():
            self._in_q.put_nowait((
                ar_canvas, raw_frame, state,
                self.projector_window is not None,
                self.calibration_mode,
            ))

    # ── Button callbacks ──────────────────────────────────────────────────────
    def _on_scan(self):
        self.kernel.pending_scan = True
        self.sidebar.terminal.push("Object scan triggered", "SYS")

    def _on_voice(self):
        self.kernel.trigger_voice()
        self.sidebar.terminal.push("Voice listener activated", "SYS")

    def _toggle_circuit(self):
        self._circuit_active = not self._circuit_active
        self.kernel.app_state['circuit_engine_enabled'] = self._circuit_active
        self.sidebar.btn_circuit.set_active(self._circuit_active)
        lvl = "SYS"
        msg = f"Circuit mode {'ENABLED' if self._circuit_active else 'DISABLED'}"
        self.sidebar.terminal.push(msg, lvl)

    def _toggle_sim(self):
        self._sim_active = not self._sim_active
        self.viewport.set_sim(self._sim_active)
        self.sidebar.btn_sim.set_active(self._sim_active)
        if self._sim_active:
            self.kernel.start_simulation()
            self.sidebar.terminal.push("Simulation STARTED", "INFO")
        else:
            self.kernel.stop_simulation()
            self.sidebar.terminal.push("Simulation STOPPED", "WARN")

    def _toggle_projector(self):
        if self.projector_window is None:
            try:
                from screeninfo import get_monitors
                monitors = get_monitors()
            except Exception:
                monitors = []
            if len(monitors) < 2:
                self.sidebar.terminal.push("No secondary display detected", "ERR")
                return
            proj = monitors[1]
            self.projector_window = ProjectorWindow(proj.x, proj.y)
            self.projector_window.show()
            self.sidebar.btn_project.set_active(True)
            self.sidebar.terminal.push(f"Projector on display {proj.name}", "SYS")
        else:
            self.projector_window.close()
            self.projector_window = None
            self.sidebar.btn_project.set_active(False)
            self.sidebar.terminal.push("Projector closed", "SYS")

    def _toggle_calib(self):
        self.calibration_mode = not self.calibration_mode
        self.sidebar.btn_calib.set_active(self.calibration_mode)
        self.sidebar.terminal.push(
            f"Calibration grid {'ON' if self.calibration_mode else 'OFF'}", "SYS"
        )

    def _save_circuit(self):
        self.kernel.save_circuit("circuit.json")
        self.sidebar.terminal.push("Circuit saved → circuit.json", "INFO")

    def _open_settings(self):
        self.sidebar.terminal.push("Opening hardware settings…", "SYS")
        self._settings_dlg = SettingsPanel(self, self.kernel.app_state, self.kernel)
        self._settings_dlg.show()

    def closeEvent(self, event):
        self._worker.terminate()
        event.accept()