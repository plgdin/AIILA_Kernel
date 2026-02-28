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
#  PROJECTOR WINDOW  (redesigned)
# ─────────────────────────────────────────────────────────────────────────────
class ProjectorWindow(QWidget):
    def __init__(self, x: int, y: int):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setGeometry(x, y, 1920, 1080)
        self.setStyleSheet("background: black;")

        self._display = QLabel(self)
        self._display.setGeometry(0, 0, 1920, 1080)
        self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._t0         = time.time()
        self._state      = "INTRO"
        self._opacity    = 0.0
        self._apps_vis   = False
        self._anim_start = 0.0
        self._btn        = (860, 650, 200, 50)
        self._apps       = ["Circuit Lab", "Vision AI", "Voice CMD", "Settings", "Diagnostics", "Deploy"]

    def update_display(self, ar_rgb: np.ndarray | None):
        elapsed = time.time() - self._t0
        frame   = np.zeros((1080, 1920, 3), dtype=np.uint8)

        if self._state == "INTRO":
            self._opacity = min(1.0, elapsed / 1.8)
            if elapsed > 3.5:
                self._state = "DESKTOP"
        c = int(255 * self._opacity)

        # Background grid
        gc = int(18 * self._opacity)
        for x in range(0, 1920, 80):
            cv2.line(frame, (x,0), (x,1080), (0,gc,gc), 1)
        for y in range(0, 1080, 80):
            cv2.line(frame, (0,y), (1920,y), (0,gc,gc), 1)

        # AR content in left pane
        if ar_rgb is not None and self._state == "DESKTOP":
            resized = cv2.resize(ar_rgb, (960, 600))
            frame[80:680, 20:980] = resized
            cv2.rectangle(frame, (20,80), (980,680), (0, c//4, c//2), 1)

        # Logo
        logo_c = (0, int(200 * self._opacity), int(255 * self._opacity))
        cv2.putText(frame, "AIILA", (1020, 260),
                    cv2.FONT_HERSHEY_DUPLEX, 5, logo_c, 8, cv2.LINE_AA)
        cv2.putText(frame, "NEURAL INTERFACE TERMINAL",
                    (1020, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, c//2, c//2), 1, cv2.LINE_AA)

        if self._state == "DESKTOP":
            d = min(1.0, (elapsed - 3.5) / 1.2)
            dc = int(255 * d)

            # Taskbar
            cv2.rectangle(frame, (0, 1028), (1920, 1080), (0, int(25*d), int(30*d)), -1)
            cv2.line(frame, (0,1028), (1920,1028), (0, int(60*d), int(80*d)), 1)
            cv2.putText(frame, "AIILA OS  ▪  KERNEL READY  ▪  " + time.strftime("%H:%M:%S"),
                        (40, 1058), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, dc//2, dc//2), 1)

            # Open Apps button
            bx, by, bw, bh = self._btn
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, dc//2, dc), 1)
            (tw, th), _ = cv2.getTextSize("OPEN APPS", cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
            cv2.putText(frame, "OPEN APPS",
                        (bx + (bw-tw)//2, by + (bh+th)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, dc//2, dc), 1, cv2.LINE_AA)

            # Circular app menu
            if self._apps_vis or (time.time() - self._anim_start < 0.7):
                t = min(1.0, (time.time() - self._anim_start) / 0.5)
                if not self._apps_vis:
                    t = 1.0 - t
                r     = int(300 * t)
                cx, cy = 1920//2, 500
                sx, sy = bx + bw//2, by + bh//2
                for i, name in enumerate(self._apps):
                    ang = -150 + i * (360 / len(self._apps))
                    rad = math.radians(ang)
                    tx_ = int(cx + 300 * math.cos(rad))
                    ty_ = int(cy + 300 * math.sin(rad))
                    ax  = int(sx + (tx_ - sx) * t)
                    ay  = int(sy + (ty_ - sy) * t)
                    ac  = int(255 * t)
                    cv2.circle(frame, (ax, ay), int(46*t), (0, int(200*t), int(255*t)), 2)
                    if t > 0.75:
                        (nw, nh), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.putText(frame, name, (ax - nw//2, ay + 64),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (ac, ac, ac), 1)

        h, w = frame.shape[:2]
        qi   = QImage(frame.data, w, h, w*3, QImage.Format.Format_BGR888)
        self._display.setPixmap(QPixmap.fromImage(qi.copy()))

    def mousePressEvent(self, event):
        if self._state == "DESKTOP":
            x, y = event.pos().x(), event.pos().y()
            bx, by, bw, bh = self._btn
            if bx <= x <= bx+bw and by <= y <= by+bh:
                self._apps_vis   = not self._apps_vis
                self._anim_start = time.time()

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

                # Projector
                if p_rgb is not None and self.projector_window:
                    self.projector_window.update_display(p_rgb)
                elif self.projector_window:
                    self.projector_window.update_display(None)

                # Main viewport
                self.viewport.set_frame(_ndarray_to_pixmap(ar_rgb))

                # Camera thumbnail
                self.sidebar.cam_preview.set_frame(_ndarray_to_pixmap(cam_rgb))

                # Terminal
                fb = state.get('voice_feedback', '')
                if fb:
                    self.sidebar.terminal.push(fb)

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