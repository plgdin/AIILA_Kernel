import time
import math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QFrame, QSizePolicy,
    QScrollArea
)
from PyQt6.QtGui  import QPixmap, QFont, QColor, QPainter, QPen
from PyQt6.QtCore import Qt

from interface.ui_styles import C, MONO, TITLE
from interface.ui_utils import _glow

# ─────────────────────────────────────────────────────────────────────────────
#  REUSABLE WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
class HexButton(QPushButton):
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
            f"GESTURE {g.upper():<12}   ▪   AIILA OS v2.4.1"
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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 190)
        self.setStyleSheet(f"background: {C['bg']}; border-radius: 4px;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap = None

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
        self.setFixedHeight(48)
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
        # Main Layout
        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # 1. FIXED TOP SECTION (Logo, Metrics, Camera)
        top_container = QWidget()
        top_lay = QVBoxLayout(top_container)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(0)

        logo_frame = QFrame()
        logo_frame.setStyleSheet(f"background:{C['bg']};border-bottom:1px solid {C['border_hi']};padding:0;")
        logo_lay = QVBoxLayout(logo_frame)
        logo_lay.setContentsMargins(16, 18, 16, 14)
        logo_lay.setSpacing(2)
        logo = QLabel("AIILA")
        logo.setFont(QFont(TITLE, 28, QFont.Weight.Bold))
        logo.setStyleSheet(f"color:{C['accent']};letter-spacing:8px;border:none;background:transparent;")
        _glow(logo, C['accent'], 24)
        sub = QLabel("NEURAL INTERFACE TERMINAL  v2.4.1")
        sub.setStyleSheet(f"color:{C['text_dim']};font-size:8px;letter-spacing:3px;border:none;background:transparent;")
        logo_lay.addWidget(logo)
        logo_lay.addWidget(sub)
        top_lay.addWidget(logo_frame)

        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("background:transparent;border:none;")
        m_lay = QHBoxLayout(metrics_frame)
        m_lay.setContentsMargins(10, 10, 10, 6)
        m_lay.setSpacing(6)
        self.tile_fps     = MetricTile("FPS",     "—",     C['accent'])
        self.tile_gesture = MetricTile("GESTURE", "IDLE", C['accent3'])
        self.tile_mode    = MetricTile("AR MODE", "DEF",  C['warn'])
        for t in (self.tile_fps, self.tile_gesture, self.tile_mode):
            m_lay.addWidget(t)
        top_lay.addWidget(metrics_frame)

        top_lay.addWidget(SectionLabel("◈  CAMERA FEED"))
        cam_wrap = QFrame()
        cam_wrap.setStyleSheet("background:transparent;border:none;")
        cw_lay = QHBoxLayout(cam_wrap)
        cw_lay.setContentsMargins(10, 0, 10, 6)
        self.cam_preview = CamPreview()
        cw_lay.addWidget(self.cam_preview, alignment=Qt.AlignmentFlag.AlignCenter)
        top_lay.addWidget(cam_wrap)

        self.mode_indicator = QLabel("MODE: DEFAULT")
        self.mode_indicator.setStyleSheet(f"""
            color: {C['warn']};
            font-size: 9px;
            letter-spacing: 2px;
            padding: 4px 14px;
            border-bottom: 1px solid {C['border']};
            background: transparent;
        """)
        top_lay.addWidget(self.mode_indicator)
        
        main_lay.addWidget(top_container)

        # 2. SCROLLABLE BOTTOM SECTION (Actions & Log) [FIX 7]
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("background: transparent; border: none;")

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_lay = QVBoxLayout(scroll_content)
        scroll_lay.setContentsMargins(0, 0, 0, 0)
        scroll_lay.setSpacing(0)

        scroll_lay.addWidget(SectionLabel("◈  ACTIONS"))
        btn_frame = QFrame()
        btn_frame.setStyleSheet("background:transparent;border:none;")
        btn_lay = QVBoxLayout(btn_frame)
        btn_lay.setContentsMargins(10, 4, 10, 4)
        btn_lay.setSpacing(3)

        self.btn_scan     = HexButton("SCAN UNIT",           "⊕")
        self.btn_voice    = HexButton("ACTIVATE JARVIS",     "◉")
        self.btn_circuit  = HexButton("CIRCUIT MODE",        "⎔")
        self.btn_draw     = HexButton("WIRE DRAW MODE",      "⌁")
        self.btn_sim      = HexButton("RUN SIMULATION",      "▶")
        self.btn_project  = HexButton("PROJECT TO SCREEN",   "⊞")
        self.btn_calib    = HexButton("CALIBRATION GRID",    "⊟")
        self.btn_save     = HexButton("SAVE CIRCUIT",        "◧")
        self.btn_undo     = HexButton("UNDO  [Ctrl+Z]",      "↩")
        self.btn_settings = HexButton("HARDWARE SETTINGS",   "⚙")

        for b in (self.btn_scan, self.btn_voice, self.btn_circuit, self.btn_draw,
                  self.btn_sim, self.btn_project, self.btn_calib,
                  self.btn_save, self.btn_undo, self.btn_settings):
            btn_lay.addWidget(b)
        scroll_lay.addWidget(btn_frame)

        scroll_lay.addWidget(SectionLabel("◈  KERNEL LOG"))
        self.terminal = TerminalLog()
        self.terminal.setMinimumHeight(250)
        scroll_lay.addWidget(self.terminal, 1)

        scroll.setWidget(scroll_content)
        main_lay.addWidget(scroll, 1)

    def update_metrics(self, fps: float, state: dict):
        self.tile_fps.set_value(f"{fps:.0f}")
        g = state.get('gesture_active') or 'IDLE'
        self.tile_gesture.set_value(g[:8].upper())
        m = state.get('ar_mode', 'default')
        self.tile_mode.set_value(m[:5].upper())

        mode_colours = {
            'default': C['warn'],
            'draw':    C['accent'],
            'inspect': C['accent3'],
            'measure': C['accent2'],
        }
        col = mode_colours.get(m, C['warn'])
        self.mode_indicator.setText(f"MODE: {m.upper()}  |  ✌ PEACE to cycle")
        self.mode_indicator.setStyleSheet(f"""
            color: {col};
            font-size: 9px;
            letter-spacing: 2px;
            padding: 4px 14px;
            border-bottom: 1px solid {C['border']};
            background: transparent;
        """)

# ─────────────────────────────────────────────────────────────────────────────
#  VIEWPORT
# ─────────────────────────────────────────────────────────────────────────────
class Viewport(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background: {C['bg']}; border: none;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._scan_y = 0
        self._sim_on = False
        self._t0     = time.time()

    def set_sim(self, on: bool):
        self._sim_on = on

    def set_frame(self, px: QPixmap):
        self.setPixmap(
            px.scaled(self.size(),
                      Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.FastTransformation))
        self._scan_y = (self._scan_y + 3) % max(self.height(), 1)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        S = 22
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
        if self._sim_on:
            scan_col = QColor(C['accent'])
            scan_col.setAlpha(30)
            p.setPen(QPen(scan_col, 1))
            p.drawLine(0, self._scan_y, W, self._scan_y)
            elapsed = time.time() - self._t0
            pulse   = abs(math.sin(elapsed * 3))
            bc = QColor(C['accent3'])
            bc.setAlpha(int(180 + 75 * pulse))
            p.setPen(QPen(bc, 1))
            from PyQt6.QtGui import QBrush
            p.setBrush(QBrush(bc))
            p.drawEllipse(W - 18, 8, 8, 8)
            p.setPen(QPen(QColor(C['text_bright']), 1))
            p.setFont(QFont(MONO, 8))
            p.drawText(W - 80, 18, "SIM LIVE")
        p.end()
