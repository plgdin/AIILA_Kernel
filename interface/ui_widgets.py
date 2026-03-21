import time
import math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QFrame, QSizePolicy,
    QScrollArea, QGraphicsOpacityEffect
)
from PyQt6.QtGui  import (
    QPixmap, QFont, QColor, QPainter, QPen, QBrush,
    QLinearGradient, QPainterPath
)
from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve,
    QPoint, pyqtSignal, QSize, QRect
)

from interface.ui_styles import C, MONO, TITLE
from interface.ui_utils  import _glow


# ─────────────────────────────────────────────────────────────────────────────
#  REUSABLE WIDGETS  (unchanged)
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
        super().__init__(parent)
        self._prefix = prefix
        self._label = text
        self._apply_text()
        self.setStyleSheet(self._NORMAL)
        self._active = False

    def _apply_text(self):
        self.setText(f"  {self._prefix}  {self._label}")

    def set_label(self, text: str):
        self._label = text
        self._apply_text()

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
        self.setFixedSize(240, 160)
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
            [(L, L+S), (L, L), (L+S, L)],
            [(R-S, T), (R, T), (R, T+S)],
            [(L, B-S), (L, B), (L+S, B)],
            [(R-S, B), (R, B), (R, B-S)],
        ]:
            for i in range(len(pts)-1):
                p.drawLine(*pts[i], *pts[i+1])
        p.end()


class MetricTile(QFrame):
    def __init__(self, label: str, value: str = "—",
                 accent: str = C['accent'], parent=None):
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
        self._lbl.setStyleSheet(
            f"color:{C['text_dim']};font-size:9px;letter-spacing:2px;"
            f"border:none;background:transparent;")
        self._val = QLabel(value)
        self._val.setStyleSheet(
            f"color:{accent};font-size:14px;font-weight:bold;"
            f"border:none;background:transparent;")
        lay.addWidget(self._lbl)
        lay.addWidget(self._val)

    def set_value(self, v: str):
        self._val.setText(v)


# ─────────────────────────────────────────────────────────────────────────────
#  NOTCH TAB  — the always-visible vertical grip strip
# ─────────────────────────────────────────────────────────────────────────────

class NotchTab(QFrame):
    """
    The slim vertical strip that remains visible when the panel is collapsed.
    Clicking or dragging it left opens the full panel.
    Emits `clicked` so Sidebar.toggle() can be connected to it.
    """
    clicked = pyqtSignal()

    NOTCH_W  = 28
    DOT_ROWS = 8      # grip dots per column
    DOT_COLS = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(self.NOTCH_W)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C['surface']};
                border-right: 1px solid {C['border_hi']};
            }}
        """)

        # Live stat labels shown vertically in the notch
        self._fps_val:     str = "—"
        self._gesture_val: str = "·"
        self._expanded:    bool = False

        # Drag tracking
        self._drag_start:  QPoint | None = None
        self._drag_opened: bool = False

    def set_stats(self, fps: str, gesture: str, expanded: bool):
        self._fps_val     = fps
        self._gesture_val = gesture[:3].upper()
        self._expanded    = expanded
        self.update()

    # ── Mouse events for drag-to-open / drag-to-close ─────────────────────

    def mousePressEvent(self, ev):
        self._drag_start  = ev.pos()
        self._drag_opened = False
        ev.accept()

    def mouseMoveEvent(self, ev):
        if self._drag_start is None:
            return
        dx = self._drag_start.x() - ev.pos().x()   # negative = drag right
        if dx > 14 and not self._drag_opened:       # dragged LEFT → open
            self._drag_opened = True
            self.clicked.emit()
        ev.accept()

    def mouseReleaseEvent(self, ev):
        if (self._drag_start is not None
                and not self._drag_opened
                and abs(self._drag_start.x() - ev.pos().x()) < 8):
            self.clicked.emit()   # tap without drag
        self._drag_start  = None
        self._drag_opened = False
        ev.accept()

    # ── Custom paint ─────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()

        # Background gradient (subtle left-edge glow)
        grad = QLinearGradient(0, 0, W, 0)
        grad.setColorAt(0.0, QColor(C['accent']).darker(700))
        grad.setColorAt(1.0, QColor(C['surface']))
        p.fillRect(0, 0, W, H, grad)

        # Left accent line
        p.setPen(QPen(QColor(C['accent']), 2))
        p.drawLine(0, 0, 0, H)

        # Arrow indicator (▶ when collapsed, ◀ when expanded)
        arrow = "◀" if self._expanded else "▶"
        p.setPen(QColor(C['accent']))
        p.setFont(QFont(MONO, 8, QFont.Weight.Bold))
        arrow_rect = QRect(0, H // 2 - 16, W, 18)
        p.drawText(arrow_rect, Qt.AlignmentFlag.AlignHCenter, arrow)

        # Grip dots
        dot_r   = 1
        dot_gap = 5
        total_h = self.DOT_ROWS * dot_gap
        start_y = H // 2 + 20
        start_x = W // 2 - (self.DOT_COLS * dot_gap) // 2

        dot_col = QColor(C['border_hi'])
        p.setBrush(QBrush(dot_col))
        p.setPen(Qt.PenStyle.NoPen)
        for row in range(self.DOT_ROWS):
            for col in range(self.DOT_COLS):
                x = start_x + col * dot_gap
                y = start_y + row * dot_gap
                p.drawEllipse(x - dot_r, y - dot_r, dot_r*2, dot_r*2)

        # FPS / gesture mini-readout at bottom
        p.save()
        p.translate(W // 2 + 4, H - 60)
        p.rotate(-90)
        p.setPen(QColor(C['text_dim']))
        p.setFont(QFont(MONO, 7))
        p.drawText(0, 0, f"{self._fps_val}fps  {self._gesture_val}")
        p.restore()

        # "AIILA" vertical text at top when collapsed
        p.save()
        p.translate(W // 2 + 4, 80)
        p.rotate(90)
        p.setPen(QColor(C['accent']))
        p.setFont(QFont(MONO, 8, QFont.Weight.Bold))
        p.drawText(0, 0, "AIILA")
        p.restore()

        p.end()


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR  — collapsible notch drawer
# ─────────────────────────────────────────────────────────────────────────────

class Sidebar(QFrame):
    """
    Collapsible sidebar panel that lives as a narrow notch on the left edge.

    • Click or drag-left the notch tab  → slides open to full width
    • Click the notch again, drag-right the panel, or call on_swipe('right')
      → slides back to notch width
    • on_swipe(direction) can be wired to GestureEngine swipe events:
        sidebar.on_swipe('left')   # opens
        sidebar.on_swipe('right')  # closes
    """

    NOTCH_W = 28
    FULL_W  = 280
    ANIM_MS = 260

    def __init__(self, kernel, parent=None):
        super().__init__(parent)
        self.kernel   = kernel
        self._expanded = False

        # Animation on maximumWidth
        self.setMinimumWidth(self.NOTCH_W)
        self.setMaximumWidth(self.NOTCH_W)     # start collapsed
        self.setSizePolicy(QSizePolicy.Policy.Fixed,
                           QSizePolicy.Policy.Expanding)
        self.setStyleSheet(f"""
            QFrame {{
                background: {C['surface']};
                border-right: 1px solid {C['border']};
            }}
        """)

        self._anim = QPropertyAnimation(self, b"maximumWidth")
        self._anim.setDuration(self.ANIM_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # ── Drag-to-close tracking on the panel body ──────────────────────
        self._panel_drag_start: QPoint | None = None

        # ── Top-level horizontal layout ───────────────────────────────────
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Notch tab (always visible)
        self._notch = NotchTab()
        self._notch.clicked.connect(self.toggle)
        root.addWidget(self._notch)

        # Full content container (hidden width until open)
        self._content_w = QWidget()
        self._content_w.setFixedWidth(self.FULL_W - self.NOTCH_W)
        self._content_w.setStyleSheet("background: transparent;")
        # Fade-in effect for content
        self._opacity_fx = QGraphicsOpacityEffect(self._content_w)
        self._opacity_fx.setOpacity(0.0)
        self._content_w.setGraphicsEffect(self._opacity_fx)

        self._fade = QPropertyAnimation(self._opacity_fx, b"opacity")
        self._fade.setDuration(self.ANIM_MS + 60)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._build_content(self._content_w)
        root.addWidget(self._content_w)

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    # ─────────────────────────────────────────────────────────────────────────
    #  Drag-to-close on the panel body
    # ─────────────────────────────────────────────────────────────────────────

    def mousePressEvent(self, ev):
        self._panel_drag_start = ev.pos()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panel_drag_start and self._expanded:
            dx = ev.pos().x() - self._panel_drag_start.x()
            if dx > 40:                # dragged RIGHT → close
                self._panel_drag_start = None
                self.close_panel()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self._panel_drag_start = None
        super().mouseReleaseEvent(ev)

    # ─────────────────────────────────────────────────────────────────────────
    #  Public gesture hook
    # ─────────────────────────────────────────────────────────────────────────

    def on_swipe(self, direction: str):
        """
        Wire this to the kernel gesture output:
            sidebar.on_swipe('left')   → open panel
            sidebar.on_swipe('right')  → close panel
        """
        if direction == 'left' and not self._expanded:
            self.open_panel()
        elif direction == 'right' and self._expanded:
            self.close_panel()

    # ─────────────────────────────────────────────────────────────────────────
    #  Open / close / toggle
    # ─────────────────────────────────────────────────────────────────────────

    def toggle(self):
        if self._expanded:
            self.close_panel()
        else:
            self.open_panel()

    def open_panel(self):
        if self._expanded:
            return
        self._expanded = True
        self._notch.set_stats(self._notch._fps_val,
                               self._notch._gesture_val, True)

        self._anim.stop()
        self._anim.setStartValue(self.maximumWidth())
        self._anim.setEndValue(self.FULL_W)
        self._anim.start()

        self._fade.stop()
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

    def close_panel(self):
        if not self._expanded:
            return
        self._expanded = False
        self._notch.set_stats(self._notch._fps_val,
                               self._notch._gesture_val, False)

        self._fade.stop()
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.start()

        self._anim.stop()
        self._anim.setStartValue(self.maximumWidth())
        self._anim.setEndValue(self.NOTCH_W)
        self._anim.start()

    # ─────────────────────────────────────────────────────────────────────────
    #  Build the full panel content
    # ─────────────────────────────────────────────────────────────────────────

    def _build_content(self, parent: QWidget):
        main_lay = QVBoxLayout(parent)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # ── Fixed top: logo + metrics + camera ────────────────────────────
        top = QWidget()
        top_lay = QVBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(0)

        logo_frame = QFrame()
        logo_frame.setStyleSheet(
            f"background:{C['bg']};"
            f"border-bottom:1px solid {C['border_hi']};padding:0;")
        logo_lay = QVBoxLayout(logo_frame)
        logo_lay.setContentsMargins(16, 18, 16, 14)
        logo_lay.setSpacing(2)
        logo = QLabel("AIILA")
        logo.setFont(QFont(TITLE, 24, QFont.Weight.Bold))
        logo.setStyleSheet(
            f"color:{C['accent']};letter-spacing:8px;"
            f"border:none;background:transparent;")
        _glow(logo, C['accent'], 24)
        sub = QLabel("NEURAL INTERFACE TERMINAL  v2.4.1")
        sub.setStyleSheet(
            f"color:{C['text_dim']};font-size:8px;letter-spacing:3px;"
            f"border:none;background:transparent;")
        logo_lay.addWidget(logo)
        logo_lay.addWidget(sub)
        top_lay.addWidget(logo_frame)

        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("background:transparent;border:none;")
        m_lay = QHBoxLayout(metrics_frame)
        m_lay.setContentsMargins(10, 10, 10, 6)
        m_lay.setSpacing(6)
        self.tile_fps     = MetricTile("FPS",     "—",    C['accent'])
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
        cw_lay.addWidget(self.cam_preview,
                         alignment=Qt.AlignmentFlag.AlignCenter)
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
        main_lay.addWidget(top)

        # ── Scrollable bottom: actions + log ──────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("background: transparent; border: none;")

        sc = QWidget()
        sc.setStyleSheet("background: transparent;")
        sc_lay = QVBoxLayout(sc)
        sc_lay.setContentsMargins(0, 0, 0, 0)
        sc_lay.setSpacing(0)

        sc_lay.addWidget(SectionLabel("◈  ACTIONS"))
        btn_frame = QFrame()
        btn_frame.setStyleSheet("background:transparent;border:none;")
        btn_lay = QVBoxLayout(btn_frame)
        btn_lay.setContentsMargins(10, 4, 10, 4)
        btn_lay.setSpacing(3)

        self.btn_scan     = HexButton("SCAN UNIT",         "⊕")
        self.btn_voice    = HexButton("ACTIVATE AIILA",    "◉")
        self.btn_circuit  = HexButton("CIRCUIT MODE",      "⎔")
        self.btn_draw     = HexButton("WIRE DRAW MODE",    "⌁")
        self.btn_sim      = HexButton("RUN SIMULATION",    "▶")
        self.btn_project  = HexButton("PROJECT TO SCREEN", "⊞")
        self.btn_calib    = HexButton("CALIBRATION GRID",  "⊟")
        self.btn_save     = HexButton("SAVE CIRCUIT",      "◧")
        self.btn_undo     = HexButton("UNDO  [Ctrl+Z]",    "↩")
        self.btn_settings = HexButton("HARDWARE SETTINGS", "⚙")

        for b in (self.btn_scan, self.btn_voice, self.btn_circuit,
                  self.btn_draw, self.btn_sim, self.btn_project,
                  self.btn_calib, self.btn_save, self.btn_undo,
                  self.btn_settings):
            btn_lay.addWidget(b)
        sc_lay.addWidget(btn_frame)

        sc_lay.addWidget(SectionLabel("◈  KERNEL LOG"))
        self.terminal = TerminalLog()
        self.terminal.setMinimumHeight(220)
        sc_lay.addWidget(self.terminal, 1)

        scroll.setWidget(sc)
        main_lay.addWidget(scroll, 1)

    # ─────────────────────────────────────────────────────────────────────────
    #  Metrics update  (same public API as before)
    # ─────────────────────────────────────────────────────────────────────────

    def update_metrics(self, fps: float, state: dict):
        fps_str = f"{fps:.0f}"
        g       = state.get('gesture_active') or 'IDLE'
        m       = state.get('ar_mode', 'default')

        self.tile_fps.set_value(fps_str)
        self.tile_gesture.set_value(g[:8].upper())
        self.tile_mode.set_value(m[:5].upper())

        # Keep notch live readout current
        self._notch.set_stats(fps_str, g, self._expanded)

        mode_colours = {
            'default': C['warn'],
            'draw':    C['accent'],
            'inspect': C['accent3'],
            'measure': C['accent2'],
        }
        col = mode_colours.get(m, C['warn'])
        self.mode_indicator.setText(
            f"MODE: {m.upper()}  |  ✌ PEACE to cycle")
        self.mode_indicator.setStyleSheet(f"""
            color: {col};
            font-size: 9px;
            letter-spacing: 2px;
            padding: 4px 14px;
            border-bottom: 1px solid {C['border']};
            background: transparent;
        """)


# ─────────────────────────────────────────────────────────────────────────────
#  VIEWPORT  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class Viewport(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"background: {C['bg']}; border: none;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
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
        S    = 22
        col  = QColor(C['accent'])
        col.setAlpha(200)
        pen = QPen(col, 2)
        p.setPen(pen)
        for pts in [
            [(S, 2),   (2, 2),   (2, S)],
            [(W-S, 2), (W-2, 2), (W-2, S)],
            [(2, H-S), (2, H-2), (S, H-2)],
            [(W-S, H-2),(W-2, H-2),(W-2, H-S)],
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
            bc      = QColor(C['accent3'])
            bc.setAlpha(int(180 + 75 * pulse))
            p.setPen(QPen(bc, 1))
            from PyQt6.QtGui import QBrush
            p.setBrush(QBrush(bc))
            p.drawEllipse(W - 18, 8, 8, 8)
            p.setPen(QPen(QColor(C['text_bright']), 1))
            p.setFont(QFont(MONO, 8))
            p.drawText(W - 80, 18, "SIM LIVE")
        p.end()
