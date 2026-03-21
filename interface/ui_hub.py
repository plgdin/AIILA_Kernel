"""
ui_hub.py  ·  AIILA OS  v2.4.1  (WORKBENCH EDITION)
=====================================================
NOTE: This file has been refactored. The internal classes and logic
have been moved to `ui_styles.py`, `ui_utils.py`, `ui_workers.py`, 
`ui_widgets.py`, and `ui_projector.py` for maintainability.

FIXES over v2.4 (Retained in extracted modules):
  [FIX 1]  _draw_circuit_panel() uses render_board_only(tmp, override_w, override_h)
  [FIX 2]  CATALOG import moved to module top
  [FIX 3]  Projector shows active pin connections as coloured wire overlay
  [FIX 4]  Added Ctrl+Z QShortcut wired to kernel.undo()
  [FIX 5]  AR mode badge on sidebar shows current mode with colour coding
  [FIX 6]  Wire-draw instructions shown in projector taskbar when in 'draw' mode
  [FIX 7]  Sidebar wrapped in QScrollArea
"""

import multiprocessing as mp
import time
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFrame, QLabel
)
from PyQt6.QtGui  import QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QTimer

from core.app_defaults import DEFAULT_KEYBINDS
from interface.settings_panel import SettingsPanel
from interface.ui_styles import C, _SS_BASE
from interface.ui_utils import _ndarray_to_pixmap
from interface.ui_workers import _image_worker
from interface.ui_widgets import Sidebar, Viewport, StatusBar
from interface.ui_projector import ProjectorWindow


SHORTCUT_BUTTONS = {
    'scan_unit': 'SCAN UNIT',
    'activate_aiila': 'ACTIVATE AIILA',
    'circuit_mode': 'CIRCUIT MODE',
    'wire_draw_mode': 'WIRE DRAW MODE',
    'run_simulation': 'RUN SIMULATION',
    'project_to_screen': 'PROJECT TO SCREEN',
}


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
        self._action_shortcuts: dict[str, QShortcut] = {}
        self._shutting_down   = False

        self._in_q  = mp.Queue(maxsize=2)
        self._out_q = mp.Queue(maxsize=2)
        self._worker = mp.Process(
            target=_image_worker,
            args=(self._in_q, self._out_q),
            daemon=True,
        )
        self._worker.start()

        self._fps_times: list[float] = []

        self.setWindowTitle("AIILA OS — NEURAL INTERFACE TERMINAL  v2.4.1")
        self.setGeometry(80, 60, 1600, 920)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(_SS_BASE)

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._loop)
        self._timer.start(16)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        title_bar = QFrame()
        title_bar.setFixedHeight(36)
        title_bar.setStyleSheet(f"background:{C['bg']};border-bottom:1px solid {C['border_hi']};")
        tb_lay = QHBoxLayout(title_bar)
        tb_lay.setContentsMargins(16, 0, 16, 0)
        lbl = QLabel("◈  AIILA OS  —  NEURAL INTERFACE TERMINAL  v2.4.1  WORKBENCH")
        lbl.setStyleSheet(f"color:{C['text_dim']};font-size:10px;letter-spacing:2px;")
        tb_lay.addWidget(lbl)
        tb_lay.addStretch()
        self._clock = QLabel()
        self._clock.setStyleSheet(f"color:{C['accent']};font-size:10px;letter-spacing:2px;")
        tb_lay.addWidget(self._clock)
        outer.addWidget(title_bar)

        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(8, 8, 8, 8)
        body_lay.setSpacing(10)

        self.sidebar  = Sidebar(self.kernel)
        body_lay.addWidget(self.sidebar)

        viewport_shell = QFrame()
        viewport_shell.setStyleSheet(
            f"background:{C['surface']};"
            f"border:1px solid {C['border']};"
            f"border-left:1px solid {C['border_hi']};"
        )
        viewport_lay = QVBoxLayout(viewport_shell)
        viewport_lay.setContentsMargins(10, 10, 10, 10)
        viewport_lay.setSpacing(0)

        self.viewport = Viewport()
        self.viewport.setMinimumSize(760, 520)
        viewport_lay.addWidget(self.viewport, 1)
        body_lay.addWidget(viewport_shell, 1)

        outer.addWidget(body, 1)

        self._status = StatusBar()
        outer.addWidget(self._status)

        # Wire up buttons
        self.sidebar.btn_scan.clicked.connect(self._on_scan)
        self.sidebar.btn_voice.clicked.connect(self._on_voice)
        self.sidebar.btn_circuit.clicked.connect(self._toggle_circuit)
        self.sidebar.btn_draw.clicked.connect(self._toggle_draw_mode)
        self.sidebar.btn_sim.clicked.connect(self._toggle_sim)
        self.sidebar.btn_project.clicked.connect(self._toggle_projector)
        self.sidebar.btn_calib.clicked.connect(self._toggle_calib)
        self.sidebar.btn_save.clicked.connect(self._save_circuit)
        self.sidebar.btn_undo.clicked.connect(self._on_undo)
        self.sidebar.btn_settings.clicked.connect(self._open_settings)

        # [FIX 4] Ctrl+Z shortcut
        undo_sc = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_sc.activated.connect(self._on_undo)

        self.refresh_shortcuts()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.width() < 1280 and self.sidebar.is_expanded:
            self.sidebar.close_panel()

    def _ensure_keybind_state(self) -> dict:
        keybinds = self.kernel.app_state.setdefault('keybinds', {})
        for action, default in DEFAULT_KEYBINDS.items():
            keybinds.setdefault(action, default)
        return keybinds

    def _set_button_shortcuts(self, keybinds: dict):
        self.sidebar.btn_scan.set_label(
            f"{SHORTCUT_BUTTONS['scan_unit']} [{keybinds['scan_unit']}]")
        self.sidebar.btn_voice.set_label(
            f"{SHORTCUT_BUTTONS['activate_aiila']} [{keybinds['activate_aiila']}]")
        self.sidebar.btn_circuit.set_label(
            f"{SHORTCUT_BUTTONS['circuit_mode']} [{keybinds['circuit_mode']}]")
        self.sidebar.btn_draw.set_label(
            f"{SHORTCUT_BUTTONS['wire_draw_mode']} [{keybinds['wire_draw_mode']}]")
        self.sidebar.btn_sim.set_label(
            f"{SHORTCUT_BUTTONS['run_simulation']} [{keybinds['run_simulation']}]")
        self.sidebar.btn_project.set_label(
            f"{SHORTCUT_BUTTONS['project_to_screen']} [{keybinds['project_to_screen']}]")

    def _bind_shortcut(self, action: str, sequence: str, handler):
        shortcut = self._action_shortcuts.get(action)
        if shortcut is None:
            shortcut = QShortcut(QKeySequence(), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)
            self._action_shortcuts[action] = shortcut
        shortcut.setKey(QKeySequence(sequence))

    def refresh_shortcuts(self):
        keybinds = self._ensure_keybind_state()
        self._set_button_shortcuts(keybinds)
        self._bind_shortcut('scan_unit', keybinds['scan_unit'], self._on_scan)
        self._bind_shortcut('activate_aiila', keybinds['activate_aiila'], self._on_voice)
        self._bind_shortcut('circuit_mode', keybinds['circuit_mode'], self._toggle_circuit)
        self._bind_shortcut('wire_draw_mode', keybinds['wire_draw_mode'], self._toggle_draw_mode)
        self._bind_shortcut('run_simulation', keybinds['run_simulation'], self._toggle_sim)
        self._bind_shortcut('project_to_screen', keybinds['project_to_screen'], self._toggle_projector)

    def _loop(self):
        self._clock.setText(time.strftime("  %Y-%m-%d   %H:%M:%S  "))

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

                if self.projector_window:
                    exploded_view = self.kernel.exploded_view_engine.get_view_state()
                    self.projector_window.update_display(
                        p_rgb,
                        self.kernel.circuit_engine if self._circuit_active else None,
                        self._circuit_active,
                        ar_mode=state.get('ar_mode', 'default'),
                        exploded_view=exploded_view,
                    )

                self.viewport.set_frame(_ndarray_to_pixmap(ar_rgb))
                self.sidebar.cam_preview.set_frame(_ndarray_to_pixmap(cam_rgb))

                fb = state.get('voice_feedback', '')
                if fb:
                    self.sidebar.terminal.push(fb)
                    if self.projector_window:
                        self.projector_window.push_log(fb)
                    # Clear it so it doesn't repeat next frame
                    self.kernel.app_state['voice_feedback'] = ''

                self.sidebar.update_metrics(fps, state)
                self._status.tick(fps, state.get('gesture_active'))

        except Exception:
            pass

    def update_display(self, ar_canvas: np.ndarray, raw_frame: np.ndarray, state: dict):
        if not self._in_q.full():
            self._in_q.put_nowait((
                ar_canvas, raw_frame, state,
                self.projector_window is not None,
                self.calibration_mode,
            ))

    def _on_scan(self):
        self.kernel.pending_scan = True
        self.sidebar.terminal.push("Object scan triggered", "SYS")

    def _on_voice(self):
        self.kernel.trigger_voice()
        self.sidebar.terminal.push("AIILA voice listener activated", "SYS")

    def _toggle_circuit(self):
        self._circuit_active = not self._circuit_active
        self.kernel.app_state['circuit_engine_enabled'] = self._circuit_active
        self.sidebar.btn_circuit.set_active(self._circuit_active)
        msg = f"Circuit mode {'ENABLED' if self._circuit_active else 'DISABLED'}"
        self.sidebar.terminal.push(msg, "SYS")

    def _toggle_draw_mode(self):
        """Toggle between 'default' and 'draw' AR modes."""
        current = self.kernel.app_state.get('ar_mode', 'default')
        new_mode = 'draw' if current != 'draw' else 'default'
        self.kernel.app_state['ar_mode'] = new_mode
        self.sidebar.btn_draw.set_active(new_mode == 'draw')
        self.sidebar.terminal.push(
            f"Wire-draw mode {'ON — pinch a GPIO pin to start' if new_mode == 'draw' else 'OFF'}",
            "SYS"
        )

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
            f"Calibration grid {'ON' if self.calibration_mode else 'OFF'}", "SYS")

    def _save_circuit(self):
        self.kernel.save_circuit("circuit.json")
        self.sidebar.terminal.push("Circuit saved → circuit.json", "INFO")

    def _on_undo(self):
        """[FIX 4] Keyboard/button undo — delegates to engine."""
        self.kernel.undo()
        self.sidebar.terminal.push("Undo performed", "SYS")

    def _open_settings(self):
        self.sidebar.terminal.push("Opening hardware settings…", "SYS")
        self._settings_dlg = SettingsPanel(self, self.kernel.app_state, self.kernel)
        self._settings_dlg.show()

    def shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True

        try:
            self._timer.stop()
        except Exception:
            pass

        self.kernel.running = False

        if self.projector_window is not None:
            try:
                self.projector_window.close()
            except Exception:
                pass
            self.projector_window = None

        try:
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join(timeout=1500/1000)
        except Exception:
            pass

        for queue in (self._in_q, self._out_q):
            try:
                queue.close()
            except Exception:
                pass
            try:
                queue.cancel_join_thread()
            except Exception:
                pass

    def closeEvent(self, event):
        self.shutdown()
        event.accept()
