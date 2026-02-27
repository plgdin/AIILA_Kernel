import sys
import cv2
import numpy as np
import multiprocessing as mp
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QTextEdit, QPushButton, QFrame)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer
from screeninfo import get_monitors
# Ensure your ar_overlay.py logic is accessible
from interface.ar_overlay import assemble_final_os_view
from interface.settings_panel import SettingsPanel

def image_engine_process(input_q, output_q):
    """
    Dedicated Worker Process: Handles heavy math on a separate CPU core.
    Bypasses GIL for buttery smooth performance.
    """
    while True:
        try:
            ar_canvas, raw_frame, state, proj_active, calib_mode = input_q.get()
            
            # 1. Projector Frame Preparation (Raw AR Blueprint)
            p_img_rgb = None
            if proj_active:
                p_rgb = cv2.cvtColor(ar_canvas, cv2.COLOR_BGR2RGB)
                if calib_mode:
                    h, w = p_rgb.shape[:2]
                    for x in range(0, w, 100): cv2.line(p_rgb, (x, 0), (x, h), (0, 255, 0), 1)
                    for y in range(0, h, 100): cv2.line(p_rgb, (0, y), (w, y), (0, 255, 0), 1)
                p_img_rgb = cv2.resize(p_rgb, (1920, 1080), interpolation=cv2.INTER_NEAREST)

            # 2. Dashboard OS Compositing (The Combined View)
            # We use your assemble_final_os_view logic here on the worker thread
            combined_bgr = assemble_final_os_view(ar_canvas, raw_frame)
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
            
            # Dashboard scale (Matching UI geometry)
            ar_res_rgb = cv2.resize(combined_rgb, (1100, 600), interpolation=cv2.INTER_NEAREST)
            
            # 3. Micro Sidebar Preview (Raw Feed)
            cam_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            cam_res_rgb = cv2.resize(cam_rgb, (260, 180), interpolation=cv2.INTER_NEAREST)

            # Send completed RGB snapshots back to UI
            output_q.put((p_img_rgb, ar_res_rgb, cam_res_rgb, state))
        except Exception:
            continue

class UIHub(QMainWindow):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        
        # --- MULTIPROCESSING QUEUES ---
        self.input_q = mp.Queue(maxsize=1)
        self.output_q = mp.Queue(maxsize=1)
        
        self.projector_window = None
        self.calibration_mode = False
        self.circuit_active = False

        # Modern UI Config
        self.setWindowTitle("AIILA OS - PRO KERNEL")
        self.setGeometry(100, 100, 1500, 850)
        self.setStyleSheet("background-color: #0D0D0D; color: white;")

        self._setup_ui()

        # START INDEPENDENT WORKER
        self.worker = mp.Process(target=image_engine_process, args=(self.input_q, self.output_q), daemon=True)
        self.worker.start()
        
        # High-Speed Precise Timer for UI Updates (60 FPS)
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.update_ui_loop)
        self.timer.start(16)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 1. Sidebar Setup
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(320)
        self.sidebar.setStyleSheet("background-color: #151515; border-right: 1px solid #222;")
        sidebar_layout = QVBoxLayout(self.sidebar)

        logo = QLabel("AIILA KERNEL")
        logo.setFont(QFont("Consolas", 22, QFont.Weight.Bold))
        logo.setStyleSheet("color: #00D4FF; margin-top: 20px;")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(logo)

        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("background-color: #050505; color: #00FF9D; font-family: Consolas; border: 1px solid #333;")
        self.terminal.setFixedHeight(180)
        sidebar_layout.addWidget(self.terminal)

        # sidebar Micro-Preview
        self.cam_preview = QLabel()
        self.cam_preview.setFixedSize(260, 180)
        self.cam_preview.setStyleSheet("background-color: black; border: 1px solid #333; border-radius: 10px;")
        sidebar_layout.addWidget(self.cam_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        # Action Buttons
        self.btn_style = "height: 40px; font-weight: bold;"
        
        self.btn_scan = QPushButton("SCAN UNIT [S]")
        self.btn_scan.setStyleSheet(self.btn_style)
        self.btn_scan.clicked.connect(self.on_scan)
        
        self.btn_voice = QPushButton("ACTIVATE JARVIS [V]")
        self.btn_voice.setStyleSheet(self.btn_style + "border: 2px solid #2ecc71; color: #2ecc71;")
        self.btn_voice.clicked.connect(self.on_voice)

        self.btn_circuit = QPushButton("ACTIVATE CIRCUIT")
        self.btn_circuit.setStyleSheet(self.btn_style)
        self.btn_circuit.clicked.connect(self.toggle_circuit)

        self.btn_settings = QPushButton("HARDWARE SETTINGS")
        self.btn_settings.setStyleSheet(self.btn_style)
        self.btn_settings.clicked.connect(self.open_settings)

        self.btn_project = QPushButton("PROJECT TO SCREEN")
        self.btn_project.setStyleSheet(self.btn_style + "background-color: #333;")
        self.btn_project.clicked.connect(self.open_projector_window)

        self.btn_calib = QPushButton("CALIBRATION GRID")
        self.btn_calib.setStyleSheet(self.btn_style + "background-color: #8e44ad;")
        self.btn_calib.clicked.connect(self.toggle_calibration)

        for btn in [self.btn_scan, self.btn_voice, self.btn_circuit, self.btn_settings, self.btn_project, self.btn_calib]:
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()
        main_layout.addWidget(self.sidebar)

        # 2. Main AR Viewport (Hardware Accelerated OS Compositor)
        self.main_viewport = QLabel()
        self.main_viewport.setStyleSheet("background-color: #000; border-radius: 20px; border: 1px solid #1A1A1A;")
        self.main_viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.main_viewport, 1)

    def update_ui_loop(self):
        """Hardware-accelerated polling loop"""
        try:
            data = None
            # Drain queue to stay real-time
            while not self.output_q.empty():
                data = self.output_q.get_nowait()
            
            if data:
                p_rgb, ar_rgb, cam_rgb, state = data
                
                # Update Projector
                if p_rgb is not None and self.projector_window:
                    self.projector_window.update_display(p_rgb)

                # Update Laptop Displays
                self.main_viewport.setPixmap(self.pixmap_from_rgb(ar_rgb))
                self.cam_preview.setPixmap(self.pixmap_from_rgb(cam_rgb))

                # Terminal Updates
                if state.get('voice_feedback'):
                    self.terminal.append(f"> {state['voice_feedback']}")
                    state['voice_feedback'] = ""
        except Exception:
            pass

    def pixmap_from_rgb(self, rgb_img):
        """Direct GPU upload for snapshots"""
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        # CRITICAL FIX: .copy() creates a unique memory snapshot
        # This stops the black-screen bug by protecting the buffer from overwrites.
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(q_img)

    def update_display(self, ar_canvas, raw_frame, state):
        """Kernel callback: feeds the separate worker process"""
        if not self.input_q.full():
            proj_active = self.projector_window is not None
            self.input_q.put((ar_canvas, raw_frame, state, proj_active, self.calibration_mode))

    def open_projector_window(self):
        if self.projector_window is None:
            monitors = get_monitors()
            if len(monitors) < 2:
                self.terminal.append("> ERROR: HDMI NOT DETECTED")
                return
            proj = monitors[1]
            self.projector_window = ProjectorWindow(proj.x, proj.y)
            self.projector_window.show()
            self.terminal.append("> PROJECTION ACTIVE")
        else:
            self.projector_window.close()
            self.projector_window = None

    def toggle_calibration(self):
        self.calibration_mode = not self.calibration_mode
        self.terminal.append(f"> GRID {'ENABLED' if self.calibration_mode else 'DISABLED'}")

    def toggle_circuit(self):
        self.circuit_active = not self.circuit_active
        self.kernel.app_state['circuit_engine_enabled'] = self.circuit_active
        color = "#e67e22" if self.circuit_active else "#34495e"
        self.btn_circuit.setStyleSheet(f"background-color: {color}; font-weight: bold;")
        self.btn_circuit.setText("CIRCUIT: ARMED" if self.circuit_active else "ACTIVATE CIRCUIT")

    def open_settings(self):
        self.settings_dlg = SettingsPanel(self, self.kernel.app_state, self.kernel)
        self.settings_dlg.show()

    def on_scan(self): self.kernel.pending_scan = True
    def on_voice(self): self.kernel.trigger_voice()

class ProjectorWindow(QWidget):
    """Hardware-accelerated borderless HDMI target"""
    def __init__(self, x, y):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(x, y, 1920, 1080)
        self.setStyleSheet("background-color: black;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.display_label = QLabel()
        layout.addWidget(self.display_label)

    def update_display(self, rgb_array):
        h, w, ch = rgb_array.shape
        # Ensure HDMI output also has its own memory snapshot
        q_img = QImage(rgb_array.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.display_label.setPixmap(QPixmap.fromImage(q_img))

    def mouseDoubleClickEvent(self, e): 
        self.close()