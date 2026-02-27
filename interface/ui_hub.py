import sys
import cv2
import numpy as np
import multiprocessing as mp
import time
import math
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QTextEdit, QPushButton, QFrame)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer, QPoint
from screeninfo import get_monitors

# Ensure your ar_overlay.py logic is accessible
from interface.ar_overlay import assemble_final_os_view
from interface.settings_panel import SettingsPanel

def image_engine_process(input_q, output_q):
    """
    Dedicated Worker Process: Handles heavy math on a separate CPU core.
    """
    while True:
        try:
            ar_canvas, raw_frame, state, proj_active, calib_mode = input_q.get()
            
            p_img_rgb = None
            if proj_active:
                p_rgb = cv2.cvtColor(ar_canvas, cv2.COLOR_BGR2RGB)
                if calib_mode:
                    h, w = p_rgb.shape[:2]
                    for x in range(0, w, 100): cv2.line(p_rgb, (x, 0), (x, h), (0, 255, 0), 1)
                    for y in range(0, h, 100): cv2.line(p_rgb, (0, y), (w, y), (0, 255, 0), 1)
                p_img_rgb = p_rgb

            combined_bgr = assemble_final_os_view(ar_canvas, raw_frame)
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
            ar_res_rgb = cv2.resize(combined_rgb, (1100, 600), interpolation=cv2.INTER_NEAREST)
            
            cam_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            cam_res_rgb = cv2.resize(cam_rgb, (260, 180), interpolation=cv2.INTER_NEAREST)

            output_q.put((p_img_rgb, ar_res_rgb, cam_res_rgb, state))
        except Exception:
            continue

class UIHub(QMainWindow):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.input_q = mp.Queue(maxsize=1)
        self.output_q = mp.Queue(maxsize=1)
        self.projector_window = None
        self.calibration_mode = False
        self.circuit_active = False

        self.setWindowTitle("AIILA OS - PRO KERNEL")
        self.setGeometry(100, 100, 1500, 850)
        self.setStyleSheet("background-color: #0D0D0D; color: white;")

        self._setup_ui()
        self.worker = mp.Process(target=image_engine_process, args=(self.input_q, self.output_q), daemon=True)
        self.worker.start()
        
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.update_ui_loop)
        self.timer.start(16)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

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
        sidebar_layout.addWidget(self.terminal)

        self.cam_preview = QLabel()
        self.cam_preview.setFixedSize(260, 180)
        self.cam_preview.setStyleSheet("background-color: black; border: 1px solid #333; border-radius: 10px;")
        sidebar_layout.addWidget(self.cam_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        self.btn_style = "height: 40px; font-weight: bold;"
        
        buttons = [
            ("SCAN UNIT [S]", self.on_scan),
            ("ACTIVATE JARVIS [V]", self.on_voice),
            ("ACTIVATE CIRCUIT", self.toggle_circuit),
            ("HARDWARE SETTINGS", self.open_settings),
            ("PROJECT TO SCREEN", self.open_projector_window),
            ("CALIBRATION GRID", self.toggle_calibration)
        ]

        for text, callback in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(self.btn_style)
            btn.clicked.connect(callback)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()
        main_layout.addWidget(self.sidebar)

        self.main_viewport = QLabel()
        self.main_viewport.setStyleSheet("background-color: #000; border-radius: 20px; border: 1px solid #1A1A1A;")
        self.main_viewport.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.main_viewport, 1)

    def update_ui_loop(self):
        try:
            data = None
            while not self.output_q.empty():
                data = self.output_q.get_nowait()
            
            if data:
                p_rgb, ar_rgb, cam_rgb, state = data
                if p_rgb is not None and self.projector_window:
                    self.projector_window.update_display(p_rgb)
                self.main_viewport.setPixmap(self.pixmap_from_rgb(ar_rgb))
                self.cam_preview.setPixmap(self.pixmap_from_rgb(cam_rgb))
                if state.get('voice_feedback'):
                    self.terminal.append(f"> {state['voice_feedback']}")
                    state['voice_feedback'] = ""
        except Exception: pass

    def pixmap_from_rgb(self, rgb_img):
        h, w, ch = rgb_img.shape
        q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(q_img)

    def update_display(self, ar_canvas, raw_frame, state):
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
        else:
            self.projector_window.close()
            self.projector_window = None

    def toggle_calibration(self): self.calibration_mode = not self.calibration_mode
    def toggle_circuit(self):
        self.circuit_active = not self.circuit_active
        self.kernel.app_state['circuit_engine_enabled'] = self.circuit_active
    def open_settings(self):
        self.settings_dlg = SettingsPanel(self, self.kernel.app_state, self.kernel)
        self.settings_dlg.show()
    def on_scan(self): self.kernel.pending_scan = True
    def on_voice(self): self.kernel.trigger_voice()

class ProjectorWindow(QWidget):
    def __init__(self, x, y):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setGeometry(x, y, 1920, 1080)
        self.setStyleSheet("background-color: black;")
        
        self.display_label = QLabel(self)
        self.display_label.setGeometry(0, 0, 1920, 1080)
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_time = time.time()
        self.state = "INTRO" 
        self.apps_visible = False
        self.app_anim_start = 0
        self.opacity = 0.0
        
        # Interactive Button Rect [x, y, w, h]
        self.btn_rect = [860, 650, 200, 50] 
        self.apps = ["Circuit Lab", "Vision AI", "Voice CMD", "Settings"]

    def update_display(self, ar_content):
        elapsed = time.time() - self.start_time
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:] = (10, 10, 10)

        # 1. Permanent Intro Text Alignment (Centered)
        text_logo = "AIILA"
        text_tag = "An Engineer's Tool"
        
        (l_w, l_h), _ = cv2.getTextSize(text_logo, cv2.FONT_HERSHEY_DUPLEX, 5, 10)
        (t_w, t_h), _ = cv2.getTextSize(text_tag, cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)
        
        lx, ly = (1920 - l_w) // 2, 500
        tx, ty = (1920 - t_w) // 2, 580

        if self.state == "INTRO":
            self.opacity = min(1.0, elapsed / 2.0)
            if elapsed > 4.0: self.state = "DESKTOP"
        
        c = int(255 * self.opacity)
        cv2.putText(frame, text_logo, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, 5, (c,c,c), 10, cv2.LINE_AA)
        cv2.putText(frame, text_tag, (tx, ty), cv2.FONT_HERSHEY_COMPLEX, 1.2, (c,c,c), 2, cv2.LINE_AA)

        # 2. Desktop Elements
        if self.state == "DESKTOP":
            d_op = min(1.0, (elapsed - 4.0) / 1.5)
            dc = int(255 * d_op)
            
            # Faded Taskbar
            cv2.rectangle(frame, (0, 1020), (1920, 1080), (int(25*d_op), int(25*d_op), int(25*d_op)), -1)
            cv2.line(frame, (0, 1020), (1920, 1020), (int(50*d_op), int(50*d_op), int(50*d_op)), 1)
            cv2.putText(frame, "SYSTEM KERNEL READY", (50, 1055), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (dc, dc, dc), 1)

            # Perfect Button Alignment
            bx, by, bw, bh = self.btn_rect
            cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (dc, dc, dc), 1)
            
            btn_txt = "OPEN APPS"
            (tw, th), _ = cv2.getTextSize(btn_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Center text inside the rect: rect_x + (rect_w - text_w)//2
            cv2.putText(frame, btn_txt, (bx + (bw - tw)//2, by + (bh + th)//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (dc, dc, dc), 2, cv2.LINE_AA)

            # 3. Animated Circular App Menu
            if self.apps_visible or (time.time() - self.app_anim_start < 0.6):
                anim_elapsed = time.time() - self.app_anim_start
                # Expansion factor (0.0 to 1.0)
                t = min(1.0, anim_elapsed / 0.5) 
                
                # If closing, reverse the factor
                if not self.apps_visible: t = 1.0 - t
                
                radius = int(350 * t)
                center_x, center_y = 1920 // 2, 500
                start_x, start_y = bx + bw//2, by + bh//2 # Start from button center
                
                for i, app_name in enumerate(self.apps):
                    angle = -180 + (i * (360 / max(len(self.apps), 4)))
                    rad_val = math.radians(angle)
                    
                    # Target circular position
                    target_x = int(center_x + 350 * math.cos(rad_val))
                    target_y = int(center_y + 350 * math.sin(rad_val))
                    
                    # Lerp from button center to target
                    ax = int(start_x + (target_x - start_x) * t)
                    ay = int(start_y + (target_y - start_y) * t)
                    
                    # Draw with fading alpha based on expansion
                    app_c = int(255 * t)
                    cv2.circle(frame, (ax, ay), int(50*t), (0, 212, int(255*t)), 2)
                    if t > 0.8:
                        cv2.putText(frame, app_name, (ax-45, ay+75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (app_c, app_c, app_c), 1)

        self._render(frame)

    def mousePressEvent(self, event):
        if self.state == "DESKTOP":
            x, y = event.pos().x(), event.pos().y()
            bx, by, bw, bh = self.btn_rect
            if bx <= x <= bx+bw and by <= y <= by+bh:
                self.apps_visible = not self.apps_visible
                self.app_anim_start = time.time() # Trigger animation

    def _render(self, frame):
        h, w, ch = frame.shape
        q_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.display_label.setPixmap(QPixmap.fromImage(q_img))

    def mouseDoubleClickEvent(self, e): self.close()