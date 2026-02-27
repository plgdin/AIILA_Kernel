import sys
import speech_recognition as sr
import sounddevice as sd
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QComboBox, 
                             QPushButton, QFrame, QWidget)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont

class SettingsPanel(QDialog):
    def __init__(self, parent, state, kernel):
        super().__init__(parent)
        self.setWindowTitle("AIILA Hardware Config")
        self.setFixedSize(440, 600)
        self.setStyleSheet("background-color: #121212; color: white;")
        
        # Reference to the kernel's app_state
        self.state_ref = state 
        self.kernel = kernel
        
        # Force window to top
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        # Main Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setSpacing(15)

        # Header
        header = QLabel("HARDWARE CONTROL CENTER")
        header.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00D4FF; margin-bottom: 10px;")
        self.main_layout.addWidget(header)

        # --- CAMERA SELECTION ---
        self.add_section_header("SELECT CAMERA")
        self.cam_menu = QComboBox()
        self.cam_menu.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.cam_menu.setCurrentText(f"Camera {state['camera_index']}")
        self.cam_menu.currentTextChanged.connect(self.change_camera)
        self.apply_combo_style(self.cam_menu)
        self.main_layout.addWidget(self.cam_menu)

        # --- MICROPHONE SELECTION ---
        self.add_section_header("SELECT MICROPHONE")
        mics = sr.Microphone.list_microphone_names()
        self.mic_menu = QComboBox()
        self.mic_menu.addItems(mics)
        self.mic_menu.setCurrentText(state['mic_name'])
        self.mic_menu.currentTextChanged.connect(self.change_mic)
        self.apply_combo_style(self.mic_menu)
        self.main_layout.addWidget(self.mic_menu)

        # --- SPEAKER SELECTION ---
        self.add_section_header("SELECT SPEAKER")
        devices = sd.query_devices()
        speakers = [d['name'] for d in devices if d['max_output_channels'] > 0]
        self.spk_menu = QComboBox()
        self.spk_menu.addItems(speakers)
        self.spk_menu.setCurrentText(state['speaker_name'])
        self.spk_menu.currentTextChanged.connect(self.change_speaker)
        self.apply_combo_style(self.spk_menu)
        self.main_layout.addWidget(self.spk_menu)

        self.main_layout.addStretch()

        # Close Button
        self.close_btn = QPushButton("APPLY & CLOSE")
        self.close_btn.setFixedHeight(45)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71; 
                color: black; 
                font-weight: bold; 
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #27ae60; }
        """)
        self.close_btn.clicked.connect(self.accept) # 'accept' closes the QDialog
        self.main_layout.addWidget(self.close_btn)

    def add_section_header(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #888; margin-top: 10px;")
        self.main_layout.addWidget(lbl)

    def apply_combo_style(self, combo):
        combo.setFixedHeight(35)
        combo.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #333;
                border-radius: 5px;
                padding-left: 10px;
                color: white;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                selection-background-color: #00D4FF;
                color: white;
            }
        """)

    # --- LOGIC RETAINED FROM ORIGINAL ---
    def change_camera(self, choice):
        try:
            index = int(choice.split(" ")[1])
            self.state_ref['camera_index'] = index
            self.kernel.restart_camera = True
            self.state_ref['voice_feedback'] = f"Switched to Camera {index}"
        except Exception: pass

    def change_mic(self, choice):
        mics = sr.Microphone.list_microphone_names()
        if choice in mics:
            new_index = mics.index(choice)
            self.state_ref['mic_index'] = new_index
            self.state_ref['mic_name'] = choice
            self.state_ref['voice_feedback'] = "Microphone Updated"

    def change_speaker(self, choice):
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d['name'] == choice:
                self.state_ref['speaker_index'] = i
                self.state_ref['speaker_name'] = choice
                break
        self.state_ref['voice_feedback'] = "Speaker Updated"