import numpy as np
import sounddevice as sd

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QKeySequenceEdit,
)

from core.app_defaults import DEFAULT_KEYBINDS
from core.voice_engine import list_working_microphones, list_working_speakers


KEYBIND_FIELDS = [
    ('scan_unit', 'Scan Unit'),
    ('activate_aiila', 'Activate AIILA'),
    ('circuit_mode', 'Circuit Mode'),
    ('wire_draw_mode', 'Wire Draw Mode'),
    ('run_simulation', 'Run Simulation'),
    ('project_to_screen', 'Project To Screen'),
]


class SettingsPanel(QDialog):
    def __init__(self, parent, state, kernel):
        super().__init__(parent)
        self.parent_hub = parent
        self.state_ref = state
        self.kernel = kernel

        self._mic_devices: list[dict] = []
        self._speaker_devices: list[dict] = []
        self._keybind_edits: dict[str, QKeySequenceEdit] = {}
        self._mic_test_stream = None
        self._mic_level = 0.0
        self._mic_test_active = False

        self._meter_timer = QTimer(self)
        self._meter_timer.setInterval(60)
        self._meter_timer.timeout.connect(self._update_mic_meter)

        self.setWindowTitle("AIILA Hardware Config")
        self.setFixedSize(560, 860)
        self.setStyleSheet("background-color: #121212; color: white;")
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint
        )

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(28, 28, 28, 28)
        self.main_layout.setSpacing(14)

        header = QLabel("HARDWARE + KEYBIND CONTROL CENTER")
        header.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00D4FF; margin-bottom: 6px;")
        self.main_layout.addWidget(header)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(10)

        title = QLabel("WORKING DEVICES ONLY")
        title.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        title.setStyleSheet("color: #888;")
        top_row.addWidget(title)
        top_row.addStretch()

        self.refresh_btn = QPushButton("REFRESH AUDIO")
        self.refresh_btn.setFixedHeight(34)
        self.refresh_btn.clicked.connect(self.refresh_audio_devices)
        self.apply_secondary_button_style(self.refresh_btn)
        top_row.addWidget(self.refresh_btn)
        self.main_layout.addLayout(top_row)

        self.add_section_header("SELECT CAMERA")
        self.cam_menu = QComboBox()
        self.cam_menu.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.cam_menu.setCurrentText(f"Camera {state['camera_index']}")
        self.cam_menu.currentTextChanged.connect(self.change_camera)
        self.apply_combo_style(self.cam_menu)
        self.main_layout.addWidget(self.cam_menu)

        self.add_section_header("VOICE DEVICES")
        audio_grid = QGridLayout()
        audio_grid.setContentsMargins(0, 0, 0, 0)
        audio_grid.setHorizontalSpacing(12)
        audio_grid.setVerticalSpacing(10)

        mic_label = QLabel("MICROPHONE")
        mic_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        mic_label.setStyleSheet("color: #A0A0A0;")
        audio_grid.addWidget(mic_label, 0, 0)

        speaker_label = QLabel("SPEAKER")
        speaker_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        speaker_label.setStyleSheet("color: #A0A0A0;")
        audio_grid.addWidget(speaker_label, 0, 1)

        self.mic_menu = QComboBox()
        self.mic_menu.currentIndexChanged.connect(self.change_mic)
        self.apply_combo_style(self.mic_menu)
        audio_grid.addWidget(self.mic_menu, 1, 0)

        self.spk_menu = QComboBox()
        self.spk_menu.currentIndexChanged.connect(self.change_speaker)
        self.apply_combo_style(self.spk_menu)
        audio_grid.addWidget(self.spk_menu, 1, 1)

        self.main_layout.addLayout(audio_grid)

        hint = QLabel(
            "Only verified microphones and speakers are listed here. Disabled and duplicate audio endpoints are filtered out."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #6f6f6f; font-size: 11px;")
        self.main_layout.addWidget(hint)

        self.add_section_header("MIC TESTER")
        mic_test_row = QHBoxLayout()
        mic_test_row.setContentsMargins(0, 0, 0, 0)
        mic_test_row.setSpacing(10)

        self.mic_test_btn = QPushButton("START MIC TEST")
        self.mic_test_btn.setFixedHeight(40)
        self.mic_test_btn.clicked.connect(self.toggle_mic_test)
        self.apply_action_button_style(self.mic_test_btn)
        mic_test_row.addWidget(self.mic_test_btn)

        self.mic_test_status = QLabel("Select a microphone and speak to see the level meter.")
        self.mic_test_status.setWordWrap(True)
        self.mic_test_status.setStyleSheet("color: #7f7f7f;")
        mic_test_row.addWidget(self.mic_test_status, 1)

        self.main_layout.addLayout(mic_test_row)

        self.mic_meter = QProgressBar()
        self.mic_meter.setRange(0, 100)
        self.mic_meter.setValue(0)
        self.mic_meter.setTextVisible(False)
        self.mic_meter.setFixedHeight(18)
        self.apply_meter_style(self.mic_meter)
        self.main_layout.addWidget(self.mic_meter)

        self.mic_meter_value = QLabel("MIC LEVEL: 0%")
        self.mic_meter_value.setStyleSheet("color: #00D4FF; font-size: 11px;")
        self.main_layout.addWidget(self.mic_meter_value)

        self.add_section_header("DEFAULT KEYBINDS")
        keybind_grid = QGridLayout()
        keybind_grid.setContentsMargins(0, 0, 0, 0)
        keybind_grid.setHorizontalSpacing(12)
        keybind_grid.setVerticalSpacing(10)

        keybinds = self.state_ref.setdefault('keybinds', DEFAULT_KEYBINDS.copy())
        for row, (action, label) in enumerate(KEYBIND_FIELDS):
            title = QLabel(label.upper())
            title.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            title.setStyleSheet("color: #A0A0A0;")

            edit = QKeySequenceEdit(QKeySequence(keybinds.get(action, DEFAULT_KEYBINDS[action])))
            self.apply_keybind_style(edit)
            self._keybind_edits[action] = edit

            keybind_grid.addWidget(title, row, 0)
            keybind_grid.addWidget(edit, row, 1)

        self.main_layout.addLayout(keybind_grid)

        keybind_hint = QLabel(
            "Shortcuts are active in the main AIILA window. Keep each keybind unique."
        )
        keybind_hint.setWordWrap(True)
        keybind_hint.setStyleSheet("color: #888; font-size: 11px; margin-top: 4px;")
        self.main_layout.addWidget(keybind_hint)

        self.main_layout.addStretch()

        self.close_btn = QPushButton("APPLY & CLOSE")
        self.close_btn.setFixedHeight(46)
        self.close_btn.clicked.connect(self.apply_and_close)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: black;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover { background-color: #27ae60; }
        """)
        self.main_layout.addWidget(self.close_btn)

        self.refresh_audio_devices()

    def add_section_header(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #888; margin-top: 8px;")
        self.main_layout.addWidget(lbl)

    def apply_combo_style(self, combo):
        combo.setFixedHeight(38)
        combo.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #333;
                border-radius: 5px;
                padding-left: 10px;
                color: white;
            }
            QComboBox:disabled {
                color: #666;
                border-color: #222;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                selection-background-color: #00D4FF;
                color: white;
            }
        """)

    def apply_keybind_style(self, edit):
        edit.setFixedHeight(35)
        edit.setStyleSheet("""
            QKeySequenceEdit {
                background-color: #1e1e1e;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 6px 10px;
                color: white;
            }
        """)

    def apply_action_button_style(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: #00D4FF;
                color: black;
                font-weight: bold;
                border-radius: 8px;
                padding: 0 14px;
            }
            QPushButton:hover { background-color: #34ddff; }
        """)

    def apply_secondary_button_style(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: #1e1e1e;
                color: #00D4FF;
                border: 1px solid #00D4FF;
                border-radius: 7px;
                padding: 0 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #172026; }
        """)

    def apply_meter_style(self, meter):
        meter.setStyleSheet("""
            QProgressBar {
                background-color: #171717;
                border: 1px solid #252525;
                border-radius: 8px;
            }
            QProgressBar::chunk {
                background-color: #00D4FF;
                border-radius: 7px;
            }
        """)

    def _device_label(self, device: dict) -> str:
        name = device.get('name', 'Unknown Device')
        if device.get('is_default'):
            return f"Windows Default ({name})"
        return name

    def _populate_device_menu(self, combo, devices, selected_name):
        combo.blockSignals(True)
        combo.clear()

        if not devices:
            combo.addItem("No Working Device Detected", None)
            combo.setEnabled(False)
            combo.blockSignals(False)
            return

        combo.setEnabled(True)
        selected_index = 0
        wanted = (selected_name or "").strip().casefold()
        for idx, device in enumerate(devices):
            combo.addItem(self._device_label(device), device)
            if wanted and device.get('name', '').casefold() == wanted:
                selected_index = idx

        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def refresh_audio_devices(self):
        current_mic_name = self.state_ref.get('mic_name', '')
        current_speaker_name = self.state_ref.get('speaker_name', '')

        was_testing = self._mic_test_active
        self._stop_mic_test(status="Mic test stopped for device refresh.")

        self._mic_devices = list_working_microphones()
        self._speaker_devices = list_working_speakers()

        self._populate_device_menu(self.mic_menu, self._mic_devices, current_mic_name)
        self._populate_device_menu(self.spk_menu, self._speaker_devices, current_speaker_name)

        self.change_mic()
        self.change_speaker()
        self.state_ref['voice_feedback'] = "Audio Devices Refreshed"

        if was_testing and self.mic_menu.currentData() is not None:
            self._start_mic_test()

    def _sequence_text(self, edit: QKeySequenceEdit) -> str:
        return edit.keySequence().toString(QKeySequence.SequenceFormat.PortableText).strip()

    def apply_and_close(self):
        updated_keybinds = {}
        seen = {'ctrl+z': 'Undo'}

        for action, label in KEYBIND_FIELDS:
            sequence = self._sequence_text(self._keybind_edits[action])
            if not sequence:
                QMessageBox.warning(
                    self,
                    "Missing Keybind",
                    f"Set a keybind for {label} before closing.",
                )
                return

            normalized = sequence.casefold()
            if normalized in seen:
                QMessageBox.warning(
                    self,
                    "Duplicate Keybind",
                    f"{label} conflicts with {seen[normalized]}. Use a unique shortcut.",
                )
                return

            seen[normalized] = label
            updated_keybinds[action] = sequence

        self._stop_mic_test(status="")
        self.state_ref['keybinds'] = updated_keybinds
        self.state_ref['voice_feedback'] = "Hardware + Keybinds Updated"

        if hasattr(self.parent_hub, 'refresh_shortcuts'):
            self.parent_hub.refresh_shortcuts()

        self.accept()

    def change_camera(self, choice):
        try:
            index = int(choice.split(" ")[1])
            self.state_ref['camera_index'] = index
            self.kernel.restart_camera = True
            self.state_ref['voice_feedback'] = f"Switched to Camera {index}"
        except Exception:
            pass

    def change_mic(self, _index=None):
        device = self.mic_menu.currentData()
        if not isinstance(device, dict):
            self.state_ref['mic_index'] = None
            self.state_ref['mic_name'] = "No Microphone Detected"
            return

        self.state_ref['mic_index'] = device.get('index')
        self.state_ref['mic_name'] = device.get('name', '')
        self.state_ref['voice_feedback'] = "Microphone Updated"

        if self._mic_test_active:
            self._start_mic_test(restart=True)

    def change_speaker(self, _index=None):
        device = self.spk_menu.currentData()
        if not isinstance(device, dict):
            self.state_ref['speaker_index'] = None
            self.state_ref['speaker_name'] = "No Speaker Detected"
            return

        self.state_ref['speaker_index'] = device.get('index')
        self.state_ref['speaker_name'] = device.get('name', '')
        self.state_ref['voice_feedback'] = "Speaker Updated"

    def toggle_mic_test(self):
        if self._mic_test_active:
            self._stop_mic_test(status="Mic test stopped.")
        else:
            self._start_mic_test()

    def _start_mic_test(self, restart: bool = False):
        device = self.mic_menu.currentData()
        if not isinstance(device, dict):
            self.mic_test_status.setText("No working microphone is available for testing.")
            return

        self._stop_mic_test(status="" if restart else None)

        stream_device = device.get('sd_index')
        if stream_device is None:
            stream_device = device.get('name')

        try:
            device_info = sd.query_devices(stream_device)
            sample_rate = int(device_info.get('default_samplerate') or 44100)
            self._mic_test_stream = sd.InputStream(
                device=stream_device,
                channels=1,
                samplerate=sample_rate,
                callback=self._mic_test_callback,
            )
            self._mic_test_stream.start()
        except Exception as exc:
            self._mic_test_stream = None
            self._mic_test_active = False
            self.mic_meter.setValue(0)
            self.mic_meter_value.setText("MIC LEVEL: 0%")
            self.mic_test_status.setText(f"Mic test unavailable: {exc}")
            self.mic_test_btn.setText("START MIC TEST")
            return

        self._mic_level = 0.0
        self._mic_test_active = True
        self._meter_timer.start()
        self.mic_test_btn.setText("STOP MIC TEST")
        self.mic_test_status.setText(f"Testing {device.get('name', 'microphone')}...")

    def _stop_mic_test(self, status=None):
        self._meter_timer.stop()
        self._mic_test_active = False

        if self._mic_test_stream is not None:
            try:
                self._mic_test_stream.stop()
            except Exception:
                pass
            try:
                self._mic_test_stream.close()
            except Exception:
                pass
            self._mic_test_stream = None

        self._mic_level = 0.0
        self.mic_meter.setValue(0)
        self.mic_meter_value.setText("MIC LEVEL: 0%")
        self.mic_test_btn.setText("START MIC TEST")

        if status is not None:
            self.mic_test_status.setText(status)

    def _mic_test_callback(self, indata, _frames, _time_info, status):
        if status:
            return
        samples = np.asarray(indata, dtype=np.float32)
        if samples.size == 0:
            return

        rms = float(np.sqrt(np.mean(np.square(samples))))
        boosted_level = min(1.0, rms * 12.0)
        self._mic_level = max(self._mic_level * 0.78, boosted_level)

    def _update_mic_meter(self):
        level = max(0.0, min(1.0, self._mic_level))
        value = int(level * 100)
        self.mic_meter.setValue(value)
        self.mic_meter_value.setText(f"MIC LEVEL: {value}%")
        self._mic_level *= 0.82

    def closeEvent(self, event):
        self._stop_mic_test(status="")
        super().closeEvent(event)
