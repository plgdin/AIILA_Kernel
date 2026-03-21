import speech_recognition as sr
import sounddevice as sd

from PyQt6.QtWidgets import (
    QComboBox, QDialog, QGridLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout, QKeySequenceEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QKeySequence

from core.app_defaults import DEFAULT_KEYBINDS


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

        self._mic_devices = self._get_microphone_devices()
        self._speaker_devices = [
            (idx, dev['name'])
            for idx, dev in enumerate(sd.query_devices())
            if dev['max_output_channels'] > 0
        ]
        self._keybind_edits: dict[str, QKeySequenceEdit] = {}

        self.setWindowTitle("AIILA Hardware Config")
        self.setFixedSize(480, 760)
        self.setStyleSheet("background-color: #121212; color: white;")
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint
        )

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setSpacing(15)

        header = QLabel("HARDWARE + KEYBIND CONTROL CENTER")
        header.setFont(QFont("Consolas", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00D4FF; margin-bottom: 10px;")
        self.main_layout.addWidget(header)

        self.add_section_header("SELECT CAMERA")
        self.cam_menu = QComboBox()
        self.cam_menu.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.cam_menu.setCurrentText(f"Camera {state['camera_index']}")
        self.cam_menu.currentTextChanged.connect(self.change_camera)
        self.apply_combo_style(self.cam_menu)
        self.main_layout.addWidget(self.cam_menu)

        self.add_section_header("SELECT MICROPHONE")
        self.mic_menu = QComboBox()
        self.mic_menu.addItems([name for _, name in self._mic_devices])
        if state.get('mic_name') in [name for _, name in self._mic_devices]:
            self.mic_menu.setCurrentText(state['mic_name'])
        self.mic_menu.currentTextChanged.connect(self.change_mic)
        self.apply_combo_style(self.mic_menu)
        self.main_layout.addWidget(self.mic_menu)

        self.add_section_header("SELECT SPEAKER")
        self.spk_menu = QComboBox()
        self.spk_menu.addItems([name for _, name in self._speaker_devices])
        if state.get('speaker_name') in [name for _, name in self._speaker_devices]:
            self.spk_menu.setCurrentText(state['speaker_name'])
        self.spk_menu.currentTextChanged.connect(self.change_speaker)
        self.apply_combo_style(self.spk_menu)
        self.main_layout.addWidget(self.spk_menu)

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

        hint = QLabel(
            "Shortcuts are active in the main AIILA window. Keep each keybind unique."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px; margin-top: 4px;")
        self.main_layout.addWidget(hint)

        self.main_layout.addStretch()

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
        self.close_btn.clicked.connect(self.apply_and_close)
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

    def _get_microphone_devices(self):
        try:
            names = sr.Microphone.list_microphone_names()
            if names:
                return list(enumerate(names))
        except Exception:
            pass

        devices = [
            (idx, dev['name'])
            for idx, dev in enumerate(sd.query_devices())
            if dev['max_input_channels'] > 0
        ]
        return devices or [(None, "No Microphone Detected")]

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

        self.state_ref['keybinds'] = updated_keybinds
        self.state_ref['voice_feedback'] = "Keybinds Updated"

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

    def change_mic(self, choice):
        for index, name in self._mic_devices:
            if name == choice:
                self.state_ref['mic_index'] = index
                self.state_ref['mic_name'] = choice
                self.state_ref['voice_feedback'] = "Microphone Updated"
                break

    def change_speaker(self, choice):
        for index, name in self._speaker_devices:
            if name == choice:
                self.state_ref['speaker_index'] = index
                self.state_ref['speaker_name'] = choice
                self.state_ref['voice_feedback'] = "Speaker Updated"
                break
