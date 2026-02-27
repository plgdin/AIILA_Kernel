import cv2
import numpy as np
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.vision_engine import scan_object
from core.voice_engine import (
    listen_and_process_command, 
    WORKING_MIC_INDEX, WORKING_MIC_NAME, 
    SPEAKER_NAME, SPEAKER_INDEX
)
from core.circuit_engine import CircuitEngine

class AIILAKernel:
    def __init__(self):
        self.gui_callback = None # Bridge to UIHub.update_display
        self.running = True
        self.restart_camera = False 
        
        self.app_state = {
            'active_category': None, 'active_model': None,
            'current_layer_view': 1, 'is_listening': False,
            'voice_feedback': "", 'dynamic_ar_text': "",
            'selected_tool': 'resistor', 'is_pinching': False,
            'camera_index': 0, 'mic_index': WORKING_MIC_INDEX,
            'mic_name': WORKING_MIC_NAME, 'speaker_index': SPEAKER_INDEX,
            'speaker_name': SPEAKER_NAME, 'circuit_engine_enabled': False
        }

        self.circuit_engine = CircuitEngine()
        self.pending_scan = False

    def run(self):
        """Main Logic Loop: Runs in a background thread"""
        
        # Detector initialization inside run() avoids GIL deadlocks
        model_path = 'assets/hand_landmarker.task'
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU 
        )
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            running_mode=vision.RunningMode.VIDEO, 
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = vision.HandLandmarker.create_from_options(options)
        
        cap = cv2.VideoCapture(self.app_state['camera_index'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_timestamp_ms = 0

        while self.running:
            if self.restart_camera:
                cap.release()
                cap = cv2.VideoCapture(self.app_state['camera_index'])
                self.restart_camera = False

            ret, frame = cap.read()
            if not ret: continue

            # Create the AR workspace
            ar_canvas = np.zeros((700, 1000, 3), dtype=np.uint8)
            ar_canvas[:] = (15, 15, 15) 

            # Process Gestures
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms += 33 
            result = detector.detect_for_video(mp_image, frame_timestamp_ms)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                index_tip, thumb_tip = hand[8], hand[4]
                h, w, _ = frame.shape
                
                # Visual feedback on raw frame
                cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 10, (255, 0, 0), -1) 
                cv2.circle(frame, (int(index_tip.x * w), int(index_tip.y * h)), 12, (255, 255, 255), 1)

                if self.app_state['circuit_engine_enabled']:
                    dist = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
                    if dist < 0.05:
                        if not self.app_state['is_pinching']:
                            cursor_x, cursor_y = int(index_tip.x * 1000), int(index_tip.y * 700)
                            self.circuit_engine.add_component(self.app_state['selected_tool'], cursor_x, cursor_y)
                            self.app_state['is_pinching'] = True
                    else: 
                        self.app_state['is_pinching'] = False

            if self.pending_scan:
                self.perform_scan(frame)
                self.pending_scan = False

            self.circuit_engine.draw_components(ar_canvas)

            # --- CRITICAL FIX FOR LIVE PREVIEW ---
            # We call the bridge function set in main.py (kernel.gui_callback = window.update_display)
            # This pushes the frames into the Multiprocessing Queue for the UI to draw.
            if self.gui_callback:
                # .copy() is mandatory to prevent buffer-rewrite flickers
                self.gui_callback(ar_canvas.copy(), frame.copy(), self.app_state.copy())

        cap.release()

    def perform_scan(self, frame):
        self.app_state['voice_feedback'] = "SCANNING..."
        cat, model = scan_object(frame)
        self.app_state['active_category'], self.app_state['active_model'] = cat, model
        self.app_state['voice_feedback'] = f"UNIT: {model}"

    def trigger_voice(self):
        if not self.app_state['is_listening']:
            self.app_state['is_listening'] = True
            threading.Thread(target=listen_and_process_command, args=(self.app_state,), daemon=True).start()