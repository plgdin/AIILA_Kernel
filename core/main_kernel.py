"""
kernel.py  ·  AIILAKernel
==========================
Main logic + gesture routing for the AIILA AR system.

Gesture→Action map
───────────────────
  pinch   grab/drag/release  →  place / drag circuit component
  swipe   left/right/up/down →  page navigation
  crumple                    →  delete component at cursor
  throw                      →  project / cast screen
  rotate  cw/ccw             →  rotate selected component
  dwell                      →  select / confirm at cursor
  peace                      →  toggle AR mode

FIXES vs original
──────────────────
  1. Camera feed is now rendered into the final composite (was missing entirely).
  2. CircuitEngine API updated:
       add_component(type_id, wx, wy)       — world coords, grid-snapped
       drag_component_to(wx, wy)            — moves selected comp
       drop_component()                     — clears drag state
       delete_component_at(wx, wy)          — hit-test + remove
       select_component_at(wx, wy)          — hit-test + select
       rotate_selected(delta_deg)           — rotate by delta
       tick_simulation()                    — call every frame when sim running
       render(canvas)                       — draws everything onto canvas
  3. ar_canvas now starts as a live camera frame (resized to AR resolution)
     so the camera is always visible underneath circuit components.
  4. gui_callback receives the composited frame — camera + AR overlay + circuit.
"""

import cv2
import numpy as np
import threading
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.vision_engine  import scan_object
from core.voice_engine   import (
    listen_and_process_command,
    WORKING_MIC_INDEX, WORKING_MIC_NAME,
    SPEAKER_NAME, SPEAKER_INDEX,
)
from core.circuit_engine import CircuitEngine
from core.gesture_engine import GestureEngine, HAND_CONNECTIONS, GestureType, PinchState

# AR canvas size — circuit engine must match this
AR_W, AR_H = 1000, 700


class AIILAKernel:

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(self):
        self.gui_callback   = None
        self.running        = True
        self.restart_camera = False

        self.app_state = {
            # Core
            'active_category':         None,
            'active_model':            None,
            'current_layer_view':      1,
            'is_listening':            False,
            'voice_feedback':          "",
            'dynamic_ar_text':         "",
            # Circuit
            'selected_tool':           'resistor',
            'is_pinching':             False,
            'circuit_engine_enabled':  False,
            # Hardware
            'camera_index':            0,
            'mic_index':               WORKING_MIC_INDEX,
            'mic_name':                WORKING_MIC_NAME,
            'speaker_index':           SPEAKER_INDEX,
            'speaker_name':            SPEAKER_NAME,
            # AR
            'ar_rotation':             0.0,
            'ar_mode':                 'default',
            # Gesture meta
            'gesture_active':          None,
            'gesture_confidence':      0.0,
            'fingers_state':           [False] * 5,
            'dwell_progress':          0.0,
        }

        # Circuit engine — sized to match AR canvas
        self.circuit_engine = CircuitEngine(canvas_w=AR_W, canvas_h=AR_H)

        self.gesture_engine = GestureEngine(
            canvas_w=AR_W, canvas_h=AR_H,
            ema_alpha=0.55, min_confidence=0.45,
        )
        self.pending_scan = False

        # Track which component is being dragged
        self._dragging_id: int | None = None

    # ─────────────────────────────────────────────────────────────────────────
    def run(self):
        """Main loop — call in a daemon thread from main.py."""

        model_path   = 'assets/hand_landmarker.task'
        base_options = python.BaseOptions(
            model_asset_path=model_path,
            delegate=python.BaseOptions.Delegate.CPU,
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(self.app_state['camera_index'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_ts = 0

        while self.running:

            # ── Camera hot-swap ───────────────────────────────────────────────
            if self.restart_camera:
                cap.release()
                cap = cv2.VideoCapture(self.app_state['camera_index'])
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.restart_camera = False

            ret, frame = cap.read()
            if not ret:
                # Camera not ready yet — push a blank frame so UI doesn't hang
                blank = np.zeros((AR_H, AR_W, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera not available", (AR_W//2 - 140, AR_H//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 180), 2)
                if self.gui_callback:
                    self.gui_callback(blank, blank, self.app_state.copy())
                continue

            # ── FIX 1: use camera frame as AR base ───────────────────────────
            # Resize camera frame to AR resolution so it fills the canvas.
            # Circuit components + HUD are drawn on top of this.
            ar_canvas = cv2.resize(frame, (AR_W, AR_H))

            # ── MediaPipe ─────────────────────────────────────────────────────
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts += 33
            result = detector.detect_for_video(mp_img, frame_ts)

            # ── Gesture pipeline ──────────────────────────────────────────────
            if result.hand_landmarks:
                # Draw skeleton on the ar_canvas (already in camera space after resize)
                self._draw_skeleton(ar_canvas, result.hand_landmarks)
                events = self.gesture_engine.update(result, ar_canvas.shape)

                if events:
                    top = events[0]
                    self.app_state['gesture_active']     = top['gesture']
                    self.app_state['gesture_confidence'] = top['confidence']
                    self.app_state['fingers_state']      = top['fingers']

                for ev in events:
                    self._route(ev)
            else:
                self.app_state['gesture_active']     = None
                self.app_state['gesture_confidence'] = 0.0
                self.app_state['fingers_state']      = [False] * 5
                self.app_state['is_pinching']        = False
                self.app_state['dwell_progress']     = 0.0

            # ── Object scan ───────────────────────────────────────────────────
            if self.pending_scan:
                self.perform_scan(frame)
                self.pending_scan = False

            # ── Simulation tick ───────────────────────────────────────────────
            if self.circuit_engine.sim_running:
                self.circuit_engine.tick_simulation()

            # ── FIX 2: render circuit onto the camera-based canvas ────────────
            # render() draws grid + wires + components + HUD on top of whatever
            # is already in ar_canvas (the camera feed).
            if self.app_state['circuit_engine_enabled']:
                self.circuit_engine.render(ar_canvas)
            else:
                # Still draw HUD even when circuit mode is off
                self._draw_hud(ar_canvas)

            # ── Push composite frame to UI ────────────────────────────────────
            if self.gui_callback:
                self.gui_callback(ar_canvas.copy(), frame.copy(), self.app_state.copy())

        cap.release()

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_skeleton(self, frame: np.ndarray, all_hands: list):
        """Render full 21-landmark skeleton scaled to frame dimensions."""
        h, w = frame.shape[:2]
        TIP_IDX = {4, 8, 12, 16, 20}

        for hand in all_hands:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 210, 110), 1, cv2.LINE_AA)

            for i, pt in enumerate(pts):
                if i in TIP_IDX:
                    cv2.circle(frame, pt,  8, (255,  80,  0), -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 10, (255, 255, 255),  1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt,  4, (90, 190, 255), -1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    def _draw_hud(self, canvas: np.ndarray):
        """Gesture-state overlay — used when circuit engine is OFF."""
        gesture = self.app_state.get('gesture_active')
        conf    = self.app_state.get('gesture_confidence', 0.0)
        if gesture:
            label = f"{gesture.upper()}  {conf:.0%}"
            cv2.putText(canvas, label, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 120), 2, cv2.LINE_AA)

        # Finger indicators
        fingers = self.app_state.get('fingers_state', [False] * 5)
        names   = ['T', 'I', 'M', 'R', 'P']
        for i, (ext, name) in enumerate(zip(fingers, names)):
            x = 14 + i * 28
            y = canvas.shape[0] - 14
            colour = (0, 220, 100) if ext else (60, 60, 60)
            cv2.circle(canvas, (x, y), 10, colour, -1, cv2.LINE_AA)
            cv2.putText(canvas, name, (x - 5, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Dwell arc
        prog = self.app_state.get('dwell_progress', 0.0)
        if 0 < prog < 1.0:
            cx, cy = canvas.shape[1] // 2, canvas.shape[0] // 2
            angle  = int(prog * 360)
            cv2.ellipse(canvas, (cx, cy), (40, 40), -90, 0, angle, (80, 200, 255), 3)

        # AR mode badge
        ar_mode = self.app_state.get('ar_mode', 'default')
        cv2.putText(canvas, f"AR:{ar_mode.upper()}", (canvas.shape[1] - 120, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    def _route(self, ev: dict):
        """Route one GestureEvent dict to the correct AIILA action."""
        g      = ev['gesture']
        cursor = ev['cursor']          # (x, y) in AR-canvas pixel space
        data   = ev['data']
        conf   = ev['confidence']

        # Convert cursor pixels → circuit world coords
        wx, wy = self.circuit_engine.to_world(*cursor)

        # ── PINCH ──────────────────────────────────────────────────────────
        if g == GestureType.PINCH:
            state = data.get('state')

            if state == 'grab':
                self.app_state['is_pinching']    = True
                self.app_state['voice_feedback'] = f"✊ GRAB  ({conf:.0%})"
                if self.app_state['circuit_engine_enabled']:
                    # Try to pick up an existing component first
                    hit = self.circuit_engine.hit_test(wx, wy)
                    if hit:
                        self.circuit_engine.selected_id = hit.id
                        self._dragging_id = hit.id
                    else:
                        # Place a new one
                        comp = self.circuit_engine.add_component(
                            self.app_state['selected_tool'], wx, wy)
                        if comp:
                            self._dragging_id = comp.id

            elif state == 'drag':
                self.app_state['is_pinching'] = True
                if self.app_state['circuit_engine_enabled'] and self._dragging_id is not None:
                    c = self.circuit_engine.get_component(self._dragging_id)
                    if c:
                        c.x = self.circuit_engine.snap(wx)
                        c.y = self.circuit_engine.snap(wy)
                dpx = data.get('delta_px', (0, 0))
                self.app_state['dynamic_ar_text'] = f"DRAG Δ{dpx}"

            elif state == 'release':
                self.app_state['is_pinching']    = False
                self.app_state['voice_feedback'] = "✋ RELEASE"
                self._dragging_id = None

            elif state == 'holding':
                self.app_state['is_pinching'] = True

        # ── SWIPE ──────────────────────────────────────────────────────────
        elif g == GestureType.SWIPE:
            direction = data['direction']
            layer     = self.app_state['current_layer_view']
            layer    += {'left': 1, 'right': -1, 'down': 1, 'up': -1}.get(direction, 0)
            layer     = max(1, layer)
            self.app_state['current_layer_view'] = layer
            self.app_state['voice_feedback']     = f"⟵⟶ SWIPE {direction.upper()}  PAGE {layer}"
            self.app_state['dynamic_ar_text']    = f"Page {layer}"

        # ── CRUMPLE ────────────────────────────────────────────────────────
        elif g == GestureType.CRUMPLE:
            self.app_state['voice_feedback'] = "🗑 DELETED"
            if self.app_state['circuit_engine_enabled']:
                hit = self.circuit_engine.hit_test(wx, wy)
                if hit:
                    self.circuit_engine.remove_component(hit.id)

        # ── THROW ──────────────────────────────────────────────────────────
        elif g == GestureType.THROW:
            vel  = data.get('velocity', 0.0)
            dvec = data.get('direction_vec', [0, 0])
            self.app_state['voice_feedback']  = f"📡 PROJECTING  v={vel:.3f}"
            self.app_state['dynamic_ar_text'] = "⟶ SCREEN CAST"
            self._on_project(vel, dvec)

        # ── ROTATE ─────────────────────────────────────────────────────────
        elif g == GestureType.ROTATE:
            delta     = data.get('delta_deg', 0.0)
            direction = data.get('direction', '')
            self.app_state['ar_rotation'] = (self.app_state['ar_rotation'] + delta) % 360
            self.app_state['voice_feedback'] = (
                f"↻ ROTATE {direction.upper()}  {self.app_state['ar_rotation']:.1f}°"
            )
            if self.app_state['circuit_engine_enabled']:
                sel = self.circuit_engine.selected_id
                if sel is not None:
                    self.circuit_engine.rotate_component(sel, int(delta))

        # ── CLAW ROTATE  (one-hand claw twist → rotate selected component) ──
        elif g == GestureType.CLAW_ROTATE:
            delta     = data.get('delta_deg', 0.0)
            direction = data.get('direction', '')
            self.app_state['ar_rotation'] = (self.app_state['ar_rotation'] + delta) % 360
            self.app_state['voice_feedback'] = (
                f"✊↻ CLAW {direction.upper()}  {delta:.1f}°"
            )
            if self.app_state['circuit_engine_enabled']:
                sel = self.circuit_engine.selected_id
                if sel is not None:
                    # Map continuous degrees to 90° snapped steps
                    steps = int(delta / 45.0)
                    if steps != 0:
                        self.circuit_engine.rotate_component(sel, steps * 90)

        # ── DWELL ──────────────────────────────────────────────────────────
        elif g == GestureType.DWELL:
            prog = data.get('progress', 0.0)
            self.app_state['dwell_progress'] = prog
            if prog >= 1.0:
                self.app_state['voice_feedback'] = "● SELECTED"
                if self.app_state['circuit_engine_enabled']:
                    hit = self.circuit_engine.hit_test(wx, wy)
                    if hit:
                        self.circuit_engine.selected_id = hit.id

        # ── PEACE ──────────────────────────────────────────────────────────
        elif g == GestureType.PEACE:
            modes = ['default', 'draw', 'inspect', 'measure']
            cur   = self.app_state.get('ar_mode', 'default')
            nxt   = modes[(modes.index(cur) + 1) % len(modes)] if cur in modes else 'default'
            self.app_state['ar_mode']        = nxt
            self.app_state['voice_feedback'] = f"✌ MODE → {nxt.upper()}"

    # ─────────────────────────────────────────────────────────────────────────
    def _on_project(self, velocity: float, direction_vec: list):
        """Override to integrate with your projector / screen-cast system."""
        pass

    # ─────────────────────────────────────────────────────────────────────────
    def perform_scan(self, frame: np.ndarray):
        self.app_state['voice_feedback'] = "🔍 SCANNING..."
        cat, model = scan_object(frame)
        self.app_state['active_category'] = cat
        self.app_state['active_model']    = model
        self.app_state['voice_feedback']  = f"UNIT: {model}"

    # ─────────────────────────────────────────────────────────────────────────
    def trigger_voice(self):
        if not self.app_state['is_listening']:
            self.app_state['is_listening'] = True
            threading.Thread(
                target=listen_and_process_command,
                args=(self.app_state,),
                daemon=True,
            ).start()

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience helpers called from your GUI / settings panel
    # ─────────────────────────────────────────────────────────────────────────
    def enable_circuit_mode(self, enabled: bool):
        self.app_state['circuit_engine_enabled'] = enabled
        if enabled:
            self.circuit_engine._log("[CIRCUIT] Circuit mode ON")
        else:
            self.circuit_engine._log("[CIRCUIT] Circuit mode OFF")

    def set_selected_tool(self, type_id: str):
        """Change which component type gets placed on next pinch-grab."""
        if type_id in self.circuit_engine.search(type_id) or type_id in __import__('core.circuit_engine', fromlist=['CATALOG']).CATALOG:
            self.app_state['selected_tool'] = type_id

    def start_simulation(self):
        self.circuit_engine.start_simulation()

    def stop_simulation(self):
        self.circuit_engine.stop_simulation()

    def save_circuit(self, path: str = "circuit.json"):
        self.circuit_engine.save(path)

    def load_circuit(self, path: str = "circuit.json"):
        self.circuit_engine.load(path)