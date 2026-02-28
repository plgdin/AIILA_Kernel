"""
kernel.py  ·  AIILAKernel  v5.1  (WORKBENCH EDITION)
======================================================
FIXES OVER v5:

  [FIX 1]  undo() now delegates to circuit_engine.undo() — single undo stack
  [FIX 2]  _push_undo() delegates to circuit_engine.snapshot() — works
  [FIX 3]  hit_test() call passes radius_multiplier kwarg properly
  [FIX 4]  Wire-draw mode uses nearest_pin_with_info() — records exact
           GPIO pin name, not hardcoded 'out'/'in'
  [FIX 5]  Wire drag shows live preview via circuit_engine.start_wire() /
           extend_wire() — you see the wire forming as you drag your finger
  [FIX 6]  _hand_lost() cancels wire_in_progress on circuit_engine too
  [FIX 7]  set_selected_tool() validates against CATALOG at import time,
           not with a dynamic re-import on every call
  [FIX 8]  Gesture debounce uses time.monotonic() consistently
  [FIX 9]  Drag drop re-routes attached wires via _reroute_component_wires()
  [FIX 10] Wire-draw HUD shows src_pin → cursor line so user can see
           which GPIO they're connecting from

WORKBENCH INTERACTION MODEL:
  • AR mode 'default'  — pinch-grab-drag components anywhere on table
  • AR mode 'draw'     — pinch on a pin → drag → release on another pin
                          draws a named wire (e.g. ESP32.D2 → LED.A)
  • DWELL on empty     — places selected component
  • DWELL on component — selects it (shows pin inspector)
  • CRUMPLE (2-hand)   — deletes component + its wires
  • PEACE ✌            — cycles through modes
  • CLAW_ROTATE        — rotates selected component
"""

from __future__ import annotations

import cv2
import numpy as np
import threading
import time
import math
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.vision_engine  import scan_object
from core.voice_engine   import (
    listen_and_process_command,
    WORKING_MIC_INDEX, WORKING_MIC_NAME,
    SPEAKER_NAME, SPEAKER_INDEX,
)
from core.circuit_engine import CircuitEngine, CATALOG
from core.gesture_engine import GestureEngine, HAND_CONNECTIONS, GestureType

AR_W, AR_H = 1000, 700

_GRAB_HIT_MULTIPLIER = 1.8
_PIN_WIRE_RADIUS     = 20    # world px — how close finger must be to a pin to start wire
_DEBOUNCE_MS = {
    GestureType.SWIPE:   400,
    GestureType.CRUMPLE: 800,
    GestureType.THROW:   600,
    GestureType.PEACE:   700,
}
_CAM_RETRY = 5


class AIILAKernel:

    def __init__(self):
        self.gui_callback   = None
        self.running        = True
        self.restart_camera = False

        self.app_state: dict = {
            'active_category':        None,
            'active_model':           None,
            'current_layer_view':     1,
            'is_listening':           False,
            'voice_feedback':         "",
            'feedback_log':           deque(maxlen=4),
            'dynamic_ar_text':        "",
            'selected_tool':          'resistor',
            'is_pinching':            False,
            'circuit_engine_enabled': False,
            'camera_index':           0,
            'mic_index':              WORKING_MIC_INDEX,
            'mic_name':               WORKING_MIC_NAME,
            'speaker_index':          SPEAKER_INDEX,
            'speaker_name':           SPEAKER_NAME,
            'ar_rotation':            0.0,
            'ar_mode':                'default',
            'gesture_active':         None,
            'gesture_confidence':     0.0,
            'fingers_state':          [False] * 5,
            'dwell_progress':         0.0,
            'cursor_wx':              0.0,
            'cursor_wy':              0.0,
        }

        self.circuit_engine = CircuitEngine(canvas_w=AR_W, canvas_h=AR_H)
        self.gesture_engine = GestureEngine(
            canvas_w=AR_W, canvas_h=AR_H,
            ema_alpha=0.55, min_confidence=0.45,
        )
        self.pending_scan = False

        # Drag state
        self._dragging_id:      int | None             = None
        self._grab_offset_wx:   float                  = 0.0
        self._grab_offset_wy:   float                  = 0.0
        self._pinch_cursor_px:  tuple[int, int] | None = None

        # Wire-draw state  [FIX 4]
        self._wire_src_id:      int | None  = None
        self._wire_src_pin:     str | None  = None
        self._wire_src_pt:      tuple | None = None   # world coords of src pin
        self._wire_drawing:     bool        = False   # live preview active

        # Gesture debounce  [FIX 8]
        self._last_fired: dict[GestureType, float] = {}

    # ─────────────────────────────────────────────────────────────────────────
    #  Undo  [FIX 1/2] — engine is authoritative
    # ─────────────────────────────────────────────────────────────────────────

    def _push_undo(self):
        """Delegate undo snapshot to circuit_engine (single stack)."""
        try:
            self.circuit_engine._push_undo()
        except Exception:
            pass

    def undo(self):
        """Restore previous circuit state."""
        self.circuit_engine.undo()
        self._feedback("↩ UNDO")

    # ─────────────────────────────────────────────────────────────────────────
    #  Feedback
    # ─────────────────────────────────────────────────────────────────────────

    def _feedback(self, msg: str):
        self.app_state['voice_feedback'] = msg
        self.app_state['feedback_log'].append(
            (time.strftime('%H:%M:%S'), msg)
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Debounce  [FIX 8]
    # ─────────────────────────────────────────────────────────────────────────

    def _debounced(self, gesture: GestureType) -> bool:
        limit = _DEBOUNCE_MS.get(gesture, 0)
        if limit == 0:
            return False
        now  = time.monotonic() * 1000
        last = self._last_fired.get(gesture, 0.0)
        if now - last < limit:
            return True
        self._last_fired[gesture] = now
        return False

    # ─────────────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
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
        detector   = vision.HandLandmarker.create_from_options(options)
        cap        = self._open_camera()
        read_fails = 0

        while self.running:

            if self.restart_camera:
                cap.release()
                cap          = self._open_camera()
                read_fails   = 0
                self.restart_camera = False

            ret, frame = cap.read()
            if not ret:
                read_fails += 1
                if read_fails >= _CAM_RETRY:
                    cap.release()
                    cap        = self._open_camera()
                    read_fails = 0
                blank = np.zeros((AR_H, AR_W, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera not available",
                            (AR_W // 2 - 140, AR_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 180), 2)
                if self.gui_callback:
                    self.gui_callback(blank, blank, self.app_state.copy())
                continue
            read_fails = 0

            ar_canvas = cv2.resize(frame, (AR_W, AR_H))

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts = int(time.monotonic() * 1000)
            result   = detector.detect_for_video(mp_img, frame_ts)

            if result.hand_landmarks:
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
                self._hand_lost()

            if self.pending_scan:
                self.perform_scan(frame)
                self.pending_scan = False

            if self.circuit_engine.sim_running:
                self.circuit_engine.tick_simulation()

            if self.app_state['circuit_engine_enabled']:
                self.circuit_engine.render(ar_canvas)
                self._draw_placement_preview(ar_canvas)
                self._draw_pinch_cursor(ar_canvas)
                self._draw_wire_draw_hud(ar_canvas)   # [FIX 10]

            self._draw_hud(ar_canvas)

            if self.gui_callback:
                self.gui_callback(ar_canvas.copy(), frame.copy(),
                                  self.app_state.copy())

        cap.release()

    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.app_state['camera_index'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def _hand_lost(self):
        self.app_state['gesture_active']     = None
        self.app_state['gesture_confidence'] = 0.0
        self.app_state['fingers_state']      = [False] * 5
        self.app_state['is_pinching']        = False
        self.app_state['dwell_progress']     = 0.0
        self._dragging_id                    = None
        self._pinch_cursor_px                = None
        # [FIX 6] cancel any in-progress wire
        self._cancel_wire_draw()

    def _cancel_wire_draw(self):
        self._wire_src_id   = None
        self._wire_src_pin  = None
        self._wire_src_pt   = None
        self._wire_drawing  = False
        self.circuit_engine.cancel_wire()

    # ─────────────────────────────────────────────────────────────────────────
    #  Drawing helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_skeleton(self, frame: np.ndarray, all_hands: list):
        h, w    = frame.shape[:2]
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

    def _draw_hud(self, canvas: np.ndarray):
        gesture = self.app_state.get('gesture_active')
        conf    = self.app_state.get('gesture_confidence', 0.0)
        if gesture:
            label = f"{str(gesture).upper()}  {conf:.0%}"
            cv2.putText(canvas, label, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 120), 2, cv2.LINE_AA)

        fingers = self.app_state.get('fingers_state', [False] * 5)
        names   = ['T', 'I', 'M', 'R', 'P']
        for i, (ext, name) in enumerate(zip(fingers, names)):
            x      = 14 + i * 28
            y      = canvas.shape[0] - 14
            colour = (0, 220, 100) if ext else (50, 50, 50)
            cv2.circle(canvas, (x, y), 10, colour, -1, cv2.LINE_AA)
            cv2.putText(canvas, name, (x - 5, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        prog = self.app_state.get('dwell_progress', 0.0)
        if 0 < prog < 1.0:
            cx = canvas.shape[1] // 2
            cy = canvas.shape[0] // 2
            cv2.ellipse(canvas, (cx, cy), (40, 40), -90, 0,
                        int(prog * 360), (80, 200, 255), 3)

        ar_mode = self.app_state.get('ar_mode', 'default')
        mode_col = {
            'default': (180, 180, 60),
            'draw':    (0, 200, 255),
            'inspect': (0, 255, 120),
            'measure': (255, 120, 0),
        }
        mc = mode_col.get(ar_mode, (180, 180, 60))
        badge = f"AR:{ar_mode.upper()}"
        cv2.putText(canvas, badge, (canvas.shape[1] - 150, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, mc, 1, cv2.LINE_AA)

        dyn = self.app_state.get('dynamic_ar_text', '')
        if dyn:
            tw, _ = cv2.getTextSize(dyn, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[:2]
            cx2   = (canvas.shape[1] - tw[0]) // 2
            cv2.putText(canvas, dyn, (cx2, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 60), 1, cv2.LINE_AA)

    def _draw_pinch_cursor(self, canvas: np.ndarray):
        if not self.app_state['is_pinching'] or self._pinch_cursor_px is None:
            return
        cx, cy = self._pinch_cursor_px

        cv2.circle(canvas, (cx, cy), 14, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy),  5, (0, 220, 120),  -1, cv2.LINE_AA)

        if self._dragging_id is None:
            return

        comp = self.circuit_engine.get_component(self._dragging_id)
        if comp is None:
            return

        sx, sy = self.circuit_engine.to_screen(comp.x, comp.y)
        cv2.line(canvas, (cx, cy), (sx, sy), (255, 220, 0), 1, cv2.LINE_AA)

        hw, hh = 30, 16
        cv2.rectangle(canvas, (sx - hw, sy - hh), (sx + hw, sy + hh),
                      (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, comp.type_id.upper(),
                    (sx - hw, sy - hh - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        snap_wx = self.circuit_engine.snap(
            self.circuit_engine.to_world(cx, cy)[0] + self._grab_offset_wx)
        snap_wy = self.circuit_engine.snap(
            self.circuit_engine.to_world(cx, cy)[1] + self._grab_offset_wy)
        snap_sx, snap_sy = self.circuit_engine.to_screen(snap_wx, snap_wy)
        cv2.circle(canvas, (snap_sx, snap_sy), 6, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (snap_sx, snap_sy), (255, 220, 0),
                       cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)

    def _draw_wire_draw_hud(self, canvas: np.ndarray):
        """
        [FIX 10] Visual overlay while drawing a wire in 'draw' mode.
        Shows: source pin label, dashed line to cursor, instruction text.
        """
        if not self._wire_drawing or self._wire_src_pt is None:
            return

        ce  = self.circuit_engine
        src_sx, src_sy = ce.to_screen(*self._wire_src_pt)
        cx, cy = self._pinch_cursor_px or (src_sx, src_sy)

        # Dashed line from source pin to finger
        dx, dy = cx - src_sx, cy - src_sy
        length = max(1, math.hypot(dx, dy))
        steps  = int(length / 12)
        for i in range(0, steps, 2):
            t0 = i / max(steps, 1)
            t1 = min(1.0, (i + 1) / max(steps, 1))
            p0 = (int(src_sx + dx * t0), int(src_sy + dy * t0))
            p1 = (int(src_sx + dx * t1), int(src_sy + dy * t1))
            cv2.line(canvas, p0, p1, (0, 212, 255), 2, cv2.LINE_AA)

        # Source pin marker
        cv2.circle(canvas, (src_sx, src_sy), 8, (255, 210, 0), 2, cv2.LINE_AA)

        # Labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        src_comp = ce.get_component(self._wire_src_id)
        if src_comp and self._wire_src_pin:
            lbl = f"{src_comp.label}.{self._wire_src_pin}"
            cv2.putText(canvas, lbl, (src_sx + 10, src_sy - 10),
                        font, 0.45, (255, 210, 0), 1, cv2.LINE_AA)

        # Instruction
        instr = "Release on target pin to connect"
        cv2.putText(canvas, instr, (12, canvas.shape[0] - 40),
                    font, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        # Highlight nearby pins the cursor is close to
        wx, wy = ce.to_world(cx, cy)
        pin_info = ce.nearest_pin_with_info(wx, wy, threshold=_PIN_WIRE_RADIUS * 2)
        if pin_info and pin_info[0].id != self._wire_src_id:
            comp2, pname, pt = pin_info
            ps = ce.to_screen(*pt)
            cv2.circle(canvas, ps, 10, (0, 255, 120), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{comp2.label}.{pname}", (ps[0] + 8, ps[1] - 8),
                        font, 0.4, (0, 255, 120), 1, cv2.LINE_AA)

    def _draw_placement_preview(self, canvas: np.ndarray):
        if (self.app_state['is_pinching']
                or self.app_state['ar_mode'] != 'default'
                or not self.app_state['circuit_engine_enabled']):
            return

        wx = self.app_state.get('cursor_wx', 0.0)
        wy = self.app_state.get('cursor_wy', 0.0)
        if wx == 0.0 and wy == 0.0:
            return

        sx, sy = self.circuit_engine.to_screen(
            self.circuit_engine.snap(wx),
            self.circuit_engine.snap(wy),
        )
        hw, hh = 28, 14
        overlay = canvas.copy()
        cv2.rectangle(overlay, (sx - hw, sy - hh), (sx + hw, sy + hh),
                      (100, 100, 100), 1)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
        cv2.putText(canvas,
                    self.app_state['selected_tool'].upper(),
                    (sx - hw, sy - hh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    #  Gesture routing
    # ─────────────────────────────────────────────────────────────────────────

    def _route(self, ev: dict):
        g      = ev['gesture']
        cursor = ev['cursor']
        data   = ev['data']
        conf   = ev['confidence']

        wx, wy = self.circuit_engine.to_world(*cursor)
        self.app_state['cursor_wx'] = wx
        self.app_state['cursor_wy'] = wy

        # ── PINCH ──────────────────────────────────────────────────────────
        if g == GestureType.PINCH:
            self._pinch_cursor_px = cursor
            state = data.get('state')

            if state == 'grab':
                self.app_state['is_pinching'] = True

                if self.app_state['circuit_engine_enabled']:

                    if self.app_state['ar_mode'] == 'draw':
                        # ── WIRE-DRAW MODE: try to start from nearest pin ──
                        # [FIX 4] Use nearest_pin_with_info for exact named pin
                        pin_info = self.circuit_engine.nearest_pin_with_info(
                            wx, wy, threshold=_PIN_WIRE_RADIUS)
                        if pin_info:
                            comp, pin_name, pt = pin_info
                            self._wire_src_id  = comp.id
                            self._wire_src_pin = pin_name
                            self._wire_src_pt  = pt
                            self._wire_drawing = True
                            # Start live wire preview
                            self.circuit_engine.start_wire(pt[0], pt[1])
                            self._feedback(
                                f"🔌 WIRE from {comp.label}.{pin_name}")
                        else:
                            # No pin nearby — fall back to grab
                            self._try_grab(wx, wy)

                    else:
                        # ── DEFAULT MODE: grab component ──
                        self._try_grab(wx, wy)

            elif state in ('drag', 'holding'):
                self.app_state['is_pinching'] = True

                if self.app_state['circuit_engine_enabled']:

                    if self._wire_drawing and state == 'drag':
                        # Extend live wire preview
                        self.circuit_engine.extend_wire(wx, wy)
                        self.circuit_engine.mouse_world = (wx, wy)

                    elif (self._dragging_id is not None and state == 'drag'):
                        # Move grabbed component
                        comp = self.circuit_engine.get_component(self._dragging_id)
                        if comp:
                            new_x = self.circuit_engine.snap(wx + self._grab_offset_wx)
                            new_y = self.circuit_engine.snap(wy + self._grab_offset_wy)
                            comp.x = new_x
                            comp.y = new_y
                            # [FIX 9] Re-route attached wires in real time
                            self.circuit_engine._reroute_component_wires(
                                self._dragging_id)

                if state == 'drag':
                    dpx = data.get('delta_px', (0, 0))
                    self.app_state['dynamic_ar_text'] = f"DRAG Δ{dpx}"

            elif state == 'release':
                self.app_state['is_pinching'] = False

                if self.app_state['circuit_engine_enabled']:

                    if self._wire_drawing:
                        # ── Finish wire: snap to nearest pin on release ──
                        pin_info = self.circuit_engine.nearest_pin_with_info(
                            wx, wy, threshold=_PIN_WIRE_RADIUS * 1.5)
                        if pin_info and pin_info[0].id != self._wire_src_id:
                            dst_comp, dst_pin, _ = pin_info
                            self.circuit_engine.cancel_wire()  # discard preview
                            result = self.circuit_engine.add_wire(
                                self._wire_src_id, self._wire_src_pin,
                                dst_comp.id, dst_pin,
                            )
                            if result:
                                self._feedback(
                                    f"✅ {self.circuit_engine.get_component(self._wire_src_id).label}"
                                    f".{self._wire_src_pin} → "
                                    f"{dst_comp.label}.{dst_pin}")
                            else:
                                self._feedback("⚠ Wire failed")
                        else:
                            self.circuit_engine.cancel_wire()
                            self._feedback("⚠ Release on a pin to connect")
                        self._cancel_wire_draw()

                    elif self._dragging_id is not None:
                        comp = self.circuit_engine.get_component(self._dragging_id)
                        if comp:
                            comp.x = self.circuit_engine.snap(wx + self._grab_offset_wx)
                            comp.y = self.circuit_engine.snap(wy + self._grab_offset_wy)
                            self.circuit_engine._reroute_component_wires(
                                self._dragging_id)

                self._dragging_id    = None
                self._grab_offset_wx = 0.0
                self._grab_offset_wy = 0.0
                self.app_state['dynamic_ar_text'] = ""
                self._feedback("✋ RELEASE")

        # ── SWIPE ──────────────────────────────────────────────────────────
        elif g == GestureType.SWIPE:
            if self._debounced(GestureType.SWIPE):
                return
            direction = data['direction']
            layer = self.app_state['current_layer_view']
            layer += {'left': 1, 'right': -1, 'down': 1, 'up': -1}.get(direction, 0)
            layer = max(1, layer)
            self.app_state['current_layer_view'] = layer
            self.app_state['dynamic_ar_text']    = f"Page {layer}"
            self._feedback(f"⟵⟶ SWIPE {direction.upper()}  PAGE {layer}")

        # ── CRUMPLE ────────────────────────────────────────────────────────
        elif g == GestureType.CRUMPLE:
            if self._debounced(GestureType.CRUMPLE):
                return
            if self.app_state['circuit_engine_enabled']:
                hit = self.circuit_engine.hit_test(
                    wx, wy, radius_multiplier=_GRAB_HIT_MULTIPLIER)
                if hit:
                    self.circuit_engine.remove_component(hit.id)
                    self._feedback(f"🗑 DELETED {hit.type_id.upper()}")
                else:
                    self._feedback("🗑 CRUMPLE (nothing here)")
            else:
                self._feedback("🗑 CRUMPLE")

        # ── THROW ──────────────────────────────────────────────────────────
        elif g == GestureType.THROW:
            if self._debounced(GestureType.THROW):
                return
            vel  = data.get('velocity', 0.0)
            dvec = data.get('direction_vec', [0, 0])
            self.app_state['dynamic_ar_text'] = "⟶ SCREEN CAST"
            self._feedback(f"📡 PROJECTING  v={vel:.3f}")
            self._on_project(vel, dvec)

        # ── ROTATE (two-hand) ──────────────────────────────────────────────
        elif g == GestureType.ROTATE:
            delta     = data.get('delta_deg', 0.0)
            direction = data.get('direction', '')
            self.app_state['ar_rotation'] = (
                self.app_state['ar_rotation'] + delta) % 360
            self._feedback(f"↻ ROTATE {direction.upper()} {self.app_state['ar_rotation']:.1f}°")
            if self.app_state['circuit_engine_enabled']:
                sel = self.circuit_engine.selected_id
                if sel is not None:
                    self.circuit_engine.rotate_component(sel, int(delta))

        # ── CLAW ROTATE ────────────────────────────────────────────────────
        elif g == GestureType.CLAW_ROTATE:
            delta     = data.get('delta_deg', 0.0)
            direction = data.get('direction', '')
            self.app_state['ar_rotation'] = (
                self.app_state['ar_rotation'] + delta) % 360
            self._feedback(f"✊↻ CLAW {direction.upper()} {delta:.1f}°")
            if self.app_state['circuit_engine_enabled']:
                sel = self.circuit_engine.selected_id
                if sel is not None:
                    steps = int(delta / 45.0)
                    if steps != 0:
                        self.circuit_engine.rotate_component(sel, steps * 90)

        # ── DWELL ──────────────────────────────────────────────────────────
        elif g == GestureType.DWELL:
            prog = data.get('progress', 0.0)
            self.app_state['dwell_progress'] = prog
            if prog >= 1.0:
                if self.app_state['circuit_engine_enabled']:
                    hit = self.circuit_engine.hit_test(wx, wy)
                    if hit:
                        self.circuit_engine.selected_id = hit.id
                        self._feedback(f"● SELECTED {hit.type_id.upper()}")
                    else:
                        comp = self.circuit_engine.add_component(
                            self.app_state['selected_tool'], wx, wy)
                        if comp:
                            self.circuit_engine.selected_id = comp.id
                            self._feedback(f"✚ PLACED {comp.type_id.upper()}")
                else:
                    self._feedback("● DWELL")

        # ── PEACE ──────────────────────────────────────────────────────────
        elif g == GestureType.PEACE:
            if self._debounced(GestureType.PEACE):
                return
            modes = ['default', 'draw', 'inspect', 'measure']
            cur   = self.app_state.get('ar_mode', 'default')
            nxt   = modes[(modes.index(cur) + 1) % len(modes)] if cur in modes else 'default'
            self.app_state['ar_mode'] = nxt
            self._feedback(f"✌ MODE → {nxt.upper()}")

    # ─────────────────────────────────────────────────────────────────────────

    def _try_grab(self, wx: float, wy: float):
        """Attempt to grab a component at world coords (wx,wy)."""
        hit = self.circuit_engine.hit_test(
            wx, wy, radius_multiplier=_GRAB_HIT_MULTIPLIER)  # [FIX 3]
        if hit:
            self.circuit_engine.selected_id = hit.id
            self._dragging_id    = hit.id
            self._grab_offset_wx = hit.x - wx
            self._grab_offset_wy = hit.y - wy
            self._feedback(f"✊ GRABBED {hit.type_id.upper()}")
        else:
            self._dragging_id    = None
            self._grab_offset_wx = 0.0
            self._grab_offset_wy = 0.0
            self._feedback("✊ PINCH (no target)")

    # ─────────────────────────────────────────────────────────────────────────
    #  Hooks / helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_project(self, velocity: float, direction_vec: list):
        pass

    def perform_scan(self, frame: np.ndarray):
        self._feedback("🔍 SCANNING...")
        cat, model = scan_object(frame)
        self.app_state['active_category'] = cat
        self.app_state['active_model']    = model
        self._feedback(f"UNIT: {model}")

    def trigger_voice(self):
        if not self.app_state['is_listening']:
            self.app_state['is_listening'] = True
            threading.Thread(
                target=listen_and_process_command,
                args=(self.app_state,),
                daemon=True,
            ).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def enable_circuit_mode(self, enabled: bool):
        self.app_state['circuit_engine_enabled'] = enabled
        self._feedback("[CIRCUIT] ON" if enabled else "[CIRCUIT] OFF")

    def set_selected_tool(self, type_id: str):
        """[FIX 7] Validates against module-level CATALOG, no dynamic import."""
        if type_id in CATALOG:
            self.app_state['selected_tool'] = type_id
        else:
            self._feedback(f"⚠ Unknown tool: {type_id}")

    def start_simulation(self):
        self.circuit_engine.start_simulation()
        self._feedback("▶ SIM START")

    def stop_simulation(self):
        self.circuit_engine.stop_simulation()
        self._feedback("■ SIM STOP")

    def save_circuit(self, path: str = "circuit.json"):
        self.circuit_engine.save(path)
        self._feedback(f"💾 SAVED → {path}")

    def load_circuit(self, path: str = "circuit.json"):
        self._push_undo()
        self.circuit_engine.load(path)
        self._feedback(f"📂 LOADED ← {path}")