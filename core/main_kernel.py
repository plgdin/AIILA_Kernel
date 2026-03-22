"""
kernel.py  ·  AIILAKernel  v5.2  (WORKBENCH EDITION)
======================================================
CHANGES OVER v5.1:

  [BRIDGE 1]  GestureCircuitBridge imported and instantiated alongside the
              two existing engines.  bridge.process(events) is called each
              frame when circuit mode is active; bridge.draw_overlay() paints
              ghost, dwell-ring and status toast on top of the rendered canvas.

  [BRIDGE 2]  SWIPE handler is now context-aware:
                • UP / DOWN while panel is visible  → panel scroll (bridge)
                • LEFT / RIGHT  OR  panel hidden    → layer-view navigation
              This prevents the same swipe from doing two things at once.

  [BRIDGE 3]  DWELL handler is now context-aware:
                • Cursor inside panel               → bridge handles selection
                • Cursor on board                   → existing place/select
              Avoids placing a component while you're just hovering the list.

  [BRIDGE 4]  PEACE handler skips the mode-cycle when circuit engine is
              active and the cursor is on the board so that quick-place
              (bridge) works without also changing the AR mode.
"""

from __future__ import annotations

import cv2
import numpy as np
import re
import threading
import time
import math
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from core.app_defaults          import DEFAULT_KEYBINDS
from core.vision_engine          import scan_object
from core.voice_engine           import (
    listen_and_process_command,
    WORKING_MIC_INDEX, WORKING_MIC_NAME,
    SPEAKER_NAME, SPEAKER_INDEX,
)
from core.circuit_engine         import CircuitEngine, CATALOG
from core.exploded_view_engine   import ExplodedViewEngine
from core.gesture_engine         import GestureEngine, HAND_CONNECTIONS, GestureType
from core.gesture_circuit_bridge import GestureCircuitBridge          # [BRIDGE 1]

AR_W, AR_H = 1000, 700

_GRAB_HIT_MULTIPLIER = 1.8
_PIN_WIRE_RADIUS     = 20
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
            'simulation_running':     False,
            'camera_index':           0,
            'mic_index':              WORKING_MIC_INDEX,
            'mic_name':               WORKING_MIC_NAME,
            'speaker_index':          SPEAKER_INDEX,
            'speaker_name':           SPEAKER_NAME,
            'ar_rotation':            0.0,
            'ar_mode':                'default',
            'projector_enabled':      False,
            'calibration_mode':       False,
            'gesture_active':         None,
            'gesture_confidence':     0.0,
            'fingers_state':          [False] * 5,
            'dwell_progress':         0.0,
            'cursor_wx':              0.0,
            'cursor_wy':              0.0,
            'keybinds':               DEFAULT_KEYBINDS.copy(),
            'exploded_view_visible':  False,
            'exploded_view_index':    0,
            'exploded_view_total':    0,
            'exploded_view_caption':  "",
            'ui_command_request':     None,
        }

        self.circuit_engine = CircuitEngine(canvas_w=AR_W, canvas_h=AR_H)
        self.gesture_engine = GestureEngine(
            canvas_w=AR_W, canvas_h=AR_H,
            ema_alpha=0.55, min_confidence=0.45,
        )
        self.exploded_view_engine = ExplodedViewEngine()

        # [BRIDGE 1] — Bridge lives here; owns no engines, just links them
        self.bridge = GestureCircuitBridge(
            self.circuit_engine, canvas_w=AR_W, canvas_h=AR_H)

        self.pending_scan = False

        # Drag state
        self._dragging_id:      int | None              = None
        self._grab_offset_wx:   float                   = 0.0
        self._grab_offset_wy:   float                   = 0.0
        self._pinch_cursor_px:  tuple[int, int] | None  = None

        # Wire-draw state
        self._wire_src_id:      int | None   = None
        self._wire_src_pin:     str | None   = None
        self._wire_src_pt:      tuple | None = None
        self._wire_drawing:     bool         = False

        # Gesture debounce
        self._last_fired: dict[GestureType, float] = {}

        # [BRIDGE 1] overlay info carried frame-to-frame
        self._overlay_info: dict = {}

    # ─────────────────────────────────────────────────────────────────────────
    #  Undo
    # ─────────────────────────────────────────────────────────────────────────

    def _push_undo(self):
        try:
            self.circuit_engine._push_undo()
        except Exception:
            pass

    def undo(self):
        self.circuit_engine.undo()
        self._feedback("↩ UNDO")

    # ─────────────────────────────────────────────────────────────────────────
    #  Feedback / debounce
    # ─────────────────────────────────────────────────────────────────────────

    def _feedback(self, msg: str):
        self.app_state['voice_feedback'] = msg
        self.app_state['feedback_log'].append(
            (time.strftime('%H:%M:%S'), msg)
        )

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
    #  Helpers — panel context
    # ─────────────────────────────────────────────────────────────────────────

    def _cursor_in_panel(self, cursor: tuple[int, int]) -> bool:
        """True when the screen cursor is over the component panel."""
        return (self.app_state['circuit_engine_enabled']
                and self.circuit_engine.panel_visible
                and self.circuit_engine.in_panel(cursor[0], cursor[1]))

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

                # [BRIDGE 1] — Let bridge process panel/board interactions
                # after _route so kernel-owned state (selected_id, mode) is
                # already up to date when the bridge reads it.
                if self.app_state['circuit_engine_enabled']:
                    self._overlay_info = self.bridge.process(events)
                else:
                    self._overlay_info = {}

            else:
                self._hand_lost()
                self._overlay_info = {}

            if self.pending_scan:
                self.perform_scan(frame)
                self.pending_scan = False

            if self.circuit_engine.sim_running:
                self.circuit_engine.tick_simulation()

            if self.app_state['circuit_engine_enabled']:
                self.circuit_engine.render(ar_canvas)
                self._draw_placement_preview(ar_canvas)
                self._draw_pinch_cursor(ar_canvas)
                self._draw_wire_draw_hud(ar_canvas)

                # [BRIDGE 1] — Paint ghost, dwell ring and status toast
                if self._overlay_info:
                    self.bridge.draw_overlay(ar_canvas, self._overlay_info)

            self._draw_exploded_view(ar_canvas)
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 120), 2,
                        cv2.LINE_AA)

        fingers = self.app_state.get('fingers_state', [False] * 5)
        names   = ['T', 'I', 'M', 'R', 'P']
        for i, (ext, name) in enumerate(zip(fingers, names)):
            x      = 14 + i * 28
            y      = canvas.shape[0] - 14
            colour = (0, 220, 100) if ext else (50, 50, 50)
            cv2.circle(canvas, (x, y), 10, colour, -1, cv2.LINE_AA)
            cv2.putText(canvas, name, (x - 5, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                        cv2.LINE_AA)

        prog = self.app_state.get('dwell_progress', 0.0)
        if 0 < prog < 1.0:
            cx = canvas.shape[1] // 2
            cy = canvas.shape[0] // 2
            cv2.ellipse(canvas, (cx, cy), (40, 40), -90, 0,
                        int(prog * 360), (80, 200, 255), 3)

        ar_mode  = self.app_state.get('ar_mode', 'default')
        mode_col = {
            'default': (180, 180, 60),
            'draw':    (0, 200, 255),
            'inspect': (0, 255, 120),
            'measure': (255, 120, 0),
        }
        mc = mode_col.get(ar_mode, (180, 180, 60))
        cv2.putText(canvas, f"AR:{ar_mode.upper()}",
                    (canvas.shape[1] - 150, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, mc, 1, cv2.LINE_AA)

        dyn = self.app_state.get('dynamic_ar_text', '')
        if dyn:
            tw, _ = cv2.getTextSize(dyn, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[:2]
            cx2   = (canvas.shape[1] - tw[0]) // 2
            cv2.putText(canvas, dyn, (cx2, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 60), 1,
                        cv2.LINE_AA)

    def _draw_exploded_view(self, canvas: np.ndarray):
        view = self.exploded_view_engine.get_view_state()
        if not view.get('visible'):
            return

        img = view.get('image')
        if img is None:
            return

        pad = 18
        panel_w = min(420, canvas.shape[1] - pad * 2)
        panel_h = min(320, canvas.shape[0] - 120)
        x2 = canvas.shape[1] - pad
        x1 = x2 - panel_w
        y1 = 74
        y2 = y1 + panel_h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (5, 12, 20), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 180, 255), 1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        header = f"INTERNAL VIEW  {view['index']}/{view['total']}"
        cv2.putText(canvas, header, (x1 + 12, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1,
                    cv2.LINE_AA)

        model_name = view.get('model_name', '')
        cv2.putText(canvas, model_name[:34], (x1 + 12, y1 + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                    cv2.LINE_AA)

        img_x1, img_y1 = x1 + 12, y1 + 58
        img_x2, img_y2 = x2 - 12, y2 - 54
        target_w = max(1, img_x2 - img_x1)
        target_h = max(1, img_y2 - img_y1)
        ih, iw = img.shape[:2]
        scale = min(target_w / iw, target_h / ih)
        resized = cv2.resize(img, (max(1, int(iw * scale)), max(1, int(ih * scale))),
                             interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape[:2]
        px = img_x1 + (target_w - rw) // 2
        py = img_y1 + (target_h - rh) // 2
        canvas[py:py + rh, px:px + rw] = resized
        cv2.rectangle(canvas, (img_x1, img_y1), (img_x2, img_y2),
                      (30, 120, 180), 1)

        caption = (view.get('caption', '') or '').replace('\n', ' ').strip()
        if len(caption) > 60:
            caption = caption[:57] + "..."
        cv2.putText(canvas, caption, (x1 + 12, y2 - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 230, 235), 1,
                    cv2.LINE_AA)
        cv2.putText(canvas, "SWIPE L/R  or  voice: next / previous / hide",
                    (x1 + 12, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 190, 220), 1,
                    cv2.LINE_AA)

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
        if not self._wire_drawing or self._wire_src_pt is None:
            return

        ce              = self.circuit_engine
        src_sx, src_sy  = ce.to_screen(*self._wire_src_pt)
        cx, cy          = self._pinch_cursor_px or (src_sx, src_sy)

        dx, dy = cx - src_sx, cy - src_sy
        length = max(1, math.hypot(dx, dy))
        steps  = int(length / 12)
        for i in range(0, steps, 2):
            t0 = i / max(steps, 1)
            t1 = min(1.0, (i + 1) / max(steps, 1))
            p0 = (int(src_sx + dx * t0), int(src_sy + dy * t0))
            p1 = (int(src_sx + dx * t1), int(src_sy + dy * t1))
            cv2.line(canvas, p0, p1, (0, 212, 255), 2, cv2.LINE_AA)

        cv2.circle(canvas, (src_sx, src_sy), 8, (255, 210, 0), 2, cv2.LINE_AA)

        font     = cv2.FONT_HERSHEY_SIMPLEX
        src_comp = ce.get_component(self._wire_src_id)
        if src_comp and self._wire_src_pin:
            lbl = f"{src_comp.label}.{self._wire_src_pin}"
            cv2.putText(canvas, lbl, (src_sx + 10, src_sy - 10),
                        font, 0.45, (255, 210, 0), 1, cv2.LINE_AA)

        cv2.putText(canvas, "Release on target pin to connect",
                    (12, canvas.shape[0] - 40),
                    font, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        wx, wy   = ce.to_world(cx, cy)
        pin_info = ce.nearest_pin_with_info(wx, wy,
                                            threshold=_PIN_WIRE_RADIUS * 2)
        if pin_info and pin_info[0].id != self._wire_src_id:
            comp2, pname, pt = pin_info
            ps = ce.to_screen(*pt)
            cv2.circle(canvas, ps, 10, (0, 255, 120), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{comp2.label}.{pname}",
                        (ps[0] + 8, ps[1] - 8),
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
        hw, hh  = 28, 14
        overlay = canvas.copy()
        cv2.rectangle(overlay, (sx - hw, sy - hh), (sx + hw, sy + hh),
                      (100, 100, 100), 1)
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
        cv2.putText(canvas,
                    self.app_state['selected_tool'].upper(),
                    (sx - hw, sy - hh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1,
                    cv2.LINE_AA)

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

        in_panel = self._cursor_in_panel(cursor)   # convenience flag

        # ── PINCH ──────────────────────────────────────────────────────────
        if g == GestureType.PINCH:
            # [BRIDGE 2] Bridge owns all panel pinch interactions.
            # _route only handles board-level pinch (wire-draw and component drag).
            if in_panel:
                # Keep is_pinching in sync so HUD is correct, then return —
                # bridge.process() will take care of the panel grab/drag/drop.
                state = data.get('state')
                if state == 'grab':
                    self.app_state['is_pinching'] = True
                elif state == 'release':
                    self.app_state['is_pinching'] = False
                self._pinch_cursor_px = cursor
                return

            self._pinch_cursor_px = cursor
            state = data.get('state')

            if state == 'grab':
                self.app_state['is_pinching'] = True

                if self.app_state['circuit_engine_enabled']:
                    if self.app_state['ar_mode'] == 'draw':
                        pin_info = self.circuit_engine.nearest_pin_with_info(
                            wx, wy, threshold=_PIN_WIRE_RADIUS)
                        if pin_info:
                            comp, pin_name, pt = pin_info
                            self._wire_src_id  = comp.id
                            self._wire_src_pin = pin_name
                            self._wire_src_pt  = pt
                            self._wire_drawing = True
                            self.circuit_engine.start_wire(pt[0], pt[1])
                            self._feedback(
                                f"🔌 WIRE from {comp.label}.{pin_name}")
                        else:
                            self._try_grab(wx, wy)
                    else:
                        self._try_grab(wx, wy)

            elif state in ('drag', 'holding'):
                self.app_state['is_pinching'] = True

                if self.app_state['circuit_engine_enabled']:
                    if self._wire_drawing and state == 'drag':
                        self.circuit_engine.extend_wire(wx, wy)
                        self.circuit_engine.mouse_world = (wx, wy)
                    elif self._dragging_id is not None and state == 'drag':
                        comp = self.circuit_engine.get_component(
                            self._dragging_id)
                        if comp:
                            comp.x = self.circuit_engine.snap(
                                wx + self._grab_offset_wx)
                            comp.y = self.circuit_engine.snap(
                                wy + self._grab_offset_wy)
                            self.circuit_engine._reroute_component_wires(
                                self._dragging_id)

                if state == 'drag':
                    dpx = data.get('delta_px', (0, 0))
                    self.app_state['dynamic_ar_text'] = f"DRAG Δ{dpx}"

            elif state == 'release':
                self.app_state['is_pinching'] = False

                if self.app_state['circuit_engine_enabled']:
                    if self._wire_drawing:
                        pin_info = self.circuit_engine.nearest_pin_with_info(
                            wx, wy, threshold=_PIN_WIRE_RADIUS * 1.5)
                        if pin_info and pin_info[0].id != self._wire_src_id:
                            dst_comp, dst_pin, _ = pin_info
                            self.circuit_engine.cancel_wire()
                            result = self.circuit_engine.add_wire(
                                self._wire_src_id, self._wire_src_pin,
                                dst_comp.id, dst_pin,
                            )
                            if result:
                                src_c = self.circuit_engine.get_component(
                                    self._wire_src_id)
                                self._feedback(
                                    f"✅ {src_c.label}.{self._wire_src_pin}"
                                    f" → {dst_comp.label}.{dst_pin}")
                            else:
                                self._feedback("⚠ Wire failed")
                        else:
                            self.circuit_engine.cancel_wire()
                            self._feedback("⚠ Release on a pin to connect")
                        self._cancel_wire_draw()

                    elif self._dragging_id is not None:
                        comp = self.circuit_engine.get_component(
                            self._dragging_id)
                        if comp:
                            comp.x = self.circuit_engine.snap(
                                wx + self._grab_offset_wx)
                            comp.y = self.circuit_engine.snap(
                                wy + self._grab_offset_wy)
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

            # [BRIDGE 2] UP/DOWN while panel visible → panel scroll (bridge
            # handles it via bridge.process()).  Only navigate layers for
            # LEFT/RIGHT, or when the panel is hidden.
            if (self.app_state['circuit_engine_enabled']
                    and self.circuit_engine.panel_visible
                    and direction in ('up', 'down')):
                # Bridge will call panel_scroll_by(); nothing to do here.
                return

            if self.app_state.get('exploded_view_visible') and direction in ('left', 'right'):
                moved = (self.next_exploded_view()
                         if direction == 'left' else
                         self.previous_exploded_view())
                if not moved and self.app_state.get('exploded_view_total', 0) <= 1:
                    self._feedback("⚠ Only one internal image available")
                return

            layer = self.app_state['current_layer_view']
            layer += {'left': 1, 'right': -1,
                      'down': 1, 'up':   -1}.get(direction, 0)
            layer = max(1, layer)
            self.app_state['current_layer_view'] = layer
            self.app_state['dynamic_ar_text']    = f"Page {layer}"
            self._feedback(
                f"⟵⟶ SWIPE {direction.upper()}  PAGE {layer}")

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
            self._feedback(
                f"↻ ROTATE {direction.upper()}"
                f" {self.app_state['ar_rotation']:.1f}°")
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

            # [BRIDGE 3] If cursor is over the panel, bridge handles the
            # hover-highlight and eventual selection — skip board logic.
            if in_panel:
                return

            if prog >= 1.0 and self.app_state['circuit_engine_enabled']:
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
            elif prog >= 1.0:
                self._feedback("● DWELL")

        # ── PEACE ──────────────────────────────────────────────────────────
        elif g == GestureType.PEACE:
            # [BRIDGE 4] When circuit mode is on and cursor is on the board,
            # bridge handles quick-place ✌ → mode-cycle is skipped so the
            # user doesn't accidentally switch modes while placing parts.
            if (self.app_state['circuit_engine_enabled']
                    and not in_panel):
                return   # bridge.process() fires quick-place

            if self._debounced(GestureType.PEACE):
                return
            modes = ['default', 'draw', 'inspect', 'measure']
            cur   = self.app_state.get('ar_mode', 'default')
            nxt   = (modes[(modes.index(cur) + 1) % len(modes)]
                     if cur in modes else 'default')
            self.app_state['ar_mode'] = nxt
            self._feedback(f"✌ MODE → {nxt.upper()}")

    # ─────────────────────────────────────────────────────────────────────────

    def _try_grab(self, wx: float, wy: float):
        hit = self.circuit_engine.hit_test(
            wx, wy, radius_multiplier=_GRAB_HIT_MULTIPLIER)
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

    def _sync_exploded_state(self):
        view = self.exploded_view_engine.get_view_state()
        self.app_state['exploded_view_visible'] = bool(view.get('visible'))
        self.app_state['exploded_view_index'] = int(view.get('index', 0))
        self.app_state['exploded_view_total'] = int(view.get('total', 0))
        self.app_state['exploded_view_caption'] = view.get('caption', '') or ""

    def show_exploded_view(self) -> bool:
        model_name = (self.app_state.get('active_model') or '').strip()
        if not model_name or model_name in {'unknown', 'error'}:
            self._feedback("⚠ Scan a supported device first")
            return False

        self._feedback(f"🧩 Loading internal view for {model_name}...")
        ok, msg = self.exploded_view_engine.load_for_model(
            model_name,
            self.app_state.get('active_category', ''),
        )
        self._sync_exploded_state()
        if ok:
            self.app_state['dynamic_ar_text'] = (
                f"INTERNAL VIEW {self.app_state['exploded_view_index']}/"
                f"{self.app_state['exploded_view_total']}"
            )
            self._feedback(msg)
            return True

        self.app_state['dynamic_ar_text'] = ""
        self._feedback(msg)
        return False

    def hide_exploded_view(self) -> bool:
        if not self.app_state.get('exploded_view_visible'):
            return False
        self.exploded_view_engine.clear()
        self._sync_exploded_state()
        self.app_state['dynamic_ar_text'] = ""
        self._feedback("🗂 Internal view closed")
        return True

    def next_exploded_view(self) -> bool:
        moved = self.exploded_view_engine.next_image()
        self._sync_exploded_state()
        if moved:
            self.app_state['dynamic_ar_text'] = (
                f"PART {self.app_state['exploded_view_index']}/"
                f"{self.app_state['exploded_view_total']}"
            )
            self._feedback(
                f"➡ PART {self.app_state['exploded_view_index']}/"
                f"{self.app_state['exploded_view_total']}"
            )
        return moved

    def previous_exploded_view(self) -> bool:
        moved = self.exploded_view_engine.previous_image()
        self._sync_exploded_state()
        if moved:
            self.app_state['dynamic_ar_text'] = (
                f"PART {self.app_state['exploded_view_index']}/"
                f"{self.app_state['exploded_view_total']}"
            )
            self._feedback(
                f"⬅ PART {self.app_state['exploded_view_index']}/"
                f"{self.app_state['exploded_view_total']}"
            )
        return moved

    def set_page(self, page: int) -> bool:
        page = max(1, int(page))
        self.app_state['current_layer_view'] = page
        self.app_state['dynamic_ar_text'] = f"Page {page}"
        self._feedback(f"PAGE {page}")
        return True

    def shift_page(self, delta: int) -> bool:
        current = int(self.app_state.get('current_layer_view', 1))
        return self.set_page(current + delta)

    def request_ui_command(self, command: str):
        self.app_state['ui_command_request'] = command

    def _handle_voice_command(self, text: str) -> bool:
        cmd = (text or '').strip().lower()
        if not cmd:
            return False

        settings_words = (
            "open settings",
            "open hardware settings",
            "hardware settings",
            "settings panel",
        )
        scan_words = (
            "scan unit",
            "scan object",
            "scan device",
            "scan now",
        )
        next_words = ("next part", "next image", "show next", "next one", "next internal")
        prev_words = ("previous part", "prev part", "previous image", "show previous", "go back")
        hide_words = ("hide exploded", "close exploded", "hide internal", "close internal", "hide parts")
        next_page_words = ("next page", "next layer", "page forward", "go next page")
        prev_page_words = ("previous page", "prev page", "back page", "go previous page", "last page")
        save_words = ("save circuit", "save project", "save board")
        undo_words = ("undo", "undo that", "go undo")
        circuit_on_words = ("enable circuit mode", "turn on circuit mode", "start circuit mode")
        circuit_off_words = ("disable circuit mode", "turn off circuit mode", "stop circuit mode")
        circuit_toggle_words = ("toggle circuit mode",)
        draw_on_words = ("enable wire draw mode", "turn on wire draw mode", "enable draw mode", "turn on draw mode")
        draw_off_words = ("disable wire draw mode", "turn off wire draw mode", "disable draw mode", "turn off draw mode")
        draw_toggle_words = ("toggle wire draw mode", "toggle draw mode")
        sim_on_words = ("start simulation", "run simulation", "enable simulation")
        sim_off_words = ("stop simulation", "disable simulation", "end simulation")
        sim_toggle_words = ("toggle simulation",)
        projector_on_words = ("project to screen", "turn on projector", "open projector", "start projector", "enable projector")
        projector_off_words = ("turn off projector", "close projector", "disable projector", "stop projector")
        projector_toggle_words = ("toggle projector",)
        calib_on_words = ("show calibration grid", "open calibration grid", "enable calibration grid", "turn on calibration")
        calib_off_words = ("hide calibration grid", "close calibration grid", "disable calibration grid", "turn off calibration")
        calib_toggle_words = ("toggle calibration grid", "toggle calibration")
        show_words = (
            "show exploded",
            "exploded view",
            "exploded image",
            "internal view",
            "show internal",
            "internal parts",
            "show me inside",
        )

        if any(word in cmd for word in settings_words):
            self.request_ui_command('open_settings')
            self._feedback("OPENING SETTINGS")
            return True

        if any(word in cmd for word in scan_words):
            self.pending_scan = True
            self._feedback("SCAN REQUESTED")
            return True

        if any(word in cmd for word in save_words):
            self.save_circuit("circuit.json")
            return True

        if any(word in cmd for word in undo_words):
            self.undo()
            return True

        if any(word in cmd for word in circuit_on_words):
            self.request_ui_command('circuit_on')
            self._feedback("CIRCUIT MODE ON")
            return True

        if any(word in cmd for word in circuit_off_words):
            self.request_ui_command('circuit_off')
            self._feedback("CIRCUIT MODE OFF")
            return True

        if any(word in cmd for word in circuit_toggle_words):
            self.request_ui_command('toggle_circuit')
            self._feedback("TOGGLING CIRCUIT MODE")
            return True

        if any(word in cmd for word in draw_on_words):
            self.request_ui_command('draw_on')
            self._feedback("WIRE DRAW MODE ON")
            return True

        if any(word in cmd for word in draw_off_words):
            self.request_ui_command('draw_off')
            self._feedback("WIRE DRAW MODE OFF")
            return True

        if any(word in cmd for word in draw_toggle_words):
            self.request_ui_command('toggle_draw')
            self._feedback("TOGGLING WIRE DRAW MODE")
            return True

        if any(word in cmd for word in sim_on_words):
            self.request_ui_command('simulation_on')
            self._feedback("SIMULATION START")
            return True

        if any(word in cmd for word in sim_off_words):
            self.request_ui_command('simulation_off')
            self._feedback("SIMULATION STOP")
            return True

        if any(word in cmd for word in sim_toggle_words):
            self.request_ui_command('toggle_simulation')
            self._feedback("TOGGLING SIMULATION")
            return True

        if any(word in cmd for word in projector_on_words):
            self.request_ui_command('projector_on')
            self._feedback("PROJECTOR ON")
            return True

        if any(word in cmd for word in projector_off_words):
            self.request_ui_command('projector_off')
            self._feedback("PROJECTOR OFF")
            return True

        if any(word in cmd for word in projector_toggle_words):
            self.request_ui_command('toggle_projector')
            self._feedback("TOGGLING PROJECTOR")
            return True

        if any(word in cmd for word in calib_on_words):
            self.request_ui_command('calibration_on')
            self._feedback("CALIBRATION ON")
            return True

        if any(word in cmd for word in calib_off_words):
            self.request_ui_command('calibration_off')
            self._feedback("CALIBRATION OFF")
            return True

        if any(word in cmd for word in calib_toggle_words):
            self.request_ui_command('toggle_calibration')
            self._feedback("TOGGLING CALIBRATION")
            return True

        if any(word in cmd for word in next_words):
            if not self.next_exploded_view():
                self._feedback("⚠ No next internal image")
            return True

        if any(word in cmd for word in prev_words):
            if not self.previous_exploded_view():
                self._feedback("⚠ No previous internal image")
            return True

        if any(word in cmd for word in next_page_words):
            self.shift_page(1)
            return True

        if any(word in cmd for word in prev_page_words):
            self.shift_page(-1)
            return True

        page_match = re.search(r"\b(?:page|layer)\s+(\d+)\b", cmd)
        if page_match:
            self.set_page(int(page_match.group(1)))
            return True

        if "first page" in cmd or "first layer" in cmd:
            self.set_page(1)
            return True

        if "second page" in cmd or "second layer" in cmd:
            self.set_page(2)
            return True

        if any(word in cmd for word in hide_words):
            if not self.hide_exploded_view():
                self._feedback("⚠ Internal view is not open")
            return True

        if any(word in cmd for word in show_words):
            self.show_exploded_view()
            return True

        return False

    def perform_scan(self, frame: np.ndarray):
        self._feedback("🔍 SCANNING...")
        cat, model = scan_object(frame)
        self.exploded_view_engine.clear()
        self._sync_exploded_state()
        self.app_state['active_category'] = cat
        self.app_state['active_model']    = model
        self.app_state['dynamic_ar_text'] = ""
        self._feedback(f"UNIT: {model}")

    def trigger_voice(self):
        if not self.app_state['is_listening']:
            self.app_state['is_listening'] = True
            threading.Thread(
                target=listen_and_process_command,
                args=(self.app_state, self._handle_voice_command),
                daemon=True,
            ).start()

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def enable_circuit_mode(self, enabled: bool):
        self.app_state['circuit_engine_enabled'] = enabled
        self._feedback("[CIRCUIT] ON" if enabled else "[CIRCUIT] OFF")

    def set_selected_tool(self, type_id: str):
        if type_id in CATALOG:
            self.app_state['selected_tool'] = type_id
        else:
            self._feedback(f"⚠ Unknown tool: {type_id}")

    def start_simulation(self):
        self.circuit_engine.start_simulation()
        self.app_state['simulation_running'] = True
        self._feedback("▶ SIM START")

    def stop_simulation(self):
        self.circuit_engine.stop_simulation()
        self.app_state['simulation_running'] = False
        self._feedback("■ SIM STOP")

    def save_circuit(self, path: str = "circuit.json"):
        self.circuit_engine.save(path)
        self._feedback(f"💾 SAVED → {path}")

    def load_circuit(self, path: str = "circuit.json"):
        self._push_undo()
        self.circuit_engine.load(path)
        self._feedback(f"📂 LOADED ← {path}")
