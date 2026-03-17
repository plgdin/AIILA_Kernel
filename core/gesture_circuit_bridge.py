"""
gesture_circuit_bridge.py  —  AIILA Gesture × Circuit Bridge v1.0
==================================================================
Connects GestureEngine events to CircuitEngine actions so that:

  PANEL INTERACTION
  ─────────────────
  • SWIPE UP / DOWN  (cursor anywhere)
      → panel_scroll_by()  — scrolls the component list.
        Scroll amount scales with swipe speed for fast jumps.

  • DWELL  (cursor inside panel, only index finger extended)
      → highlights + selects the hovered component type.
        (sets engine.panel_selected and engine._panel_hovered)
        Progress ring drawn on screen as visual feedback.
        Completes when dwell progress reaches 1.0.

  COMPONENT PLACEMENT
  ───────────────────
  • PINCH GRAB  (cursor inside panel)
      → "picks up" the component under the cursor.
        Stores it as _held_type and shows a ghost while dragging.

  • PINCH DRAG  (after a panel grab)
      → moves the ghost preview to follow the cursor.

  • PINCH RELEASE  (after a panel grab)
      → if cursor is on the BOARD  → add_component() at that snapped position.
        if cursor is still in PANEL → cancel (put it back).

  • PEACE ✌️  (cursor on board, component selected in panel)
      → quick-place: drops one copy of panel_selected at the cursor.
        No drag needed — good for rapid repeated placement.
        Must hold the sign for PEACE_HOLD_FRAMES frames to fire (debounce).

  BOARD INTERACTION
  ─────────────────
  • PINCH GRAB  on board (no panel component being held)
      → hit_test() finds the component under the cursor and starts dragging it.

  • PINCH DRAG  (after a board grab)
      → moves the grabbed component in world space, live.

  • PINCH RELEASE  (after a board grab)
      → drops the component; position is grid-snapped.

  • CLAW_ROTATE  (component selected)
      → rotate_component() by delta_deg rounded to nearest 90°.

  • THROW  (component selected)
      → remove_component() — the throw gesture discards it.

  • CRUMPLE  (both hands)
      → engine.clear() — clears the entire board (undo-safe).

Usage
─────
    bridge = GestureCircuitBridge(circuit_engine, canvas_w=1280, canvas_h=720)

    # Each frame, after getting events from GestureEngine:
    overlay_info = bridge.process(gesture_events)

    # Render the circuit first, then paint gesture feedback on top:
    engine.render(canvas)
    bridge.draw_overlay(canvas, overlay_info)
"""

from __future__ import annotations

import math
import time
import cv2
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from circuit_engine import CircuitEngine


# ─────────────────────────────────────────────────────────────────────────────
#  Tuning constants
# ─────────────────────────────────────────────────────────────────────────────

# Scroll pixels per unit of swipe speed reported by GestureEngine
SWIPE_SCROLL_SCALE  = 1800

# Claw-rotate: snap to this increment (degrees)
ROTATE_SNAP         = 90

# Peace quick-place: must hold ✌️ this many frames before firing
PEACE_HOLD_FRAMES   = 12

# Ghost component translucency (0 = invisible, 1 = opaque)
GHOST_ALPHA         = 0.55

# Dwell ring display radius (screen pixels)
DWELL_RING_RADIUS   = 22


# ─────────────────────────────────────────────────────────────────────────────
#  Bridge
# ─────────────────────────────────────────────────────────────────────────────

class GestureCircuitBridge:
    """
    Stateful bridge between GestureEngine event stream and CircuitEngine.
    One instance lives for the lifetime of the session.
    """

    def __init__(self, engine: "CircuitEngine",
                 canvas_w: int = 1280, canvas_h: int = 720):
        self.engine   = engine
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        # Panel-drag state
        self._held_type:    Optional[str]       = None   # type_id being dragged from panel
        # Board-drag state
        self._held_comp_id: Optional[int]       = None   # component id being moved on board
        self._drag_cursor:  tuple[int, int]     = (0, 0)
        self._drag_offset:  tuple[int, int]     = (0, 0) # world (wx-comp.x, wy-comp.y) at grab

        # Dwell visual
        self._dwell_cursor: tuple[int, int]     = (0, 0)
        self._dwell_prog:   float               = 0.0

        # Peace debounce
        self._peace_counter: int                = 0

        # Status toast
        self._status:       str                 = ""
        self._status_ts:    float               = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    #  Main entry point — call once per frame
    # ─────────────────────────────────────────────────────────────────────────

    def process(self, gesture_events: list[dict]) -> dict:
        """
        Consume gesture event dicts from GestureEngine.update() and drive
        the CircuitEngine.  Returns overlay_info for draw_overlay().
        """
        e = self.engine
        has_dwell = False

        for ev in gesture_events:
            g      = ev.get("gesture", "")
            cursor = ev.get("cursor", (0, 0))   # screen pixels
            data   = ev.get("data", {})

            # ── SWIPE → scroll panel ──────────────────────────────────────
            if g == "swipe":
                direction = data.get("direction", "")
                speed     = float(data.get("speed", 0.02))
                scroll_px = int(speed * SWIPE_SCROLL_SCALE)

                if direction == "up":
                    e.panel_scroll_by(scroll_px)
                    self._toast(f"↑ Scroll +{scroll_px}px")
                elif direction == "down":
                    e.panel_scroll_by(-scroll_px)
                    self._toast(f"↓ Scroll -{scroll_px}px")
                # left/right swipes reserved for future undo/redo
                elif direction == "left":
                    self._toast("← (left swipe)")
                elif direction == "right":
                    self._toast("→ (right swipe)")

            # ── DWELL → hover-select panel item ──────────────────────────
            elif g == "dwell":
                has_dwell          = True
                self._dwell_cursor = cursor
                self._dwell_prog   = float(data.get("progress", 0.0))

                if e.in_panel(cursor[0], cursor[1]):
                    hovered = e.panel_hit_test(cursor[0], cursor[1])
                    if hovered:
                        e._panel_hovered = hovered
                        if self._dwell_prog >= 1.0:
                            e.panel_selected = hovered
                            self._toast(f"✔ Selected: {hovered}")
                else:
                    # Dwell outside panel — reset so the ring disappears
                    self._dwell_prog = 0.0

            # ── PINCH ─────────────────────────────────────────────────────
            elif g == "pinch":
                state = data.get("state", "")
                if state == "grab":
                    self._on_pinch_grab(cursor)
                elif state in ("drag", "holding"):
                    self._drag_cursor = cursor
                    if self._held_comp_id is not None:
                        self._move_board_comp(cursor)
                elif state == "release":
                    self._on_pinch_release(cursor)

            # ── CLAW_ROTATE → rotate selected component ───────────────────
            elif g == "claw_rotate":
                if e.selected_id is not None:
                    delta = float(data.get("delta_deg", 0.0))
                    snapped = round(delta / ROTATE_SNAP) * ROTATE_SNAP
                    if snapped != 0:
                        e.rotate_component(e.selected_id, snapped)
                        sign = "+" if snapped > 0 else ""
                        self._toast(f"↻ Rotate #{e.selected_id}  {sign}{snapped}°")

            # ── THROW → delete selected component ─────────────────────────
            elif g == "throw":
                if e.selected_id is not None:
                    cid = e.selected_id
                    e.remove_component(cid)
                    self._toast(f"🗑 Removed #{cid}")

            # ── CRUMPLE → clear board ─────────────────────────────────────
            elif g == "crumple":
                e.clear()
                self._toast("💥 Board cleared")

            # ── PEACE ✌️ → quick-place ─────────────────────────────────────
            elif g == "peace":
                self._peace_counter += 1
                if self._peace_counter >= PEACE_HOLD_FRAMES:
                    self._peace_counter = 0
                    if not e.in_panel(cursor[0], cursor[1]):
                        self._place_at(cursor)
            else:
                if g != "peace":
                    self._peace_counter = 0

        # Reset dwell ring if no dwell event arrived this frame
        if not has_dwell:
            self._dwell_prog = 0.0

        return {
            "ghost":          self._ghost_info(),
            "dwell_progress": self._dwell_prog,
            "dwell_cursor":   self._dwell_cursor,
            "status":         self._current_toast(),
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  Pinch helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_pinch_grab(self, cursor: tuple[int, int]):
        e = self.engine
        sx, sy = cursor

        if e.in_panel(sx, sy):
            # ── grab from component panel ──────────────────────────────
            tid = e.panel_hit_test(sx, sy)
            if tid:
                self._held_type    = tid
                self._held_comp_id = None
                self._drag_cursor  = cursor
                e.panel_selected   = tid
                self._toast(f"✊ Grabbed: {tid}")
        else:
            # ── grab existing board component ──────────────────────────
            wx, wy = e.to_world(sx, sy)
            comp   = e.hit_test(wx, wy, radius_multiplier=1.4)
            if comp:
                self._held_comp_id = comp.id
                self._held_type    = None
                self._drag_cursor  = cursor
                self._drag_offset  = (int(wx - comp.x), int(wy - comp.y))
                e.selected_id      = comp.id
                self._toast(f"✊ Grabbed #{comp.id} ({comp.label})")

    def _on_pinch_release(self, cursor: tuple[int, int]):
        e = self.engine
        sx, sy = cursor

        if self._held_type is not None:
            # Releasing a panel-dragged component
            if not e.in_panel(sx, sy):
                self._place_at(cursor, type_override=self._held_type)
            else:
                self._toast("↩ Cancelled")
            self._held_type = None

        elif self._held_comp_id is not None:
            # Releasing a board-dragged component
            comp = e.get_component(self._held_comp_id)
            if comp:
                e.selected_id = comp.id
                self._toast(f"📌 Placed #{comp.id} @ ({comp.x},{comp.y})")
            self._held_comp_id = None

        self._drag_cursor = cursor

    def _move_board_comp(self, cursor: tuple[int, int]):
        """Live-move a grabbed board component to follow the cursor."""
        e  = self.engine
        sx, sy = cursor
        comp   = e.get_component(self._held_comp_id)
        if comp is None:
            self._held_comp_id = None
            return
        wx, wy = e.to_world(sx, sy)
        ox, oy = self._drag_offset
        with e._lock:
            comp.x = e.snap(wx - ox)
            comp.y = e.snap(wy - oy)
        e._reroute_component_wires(comp.id)

    # ─────────────────────────────────────────────────────────────────────────
    #  Place helper
    # ─────────────────────────────────────────────────────────────────────────

    def _place_at(self, cursor: tuple[int, int], type_override: str = None):
        e   = self.engine
        tid = type_override or e.panel_selected
        if not tid:
            self._toast("⚠ No component selected")
            return
        sx, sy = cursor
        wx, wy = e.to_world(sx, sy)
        comp   = e.add_component(tid, wx, wy)
        if comp:
            e.selected_id = comp.id
            self._toast(f"✚ Placed {tid} #{comp.id} @ ({comp.x},{comp.y})")

    # ─────────────────────────────────────────────────────────────────────────
    #  Ghost info
    # ─────────────────────────────────────────────────────────────────────────

    def _ghost_info(self) -> Optional[dict]:
        if self._held_type is None:
            return None
        return {"type_id": self._held_type, "cursor": self._drag_cursor}

    # ─────────────────────────────────────────────────────────────────────────
    #  Toast status
    # ─────────────────────────────────────────────────────────────────────────

    def _toast(self, msg: str):
        self._status    = msg
        self._status_ts = time.time()

    def _current_toast(self) -> str:
        return self._status if (time.time() - self._status_ts) < 2.5 else ""

    # ─────────────────────────────────────────────────────────────────────────
    #  Overlay drawing  (call AFTER engine.render())
    # ─────────────────────────────────────────────────────────────────────────

    def draw_overlay(self, canvas: np.ndarray, overlay_info: dict):
        """
        Paint gesture feedback on top of the already-rendered circuit canvas.
        Pass the dict returned by process().
        """
        ghost  = overlay_info.get("ghost")
        dwell  = overlay_info.get("dwell_progress", 0.0)
        dcur   = overlay_info.get("dwell_cursor", (0, 0))
        status = overlay_info.get("status", "")

        # ── Ghost component preview while dragging from panel ─────────────
        if ghost:
            self._draw_ghost(canvas, ghost["type_id"], ghost["cursor"])

        # ── Dwell progress ring ───────────────────────────────────────────
        if dwell > 0.01:
            cx, cy  = int(dcur[0]), int(dcur[1])
            arc_deg = int(360 * dwell)
            col_arc = (0, 200, 255)
            ring_r  = DWELL_RING_RADIUS
            # Dim track
            cv2.circle(canvas, (cx, cy), ring_r, (40, 40, 40), 2, cv2.LINE_AA)
            # Progress arc (starts at top = -90°)
            cv2.ellipse(canvas, (cx, cy), (ring_r, ring_r),
                        -90, 0, arc_deg, col_arc, 3, cv2.LINE_AA)
            # Fill when complete
            if dwell >= 1.0:
                cv2.circle(canvas, (cx, cy), ring_r - 5,
                           (0, 255, 150), -1, cv2.LINE_AA)

        # ── Status toast ──────────────────────────────────────────────────
        if status:
            H, W = canvas.shape[:2]
            font  = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(status, font, 0.50, 1)
            bx, by = 12, H - 60
            # Semi-transparent backing
            ov = canvas.copy()
            cv2.rectangle(ov,
                          (bx - 6, by - th - 6),
                          (bx + tw + 10, by + 8),
                          (0, 8, 18), -1)
            cv2.addWeighted(ov, 0.80, canvas, 0.20, 0, canvas)
            cv2.putText(canvas, status, (bx, by),
                        font, 0.50, (0, 230, 180), 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    #  Ghost renderer
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_ghost(self, canvas: np.ndarray,
                    type_id: str, cursor: tuple[int, int]):
        """Translucent snapped-preview of the component being dragged."""
        try:
            from circuit_engine import CATALOG
        except ImportError:
            return

        d = CATALOG.get(type_id)
        if d is None:
            return

        e = self.engine
        sx, sy = cursor
        wx, wy = e.to_world(sx, sy)
        snap_x, snap_y = e.snap(wx), e.snap(wy)
        scr_x,  scr_y  = e.to_screen(snap_x, snap_y)

        cw  = int(d["w"] * e.zoom)
        ch  = int(d["h"] * e.zoom)
        col = d["color"]

        x0, y0 = scr_x, scr_y
        x1, y1 = scr_x + cw, scr_y + ch

        H, W = canvas.shape[:2]
        clip_w = e.board_w   # don't draw ghost over the panel
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(clip_w, x1), min(H, y1)

        if x1c > x0c and y1c > y0c:
            ov = canvas.copy()
            cv2.rectangle(ov, (x0c, y0c), (x1c, y1c), col, -1)
            cv2.addWeighted(ov, GHOST_ALPHA, canvas, 1.0 - GHOST_ALPHA, 0, canvas)
            cv2.rectangle(canvas, (x0c, y0c), (x1c, y1c), col, 2, cv2.LINE_AA)

        # Component label on the ghost
        lbl  = d.get("label", type_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = max(0.28, min(0.50, cw / 120.0))
        (tw, th), _ = cv2.getTextSize(lbl, font, fs, 1)
        cv2.putText(canvas, lbl,
                    (scr_x + cw // 2 - tw // 2, scr_y + ch // 2 + th // 2),
                    font, fs,
                    tuple(min(255, int(c * 1.8)) for c in col),
                    1, cv2.LINE_AA)

        # Cross-hair at snap-centre
        mid_x, mid_y = scr_x + cw // 2, scr_y + ch // 2
        arm = 10
        cv2.line(canvas, (mid_x - arm, mid_y), (mid_x + arm, mid_y),
                 (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (mid_x, mid_y - arm), (mid_x, mid_y + arm),
                 (255, 255, 255), 1, cv2.LINE_AA)