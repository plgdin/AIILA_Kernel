"""
gesture_engine.py  —  AIILA Gesture Engine v4
===============================================
Changes from v3:

FIX 1 — CLAW_ROTATE now tracks wrist ROLL (hand twist around its own axis)
  Root cause in v3: palm normal projected onto XY plane barely moves during
  a wrist roll because the normal is nearly perpendicular to XY to begin
  with. A 90° wrist roll barely changed the atan2(ny, nx) angle.

  Fix: instead of using the palm normal direction, we now track the *INDEX
  finger knuckle angle relative to the PINKY knuckle* in camera space —
  i.e. the angle of the line INDEX_MCP→PINKY_MCP in 2-D (XY). This line
  rotates exactly as much as the hand rolls, is insensitive to hand tilt
  or translation, and needs no 3-D math. We call this the "roll angle".

  Additionally we also compute the "pitch" angle using the WRIST→MIDDLE_MCP
  vector in XY and average both signals to get a robust rotation estimate
  that covers wrist roll AND tilting the wrist up/down (pronation/
  supination). Signed delta is accumulated and fires when >= CLAW_ROT_THRESH.

FIX 2 — CRUMPLE is now a TWO-HAND gesture (paper crumple)
  Root cause in v3: single-hand open↔close flips are too easy to trigger
  accidentally (e.g. during a pinch or claw) and don't match the intuitive
  "crumple a piece of paper" motion.

  Fix: CRUMPLE now requires BOTH hands to be visible. It watches for:
    Phase 1 – SPREAD: both hands open, palms apart (distance > threshold)
    Phase 2 – SQUEEZE: both hands simultaneously close into fists AND move
              toward each other (inter-palm distance shrinks by >MIN_SHRINK)
  The phase transition must happen within CRUMPLE_WINDOW seconds.
  Confidence scales with how much the hands squeezed relative to the
  initial spread distance.

Other minor improvements:
  • CLAW_ROT_THRESH lowered slightly (5° instead of 6°) since the new
    roll-angle signal is more direct and less noisy.
  • _reset_crumple now clears two-hand state fully.
  • _reset_claw clears the new roll_angle_prev field.
"""

from __future__ import annotations

import math
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════════════

class GestureType(str, Enum):
    PINCH        = "pinch"
    SWIPE        = "swipe"
    CRUMPLE      = "crumple"
    THROW        = "throw"
    CLAW_ROTATE  = "claw_rotate"
    ROTATE       = "rotate"
    DWELL        = "dwell"
    PEACE        = "peace"


class PinchState(str, Enum):
    GRAB    = "grab"
    HOLDING = "holding"
    DRAG    = "drag"
    RELEASE = "release"


class SwipeDir(str, Enum):
    LEFT  = "left"
    RIGHT = "right"
    UP    = "up"
    DOWN  = "down"


# ═══════════════════════════════════════════════════════════════════════════════
#  Landmark indices & connections
# ═══════════════════════════════════════════════════════════════════════════════

WRIST       = 0
THUMB_CMC   = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP   = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP  = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP    = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

_PRIORITY: dict[GestureType, int] = {
    GestureType.THROW:       100,
    GestureType.CLAW_ROTATE:  90,
    GestureType.CRUMPLE:      80,
    GestureType.ROTATE:       70,
    GestureType.SWIPE:        60,
    GestureType.PINCH:        40,
    GestureType.DWELL:        30,
    GestureType.PEACE:        20,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HandState:
    landmarks:   np.ndarray   # (21,3) normalised 0-1
    fingers:     list[bool]   # [thumb, index, middle, ring, pinky]
    curl:        list[float]  # 0=straight, 1=fully curled
    palm_center: np.ndarray
    palm_normal: np.ndarray   # unit vector
    spread:      float        # mean fingertip spread (0-1)
    hand_size:   float        # wrist→middle-MCP distance
    hand_open:   bool         # ≥4 non-thumb fingers extended
    is_fist:     bool         # all 4 non-thumb fingers curl > 0.50
    is_claw:     bool         # all 5 fingers semi-curled (0.28–0.76)
    roll_angle:  float        # angle (deg) of INDEX_MCP→PINKY_MCP in XY — tracks wrist roll
    handedness:  str


@dataclass
class GestureEvent:
    gesture:    GestureType
    confidence: float
    cursor:     tuple[int, int]
    fingers:    list[bool]
    hand_open:  bool
    handedness: str
    data:       dict  = field(default_factory=dict)
    timestamp:  float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "gesture":    self.gesture.value,
            "confidence": round(self.confidence, 3),
            "cursor":     self.cursor,
            "fingers":    self.fingers,
            "hand_open":  self.hand_open,
            "handedness": self.handedness,
            "data":       self.data,
            "timestamp":  self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  EMA smoother
# ═══════════════════════════════════════════════════════════════════════════════

class _Smoother:
    def __init__(self, alpha: float = 0.60):
        self._a    = alpha
        self._prev: Optional[np.ndarray] = None

    def update(self, raw: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = raw.copy()
            return raw
        out = self._a * raw + (1.0 - self._a) * self._prev
        self._prev = out
        return out

    def reset(self):
        self._prev = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _to_array(hand_lms) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32)


def _d2(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _d3(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _curl_angle(lms: np.ndarray, tip: int, pip: int, mcp: int) -> float:
    v1 = lms[pip] - lms[mcp]
    v2 = lms[tip] - lms[pip]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.5
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.clip(math.acos(cos_a) / math.pi, 0.0, 1.0))


def _palm_normal(lms: np.ndarray) -> np.ndarray:
    a = lms[INDEX_MCP] - lms[WRIST]
    b = lms[PINKY_MCP] - lms[WRIST]
    n = np.cross(a, b)
    mag = np.linalg.norm(n)
    return n / mag if mag > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _roll_angle_deg(lms: np.ndarray) -> float:
    """
    Angle of the INDEX_MCP → PINKY_MCP vector in XY (camera) space.
    This line rotates directly with wrist roll/twist and is independent of
    hand translation or depth. Returns degrees in (-180, 180].

    We blend with the WRIST→MIDDLE_MCP angle (pitch proxy) using a 70/30
    weighting so subtle wrist tilts that don't fully roll the knuckle line
    still register.
    """
    # Primary: knuckle line across hand
    dx1 = float(lms[PINKY_MCP][0] - lms[INDEX_MCP][0])
    dy1 = float(lms[PINKY_MCP][1] - lms[INDEX_MCP][1])
    a1  = math.degrees(math.atan2(dy1, dx1))

    # Secondary: wrist → middle knuckle (captures pronation/supination)
    dx2 = float(lms[MIDDLE_MCP][0] - lms[WRIST][0])
    dy2 = float(lms[MIDDLE_MCP][1] - lms[WRIST][1])
    a2  = math.degrees(math.atan2(dy2, dx2))

    return 0.70 * a1 + 0.30 * a2


def _build_state(lms: np.ndarray, handedness: str, hand_size_ref: float) -> HandState:
    curls: list[float] = []
    ext:   list[bool]  = []

    tc = _curl_angle(lms, THUMB_TIP, THUMB_IP, THUMB_MCP)
    curls.append(tc)
    ext.append(_d3(lms[THUMB_TIP], lms[WRIST]) > _d3(lms[THUMB_IP], lms[WRIST]) * 1.05)

    for tip_i, pip_i, mcp_i in [
        (INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        (RING_TIP,   RING_PIP,   RING_MCP),
        (PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
    ]:
        c = _curl_angle(lms, tip_i, pip_i, mcp_i)
        curls.append(c)
        ext.append(bool(lms[tip_i][1] < lms[pip_i][1]) and c < 0.50)

    hsize = max(_d3(lms[WRIST], lms[MIDDLE_MCP]), 1e-6)
    is_fist = all(curls[i] > 0.50 for i in range(1, 5))
    is_claw = (
        0.22 <= curls[0] <= 0.75 and
        all(0.28 <= curls[i] <= 0.78 for i in range(1, 5))
    )

    spread = float(np.mean([
        _d2(lms[INDEX_TIP],  lms[MIDDLE_TIP]),
        _d2(lms[MIDDLE_TIP], lms[RING_TIP]),
        _d2(lms[RING_TIP],   lms[PINKY_TIP]),
    ])) / max(hsize, 1e-6)

    return HandState(
        landmarks   = lms,
        fingers     = ext,
        curl        = curls,
        palm_center = np.mean(
            lms[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]], axis=0
        ),
        palm_normal = _palm_normal(lms),
        spread      = spread,
        hand_size   = hsize,
        hand_open   = sum(ext[1:]) >= 4,
        is_fist     = is_fist,
        is_claw     = is_claw,
        roll_angle  = _roll_angle_deg(lms),
        handedness  = handedness,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  GestureEngine
# ═══════════════════════════════════════════════════════════════════════════════

class GestureEngine:
    """
    Usage
    -----
        engine = GestureEngine(canvas_w=1000, canvas_h=700)
        events = engine.update(mediapipe_result, frame.shape)
        # returns list[dict] sorted by priority, highest first.
    """

    # ── Pinch ───────────────────────────────────────────────────────────────
    PINCH_GRAB_RATIO    = 0.36
    PINCH_RELEASE_RATIO = 0.60
    PINCH_MIN_OPEN      = 2

    # ── Swipe ───────────────────────────────────────────────────────────────
    SWIPE_WIN           = 10
    SWIPE_MIN_SPEED     = 0.022
    SWIPE_CONSISTENCY   = 0.60

    # ── Throw ───────────────────────────────────────────────────────────────
    THROW_FIST_FRAMES   = 5
    THROW_VEL_THRESH    = 0.045

    # ── Claw Rotate ─────────────────────────────────────────────────────────
    CLAW_MIN_FRAMES     = 4
    CLAW_ROT_THRESH     = 5.0    # lowered from 6 — roll signal is cleaner
    CLAW_MAX_ACCUM      = 90.0

    # ── Two-hand Crumple ────────────────────────────────────────────────────
    # Phase 1 (spread): both hands open, at least this far apart (0-1 coords)
    CRUMPLE_MIN_SPREAD_DIST  = 0.20   # inter-palm distance to "arm" crumple
    # Phase 2 (squeeze): distance must shrink by at least this fraction of initial
    CRUMPLE_MIN_SHRINK_FRAC  = 0.35   # 35% closer than the armed distance
    CRUMPLE_FIST_REQUIRED    = 2      # how many non-thumb fingers must curl on each hand
    CRUMPLE_WINDOW           = 2.0    # seconds from arm to fire

    # ── Dwell ───────────────────────────────────────────────────────────────
    DWELL_FRAMES        = 26
    DWELL_MOVE_RATIO    = 0.18

    # ── Peace ───────────────────────────────────────────────────────────────
    PEACE_FRAMES        = 10

    def __init__(self,
                 canvas_w: int = 1000,
                 canvas_h: int = 700,
                 ema_alpha: float = 0.60,
                 min_confidence: float = 0.40):
        self.canvas_w       = canvas_w
        self.canvas_h       = canvas_h
        self.min_confidence = min_confidence
        self._smoothers     = [_Smoother(ema_alpha) for _ in range(2)]
        self._hand_size_ref = 0.12
        self._full_reset()

    # ── Resets ──────────────────────────────────────────────────────────────

    def _full_reset(self):
        for s in self._smoothers:
            s.reset()
        self._reset_pinch()
        self._reset_swipe()
        self._reset_throw()
        self._reset_claw()
        self._reset_crumple()
        self._reset_dwell()
        self._peace_frames    = 0
        self._prev_two_angle  = None

    def _reset_pinch(self):
        self._pinching        = False
        self._pinch_anchor_px: Optional[np.ndarray] = None
        self._pinch_last_px:   Optional[np.ndarray] = None

    def _reset_swipe(self):
        self._swipe_hist: deque = deque(maxlen=self.SWIPE_WIN)

    def _reset_throw(self):
        self._fist_frames = 0
        self._throw_armed = False
        self._throw_hist: deque = deque(maxlen=14)

    def _reset_claw(self):
        self._claw_frames         = 0
        self._claw_roll_angle_prev: Optional[float] = None   # ← new: track roll angle
        self._claw_accum          = 0.0
        self._claw_active         = False

    def _reset_crumple(self):
        # Two-hand crumple state machine
        self._crumple_armed      = False
        self._crumple_armed_dist: float = 0.0
        self._crumple_armed_ts:   float = 0.0

    def _reset_dwell(self):
        self._dwell_frames = 0
        self._dwell_anchor: Optional[np.ndarray] = None

    # ── Main update ─────────────────────────────────────────────────────────

    def update(self, mediapipe_result, frame_shape) -> list[dict]:
        hands      = mediapipe_result.hand_landmarks
        handedness = getattr(mediapipe_result, 'handedness', [])

        if not hands:
            self._full_reset()
            return []

        states: list[HandState] = []
        for i, raw_hand in enumerate(hands[:2]):
            lms = self._smoothers[i].update(_to_array(raw_hand))
            h   = (handedness[i][0].category_name
                   if handedness and i < len(handedness) else "Right")
            states.append(_build_state(lms, h, self._hand_size_ref))

        self._hand_size_ref = (0.96 * self._hand_size_ref
                                + 0.04 * states[0].hand_size)

        primary = states[0]
        cursor  = self._to_cursor(primary.palm_center)

        candidates: list[GestureEvent] = []

        def _try(ev: Optional[GestureEvent]):
            if ev and ev.confidence >= self.min_confidence:
                candidates.append(ev)

        _try(self._detect_throw(primary, cursor))
        _try(self._detect_claw_rotate(primary, cursor))
        _try(self._detect_swipe(primary, cursor))
        _try(self._detect_pinch(primary, cursor))
        _try(self._detect_dwell(primary, cursor))
        _try(self._detect_peace(primary, cursor))

        # Two-hand gestures
        if len(states) >= 2:
            _try(self._detect_crumple(states[0], states[1], cursor))
            _try(self._detect_two_hand_rotate(states[0], states[1], cursor))
        else:
            # No second hand: reset crumple so it re-arms properly
            self._reset_crumple()

        candidates.sort(key=lambda e: _PRIORITY.get(e.gesture, 0), reverse=True)
        seen: set[GestureType] = set()
        final: list[GestureEvent] = []
        for ev in candidates:
            if ev.gesture not in seen:
                seen.add(ev.gesture)
                final.append(ev)

        return [ev.as_dict() for ev in final]

    # ── Coordinate helpers ───────────────────────────────────────────────────

    def _to_cursor(self, pt: np.ndarray) -> tuple[int, int]:
        return (int(pt[0] * self.canvas_w), int(pt[1] * self.canvas_h))

    def _tip_cursor(self, lms: np.ndarray, tip: int) -> tuple[int, int]:
        return (int(lms[tip][0] * self.canvas_w),
                int(lms[tip][1] * self.canvas_h))

    # ═════════════════════════════════════════════════════════════════════════
    #  PINCH
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_pinch(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        if hs.is_fist:
            if self._pinching:
                self._reset_pinch()
            return None

        if sum(hs.fingers[1:]) < self.PINCH_MIN_OPEN and not self._pinching:
            return None

        dist  = _d2(hs.landmarks[INDEX_TIP], hs.landmarks[THUMB_TIP])
        ratio = dist / max(self._hand_size_ref, 1e-6)
        cur   = self._tip_cursor(hs.landmarks, INDEX_TIP)

        base = dict(
            gesture    = GestureType.PINCH,
            confidence = 0.0,
            cursor     = cur,
            fingers    = hs.fingers,
            hand_open  = hs.hand_open,
            handedness = hs.handedness,
            data       = {},
        )

        if not self._pinching and ratio < self.PINCH_GRAB_RATIO:
            self._pinching        = True
            cur_f                 = np.array(cur, dtype=float)
            self._pinch_anchor_px = cur_f.copy()
            self._pinch_last_px   = cur_f.copy()
            return GestureEvent(**{**base,
                'confidence': 0.92,
                'data': {'state': PinchState.GRAB.value},
            })

        if self._pinching:
            cur_f = np.array(cur, dtype=float)
            if ratio > self.PINCH_RELEASE_RATIO:
                self._reset_pinch()
                return GestureEvent(**{**base,
                    'confidence': 0.95,
                    'data': {'state': PinchState.RELEASE.value},
                })

            delta = cur_f - self._pinch_last_px
            self._pinch_last_px = cur_f.copy()
            moving = np.linalg.norm(delta) > 0.8

            state_val = PinchState.DRAG.value if moving else PinchState.HOLDING.value
            return GestureEvent(**{**base,
                'confidence': 0.88,
                'data': {
                    'state':    state_val,
                    'delta_px': (int(delta[0]), int(delta[1])),
                },
            })

        return None

    # ═════════════════════════════════════════════════════════════════════════
    #  SWIPE
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_swipe(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        if self._pinching or hs.is_fist or hs.is_claw:
            self._reset_swipe()
            return None

        tip = hs.landmarks[INDEX_TIP][:2].copy()
        self._swipe_hist.append(tip)

        if len(self._swipe_hist) < self.SWIPE_WIN:
            return None

        pts    = np.array(self._swipe_hist)
        deltas = np.diff(pts, axis=0)
        speeds = np.linalg.norm(deltas, axis=1)

        mean_speed = speeds.mean()
        if mean_speed < self.SWIPE_MIN_SPEED:
            return None

        total = pts[-1] - pts[0]
        if abs(total[0]) > abs(total[1]):
            direction = SwipeDir.RIGHT if total[0] > 0 else SwipeDir.LEFT
            signs     = np.sign(deltas[:, 0])
            expected  = 1 if direction == SwipeDir.RIGHT else -1
        else:
            direction = SwipeDir.DOWN if total[1] > 0 else SwipeDir.UP
            signs     = np.sign(deltas[:, 1])
            expected  = 1 if direction == SwipeDir.DOWN else -1

        consistency = np.mean(signs == expected)
        if consistency < self.SWIPE_CONSISTENCY:
            return None

        conf = float(min(1.0, mean_speed / self.SWIPE_MIN_SPEED * 0.6 + 0.4))
        self._reset_swipe()

        return GestureEvent(
            gesture    = GestureType.SWIPE,
            confidence = conf,
            cursor     = cursor,
            fingers    = hs.fingers,
            hand_open  = hs.hand_open,
            handedness = hs.handedness,
            data       = {
                'direction': direction.value,
                'speed':     round(float(mean_speed), 4),
            }
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  THROW
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_throw(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        palm_px = np.array([
            hs.palm_center[0] * self.canvas_w,
            hs.palm_center[1] * self.canvas_h,
        ])

        if hs.is_fist:
            self._fist_frames += 1
            self._throw_hist.append(palm_px.copy())
            if self._fist_frames >= self.THROW_FIST_FRAMES:
                self._throw_armed = True
            return None

        if self._throw_armed and hs.hand_open:
            self._throw_armed = False
            hist = list(self._throw_hist)
            if len(hist) >= 3:
                arr  = np.array(hist)
                vels = np.linalg.norm(np.diff(arr, axis=0), axis=1)
                mean_v01 = vels.mean() / self.canvas_w
                if mean_v01 > self.THROW_VEL_THRESH:
                    dvec = arr[-1] - arr[0]
                    mag  = np.linalg.norm(dvec)
                    dn   = (dvec / mag).tolist() if mag > 0 else [0.0, 0.0]
                    self._fist_frames = 0
                    self._throw_hist.clear()
                    return GestureEvent(
                        gesture    = GestureType.THROW,
                        confidence = min(1.0, mean_v01 / self.THROW_VEL_THRESH),
                        cursor     = cursor,
                        fingers    = hs.fingers,
                        hand_open  = hs.hand_open,
                        handedness = hs.handedness,
                        data       = {
                            'velocity':      round(mean_v01, 4),
                            'direction_vec': [round(v, 3) for v in dn],
                        }
                    )
            self._fist_frames = 0
            self._throw_hist.clear()

        if hs.hand_open and not self._throw_armed:
            self._fist_frames = 0
            self._throw_hist.clear()

        return None

    # ═════════════════════════════════════════════════════════════════════════
    #  CLAW ROTATE  — tracks wrist ROLL via knuckle-line angle in XY
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_claw_rotate(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        if not hs.is_claw:
            self._claw_frames = max(0, self._claw_frames - 2)
            if self._claw_frames == 0:
                self._claw_roll_angle_prev = None
                self._claw_accum           = 0.0
                self._claw_active          = False
            return None

        self._claw_frames = min(self._claw_frames + 1, 200)

        if self._claw_frames < self.CLAW_MIN_FRAMES:
            # Seed the reference angle so the first real delta is valid
            self._claw_roll_angle_prev = hs.roll_angle
            return None

        self._claw_active = True

        if self._claw_roll_angle_prev is None:
            self._claw_roll_angle_prev = hs.roll_angle
            return None

        # Signed angular delta — wrap to ±180°
        raw_delta = hs.roll_angle - self._claw_roll_angle_prev
        delta = (raw_delta + 180.0) % 360.0 - 180.0

        # Reject tiny jitter and physically impossible jumps
        if abs(delta) < 0.8 or abs(delta) > 50.0:
            self._claw_roll_angle_prev = hs.roll_angle
            return None

        self._claw_accum += delta
        self._claw_accum  = max(-self.CLAW_MAX_ACCUM,
                                min(self.CLAW_MAX_ACCUM, self._claw_accum))
        self._claw_roll_angle_prev = hs.roll_angle

        if abs(self._claw_accum) < self.CLAW_ROT_THRESH:
            return None

        fired            = self._claw_accum
        self._claw_accum = 0.0

        return GestureEvent(
            gesture    = GestureType.CLAW_ROTATE,
            confidence = min(1.0, abs(fired) / 18.0 + 0.60),
            cursor     = cursor,
            fingers    = hs.fingers,
            hand_open  = hs.hand_open,
            handedness = hs.handedness,
            data       = {
                'delta_deg': round(fired, 2),
                'direction': "cw" if fired > 0 else "ccw",
            }
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  TWO-HAND ROTATE
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_two_hand_rotate(self,
                                 ha: HandState,
                                 hb: HandState,
                                 cursor: tuple) -> Optional[GestureEvent]:
        angle = math.degrees(math.atan2(
            float(hb.palm_center[1] - ha.palm_center[1]),
            float(hb.palm_center[0] - ha.palm_center[0]),
        ))
        if self._prev_two_angle is None:
            self._prev_two_angle = angle
            return None
        delta = (angle - self._prev_two_angle + 180) % 360 - 180
        self._prev_two_angle = angle
        if abs(delta) < 4.0:
            return None
        return GestureEvent(
            gesture    = GestureType.ROTATE,
            confidence = min(1.0, abs(delta) / 40.0 + 0.55),
            cursor     = cursor,
            fingers    = ha.fingers,
            hand_open  = ha.hand_open,
            handedness = "Both",
            data       = {
                'delta_deg': round(delta, 2),
                'direction': "cw" if delta > 0 else "ccw",
            }
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  CRUMPLE  — two-hand paper crumple (spread → squeeze)
    #
    #  State machine:
    #    IDLE → ARMED  when both hands open AND inter-palm dist > MIN_SPREAD_DIST
    #    ARMED → FIRE  when both hands close (≥ FIST_REQUIRED curled fingers
    #                  on each hand) AND inter-palm dist shrank ≥ MIN_SHRINK_FRAC
    #                  of the armed distance — all within CRUMPLE_WINDOW seconds.
    #    FIRE  → IDLE  immediately after emitting event
    #    ARMED → IDLE  if CRUMPLE_WINDOW expires without squeeze
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_crumple(self,
                        ha: HandState,
                        hb: HandState,
                        cursor: tuple) -> Optional[GestureEvent]:
        inter_dist = float(_d2(ha.palm_center[:2], hb.palm_center[:2]))

        # Count curled non-thumb fingers on each hand
        curled_a = sum(1 for i in range(1, 5) if ha.curl[i] > 0.50)
        curled_b = sum(1 for i in range(1, 5) if hb.curl[i] > 0.50)

        both_open   = ha.hand_open and hb.hand_open
        both_closed = (curled_a >= self.CRUMPLE_FIST_REQUIRED and
                       curled_b >= self.CRUMPLE_FIST_REQUIRED)

        now = time.time()

        # ── Phase 2: already armed, check for squeeze ────────────────────────
        if self._crumple_armed:
            if now - self._crumple_armed_ts > self.CRUMPLE_WINDOW:
                # Timed out
                self._reset_crumple()
                return None

            if both_closed:
                shrink = (self._crumple_armed_dist - inter_dist) / max(self._crumple_armed_dist, 1e-6)
                if shrink >= self.CRUMPLE_MIN_SHRINK_FRAC:
                    self._reset_crumple()
                    conf = float(min(1.0, 0.70 + shrink * 0.60))
                    return GestureEvent(
                        gesture    = GestureType.CRUMPLE,
                        confidence = conf,
                        cursor     = cursor,
                        fingers    = ha.fingers,
                        hand_open  = False,
                        handedness = "Both",
                        data       = {
                            'shrink_frac':   round(shrink, 3),
                            'initial_dist':  round(self._crumple_armed_dist, 3),
                            'final_dist':    round(inter_dist, 3),
                        }
                    )
            return None

        # ── Phase 1: watch for open-hands spread ─────────────────────────────
        if both_open and inter_dist >= self.CRUMPLE_MIN_SPREAD_DIST:
            self._crumple_armed      = True
            self._crumple_armed_dist = inter_dist
            self._crumple_armed_ts   = now

        return None

    # ═════════════════════════════════════════════════════════════════════════
    #  DWELL
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_dwell(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        only_index = (hs.fingers[1]
                      and not hs.fingers[2]
                      and not hs.fingers[3]
                      and not hs.fingers[4])
        if not only_index:
            self._reset_dwell()
            return None

        tip_2d = hs.landmarks[INDEX_TIP][:2].copy()
        thresh = self._hand_size_ref * self.DWELL_MOVE_RATIO

        if (self._dwell_anchor is None
                or _d2(tip_2d, self._dwell_anchor) > thresh):
            self._dwell_anchor = tip_2d
            self._dwell_frames = 0
            return None

        self._dwell_frames += 1
        prog   = self._dwell_frames / self.DWELL_FRAMES
        cursor = self._tip_cursor(hs.landmarks, INDEX_TIP)

        if self._dwell_frames >= self.DWELL_FRAMES:
            self._reset_dwell()
            return GestureEvent(
                gesture    = GestureType.DWELL,
                confidence = 0.94,
                cursor     = cursor,
                fingers    = hs.fingers,
                hand_open  = hs.hand_open,
                handedness = hs.handedness,
                data       = {'progress': 1.0},
            )

        return GestureEvent(
            gesture    = GestureType.DWELL,
            confidence = 0.60 + 0.3 * prog,
            cursor     = cursor,
            fingers    = hs.fingers,
            hand_open  = hs.hand_open,
            handedness = hs.handedness,
            data       = {'progress': round(prog, 3)},
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  PEACE
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_peace(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        is_peace = (
            hs.fingers[1] and hs.fingers[2]
            and not hs.fingers[3]
            and not hs.fingers[4]
            and hs.curl[0] > 0.38
        )
        if is_peace:
            self._peace_frames += 1
            if self._peace_frames >= self.PEACE_FRAMES:
                self._peace_frames = 0
                return GestureEvent(
                    gesture    = GestureType.PEACE,
                    confidence = 0.85,
                    cursor     = cursor,
                    fingers    = hs.fingers,
                    hand_open  = hs.hand_open,
                    handedness = hs.handedness,
                )
        else:
            self._peace_frames = 0
        return None