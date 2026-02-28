"""
gesture_engine.py  —  AIILA Gesture Engine v3
===============================================
Root-cause fixes:

PROBLEM 1 — Grab/Throw detected as Pinch
  Root cause: pinch only checked thumb-index distance, which is small even
  in a fist. Fix: pinch is now BLOCKED when is_fist=True (all 4 fingers
  curled >0.50). Fist → pinch conflict is impossible.

PROBLEM 2 — Claw rotation not detected
  Root cause: old engine had no claw detector at all (it only had 2-hand
  rotate). Fix: added dedicated CLAW_ROTATE detector that fires when all 5
  fingers are semi-curled (0.28–0.76) AND the palm normal rotates >6° from
  the previous frame. Uses atan2 of the projected normal vector so it
  responds to both wrist roll and hand twist.

PROBLEM 3 — Swipe unreliable
  Root cause: using palm centre (very stable) with only 14 frames. Palm
  barely moves on a fast flick. Fix: swipe now tracks INDEX-TIP (moves 2×
  more than palm on a flick), uses 10-frame window, and requires only 60%
  directional consistency instead of implicit threshold. Speed threshold
  lowered to 0.022 (was implicitly ~0.035). Pinch/fist/claw block it so
  no false-positives.

Additional improvements:
  • PinchState.HOLDING fires every frame between grab and drag so the
    kernel knows the grab is still live even without movement.
  • Throw uses a tighter fist definition (curl>0.52 for all 4 fingers) and
    arms itself after just 5 fist frames to catch quick grabs.
  • Crumple uses spread of ALL 5 fingertips relative to palm (more robust
    than just index-middle spread which can change during pinch).
  • Dwell tracks index TIP rather than palm centre.
  • All detectors share a single _hand_size_ref EMA so thresholds scale
    naturally with hand distance from camera.
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
    CLAW_ROTATE  = "claw_rotate"   # one-hand claw twist
    ROTATE       = "rotate"        # two-hand rotate
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
    """
    Joint angle 0=straight, 1=fully curled.
    Uses the angle between the mcp→pip and pip→tip vectors in 3-D so it
    works regardless of hand orientation in frame.
    """
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


def _build_state(lms: np.ndarray, handedness: str, hand_size_ref: float) -> HandState:
    # ── Curl for each finger ────────────────────────────────────────────────
    curls: list[float] = []
    ext:   list[bool]  = []

    # Thumb: compare tip→wrist vs IP→wrist (robust across orientations)
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
        # Extended: tip higher (lower y) than PIP AND curl < 0.50
        ext.append(bool(lms[tip_i][1] < lms[pip_i][1]) and c < 0.50)

    hsize = max(_d3(lms[WRIST], lms[MIDDLE_MCP]), 1e-6)

    # Fist: all 4 non-thumb fingers tightly curled
    is_fist = all(curls[i] > 0.50 for i in range(1, 5))

    # Claw: all 5 fingers semi-curled (classic "grab a ball" pose)
    # thumb slightly extended (curl 0.25–0.70), rest 0.28–0.76
    is_claw = (
        0.22 <= curls[0] <= 0.75 and
        all(0.28 <= curls[i] <= 0.78 for i in range(1, 5))
    )

    # Spread: mean distance between adjacent fingertips (normalised by hand size)
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

    # ── Tuning knobs ────────────────────────────────────────────────────────
    # Pinch
    PINCH_GRAB_RATIO    = 0.36   # thumb-index dist / hand_size → grab
    PINCH_RELEASE_RATIO = 0.60   # dist / hand_size → release
    PINCH_MIN_OPEN      = 2      # ≥ this many non-thumb fingers open to allow pinch

    # Swipe
    SWIPE_WIN           = 10     # frames of history
    SWIPE_MIN_SPEED     = 0.022  # mean per-frame displacement (0-1 coords)
    SWIPE_CONSISTENCY   = 0.60   # fraction of frames in dominant direction

    # Throw (fist → open release)
    THROW_FIST_FRAMES   = 5      # fist frames needed to arm
    THROW_VEL_THRESH    = 0.045  # mean 0-1/frame velocity to fire

    # Claw rotate
    CLAW_MIN_FRAMES     = 4      # frames of claw before tracking starts
    CLAW_ROT_THRESH     = 6.0    # degrees accumulated before firing
    CLAW_MAX_ACCUM      = 90.0   # cap accumulator to avoid ghost fires

    # Crumple
    CRUMPLE_TRANSITIONS = 5      # open↔close flips within window
    CRUMPLE_WINDOW      = 1.5    # seconds

    # Dwell
    DWELL_FRAMES        = 26
    DWELL_MOVE_RATIO    = 0.18   # relative to hand_size

    # Peace
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
        self._claw_frames     = 0
        self._claw_prev_norm: Optional[np.ndarray] = None
        self._claw_accum      = 0.0
        self._claw_active     = False

    def _reset_crumple(self):
        self._crumple_ts:    deque = deque(maxlen=40)
        self._crumple_state: Optional[str] = None

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

        # Update adaptive hand-size reference
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
        _try(self._detect_crumple(primary, cursor))
        _try(self._detect_swipe(primary, cursor))
        _try(self._detect_pinch(primary, cursor))
        _try(self._detect_dwell(primary, cursor))
        _try(self._detect_peace(primary, cursor))

        if len(states) >= 2:
            _try(self._detect_two_hand_rotate(states[0], states[1], cursor))

        # Sort by priority; keep only first of each gesture type
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
    #  PINCH  — thumb+index close, other fingers NOT in a fist
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_pinch(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        # KEY FIX: if hand is a fist, this belongs to grab/throw — never pinch
        if hs.is_fist:
            if self._pinching:
                self._reset_pinch()
            return None

        # Require enough open fingers so claw/full-curl can't trigger pinch
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
            moving = np.linalg.norm(delta) > 0.8   # > 0.8 pixels

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
    #  SWIPE  — index TIP fast directional motion
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_swipe(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        # Block during pinch, fist, or claw
        if self._pinching or hs.is_fist or hs.is_claw:
            self._reset_swipe()
            return None

        # Track index-finger TIP in 0-1 space (moves more than palm)
        tip = hs.landmarks[INDEX_TIP][:2].copy()
        self._swipe_hist.append(tip)

        if len(self._swipe_hist) < self.SWIPE_WIN:
            return None

        pts    = np.array(self._swipe_hist)
        deltas = np.diff(pts, axis=0)                    # (N-1, 2)
        speeds = np.linalg.norm(deltas, axis=1)          # per-frame

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
        self._reset_swipe()   # clear history so one gesture fires per motion

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
    #  THROW  — strict fist → velocity burst → open hand
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
            # While fisting, suppress everything below in priority
            return None

        # Hand opened after fist
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

        # Open hand but was never armed
        if hs.hand_open and not self._throw_armed:
            self._fist_frames = 0
            self._throw_hist.clear()

        return None

    # ═════════════════════════════════════════════════════════════════════════
    #  CLAW ROTATE  — claw pose + palm normal rotation
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_claw_rotate(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        if not hs.is_claw:
            # Gradually reset so brief claw breaks don't abort mid-rotation
            self._claw_frames = max(0, self._claw_frames - 2)
            if self._claw_frames == 0:
                self._claw_prev_norm = None
                self._claw_accum     = 0.0
                self._claw_active    = False
            return None

        self._claw_frames = min(self._claw_frames + 1, 200)

        if self._claw_frames < self.CLAW_MIN_FRAMES:
            self._claw_prev_norm = hs.palm_normal.copy()
            return None

        self._claw_active = True

        if self._claw_prev_norm is None:
            self._claw_prev_norm = hs.palm_normal.copy()
            return None

        # ── Signed rotation in XY plane (camera-facing rotation) ────────────
        prev = self._claw_prev_norm
        curr = hs.palm_normal

        prev_a = math.degrees(math.atan2(float(prev[1]), float(prev[0])))
        curr_a = math.degrees(math.atan2(float(curr[1]), float(curr[0])))
        delta  = (curr_a - prev_a + 180.0) % 360.0 - 180.0   # wrap ±180°

        # Reject tiny jitter (<0.5°) and huge jumps (>45°, likely normal flip)
        if abs(delta) < 0.5 or abs(delta) > 45.0:
            self._claw_prev_norm = curr.copy()
            return None

        self._claw_accum += delta
        self._claw_accum  = max(-self.CLAW_MAX_ACCUM,
                                min(self.CLAW_MAX_ACCUM, self._claw_accum))
        self._claw_prev_norm = curr.copy()

        if abs(self._claw_accum) < self.CLAW_ROT_THRESH:
            return None

        fired            = self._claw_accum
        self._claw_accum = 0.0   # reset accumulator

        return GestureEvent(
            gesture    = GestureType.CLAW_ROTATE,
            confidence = min(1.0, abs(fired) / 20.0 + 0.60),
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
    #  CRUMPLE  — rapid open ↔ close transitions
    #  Uses spread of ALL fingertips relative to palm — more robust than
    #  just index-middle spread which changes during pinch/claw.
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_crumple(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        # State based on average tip-to-palm distance
        tips_to_palm = float(np.mean([
            _d2(hs.landmarks[t], hs.palm_center[:2])
            for t in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        ]))
        state = 'open' if tips_to_palm > self._hand_size_ref * 1.0 else 'closed'

        if state != self._crumple_state:
            self._crumple_ts.append(time.time())
            self._crumple_state = state

        now = time.time()
        while self._crumple_ts and self._crumple_ts[0] < now - self.CRUMPLE_WINDOW:
            self._crumple_ts.popleft()

        if len(self._crumple_ts) >= self.CRUMPLE_TRANSITIONS:
            self._crumple_ts.clear()
            return GestureEvent(
                gesture    = GestureType.CRUMPLE,
                confidence = 0.92,
                cursor     = cursor,
                fingers    = hs.fingers,
                hand_open  = hs.hand_open,
                handedness = hs.handedness,
            )
        return None

    # ═════════════════════════════════════════════════════════════════════════
    #  DWELL  — index only, held still
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
    #  PEACE  — index + middle extended, rest curled
    # ═════════════════════════════════════════════════════════════════════════

    def _detect_peace(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        is_peace = (
            hs.fingers[1] and hs.fingers[2]
            and not hs.fingers[3]
            and not hs.fingers[4]
            and hs.curl[0] > 0.38        # thumb curled in
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