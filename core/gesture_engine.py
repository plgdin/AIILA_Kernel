from __future__ import annotations

import math
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class GestureType(str, Enum):
    PINCH   = "pinch"
    SWIPE   = "swipe"
    CRUMPLE = "crumple"
    THROW   = "throw"
    ROTATE  = "rotate"
    DWELL   = "dwell"
    PEACE   = "peace"

class PinchState(str, Enum):
    GRAB    = "grab"
    HOLDING = "holding"
    RELEASE = "release"
    DRAG    = "drag"

class SwipeDir(str, Enum):
    LEFT  = "left"
    RIGHT = "right"
    UP    = "up"
    DOWN  = "down"

# ═══════════════════════════════════════════════════════════════════════════════
# MediaPipe landmark indices & Connections (Restored)
# ═══════════════════════════════════════════════════════════════════════════════

WRIST      = 0
THUMB_CMC  = 1;  THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11; MIDDLE_TIP = 12
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15; RING_TIP   = 16
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19; PINKY_TIP  = 20

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
    GestureType.THROW:    90,
    GestureType.CRUMPLE:  80,
    GestureType.ROTATE:   60,
    GestureType.SWIPE:    50,
    GestureType.PINCH:    40,
    GestureType.DWELL:    30,
    GestureType.PEACE:    20,
}

@dataclass
class HandState:
    landmarks:   np.ndarray
    fingers:     list[bool]    
    curl:        list[float]   
    palm_center: np.ndarray    
    palm_normal: np.ndarray    
    spread:      float         
    hand_size:   float         
    hand_open:   bool          
    is_fist:     bool          
    handedness:  str           

@dataclass
class GestureEvent:
    gesture:    GestureType
    confidence: float
    cursor:     tuple[int, int]
    fingers:    list[bool]
    hand_open:  bool
    handedness: str
    data:       dict = field(default_factory=dict)
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

class _LandmarkSmoother:
    def __init__(self, alpha: float = 0.65):
        self._alpha = alpha
        self._prev: Optional[np.ndarray] = None 

    def update(self, raw: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = raw.copy()
            return raw
        out = self._alpha * raw + (1.0 - self._alpha) * self._prev
        self._prev = out
        return out

    def reset(self):
        self._prev = None

# ═══════════════════════════════════════════════════════════════════════════════
# Geometry helpers 
# ═══════════════════════════════════════════════════════════════════════════════

def _to_array(hand_landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

def _d2(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))

def _curl(lms: np.ndarray, tip: int, pip: int, mcp: int) -> float:
    v1 = lms[pip] - lms[mcp]
    v2 = lms[tip] - lms[pip]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 0.5
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.clip(math.acos(cos_a) / math.pi, 0.0, 1.0))

def _analyse_hand(lms: np.ndarray) -> tuple[list[bool], list[float]]:
    ext, curls = [], []
    wrist = lms[WRIST]
    ext.append(_d2(lms[THUMB_TIP], wrist) > _d2(lms[THUMB_IP], wrist))
    curls.append(_curl(lms, THUMB_TIP, THUMB_IP, THUMB_MCP))

    for tip_i, pip_i, mcp_i in zip(
        [INDEX_TIP,  MIDDLE_TIP,  RING_TIP,  PINKY_TIP],
        [INDEX_PIP,  MIDDLE_PIP,  RING_PIP,  PINKY_PIP],
        [INDEX_MCP,  MIDDLE_MCP,  RING_MCP,  PINKY_MCP],
    ):
        ext.append(bool(lms[tip_i][1] < lms[pip_i][1]))
        curls.append(_curl(lms, tip_i, pip_i, mcp_i))
    return ext, curls

# ═══════════════════════════════════════════════════════════════════════════════
# GestureEngine
# ═══════════════════════════════════════════════════════════════════════════════

class GestureEngine:
    def __init__(self, canvas_w=1000, canvas_h=700, ema_alpha=0.65, min_confidence=0.45):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.min_confidence = min_confidence
        self._smoothers = [_LandmarkSmoother(ema_alpha) for _ in range(2)]
        self._reset()
        self._last_fired: dict[GestureType, float] = {}
        self._hand_size_ref: float = 0.12 

    def _reset(self):
        for s in self._smoothers: s.reset()
        self._pinching = False
        self._pinch_anchor = self._last_pinch_pos = None
        self._palm_hist = deque(maxlen=14)
        self._spread_ts = deque(maxlen=40)
        self._last_spread_state = None
        self._last_spread_val = 0.0
        self._grab_frames = 0
        self._vel_hist = deque(maxlen=10)
        self._prev_two_angle = None
        self._dwell_frames = 0
        self._dwell_anchor = None
        self._peace_frames = 0

    def update(self, mediapipe_result, frame_shape) -> list[dict]:
        hands = mediapipe_result.hand_landmarks
        handedness = getattr(mediapipe_result, 'handedness', [])

        if not hands:
            self._reset()
            return []

        states: list[HandState] = []
        for i, raw_hand in enumerate(hands[:2]):
            s = self._smoothers[i].update(_to_array(raw_hand))
            ext, curls = _analyse_hand(s)
            hsize = max(_d2(s[WRIST], s[MIDDLE_MCP]), 1e-6)
            h_label = handedness[i][0].category_name if handedness and i < len(handedness) else "Right"

            states.append(HandState(
                landmarks=s, fingers=ext, curl=curls,
                palm_center=np.mean(s[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]], axis=0),
                palm_normal=self._get_normal(s),
                spread=float(np.mean([_d2(s[INDEX_TIP], s[MIDDLE_TIP]), _d2(s[MIDDLE_TIP], s[RING_TIP])])),
                hand_size=hsize,
                hand_open=sum(ext[1:]) >= 4,
                is_fist=sum(ext) <= 1,
                handedness=h_label
            ))

        self._hand_size_ref = 0.95 * self._hand_size_ref + 0.05 * states[0].hand_size

        p = states[0]
        cursor = (int(p.palm_center[0] * self.canvas_w), int(p.palm_center[1] * self.canvas_h))
        candidates: list[GestureEvent] = []

        def _try(ev: Optional[GestureEvent]):
            if ev and ev.confidence >= self.min_confidence: candidates.append(ev)

        _try(self._detect_pinch(p, cursor))
        self._palm_hist.append(p.palm_center[:2].copy())
        _try(self._detect_swipe(p, cursor))
        _try(self._detect_crumple(p, cursor))
        _try(self._detect_throw(p, cursor))
        _try(self._detect_dwell(p, cursor))
        _try(self._detect_peace(p, cursor))

        if len(states) >= 2:
            _try(self._detect_rotate(states[0], states[1], cursor))

        candidates.sort(key=lambda e: _PRIORITY.get(e.gesture, 0), reverse=True)
        seen, final = set(), []
        for ev in candidates:
            if ev.gesture not in seen:
                seen.add(ev.gesture)
                final.append(ev)

        return [ev.as_dict() for ev in final]

    def _get_normal(self, lms):
        a, b = lms[INDEX_MCP] - lms[WRIST], lms[PINKY_MCP] - lms[WRIST]
        n = np.cross(a, b)
        mag = np.linalg.norm(n)
        return (n / mag) if mag > 1e-6 else np.array([0.0, 0.0, 1.0])

    def _detect_pinch(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        dist = _d2(hs.landmarks[INDEX_TIP], hs.landmarks[THUMB_TIP])
        grab_t = self._hand_size_ref * 0.45 
        release_t = grab_t * 1.5
        
        _base = dict(gesture=GestureType.PINCH, confidence=0.0, cursor=cursor,
                     fingers=hs.fingers, hand_open=hs.hand_open, handedness=hs.handedness, data={})

        if not self._pinching and dist < grab_t:
            self._pinching = True
            self._pinch_anchor = hs.landmarks[INDEX_TIP][:2].copy()
            self._last_pinch_pos = self._pinch_anchor.copy()
            return GestureEvent(**{**_base, 'confidence': 0.9, 'data': {'state': PinchState.GRAB.value}})

        if self._pinching:
            if dist > release_t:
                self._pinching = False
                return GestureEvent(**{**_base, 'confidence': 0.95, 'data': {'state': PinchState.RELEASE.value}})
            
            cur = hs.landmarks[INDEX_TIP][:2].copy()
            delta = cur - self._last_pinch_pos
            self._last_pinch_pos = cur
            return GestureEvent(**{**_base, 'confidence': 0.85, 'data': {'state': PinchState.DRAG.value, 'delta': delta.tolist()}})
        return None

    def _detect_swipe(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        if len(self._palm_hist) < 14 or self._pinching: return None
        pts = np.array(self._palm_hist)
        delta = pts[-1] - pts[0]
        speed = np.max(np.abs(delta)) / 14
        if speed < 0.035: return None
        direction = (SwipeDir.RIGHT if delta[0] > 0 else SwipeDir.LEFT) if abs(delta[0]) > abs(delta[1]) else (SwipeDir.DOWN if delta[1] > 0 else SwipeDir.UP)
        return GestureEvent(gesture=GestureType.SWIPE, confidence=0.8, cursor=cursor, fingers=hs.fingers, 
                            hand_open=hs.hand_open, handedness=hs.handedness, data={'direction': direction.value})

    def _detect_crumple(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        state = 'open' if hs.spread > (self._hand_size_ref * 0.6) else 'closed'
        if state != self._last_spread_state:
            self._spread_ts.append(time.time())
            self._last_spread_state = state
        now = time.time()
        while self._spread_ts and self._spread_ts[0] < now - 1.3: self._spread_ts.popleft()
        if len(self._spread_ts) >= 6:
            self._spread_ts.clear()
            return GestureEvent(gesture=GestureType.CRUMPLE, confidence=0.9, cursor=cursor, fingers=hs.fingers, 
                                hand_open=hs.hand_open, handedness=hs.handedness)
        return None

    def _detect_throw(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        if hs.is_fist:
            self._grab_frames = min(self._grab_frames + 1, 120)
            self._vel_hist.append(hs.palm_center[:2].copy())
        elif hs.hand_open and self._grab_frames >= 8:
            if len(self._vel_hist) >= 3:
                vels = [np.linalg.norm(self._vel_hist[i+1]-self._vel_hist[i]) for i in range(len(self._vel_hist)-1)]
                if np.mean(vels) > 0.065:
                    self._grab_frames = 0
                    return GestureEvent(gesture=GestureType.THROW, confidence=0.88, cursor=cursor, fingers=hs.fingers, 
                                        hand_open=hs.hand_open, handedness=hs.handedness)
        return None

    def _detect_rotate(self, ha, hb, cursor) -> Optional[GestureEvent]:
        angle = math.degrees(math.atan2(hb.palm_center[1]-ha.palm_center[1], hb.palm_center[0]-ha.palm_center[0]))
        if self._prev_two_angle is not None:
            delta = (angle - self._prev_two_angle + 180) % 360 - 180
            if abs(delta) > 5.0:
                self._prev_two_angle = angle
                return GestureEvent(gesture=GestureType.ROTATE, confidence=0.7, cursor=cursor, fingers=ha.fingers, 
                                    hand_open=ha.hand_open, handedness="Both", data={'delta_deg': round(delta, 2)})
        self._prev_two_angle = angle
        return None

    def _detect_dwell(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        only_index = hs.fingers[1] and not any(hs.fingers[2:])
        if not only_index:
            self._dwell_frames = 0
            return None
        p2d = hs.palm_center[:2]
        if self._dwell_anchor is None or _d2(p2d, self._dwell_anchor) > (self._hand_size_ref * 0.15):
            self._dwell_anchor = p2d.copy()
            self._dwell_frames = 1
            return None
        self._dwell_frames += 1
        if self._dwell_frames >= 25:
            self._dwell_frames = 0
            return GestureEvent(gesture=GestureType.DWELL, confidence=0.92, cursor=cursor, fingers=hs.fingers, 
                                hand_open=hs.hand_open, handedness=hs.handedness)
        return None

    def _detect_peace(self, hs: HandState, cursor: tuple) -> Optional[GestureEvent]:
        if hs.fingers[1] and hs.fingers[2] and not any(hs.fingers[3:]) and hs.curl[0] > 0.4:
            self._peace_frames += 1
            if self._peace_frames >= 8:
                self._peace_frames = 0
                return GestureEvent(gesture=GestureType.PEACE, confidence=0.8, cursor=cursor, fingers=hs.fingers, 
                                    hand_open=hs.hand_open, handedness=hs.handedness)
        else:
            self._peace_frames = 0
        return None