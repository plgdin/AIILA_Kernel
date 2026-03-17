import cv2
import numpy as np
import time
import math
from PyQt6.QtWidgets import QWidget, QLabel
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

# CATALOG import from original ui_hub
from core.circuit_engine import CATALOG as _CIRCUIT_CATALOG

# ─────────────────────────────────────────────────────────────────────────────
#  PROJECTOR WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class ProjectorWindow(QWidget):
    W, H = 1920, 1080

    APPS = [
        ("Circuit Lab",  "⎔", (255, 200,   0)),
        ("Vision AI",    "◎", (  0, 200, 255)),
        ("Voice CMD",    "◉", (  0, 255, 140)),
        ("Settings",     "⚙", (180, 180, 180)),
        ("Diagnostics",  "▦", (255, 120,   0)),
        ("Deploy",       "▶", (100, 255, 100)),
        ("Analytics",    "▪", (200,   0, 255)),
        ("Network",      "⊞", (  0, 160, 255)),
    ]

    def __init__(self, x: int, y: int):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setGeometry(x, y, self.W, self.H)
        self.setStyleSheet("background: black;")

        self._canvas = QLabel(self)
        self._canvas.setGeometry(0, 0, self.W, self.H)

        self._t0           = time.time()
        self._phase        = "BOOT"
        self._fade         = 0.0
        self._hovered_app  = -1
        self._active_app   = -1
        self._log_lines    = [
            "AIILA KERNEL v2.4.1 — WORKBENCH EDITION",
            "Handlandmark model loaded  [OK]",
            "MediaPipe pipeline active  [OK]",
            "CircuitEngine v3.1 init   [OK]",
            "Pin-level routing enabled  [OK]",
            "Voice engine ready         [OK]",
            "AR canvas 1000×700 mapped  [OK]",
            "Projector output detected  [OK]",
            "All subsystems nominal     [READY]",
        ]
        self._log_scroll   = 0
        self._dock_rects: list[tuple[int,int,int,int]] = []
        self._metric_hist  = {k: [0.0]*80 for k in ("CPU","GPU","MEM","NET")}

        self._circuit_engine  = None
        self._circuit_active  = False
        self._ar_mode         = 'default'

    def update_display(self, ar_rgb: np.ndarray | None,
                       circuit_engine=None, circuit_active: bool = False,
                       ar_mode: str = 'default'):
        elapsed = time.time() - self._t0
        if self._phase == "BOOT"  and elapsed > 2.2:
            self._phase = "INTRO"
        if self._phase == "INTRO" and elapsed > 4.5:
            self._phase = "DESKTOP"

        self._circuit_engine = circuit_engine
        self._circuit_active = circuit_active
        self._ar_mode        = ar_mode

        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self._draw_background(frame, elapsed)

        if self._phase == "BOOT":
            self._draw_boot(frame, elapsed)
        elif self._phase == "INTRO":
            self._draw_intro(frame, elapsed)
        else:
            d = min(1.0, (elapsed - 4.5) / 1.0)
            self._draw_desktop(frame, elapsed, ar_rgb, d,
                               self._circuit_engine, self._circuit_active)

        if self._phase != "BOOT":
            self._draw_topbar(frame, elapsed)
            self._draw_taskbar(frame, elapsed)

        qi = QImage(frame.data, self.W, self.H, self.W * 3, QImage.Format.Format_BGR888)
        self._canvas.setPixmap(QPixmap.fromImage(qi.copy()))

    def _draw_background(self, f, t):
        drift_x = int(math.sin(t * 0.08) * 30)
        drift_y = int(math.cos(t * 0.05) * 20)
        gc = int(14 * self._fade)
        ac = int(8  * self._fade)
        for x in range((drift_x % 120) - 120, self.W + 120, 120):
            cv2.line(f, (x, 0), (x, self.H), (0, gc, gc), 1)
        for y in range((drift_y % 120) - 120, self.H + 120, 120):
            cv2.line(f, (0, y), (self.W, y), (0, gc, gc), 1)
        for x in range((drift_x % 40) - 40, self.W + 40, 40):
            cv2.line(f, (x, 0), (x, self.H), (0, ac, ac), 1)
        for y in range((drift_y % 40) - 40, self.H + 40, 40):
            cv2.line(f, (0, y), (self.W, y), (0, ac, ac), 1)

    def _draw_boot(self, f, t):
        lines = [
            (0.0, "AIILA NEURAL INTERFACE TERMINAL"),
            (0.3, "Initialising hardware abstraction layer..."),
            (0.6, "Loading hand landmarker model..."),
            (0.9, "Calibrating gesture pipeline..."),
            (1.2, "Mounting AR canvas..."),
            (1.5, "Projector handshake..."),
            (1.8, "All systems nominal."),
            (2.0, "BOOT COMPLETE"),
        ]
        cy = 460
        for delay, text in lines:
            if t > delay:
                a = min(1.0, (t - delay) / 0.25)
                col_g = int(255 * a) if text == "BOOT COMPLETE" else int(160 * a)
                col_b = int(255 * a)
                col   = (0, col_g, col_b) if text != "BOOT COMPLETE" else (0, 255, 80)
                th    = 2 if text == "BOOT COMPLETE" else 1
                fs    = 0.8 if text == "BOOT COMPLETE" else 0.55
                (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                cv2.putText(f, text, ((self.W - tw) // 2, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
            cy += 36

    def _draw_intro(self, f, t):
        a = min(1.0, (t - 2.2) / 0.8)
        c = int(255 * a)
        logo = "AIILA"
        (lw, lh), _ = cv2.getTextSize(logo, cv2.FONT_HERSHEY_DUPLEX, 9, 14)
        cv2.putText(f, logo, ((self.W - lw) // 2, 560),
                    cv2.FONT_HERSHEY_DUPLEX, 9, (0, int(200*a), c), 14, cv2.LINE_AA)
        cv2.putText(f, logo, ((self.W - lw) // 2, 560),
                    cv2.FONT_HERSHEY_DUPLEX, 9, (0, int(220*a), c), 2, cv2.LINE_AA)
        tag = "NEURAL INTERFACE TERMINAL  v2.4.1  —  WORKBENCH EDITION"
        (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)
        cv2.putText(f, tag, ((self.W - tw) // 2, 620),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, int(130*a), int(160*a)), 1, cv2.LINE_AA)
        rx = (self.W - 500) // 2
        cv2.line(f, (rx, 640), (rx + 500, 640), (0, int(80*a), int(120*a)), 1)

    def _draw_topbar(self, f, t):
        a   = min(1.0, (t - 2.2) / 0.6)
        c   = int(255 * a)
        dim = int(80  * a)
        cv2.rectangle(f, (0, 0), (self.W, 48), (0, int(12*a), int(16*a)), -1)
        cv2.line(f, (0, 48), (self.W, 48), (0, int(50*a), int(70*a)), 1)
        cv2.putText(f, "AIILA", (16, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, int(200*a), c), 2, cv2.LINE_AA)
        cv2.putText(f, "OS", (102, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, int(160*a), int(200*a)), 1, cv2.LINE_AA)
        cv2.line(f, (155, 10), (155, 38), (0, dim, dim), 1)
        cv2.putText(f, "NEURAL INTERFACE TERMINAL  —  WORKBENCH", (168, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, dim, dim), 1, cv2.LINE_AA)
        ts = time.strftime("%H:%M:%S")
        ds = time.strftime("%Y-%m-%d")
        (tw, _), _ = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.putText(f, ts, (self.W - tw - 16, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, int(200*a), c), 1, cv2.LINE_AA)
        (dw, _), _ = cv2.getTextSize(ds, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(f, ds, (self.W - dw - 16, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, dim, dim), 1, cv2.LINE_AA)
        for i, (lbl, col) in enumerate([
            ("KERNEL", (0, 255, 120)),
            ("AR",     (0, 200, 255)),
            ("VOICE",  (0, 255, 200)),
            ("CIRCUIT",(255, 200, 0)),
        ]):
            bx = self.W - 350 + i * 80
            pulse = abs(math.sin(t * 2 + i)) * 0.4 + 0.6
            dc = tuple(int(c2 * pulse * a) for c2 in col)
            cv2.circle(f, (bx, 24), 5, dc, -1, cv2.LINE_AA)
            cv2.putText(f, lbl, (bx + 8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, dc, 1, cv2.LINE_AA)

    def _draw_taskbar(self, f, t):
        a   = min(1.0, (t - 2.2) / 0.6)
        dim = int(80 * a)
        c   = int(255 * a)
        TY  = self.H - 40
        cv2.rectangle(f, (0, TY), (self.W, self.H), (0, int(10*a), int(14*a)), -1)
        cv2.line(f, (0, TY), (self.W, TY), (0, int(45*a), int(60*a)), 1)

        uptime = t
        h_ = int(uptime // 3600)
        m_ = int((uptime % 3600) // 60)
        s_ = int(uptime % 60)

        # [FIX 6] Show wire-draw instruction when in draw mode
        mode_info = f"AR MODE  {self._ar_mode.upper()}"
        if self._ar_mode == 'draw':
            mode_info = "DRAW MODE  |  PINCH PIN → DRAG → RELEASE PIN"

        items = [
            f"UPTIME  {h_:02d}:{m_:02d}:{s_:02d}",
            mode_info,
            "KERNEL  NOMINAL",
            "v2.4.1-WORKBENCH",
        ]
        spacing = self.W // len(items)
        for i, txt in enumerate(items):
            x = spacing * i + 20
            col_ = (0, int(200*a), c) if i == 1 and self._ar_mode == 'draw' else (0, dim, int(c * 0.7))
            cv2.putText(f, txt, (x, TY + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col_, 1, cv2.LINE_AA)
            if i > 0:
                cv2.line(f, (spacing * i, TY + 6), (spacing * i, self.H - 6),
                         (0, int(30*a), int(40*a)), 1)

    def _draw_desktop(self, f, t, ar_rgb, d, circuit_engine, circuit_active):
        c   = int(255 * d)
        dim = int(120 * d)
        PAD, TBH = 12, 48
        feed_x1, feed_y1 = PAD,  TBH + PAD
        feed_x2, feed_y2 = 980,  TBH + PAD + 600

        cv2.rectangle(f, (feed_x1, feed_y1), (feed_x2, feed_y2),
                      (0, int(40*d), int(60*d)), 1)
        self._corner_brackets(f, feed_x1, feed_y1, feed_x2, feed_y2,
                               (0, int(200*d), c), 20, 2)

        if ar_rgb is not None:
            try:
                roi = cv2.resize(ar_rgb, (feed_x2 - feed_x1 - 2, feed_y2 - feed_y1 - 2))
                f[feed_y1+1:feed_y2, feed_x1+1:feed_x2] = roi
            except Exception:
                pass
        else:
            cx_, cy_ = (feed_x1 + feed_x2) // 2, (feed_y1 + feed_y2) // 2
            r = int(60 + 10 * math.sin(t * 2))
            cv2.circle(f, (cx_, cy_), r, (0, dim//2, dim), 1, cv2.LINE_AA)
            cv2.putText(f, "AWAITING AR FEED", (cx_ - 100, cy_ + r + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, dim, dim), 1, cv2.LINE_AA)

        cv2.putText(f, "AR LIVE FEED", (feed_x1 + 4, feed_y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, dim, dim), 1, cv2.LINE_AA)
        sy = feed_y1 + int((t * 80) % (feed_y2 - feed_y1))
        scan_ov = f.copy()
        cv2.line(scan_ov, (feed_x1, sy), (feed_x2, sy), (0, 200, 255), 1)
        cv2.addWeighted(scan_ov, 0.25, f, 0.75, 0, f)

        RX = 996
        RW = self.W - RX - PAD
        if circuit_active and circuit_engine is not None:
            self._draw_circuit_panel(f, t, d, RX, TBH + PAD, RW, circuit_engine)
        else:
            self._draw_metrics(f, t, d, RX, TBH + PAD, RW)
            log_y1 = TBH + PAD + 260
            log_y2 = TBH + PAD + 530
            self._draw_kernel_log(f, d, RX, log_y1, RW, log_y2)
            dock_y = TBH + PAD + 545
            self._draw_app_dock(f, t, d, RX, dock_y, RW)

    def _draw_metrics(self, f, t, d, rx, ry, rw):
        c   = int(255 * d)
        dim = int(100 * d)
        ph  = 250
        cv2.rectangle(f, (rx, ry), (rx + rw, ry + ph), (0, int(30*d), int(45*d)), 1)
        self._corner_brackets(f, rx, ry, rx + rw, ry + ph, (0, int(180*d), c), 14, 1)
        cv2.putText(f, "SYSTEM METRICS", (rx + 6, ry - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, dim, dim), 1, cv2.LINE_AA)

        keys   = ["CPU", "GPU", "MEM", "NET"]
        colors = [(0, 200, 255), (0, 255, 140), (255, 180, 0), (200, 80, 255)]
        bases  = [45.0, 62.0, 71.0, 30.0]
        amps   = [25.0, 18.0, 8.0, 35.0]
        freqs  = [0.7,  0.5,  0.2,  1.1]
        for ki, key in enumerate(keys):
            val = bases[ki] + amps[ki] * (0.5 + 0.5 * math.sin(t * freqs[ki] + ki))
            self._metric_hist[key].append(val)
            self._metric_hist[key] = self._metric_hist[key][-80:]

        gw, gh = rw // 2 - 14, 100
        for i, (key, col) in enumerate(zip(keys, colors)):
            gx = rx + 8  + (i % 2) * (gw + 12)
            gy = ry + 14 + (i // 2) * (gh + 18)
            hist = self._metric_hist[key]
            val  = hist[-1]
            cv2.rectangle(f, (gx, gy), (gx + gw, gy + gh), (0, int(8*d), int(12*d)), -1)
            cv2.rectangle(f, (gx, gy), (gx + gw, gy + gh),
                          tuple(int(c2 * 0.25 * d) for c2 in col), 1)
            pts = []
            for j, v in enumerate(hist):
                px_ = gx + int(j * gw / max(len(hist)-1, 1))
                py_ = gy + gh - 4 - int(v / 100 * (gh - 8))
                pts.append((px_, py_))
            if len(pts) > 1:
                for j in range(len(pts) - 1):
                    cv2.line(f, pts[j], pts[j+1], tuple(int(c2 * d) for c2 in col), 1, cv2.LINE_AA)
            fill_pts = [(gx, gy+gh-4)] + pts + [(gx+gw, gy+gh-4)]
            fill_arr = np.array(fill_pts, dtype=np.int32)
            ov = f.copy()
            cv2.fillPoly(ov, [fill_arr], tuple(int(c2 * 0.12 * d) for c2 in col))
            cv2.addWeighted(ov, 0.6, f, 0.4, 0, f)
            cv2.putText(f, key, (gx + 4, gy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        tuple(int(c2 * 0.7 * d) for c2 in col), 1, cv2.LINE_AA)
            pct_str = f"{val:.0f}%"
            (pw, _), _ = cv2.getTextSize(pct_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(f, pct_str, (gx + gw - pw - 4, gy + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        tuple(int(c2 * d) for c2 in col), 1, cv2.LINE_AA)
            ty_ = gy + gh - 4 - int(0.80 * (gh - 8))
            cv2.line(f, (gx, ty_), (gx + gw, ty_), tuple(int(c2 * 0.3 * d) for c2 in col), 1)

    def _draw_kernel_log(self, f, d, rx, y1, rw, y2):
        dim = int(100 * d)
        c   = int(255 * d)
        cv2.rectangle(f, (rx, y1), (rx + rw, y2), (0, int(25*d), int(35*d)), -1)
        cv2.rectangle(f, (rx, y1), (rx + rw, y2), (0, int(40*d), int(55*d)), 1)
        self._corner_brackets(f, rx, y1, rx + rw, y2, (0, int(160*d), c), 12, 1)
        cv2.putText(f, "KERNEL LOG", (rx + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, dim, dim), 1, cv2.LINE_AA)
        line_h  = 18
        visible = (y2 - y1 - 10) // line_h
        lines   = self._log_lines[-visible:]
        for i, line in enumerate(lines):
            ly = y1 + 14 + i * line_h
            if "[OK]" in line or "[READY]" in line:
                col = (0, int(200*d), int(80*d))
            elif "[ERR]" in line or "[FAIL]" in line:
                col = (0, int(60*d), int(255*d))
            else:
                col = (0, int(140*d), int(160*d))
            cv2.putText(f, line[:int(rw / 7)], (rx + 8, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
        cy_ = y1 + 14 + len(lines) * line_h
        if int(time.time() * 2) % 2:
            cv2.putText(f, "▌", (rx + 8, cy_), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (0, int(200*d), int(80*d)), 1, cv2.LINE_AA)

    def _draw_app_dock(self, f, t, d, rx, ry, rw):
        c    = int(255 * d)
        cols = 4
        rows = math.ceil(len(self.APPS) / cols)
        cell_w = rw // cols
        cell_h = 56
        total_h = rows * cell_h + 20
        cv2.rectangle(f, (rx, ry), (rx + rw, ry + total_h), (0, int(20*d), int(28*d)), -1)
        cv2.rectangle(f, (rx, ry), (rx + rw, ry + total_h), (0, int(35*d), int(50*d)), 1)
        self._corner_brackets(f, rx, ry, rx + rw, ry + total_h, (0, int(150*d), c), 12, 1)
        cv2.putText(f, "APP DOCK", (rx + 6, ry - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, int(100*d), int(120*d)), 1, cv2.LINE_AA)
        self._dock_rects.clear()
        for i, (name, icon, accent) in enumerate(self.APPS):
            col_i = i % cols
            row_i = i // cols
            cx_   = rx + col_i * cell_w + cell_w // 2
            cy_   = ry + 14 + row_i * cell_h + cell_h // 2
            is_hov = (i == self._hovered_app)
            is_act = (i == self._active_app)
            if is_hov or is_act:
                pulse = 0.6 + 0.4 * abs(math.sin(t * 4))
                gc_ = tuple(int(v * pulse * d) for v in accent)
                cv2.circle(f, (cx_, cy_), 26, gc_, 1, cv2.LINE_AA)
                cv2.circle(f, (cx_, cy_), 22, tuple(int(v * 0.18 * d) for v in accent), -1)
            else:
                cv2.circle(f, (cx_, cy_), 22, (0, int(18*d), int(24*d)), -1)
                cv2.circle(f, (cx_, cy_), 22, tuple(int(v * 0.35 * d) for v in accent), 1, cv2.LINE_AA)
            icon_ch = name[0]
            (iw, ih), _ = cv2.getTextSize(icon_ch, cv2.FONT_HERSHEY_DUPLEX, 0.7, 1)
            cv2.putText(f, icon_ch, (cx_ - iw//2, cy_ + ih//2), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                        tuple(int(v * d) for v in accent), 1, cv2.LINE_AA)
            (nw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
            cv2.putText(f, name, (cx_ - nw//2, cy_ + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        tuple(int(v * 0.7 * d) for v in accent), 1, cv2.LINE_AA)
            self._dock_rects.append((cx_ - 24, cy_ - 24, cx_ + 24, cy_ + 24))

    # ─────────────────────────────────────────────────────────────────────────
    #  [FIX 1] CIRCUIT PANEL — safe render with override_w/override_h
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_circuit_panel(self, f, t, d, rx, ry, rw, engine):
        c    = int(255 * d)
        dim  = int(120 * d)
        font = cv2.FONT_HERSHEY_SIMPLEX

        bh = 380
        cv2.rectangle(f, (rx, ry), (rx + rw, ry + bh), (0, int(18*d), int(28*d)), -1)
        cv2.rectangle(f, (rx, ry), (rx + rw, ry + bh), (0, int(120*d), c), 1)
        self._corner_brackets(f, rx, ry, rx + rw, ry + bh, (0, int(200*d), c), 14, 1)
        cv2.putText(f, "CIRCUIT BOARD", (rx + 6, ry - 6),
                    font, 0.38, (0, dim, dim), 1, cv2.LINE_AA)

        # Component and wire count overlay
        n_comps = len(engine.components)
        n_wires = len(engine.wires)
        cv2.putText(f, f"{n_comps} comps  {n_wires} wires",
                    (rx + rw - 120, ry - 6), font, 0.32, (0, dim, dim), 1, cv2.LINE_AA)

        # [FIX 1] Render using override params — NEVER mutates engine state
        try:
            tmp = np.zeros((bh - 2, rw - 2, 3), dtype=np.uint8)
            engine.render_board_only(tmp, override_w=rw - 2, override_h=bh - 2)
            f[ry+1:ry+bh-1, rx+1:rx+rw-1] = tmp
        except Exception as e:
            cv2.putText(f, f"CIRCUIT: {str(e)[:30]}", (rx + 8, ry + bh//2),
                        font, 0.4, (0, int(100*d), int(120*d)), 1, cv2.LINE_AA)

        # ── Component picker ─────────────────────────────────────────────────
        py  = ry + bh + 10
        ph  = self.H - py - 50
        cv2.rectangle(f, (rx, py), (rx + rw, py + ph), (0, int(12*d), int(18*d)), -1)
        cv2.rectangle(f, (rx, py), (rx + rw, py + ph), (0, int(60*d), int(80*d)), 1)
        cv2.putText(f, "COMPONENTS", (rx + 6, py - 6),
                    font, 0.38, (0, dim, dim), 1, cv2.LINE_AA)

        sel      = getattr(engine, 'panel_selected', 'resistor')
        row_h    = 22
        visible  = max(1, (ph - 6) // row_h)
        items    = [it for it in engine._panel_items if it[0] == "comp"]
        sel_idx  = next((i for i, it in enumerate(items) if it[1] == sel), 0)
        start    = max(0, sel_idx - visible // 2)
        end      = min(len(items), start + visible)

        for i, (kind, tid, lbl) in enumerate(items[start:end]):
            iy = py + 4 + i * row_h
            if iy + row_h > py + ph:
                break
            is_sel = (tid == sel)
            bg = (0, int(35*d), int(55*d)) if is_sel else (0, 0, 0)
            cv2.rectangle(f, (rx + 1, iy), (rx + rw - 1, iy + row_h - 1), bg, -1)
            # [FIX 2] Use module-level _CIRCUIT_CATALOG
            d_  = _CIRCUIT_CATALOG.get(tid, {})
            col = d_.get("color", (80, 80, 80))
            cv2.rectangle(f, (rx + 4, iy + 5), (rx + 14, iy + row_h - 5),
                          tuple(int(cc * d) for cc in col), -1)
            text_c = (0, int(200*d), c) if is_sel else (int(160*d), int(200*d), int(220*d))
            cv2.putText(f, lbl[:14], (rx + 18, iy + 15), font, 0.30, text_c, 1, cv2.LINE_AA)
            # Show pin count
            n_pins = len(d_.get("pins", []))
            cv2.putText(f, f"{n_pins}p", (rx + rw - 28, iy + 15),
                        font, 0.25, (0, int(80*d), int(100*d)), 1, cv2.LINE_AA)
            if engine.sim_running and is_sel:
                cv2.putText(f, "►SIM", (rx + rw - 50, iy + 15), font, 0.25,
                            (80, 255, 80), 1, cv2.LINE_AA)

        if len(items) > visible:
            bar_h = max(8, int(ph * visible / len(items)))
            bar_y = py + int(start / len(items) * ph)
            cv2.rectangle(f, (rx + rw - 4, bar_y), (rx + rw - 2, bar_y + bar_h),
                          (0, int(100*d), c), -1)

        if engine.sim_running:
            elapsed = time.time() - engine._sim_t0
            pulse   = 0.6 + 0.4 * abs(math.sin(elapsed * 3))
            sc      = tuple(int(cc * pulse * d) for cc in (80, 255, 80))
            cv2.circle(f, (rx + rw - 14, ry + 8), 5, sc, -1, cv2.LINE_AA)
            cv2.putText(f, f"SIM {elapsed:.0f}s", (rx + rw - 60, ry + 12),
                        font, 0.28, sc, 1, cv2.LINE_AA)

    @staticmethod
    def _corner_brackets(f, x1, y1, x2, y2, col, size, thick):
        for pts in [
            [(x1, y1+size),(x1,y1),(x1+size,y1)],
            [(x2-size,y1),(x2,y1),(x2,y1+size)],
            [(x1,y2-size),(x1,y2),(x1+size,y2)],
            [(x2-size,y2),(x2,y2),(x2,y2-size)],
        ]:
            for i in range(len(pts)-1):
                cv2.line(f, pts[i], pts[i+1], col, thick, cv2.LINE_AA)

    def push_log(self, line: str):
        self._log_lines.append(line)
        if len(self._log_lines) > 200:
            self._log_lines = self._log_lines[-150:]

    def mouseMoveEvent(self, event):
        if self._phase != "DESKTOP":
            return
        x, y = event.pos().x(), event.pos().y()
        self._hovered_app = -1
        for i, (x1, y1, x2, y2) in enumerate(self._dock_rects):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._hovered_app = i
                break
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if self._phase != "DESKTOP":
            return
        x, y = event.pos().x(), event.pos().y()
        for i, (x1, y1, x2, y2) in enumerate(self._dock_rects):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._active_app = i if self._active_app != i else -1
                self.push_log(f"APP LAUNCHED: {self.APPS[i][0]}")
                return

    def mouseDoubleClickEvent(self, e):
        self.close()
