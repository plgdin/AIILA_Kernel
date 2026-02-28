import cv2
import numpy as np
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  COMPONENT CATALOG
# ─────────────────────────────────────────────
CATALOG = {
    # Microcontrollers
    "esp32":         {"category": "MCU",         "label": "ESP32",        "color": (200, 133,  61), "w": 80, "h": 100,
                      "pins": ["3V3","GND","EN","D34","D35","D32","D33","D25","D26","D27","D14","D12","D13","D2","D4","RX2","TX2","D5","D18","D19","D21","RX0","TX0","D22","D23"],
                      "props": {"board": "ESP32 Dev Module", "cpu": "240MHz", "flash": "4MB", "voltage": "3.3V"}},
    "arduino_uno":   {"category": "MCU",         "label": "Uno",          "color": (193, 122,   0), "w": 80, "h": 100,
                      "pins": ["RESET","3.3V","5V","GND","VIN","A0","A1","A2","A3","A4","A5","D0","D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13"],
                      "props": {"board": "Arduino Uno", "cpu": "16MHz", "flash": "32KB", "voltage": "5V"}},
    "esp8266":       {"category": "MCU",         "label": "ESP8266",      "color": (32,  160, 232), "w": 70, "h": 90,
                      "pins": ["RST","A0","D0","D5","D6","D7","D8","3V3","GND","D4","D3","D2","D1"],
                      "props": {"board": "NodeMCU 1.0", "cpu": "80MHz", "flash": "4MB", "voltage": "3.3V"}},
    "rpi_pico":      {"category": "MCU",         "label": "Pico",         "color": (74,  26,  197), "w": 80, "h": 100,
                      "pins": ["GP0","GP1","GND","GP2","GP3","GP4","GP5","GP6","GP7","GP8","GP9","GP10","GP11","GP12","GP13","GP14","GP15","3V3","VSYS","VBUS"],
                      "props": {"board": "Raspberry Pi Pico", "cpu": "133MHz", "flash": "2MB", "voltage": "3.3V"}},
    "stm32":         {"category": "MCU",         "label": "STM32",        "color": (184,  112,  0), "w": 80, "h": 100,
                      "pins": ["VBAT","PC13","PC14","PC15","PA0","PA1","PA2","PA3","PA4","PA5","PA6","PA7","PB0","PB1","GND","3.3V","5V"],
                      "props": {"board": "Blue Pill", "cpu": "72MHz", "flash": "64KB", "voltage": "3.3V"}},

    # Passives
    "resistor":      {"category": "Passive",     "label": "RES",          "color": (20, 105, 139), "w": 50, "h": 20,
                      "pins": ["1", "2"],
                      "props": {"resistance": "1kΩ", "tolerance": "5%", "power": "0.25W"}},
    "capacitor":     {"category": "Passive",     "label": "CAP",          "color": (100, 100, 100), "w": 24, "h": 40,
                      "pins": ["+", "-"],
                      "props": {"capacitance": "100nF", "voltage": "50V", "type": "Ceramic"}},
    "inductor":      {"category": "Passive",     "label": "IND",          "color": (130, 130, 130), "w": 50, "h": 20,
                      "pins": ["1", "2"],
                      "props": {"inductance": "10µH", "current": "1A"}},
    "potentiometer": {"category": "Passive",     "label": "POT",          "color": (110, 110, 110), "w": 40, "h": 40,
                      "pins": ["1", "W", "2"],
                      "props": {"resistance": "10kΩ", "type": "Linear"}},

    # LEDs & Displays
    "led_red":       {"category": "LED",         "label": "LED RED",      "color": (0,   0,  255), "w": 20, "h": 30,
                      "pins": ["A", "K"],
                      "props": {"Vf": "2.0V", "If": "20mA", "color": "Red"}},
    "led_green":     {"category": "LED",         "label": "LED GRN",      "color": (0, 255,   0), "w": 20, "h": 30,
                      "pins": ["A", "K"],
                      "props": {"Vf": "2.1V", "If": "20mA", "color": "Green"}},
    "led_blue":      {"category": "LED",         "label": "LED BLU",      "color": (255,  50,  50), "w": 20, "h": 30,
                      "pins": ["A", "K"],
                      "props": {"Vf": "3.2V", "If": "20mA", "color": "Blue"}},
    "led_rgb":       {"category": "LED",         "label": "RGB LED",      "color": (255,   0, 255), "w": 24, "h": 30,
                      "pins": ["R", "G", "B", "GND"],
                      "props": {"type": "Common Cathode", "If": "20mA"}},
    "ws2812":        {"category": "LED",         "label": "WS2812B",      "color": (0, 140, 255), "w": 20, "h": 20,
                      "pins": ["VCC", "GND", "DIN", "DOUT"],
                      "props": {"protocol": "NeoPixel", "voltage": "5V", "current": "60mA"}},
    "oled":          {"category": "Display",     "label": "OLED",         "color": (102,  68,  68), "w": 70, "h": 50,
                      "pins": ["GND", "VCC", "SCL", "SDA"],
                      "props": {"driver": "SSD1306", "res": "128x64", "addr": "0x3C"}},
    "lcd16x2":       {"category": "Display",     "label": "LCD 16x2",     "color": (50, 100,  50), "w": 90, "h": 40,
                      "pins": ["VSS","VDD","V0","RS","RW","E","D4","D5","D6","D7","A","K"],
                      "props": {"size": "16x2", "backlight": "Yes"}},

    # Sensors
    "dht22":         {"category": "Sensor",      "label": "DHT22",        "color": (136, 170,   0), "w": 30, "h": 50,
                      "pins": ["VCC", "DATA", "NC", "GND"],
                      "props": {"temp": "-40~80°C", "hum": "0~100%", "acc": "±0.5°C"}},
    "ldr":           {"category": "Sensor",      "label": "LDR",          "color": (34, 169,  170), "w": 20, "h": 20,
                      "pins": ["1", "2"],
                      "props": {"dark": ">1MΩ", "light": "1-10kΩ"}},
    "pir":           {"category": "Sensor",      "label": "PIR",          "color": (0,  68, 136), "w": 35, "h": 35,
                      "pins": ["VCC", "OUT", "GND"],
                      "props": {"range": "7m", "angle": "120°", "voltage": "5-20V"}},
    "ultrasonic":    {"category": "Sensor",      "label": "HC-SR04",      "color": (170, 136,  68), "w": 60, "h": 30,
                      "pins": ["VCC", "TRIG", "ECHO", "GND"],
                      "props": {"range": "2-400cm", "acc": "3mm", "freq": "40kHz"}},
    "mpu6050":       {"category": "Sensor",      "label": "MPU-6050",     "color": (85, 102,  68), "w": 40, "h": 40,
                      "pins": ["VCC","GND","SCL","SDA","AD0","INT"],
                      "props": {"axes": "6-DOF", "iface": "I2C", "addr": "0x68"}},
    "bmp280":        {"category": "Sensor",      "label": "BMP280",       "color": (102,  85,  51), "w": 35, "h": 20,
                      "pins": ["VCC","GND","SCL","SDA","CSB","SDO"],
                      "props": {"pressure": "300-1100hPa", "acc": "±1hPa"}},
    "ds18b20":       {"category": "Sensor",      "label": "DS18B20",      "color": (85,  85, 102), "w": 20, "h": 40,
                      "pins": ["GND", "DATA", "VCC"],
                      "props": {"range": "-55~125°C", "acc": "±0.5°C", "iface": "1-Wire"}},
    "soil_moisture": {"category": "Sensor",      "label": "Soil",         "color": (51, 119,  68), "w": 20, "h": 60,
                      "pins": ["VCC","GND","AOUT","DOUT"],
                      "props": {"output": "Analog+Digital", "voltage": "3.3-5V"}},

    # Actuators
    "servo":         {"category": "Actuator",    "label": "SERVO",        "color": (0, 102, 255), "w": 40, "h": 50,
                      "pins": ["GND", "VCC", "PWM"],
                      "props": {"torque": "1.8kgcm", "angle": "0-180°", "voltage": "4.8-6V"}},
    "stepper":       {"category": "Actuator",    "label": "STEPPER",      "color": (0,  68, 204), "w": 50, "h": 50,
                      "pins": ["IN1","IN2","IN3","IN4","VCC"],
                      "props": {"steps": "64", "gear": "1/64", "voltage": "5V"}},
    "dc_motor":      {"category": "Actuator",    "label": "DC MOTOR",     "color": (0,  68, 136), "w": 45, "h": 45,
                      "pins": ["M+", "M-"],
                      "props": {"voltage": "3-6V", "rpm": "200RPM", "current": "200mA"}},
    "buzzer":        {"category": "Actuator",    "label": "BUZZER",       "color": (68,  68,  68), "w": 28, "h": 28,
                      "pins": ["+", "-"],
                      "props": {"type": "Active", "freq": "2.5kHz", "voltage": "3-24V"}},
    "relay":         {"category": "Actuator",    "label": "RELAY",        "color": (153, 102,  51), "w": 60, "h": 40,
                      "pins": ["VCC","GND","IN","NC","COM","NO"],
                      "props": {"coil": "5V", "contact": "10A 250VAC", "type": "SPDT"}},
    "l298n":         {"category": "Actuator",    "label": "L298N",        "color": (0,  68, 153), "w": 70, "h": 60,
                      "pins": ["IN1","IN2","IN3","IN4","ENA","ENB","VCC","GND","12V"],
                      "props": {"maxA": "2A", "maxV": "46V", "channels": "2"}},

    # Communication
    "nrf24":         {"category": "Comms",       "label": "nRF24",        "color": (204, 136,   0), "w": 40, "h": 50,
                      "pins": ["GND","VCC","CE","CSN","SCK","MOSI","MISO","IRQ"],
                      "props": {"freq": "2.4GHz", "range": "100m", "rate": "2Mbps"}},
    "hc05":          {"category": "Comms",       "label": "HC-05 BT",     "color": (204,   0,   0), "w": 40, "h": 50,
                      "pins": ["VCC","GND","TXD","RXD","STATE","EN"],
                      "props": {"proto": "BT 2.0", "range": "10m", "baud": "9600"}},
    "lora":          {"category": "Comms",       "label": "LoRa",         "color": (102,  34, 102), "w": 45, "h": 55,
                      "pins": ["GND","VCC","MISO","MOSI","SCK","NSS","RST","DIO0"],
                      "props": {"freq": "433MHz", "range": "10km", "iface": "SPI"}},

    # Power
    "power_5v":      {"category": "Power",       "label": "5V PSU",       "color": (0,   0, 200), "w": 30, "h": 40,
                      "pins": ["V+", "GND"],
                      "props": {"voltage": "5V", "current": "2A"}},
    "power_3v3":     {"category": "Power",       "label": "3V3 REG",      "color": (0,  68, 200), "w": 20, "h": 30,
                      "pins": ["IN", "OUT", "GND"],
                      "props": {"output": "3.3V", "maxA": "1A"}},
    "battery":       {"category": "Power",       "label": "LiPo",         "color": (0, 136,  68), "w": 50, "h": 30,
                      "pins": ["+", "-"],
                      "props": {"voltage": "3.7V", "cap": "1000mAh"}},

    # Input
    "pushbutton":    {"category": "Input",       "label": "BTN",          "color": (80,  80,  80), "w": 20, "h": 20,
                      "pins": ["1A","1B","2A","2B"],
                      "props": {"rating": "50mA 12V", "type": "Momentary"}},
    "rotary_enc":    {"category": "Input",       "label": "ENCODER",      "color": (100, 100, 100), "w": 35, "h": 35,
                      "pins": ["CLK","DT","SW","VCC","GND"],
                      "props": {"ppr": "20", "type": "Incremental"}},
    "joystick":      {"category": "Input",       "label": "JOYSTICK",     "color": (68,  68,  68), "w": 40, "h": 40,
                      "pins": ["VRx","VRy","SW","VCC","GND"],
                      "props": {"axes": "2", "output": "Analog"}},
    "keypad4x4":     {"category": "Input",       "label": "KEYPAD",       "color": (51,  68,  85), "w": 60, "h": 60,
                      "pins": ["R1","R2","R3","R4","C1","C2","C3","C4"],
                      "props": {"layout": "4x4", "keys": "16"}},

    # Semiconductors
    "npn":           {"category": "Semi",        "label": "NPN",          "color": (136, 102,  68), "w": 20, "h": 30,
                      "pins": ["B", "C", "E"],
                      "props": {"Vceo": "40V", "Ic": "600mA", "hFE": "100-300"}},
    "mosfet_n":      {"category": "Semi",        "label": "N-MOS",        "color": (102,  68, 136), "w": 20, "h": 30,
                      "pins": ["G", "D", "S"],
                      "props": {"Vds": "55V", "Id": "47A", "Rds": "22mΩ"}},
    "diode":         {"category": "Semi",        "label": "1N4007",       "color": (170, 136,  68), "w": 40, "h": 16,
                      "pins": ["A", "K"],
                      "props": {"Vf": "1.1V", "If": "1A", "Vr": "1000V"}},
    "zener":         {"category": "Semi",        "label": "ZENER",        "color": (136, 136,  68), "w": 40, "h": 16,
                      "pins": ["A", "K"],
                      "props": {"Vz": "5.1V", "Iz": "5mA", "Pz": "500mW"}},
}


# ─────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────
@dataclass
class Component:
    id: int
    type_id: str
    x: int
    y: int
    rotation: int = 0              # degrees: 0, 90, 180, 270
    label: str = ""
    props: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)  # runtime sim state

    def __post_init__(self):
        if not self.label:
            self.label = CATALOG.get(self.type_id, {}).get("label", self.type_id)
        if not self.props:
            self.props = dict(CATALOG.get(self.type_id, {}).get("props", {}))


@dataclass
class Wire:
    points: list          # list of (x, y) world-coords
    color: tuple = (255, 212, 0)
    net_id: int = -1


# ─────────────────────────────────────────────
#  CIRCUIT ENGINE
# ─────────────────────────────────────────────
class CircuitEngine:

    GRID = 20   # grid snap size (pixels in world space)

    def __init__(self, canvas_w: int = 1280, canvas_h: int = 720):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        # Scene
        self.components: list[Component] = []
        self.wires: list[Wire] = []
        self._next_id = 1

        # Viewport
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.zoom: float = 1.0

        # Interaction state
        self.selected_id: Optional[int] = None
        self.mode: str = "select"          # "select" | "wire" | "pan"
        self.wire_in_progress: Optional[Wire] = None
        self.drag_offset: Optional[tuple] = None
        self.mouse_world: tuple = (0, 0)

        # Simulation
        self.sim_running: bool = False
        self.sim_tick: int = 0
        self._sim_t0: float = 0
        self.serial_log: list[str] = []

        # UI
        self._show_grid: bool = True
        self._show_pins: bool = True
        self._show_labels: bool = True

    # ────────────────── ID ──────────────────
    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    # ────────────────── GRID ──────────────────
    @staticmethod
    def snap(v: float) -> int:
        g = CircuitEngine.GRID
        return int(round(v / g) * g)

    # ────────────────── WORLD/SCREEN TRANSFORMS ──────────────────
    def to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        sx = int(wx * self.zoom + self.pan_x)
        sy = int(wy * self.zoom + self.pan_y)
        return sx, sy

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        wx = (sx - self.pan_x) / self.zoom
        wy = (sy - self.pan_y) / self.zoom
        return wx, wy

    # ────────────────── COMPONENT MANAGEMENT ──────────────────
    def add_component(self, type_id: str, wx: int, wy: int,
                      label: str = "", props: dict = None) -> Optional[Component]:
        if type_id not in CATALOG:
            return None
        comp = Component(
            id=self._new_id(),
            type_id=type_id,
            x=self.snap(wx),
            y=self.snap(wy),
            label=label or CATALOG[type_id]["label"],
            props=props or dict(CATALOG[type_id].get("props", {}))
        )
        self.components.append(comp)
        self._log(f"[PLACE] {comp.label} (id={comp.id}) at ({comp.x},{comp.y})")
        return comp

    def remove_component(self, comp_id: int):
        self.components = [c for c in self.components if c.id != comp_id]
        if self.selected_id == comp_id:
            self.selected_id = None

    def duplicate_component(self, comp_id: int) -> Optional[Component]:
        src = self.get_component(comp_id)
        if not src:
            return None
        new_c = Component(
            id=self._new_id(),
            type_id=src.type_id,
            x=src.x + self.GRID * 3,
            y=src.y + self.GRID * 3,
            rotation=src.rotation,
            label=src.label,
            props=dict(src.props)
        )
        self.components.append(new_c)
        return new_c

    def rotate_component(self, comp_id: int, delta: int = 90):
        c = self.get_component(comp_id)
        if c:
            c.rotation = (c.rotation + delta) % 360

    def get_component(self, comp_id: int) -> Optional[Component]:
        for c in self.components:
            if c.id == comp_id:
                return c
        return None

    def clear(self):
        self.components.clear()
        self.wires.clear()
        self.selected_id = None
        self.wire_in_progress = None
        self.sim_running = False
        self._log("[CLEAR] Board cleared.")

    # ────────────────── WIRES ──────────────────
    def start_wire(self, wx: float, wy: float):
        sx, sy = self.snap(wx), self.snap(wy)
        self.wire_in_progress = Wire(points=[(sx, sy)])

    def extend_wire(self, wx: float, wy: float):
        if self.wire_in_progress:
            sx, sy = self.snap(wx), self.snap(wy)
            last = self.wire_in_progress.points[-1]
            if (sx, sy) != last:
                self.wire_in_progress.points.append((sx, sy))

    def finish_wire(self):
        if self.wire_in_progress and len(self.wire_in_progress.points) >= 2:
            self.wires.append(self.wire_in_progress)
            self._log(f"[WIRE] Wire added ({len(self.wire_in_progress.points)} points)")
        self.wire_in_progress = None

    def cancel_wire(self):
        self.wire_in_progress = None

    def remove_last_wire(self):
        if self.wires:
            self.wires.pop()

    # ────────────────── HIT TEST ──────────────────
    def hit_test(self, wx: float, wy: float) -> Optional[Component]:
        for comp in reversed(self.components):
            d = CATALOG[comp.type_id]
            cw, ch = d["w"], d["h"]
            if comp.x <= wx <= comp.x + cw and comp.y <= wy <= comp.y + ch:
                return comp
        return None

    # ────────────────── VIEWPORT ──────────────────
    def pan(self, dx: float, dy: float):
        self.pan_x += dx
        self.pan_y += dy

    def zoom_at(self, screen_x: float, screen_y: float, factor: float):
        new_zoom = max(0.15, min(5.0, self.zoom * factor))
        self.pan_x = screen_x - (screen_x - self.pan_x) * (new_zoom / self.zoom)
        self.pan_y = screen_y - (screen_y - self.pan_y) * (new_zoom / self.zoom)
        self.zoom = new_zoom

    def fit_to_screen(self):
        if not self.components:
            self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
            return
        xs = [c.x for c in self.components]
        ys = [c.y for c in self.components]
        xs2 = [c.x + CATALOG[c.type_id]["w"] for c in self.components]
        ys2 = [c.y + CATALOG[c.type_id]["h"] for c in self.components]
        min_x, max_x = min(xs), max(xs2)
        min_y, max_y = min(ys), max(ys2)
        pad = 60
        zx = (self.canvas_w - pad * 2) / max(max_x - min_x, 1)
        zy = (self.canvas_h - pad * 2) / max(max_y - min_y, 1)
        self.zoom = min(zx, zy, 3.0)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        self.pan_x = self.canvas_w / 2 - cx * self.zoom
        self.pan_y = self.canvas_h / 2 - cy * self.zoom

    # ────────────────── SIMULATION ──────────────────
    def start_simulation(self):
        if self.sim_running:
            return
        self.sim_running = True
        self.sim_tick = 0
        self._sim_t0 = time.time()
        self._log("=== Simulation Started ===")
        mcus = [c for c in self.components if CATALOG[c.type_id]["category"] == "MCU"]
        if not mcus:
            self._log("[WARN] No MCU on board — add ESP32, Arduino, etc.")
        for m in mcus:
            d = CATALOG[m.type_id]
            self._log(f"[BOOT] {d['label']} @ {m.props.get('cpu','?')} starting…")
        # Initialise LED states
        for c in self.components:
            if CATALOG[c.type_id]["category"] == "LED":
                c.state["on"] = False
                c.state["brightness"] = 0.0

    def stop_simulation(self):
        self.sim_running = False
        for c in self.components:
            c.state.clear()
        self._log("=== Simulation Stopped ===")

    def tick_simulation(self):
        """Call once per frame while simRunning == True."""
        if not self.sim_running:
            return
        self.sim_tick += 1
        t = self.sim_tick

        # LEDs blink
        for c in self.components:
            cat = CATALOG[c.type_id]["category"]
            if cat == "LED":
                c.state["on"] = (t % 20) < 10
                c.state["brightness"] = 0.8 + 0.2 * math.sin(t * 0.3)

        # Sensor readings every 60 ticks
        if t % 60 == 0:
            for c in self.components:
                tid = c.type_id
                if tid == "dht22":
                    c.state["temp"] = round(22 + random.uniform(-1, 3), 1)
                    c.state["hum"] = round(55 + random.uniform(-5, 10), 1)
                    self._log(f"[DHT22] Temp:{c.state['temp']}°C  Hum:{c.state['hum']}%")
                elif tid == "bmp280":
                    c.state["pressure"] = round(1013 + random.uniform(-2, 5), 1)
                    self._log(f"[BMP280] Pressure:{c.state['pressure']} hPa")
                elif tid == "ldr":
                    c.state["lux"] = random.randint(200, 900)
                    self._log(f"[LDR] Light:{c.state['lux']} lux")
                elif tid == "mpu6050":
                    c.state["ax"] = round(random.uniform(-1, 1), 2)
                    c.state["ay"] = round(random.uniform(-1, 1), 2)
                    c.state["az"] = round(9.8 + random.uniform(-0.05, 0.05), 2)
                    self._log(f"[MPU6050] ax:{c.state['ax']} ay:{c.state['ay']} az:{c.state['az']}")
                elif tid == "ds18b20":
                    c.state["temp"] = round(23 + random.uniform(-0.5, 1.5), 1)
                    self._log(f"[DS18B20] Temp:{c.state['temp']}°C")
                elif tid == "ultrasonic":
                    c.state["dist"] = round(random.uniform(5, 200), 1)
                    self._log(f"[HC-SR04] Dist:{c.state['dist']} cm")

    # ────────────────── SERIAL LOG ──────────────────
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.serial_log.append(line)
        if len(self.serial_log) > 300:
            self.serial_log = self.serial_log[-200:]

    def get_log_tail(self, n: int = 10) -> list[str]:
        return self.serial_log[-n:]

    # ────────────────── SAVE / LOAD ──────────────────
    def save(self, path: str):
        data = {
            "version": 2,
            "zoom": self.zoom,
            "pan": [self.pan_x, self.pan_y],
            "components": [
                {"id": c.id, "type": c.type_id, "x": c.x, "y": c.y,
                 "rotation": c.rotation, "label": c.label, "props": c.props}
                for c in self.components
            ],
            "wires": [
                {"points": w.points, "color": list(w.color)}
                for w in self.wires
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._log(f"[SAVE] Saved to {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.clear()
        self.zoom = data.get("zoom", 1.0)
        px, py = data.get("pan", [0, 0])
        self.pan_x, self.pan_y = px, py
        for cd in data.get("components", []):
            c = Component(
                id=cd["id"], type_id=cd["type"],
                x=cd["x"], y=cd["y"],
                rotation=cd.get("rotation", 0),
                label=cd.get("label", ""),
                props=cd.get("props", {})
            )
            self.components.append(c)
        for wd in data.get("wires", []):
            pts = [tuple(p) for p in wd["points"]]
            col = tuple(wd.get("color", [255, 212, 0]))
            self.wires.append(Wire(points=pts, color=col))
        self._next_id = max((c.id for c in self.components), default=0) + 1
        self._log(f"[LOAD] Loaded {len(self.components)} components, {len(self.wires)} wires")

    # ────────────────── RENDERING ──────────────────
    def render(self, canvas: np.ndarray) -> np.ndarray:
        """Draw everything onto canvas (in-place + return)."""
        if self._show_grid:
            self._draw_grid(canvas)
        for wire in self.wires:
            self._draw_wire(canvas, wire)
        if self.wire_in_progress and len(self.wire_in_progress.points) >= 1:
            self._draw_wire_preview(canvas)
        for comp in self.components:
            self._draw_component(canvas, comp)
        self._draw_hud(canvas)
        return canvas

    # ── Grid ──
    def _draw_grid(self, canvas: np.ndarray):
        h, w = canvas.shape[:2]
        step = max(4, int(self.GRID * self.zoom))
        off_x = int(self.pan_x % step)
        off_y = int(self.pan_y % step)
        for x in range(off_x, w, step):
            for y in range(off_y, h, step):
                if 0 <= x < w and 0 <= y < h:
                    canvas[y, x] = (30, 45, 20)    # BGR dot

    # ── Wire ──
    def _draw_wire(self, canvas: np.ndarray, wire: Wire, alpha: float = 1.0):
        pts = [self.to_screen(p[0], p[1]) for p in wire.points]
        if len(pts) < 2:
            return
        col = tuple(int(c * alpha) for c in wire.color)
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], col, max(1, int(2 * self.zoom)), cv2.LINE_AA)
        # glow effect: second pass slightly wider, dimmer
        glow_col = tuple(int(c * 0.35) for c in wire.color)
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], glow_col, max(2, int(5 * self.zoom)), cv2.LINE_AA)
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], col, max(1, int(2 * self.zoom)), cv2.LINE_AA)
        for pt in pts:
            cv2.circle(canvas, pt, max(2, int(3 * self.zoom)), col, -1, cv2.LINE_AA)

    def _draw_wire_preview(self, canvas: np.ndarray):
        pts = [self.to_screen(p[0], p[1]) for p in self.wire_in_progress.points]
        mx, my = self.to_screen(*self.mouse_world)
        pts.append((mx, my))
        col = (200, 212, 80)
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], col, max(1, int(2 * self.zoom)), cv2.LINE_AA)

    # ── Component ──
    def _draw_component(self, canvas: np.ndarray, comp: Component):
        d = CATALOG.get(comp.type_id)
        if not d:
            return

        cw = int(d["w"] * self.zoom)
        ch = int(d["h"] * self.zoom)
        sx, sy = self.to_screen(comp.x, comp.y)
        is_sel = (comp.id == self.selected_id)
        color_bgr = d["color"]
        sim = self.sim_running

        # Rotation: we build a mini-canvas and rotate it
        comp_img = np.zeros((ch + 20, cw + 20, 3), dtype=np.uint8)
        ox, oy = 10, 10   # offset inside mini-canvas

        # Body fill
        fill = tuple(int(c * 0.18) for c in color_bgr)
        border = color_bgr if not is_sel else (50, 120, 255)
        thickness = max(1, int(2 * self.zoom)) if not is_sel else max(2, int(3 * self.zoom))
        cv2.rectangle(comp_img, (ox, oy), (ox + cw, oy + ch), fill, -1)
        cv2.rectangle(comp_img, (ox, oy), (ox + cw, oy + ch), border, thickness, cv2.LINE_AA)

        # LED glow overlay when simulating
        if sim and d["category"] == "LED" and comp.state.get("on"):
            glow_col = tuple(int(c * comp.state.get("brightness", 1.0)) for c in color_bgr)
            center = (ox + cw // 2, oy + ch // 2)
            cv2.circle(comp_img, center, int(min(cw, ch) * 0.35), glow_col, -1, cv2.LINE_AA)

        # Label
        if self._show_labels:
            lbl = comp.label[:10]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = max(0.28, 0.38 * self.zoom)
            th = max(1, int(self.zoom))
            (tw, tht), _ = cv2.getTextSize(lbl, font, fs, th)
            tx = ox + cw // 2 - tw // 2
            ty = oy + ch - max(4, int(6 * self.zoom))
            cv2.putText(comp_img, lbl, (tx, ty), font, fs, (220, 230, 240), th, cv2.LINE_AA)

        # Pin dots
        if self._show_pins and self.zoom > 0.5:
            pins = d.get("pins", [])
            if pins:
                step = ch / (len(pins) + 1)
                for i, pin in enumerate(pins):
                    py = int(oy + step * (i + 1))
                    cv2.circle(comp_img, (ox, py), max(2, int(3 * self.zoom)), (255, 210, 0), -1, cv2.LINE_AA)
                    if self.zoom > 0.9:
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fs = max(0.18, 0.22 * self.zoom)
                        cv2.putText(comp_img, pin, (ox + 4, py + 3), font, fs, (160, 200, 220), 1, cv2.LINE_AA)

        # Sim state overlay (sensor values)
        if sim and comp.state and self.zoom > 0.7:
            val_str = self._state_string(comp)
            if val_str:
                font = cv2.FONT_HERSHEY_SIMPLEX
                fs = max(0.22, 0.28 * self.zoom)
                cv2.putText(comp_img, val_str, (ox + 2, oy + 14), font, fs, (80, 255, 120), 1, cv2.LINE_AA)

        # Rotate mini-canvas
        img_rot = self._rotate_img(comp_img, comp.rotation)
        rh, rw = img_rot.shape[:2]

        # Paste onto canvas (blend non-black pixels)
        dx, dy = sx - 10, sy - 10
        self._paste(canvas, img_rot, dx, dy)

    @staticmethod
    def _state_string(comp: Component) -> str:
        s = comp.state
        if "temp" in s and "hum" in s:
            return f"{s['temp']}C {s['hum']}%"
        if "temp" in s:
            return f"{s['temp']}C"
        if "lux" in s:
            return f"{s['lux']}lx"
        if "dist" in s:
            return f"{s['dist']}cm"
        if "pressure" in s:
            return f"{s['pressure']}hPa"
        if "ax" in s:
            return f"ax{s['ax']}"
        return ""

    @staticmethod
    def _rotate_img(img: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return img
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    @staticmethod
    def _paste(canvas: np.ndarray, img: np.ndarray, dx: int, dy: int):
        ch, cw = canvas.shape[:2]
        ih, iw = img.shape[:2]
        x0 = max(0, dx)
        y0 = max(0, dy)
        x1 = min(cw, dx + iw)
        y1 = min(ch, dy + ih)
        ix0 = x0 - dx
        iy0 = y0 - dy
        ix1 = ix0 + (x1 - x0)
        iy1 = iy0 + (y1 - y0)
        if x1 <= x0 or y1 <= y0:
            return
        region = canvas[y0:y1, x0:x1]
        patch = img[iy0:iy1, ix0:ix1]
        mask = patch.sum(axis=2) > 0
        region[mask] = patch[mask]

    # ── HUD overlay ──
    def _draw_hud(self, canvas: np.ndarray):
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Zoom + coord
        zoom_str = f"Zoom: {self.zoom:.2f}x  |  Comps: {len(self.components)}  |  Wires: {len(self.wires)}"
        cv2.putText(canvas, zoom_str, (10, h - 12), font, 0.38, (80, 100, 120), 1, cv2.LINE_AA)

        mx, my = self.mouse_world
        coord_str = f"X:{int(mx)}  Y:{int(my)}"
        cv2.putText(canvas, coord_str, (w - 120, h - 12), font, 0.38, (80, 100, 120), 1, cv2.LINE_AA)

        # Mode badge
        mode_col = {"select": (255, 180, 0), "wire": (0, 200, 255), "pan": (0, 255, 100)}
        col = mode_col.get(self.mode, (180, 180, 180))
        cv2.putText(canvas, f"MODE: {self.mode.upper()}", (10, 20), font, 0.45, col, 1, cv2.LINE_AA)

        # Sim badge
        if self.sim_running:
            elapsed = time.time() - self._sim_t0
            cv2.putText(canvas, f"SIM RUNNING  {elapsed:.1f}s", (10, 38), font, 0.42, (80, 255, 80), 1, cv2.LINE_AA)

        # Serial tail (last 5 lines)
        for i, line in enumerate(self.get_log_tail(5)):
            cv2.putText(canvas, line[-80:], (10, h - 90 + i * 16), font, 0.32, (60, 180, 80), 1, cv2.LINE_AA)

    # ────────────────── SEARCH ──────────────────
    @staticmethod
    def search(query: str) -> list[str]:
        q = query.lower()
        return [
            tid for tid, d in CATALOG.items()
            if q in tid.lower()
            or q in d["label"].lower()
            or q in d["category"].lower()
        ]

    @staticmethod
    def list_categories() -> dict[str, list[str]]:
        cats: dict[str, list[str]] = {}
        for tid, d in CATALOG.items():
            cats.setdefault(d["category"], []).append(tid)
        return cats

    @staticmethod
    def get_info(type_id: str) -> Optional[dict]:
        return CATALOG.get(type_id)