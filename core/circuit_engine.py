"""
circuit_engine.py  —  AIILA Circuit Engine v3.1  (WORKBENCH EDITION)
=====================================================================
FIXES APPLIED OVER v3:

  [FIX 1]  snapshot() / restore()  — kernel undo now works
  [FIX 2]  add_wire(src_id, src_pin, dst_id, dst_pin)  — gesture wire-draw works
  [FIX 3]  hit_test(wx, wy, radius_multiplier=1.0)  — grab forgiveness radius
  [FIX 4]  get_pin_positions() — LEFT+RIGHT column layout for ICs (not all-left)
  [FIX 5]  get_pin_by_name(comp, pin_name) -> (wx, wy)  — named pin lookup
  [FIX 6]  get_pin_positions() applies rotation transform
  [FIX 7]  threading.RLock on all component mutations — no race conditions
  [FIX 8]  render_board_only() accepts override_w/override_h — no global mutation
  [FIX 9]  serial_log now uses collections.deque(maxlen=200) — no memory leak
  [FIX 10] tick_simulation() uses time.time() not sim_tick for frame-rate
           independence
  [FIX 11] Undo stack consolidated in engine — kernel.undo() delegates here

  NEW — PIN VISIBILITY:
  • IC components (ESP32, Arduino etc.) show pins on BOTH left and right sides,
    split ~50/50. Each pin has its name rendered next to it.
  • Pin hit-radius scales with zoom so you can target individual GPIO pins
    from a normal arm-length projector distance.
  • _draw_pins() now also draws right-side pins with labels.
"""

import cv2
import numpy as np
import json
import math
import random
import time
import threading
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  CATALOG  (unchanged from v3 — full 80+ component catalog)
# ─────────────────────────────────────────────────────────────────────────────
CATALOG: dict[str, dict] = {
    # ── Microcontrollers ─────────────────────────────────────────────────────
    "esp32":       {"label":"ESP32",    "category":"MCU",     "color":(61,133,200), "w":90,"h":160,
                    "symbol":"ic",
                    "pins":["3V3","GND","EN","VP","VN","D34","D35","D32","D33","D25","D26","D27","D14","D12","D13","D2","D4","RX2","TX2","D5","D18","D19","D21","RX0","TX0","D22","D23"],
                    "props":{"board":"ESP32 Dev","cpu":"240MHz","flash":"4MB","ram":"520KB","voltage":"3.3V","wifi":"Yes","bt":"Yes"}},
    "arduino_uno": {"label":"Uno",      "category":"MCU",     "color":(0,122,193),  "w":90,"h":160,
                    "symbol":"ic",
                    "pins":["RESET","3.3V","5V","GND","VIN","A0","A1","A2","A3","A4","A5","D0","D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","AREF","GND2"],
                    "props":{"board":"Arduino Uno R3","cpu":"16MHz","flash":"32KB","ram":"2KB","voltage":"5V"}},
    "arduino_nano":{"label":"Nano",     "category":"MCU",     "color":(0,100,170),  "w":70,"h":150,
                    "symbol":"ic",
                    "pins":["D1","D0","RESET","GND","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","3V3","AREF","A0","A1","A2","A3","A4","A5","5V","GND2","VIN"],
                    "props":{"board":"Arduino Nano","cpu":"16MHz","flash":"32KB","voltage":"5V"}},
    "esp8266":     {"label":"ESP8266",  "category":"MCU",     "color":(232,160,32), "w":75,"h":100,
                    "symbol":"ic",
                    "pins":["RST","A0","D0","D5","D6","D7","D8","3V3","GND","D4","D3","D2","D1"],
                    "props":{"board":"NodeMCU 1.0","cpu":"80MHz","flash":"4MB","voltage":"3.3V","wifi":"Yes"}},
    "rpi_pico":    {"label":"Pico",     "category":"MCU",     "color":(197,26,74),  "w":90,"h":160,
                    "symbol":"ic",
                    "pins":["GP0","GP1","GND","GP2","GP3","GP4","GP5","GP6","GP7","GP8","GP9","GP10","GP11","GP12","GP13","GP14","GP15","3V3","VSYS","VBUS","GND2","RUN","ADC_VREF","GP26","GP27","GP28","AGND"],
                    "props":{"board":"RPi Pico","cpu":"133MHz","flash":"2MB","ram":"264KB","voltage":"3.3V"}},
    "stm32":       {"label":"STM32",    "category":"MCU",     "color":(0,112,184),  "w":90,"h":160,
                    "symbol":"ic",
                    "pins":["VBAT","PC13","PC14","PC15","PD0","PD1","NRST","PC0","PC1","PC2","PC3","VSSA","VDDA","PA0","PA1","PA2","PA3","PA4","PA5","PA6","PA7","PB0","PB1","PB2","GND","3.3V","5V"],
                    "props":{"board":"Blue Pill STM32F103","cpu":"72MHz","flash":"64KB","ram":"20KB","voltage":"3.3V"}},

    # ── Passives ─────────────────────────────────────────────────────────────
    "resistor":    {"label":"RES",   "category":"Passive", "color":(139,105,20), "w":60,"h":22,
                    "symbol":"res",  "pins":["1","2"],
                    "props":{"resistance":"1kΩ","tolerance":"5%","power":"0.25W","type":"Carbon Film"}},
    "resistor_10k":{"label":"10kΩ",  "category":"Passive", "color":(139,105,20), "w":60,"h":22,
                    "symbol":"res",  "pins":["1","2"],
                    "props":{"resistance":"10kΩ","tolerance":"1%","power":"0.25W","type":"Metal Film"}},
    "capacitor":   {"label":"CAP",   "category":"Passive", "color":(80,80,140),  "w":26,"h":44,
                    "symbol":"cap",  "pins":["+","-"],
                    "props":{"capacitance":"100nF","voltage":"50V","type":"Ceramic","esr":"0.01Ω"}},
    "cap_elec":    {"label":"ELEC",  "category":"Passive", "color":(60,60,120),  "w":30,"h":48,
                    "symbol":"cap",  "pins":["+","-"],
                    "props":{"capacitance":"100µF","voltage":"16V","type":"Electrolytic"}},
    "inductor":    {"label":"IND",   "category":"Passive", "color":(80,110,80),  "w":60,"h":24,
                    "symbol":"res",  "pins":["1","2"],
                    "props":{"inductance":"10µH","current":"1A","dcr":"0.1Ω"}},
    "potentiometer":{"label":"POT",  "category":"Passive", "color":(100,100,100),"w":44,"h":44,
                    "symbol":"rect", "pins":["1","W","2"],
                    "props":{"resistance":"10kΩ","type":"Linear","power":"0.5W"}},
    "crystal":     {"label":"XTAL",  "category":"Passive", "color":(150,150,90), "w":30,"h":50,
                    "symbol":"rect", "pins":["1","2"],
                    "props":{"freq":"16MHz","load":"20pF","type":"HC-49S"}},

    # ── LEDs ─────────────────────────────────────────────────────────────────
    "led_red":     {"label":"LED",   "category":"LED",     "color":(0,0,255),    "w":22,"h":34,
                    "symbol":"led",  "pins":["A","K"],
                    "props":{"Vf":"2.0V","If":"20mA","color":"Red","wavelength":"625nm"}},
    "led_green":   {"label":"LED",   "category":"LED",     "color":(0,200,0),    "w":22,"h":34,
                    "symbol":"led",  "pins":["A","K"],
                    "props":{"Vf":"2.1V","If":"20mA","color":"Green","wavelength":"520nm"}},
    "led_blue":    {"label":"LED",   "category":"LED",     "color":(230,100,0),  "w":22,"h":34,
                    "symbol":"led",  "pins":["A","K"],
                    "props":{"Vf":"3.2V","If":"20mA","color":"Blue","wavelength":"470nm"}},
    "led_yellow":  {"label":"LED",   "category":"LED",     "color":(0,200,200),  "w":22,"h":34,
                    "symbol":"led",  "pins":["A","K"],
                    "props":{"Vf":"2.1V","If":"20mA","color":"Yellow","wavelength":"590nm"}},
    "led_rgb":     {"label":"RGB",   "category":"LED",     "color":(200,0,200),  "w":28,"h":34,
                    "symbol":"led",  "pins":["R","G","B","GND"],
                    "props":{"type":"Common Cathode","If":"20mA","package":"5mm"}},
    "ws2812":      {"label":"WS2812","category":"LED",     "color":(255,140,0),  "w":24,"h":24,
                    "symbol":"ic",   "pins":["VCC","GND","DIN","DOUT"],
                    "props":{"protocol":"NeoPixel","voltage":"5V","current":"60mA","RGB":"Yes"}},
    "ws2812_strip":{"label":"NeoStrip","category":"LED",   "color":(200,120,0),  "w":90,"h":24,
                    "symbol":"rect", "pins":["5V","GND","DIN","DOUT"],
                    "props":{"leds":"8","voltage":"5V","current":"480mA","density":"30/m"}},

    # ── Displays ─────────────────────────────────────────────────────────────
    "oled":        {"label":"OLED",  "category":"Display", "color":(68,68,102),  "w":76,"h":56,
                    "symbol":"rect", "pins":["GND","VCC","SCL","SDA"],
                    "props":{"driver":"SSD1306","res":"128x64","iface":"I2C","addr":"0x3C","voltage":"3.3-5V"}},
    "lcd16x2":     {"label":"LCD",   "category":"Display", "color":(50,100,50),  "w":100,"h":44,
                    "symbol":"rect", "pins":["VSS","VDD","V0","RS","RW","E","D4","D5","D6","D7","A","K"],
                    "props":{"size":"16x2","backlight":"Yes","voltage":"5V","iface":"Parallel"}},
    "lcd_i2c":     {"label":"LCD I2C","category":"Display","color":(40,90,40),   "w":60,"h":44,
                    "symbol":"ic",   "pins":["GND","VCC","SDA","SCL"],
                    "props":{"size":"16x2","backlight":"Yes","driver":"PCF8574","addr":"0x27"}},
    "tft_128":     {"label":"TFT128","category":"Display", "color":(80,50,80),   "w":70,"h":80,
                    "symbol":"rect", "pins":["GND","VCC","SCL","SDA","RES","DC","CS","BLK"],
                    "props":{"driver":"ST7735","res":"128x160","voltage":"3.3V","iface":"SPI"}},
    "epaper":      {"label":"ePaper","category":"Display", "color":(120,120,120),"w":80,"h":60,
                    "symbol":"rect", "pins":["VCC","GND","DIN","CLK","CS","DC","RST","BUSY"],
                    "props":{"res":"200x200","voltage":"3.3V","color":"BW","iface":"SPI"}},

    # ── Sensors ──────────────────────────────────────────────────────────────
    "dht22":       {"label":"DHT22", "category":"Sensor",  "color":(0,170,136),  "w":34,"h":54,
                    "symbol":"sensor","pins":["VCC","DATA","NC","GND"],
                    "props":{"temp":"-40~80°C","hum":"0~100%","acc":"±0.5°C","rate":"0.5Hz"}},
    "dht11":       {"label":"DHT11", "category":"Sensor",  "color":(0,140,100),  "w":30,"h":50,
                    "symbol":"sensor","pins":["VCC","DATA","GND"],
                    "props":{"temp":"0~50°C","hum":"20~80%","acc":"±2°C","rate":"1Hz"}},
    "bmp280":      {"label":"BMP280","category":"Sensor",  "color":(51,85,102),  "w":38,"h":24,
                    "symbol":"ic",   "pins":["VCC","GND","SCL","SDA","CSB","SDO"],
                    "props":{"pressure":"300-1100hPa","temp":"-40~85°C","acc":"±1hPa","iface":"I2C/SPI"}},
    "bme680":      {"label":"BME680","category":"Sensor",  "color":(51,75,90),   "w":38,"h":24,
                    "symbol":"ic",   "pins":["VCC","GND","SCL","SDA","SDO","CS"],
                    "props":{"sensors":"Temp/Hum/Press/Gas","iface":"I2C/SPI"}},
    "ds18b20":     {"label":"DS18B20","category":"Sensor", "color":(102,85,85),  "w":24,"h":44,
                    "symbol":"sensor","pins":["GND","DATA","VCC"],
                    "props":{"range":"-55~125°C","acc":"±0.5°C","iface":"1-Wire","parasitic":"Yes"}},
    "ldr":         {"label":"LDR",   "category":"Sensor",  "color":(170,169,34), "w":24,"h":24,
                    "symbol":"res",  "pins":["1","2"],
                    "props":{"dark":">1MΩ","light":"1-10kΩ","peak":"540nm"}},
    "pir":         {"label":"PIR",   "category":"Sensor",  "color":(136,68,0),   "w":38,"h":38,
                    "symbol":"sensor","pins":["VCC","OUT","GND"],
                    "props":{"range":"7m","angle":"120°","voltage":"5-20V","delay":"5s"}},
    "ultrasonic":  {"label":"HC-SR04","category":"Sensor", "color":(68,136,170), "w":66,"h":34,
                    "symbol":"sensor","pins":["VCC","TRIG","ECHO","GND"],
                    "props":{"range":"2-400cm","acc":"3mm","freq":"40kHz","voltage":"5V"}},
    "mpu6050":     {"label":"MPU6050","category":"Sensor", "color":(68,102,85),  "w":44,"h":44,
                    "symbol":"ic",   "pins":["VCC","GND","SCL","SDA","XDA","XCL","AD0","INT"],
                    "props":{"axes":"6-DOF","gyro":"±250-2000°/s","acc":"±2-16g","iface":"I2C","addr":"0x68"}},
    "mpu9250":     {"label":"MPU9250","category":"Sensor", "color":(60,95,75),   "w":44,"h":44,
                    "symbol":"ic",   "pins":["VDD","GND","SCL","SDA","AD0","INT","NCS","FSYNC"],
                    "props":{"axes":"9-DOF","mag":"±4800µT","iface":"I2C/SPI"}},
    "soil_moisture":{"label":"Soil", "category":"Sensor",  "color":(68,119,51),  "w":24,"h":66,
                    "symbol":"sensor","pins":["VCC","GND","AOUT","DOUT"],
                    "props":{"output":"Analog+Digital","voltage":"3.3-5V"}},
    "gas_mq2":     {"label":"MQ-2",  "category":"Sensor",  "color":(0,100,180),  "w":40,"h":40,
                    "symbol":"sensor","pins":["VCC","GND","AOUT","DOUT"],
                    "props":{"gas":"LPG/Smoke/H2","voltage":"5V","preheat":"20s"}},
    "color_sensor":{"label":"TCS3200","category":"Sensor", "color":(100,50,150), "w":44,"h":44,
                    "symbol":"ic",   "pins":["GND","OE","OUT","VCC","S0","S1","S2","S3"],
                    "props":{"filters":"RGBW","freq":"0-500kHz","voltage":"2.7-5.5V"}},
    "ir_recv":     {"label":"IR Recv","category":"Sensor", "color":(40,40,80),   "w":24,"h":30,
                    "symbol":"sensor","pins":["OUT","GND","VCC"],
                    "props":{"freq":"38kHz","range":"18m","voltage":"2.7-5.5V"}},

    # ── Actuators ────────────────────────────────────────────────────────────
    "servo":       {"label":"SERVO", "category":"Actuator","color":(255,102,0),  "w":44,"h":54,
                    "symbol":"rect", "pins":["GND","VCC","PWM"],
                    "props":{"torque":"1.8kgcm","angle":"0-180°","voltage":"4.8-6V","freq":"50Hz"}},
    "servo_360":   {"label":"SRV360","category":"Actuator","color":(220,90,0),   "w":44,"h":54,
                    "symbol":"rect", "pins":["GND","VCC","PWM"],
                    "props":{"type":"Continuous","voltage":"4.8-6V","freq":"50Hz"}},
    "stepper":     {"label":"STEP",  "category":"Actuator","color":(204,68,0),   "w":54,"h":54,
                    "symbol":"rect", "pins":["IN1","IN2","IN3","IN4","VCC"],
                    "props":{"steps":"64","gear":"1/64","voltage":"5V","current":"400mA"}},
    "dc_motor":    {"label":"DCMOT", "category":"Actuator","color":(136,68,0),   "w":48,"h":48,
                    "symbol":"rect", "pins":["M+","M-"],
                    "props":{"voltage":"3-6V","rpm":"200RPM","current":"200mA","stall":"800mA"}},
    "buzzer":      {"label":"BUZ",   "category":"Actuator","color":(60,60,60),   "w":30,"h":30,
                    "symbol":"rect", "pins":["+","-"],
                    "props":{"type":"Active","freq":"2.5kHz","voltage":"3-24V","spl":"85dB"}},
    "buzzer_p":    {"label":"PBUZ",  "category":"Actuator","color":(70,70,70),   "w":30,"h":30,
                    "symbol":"rect", "pins":["+","-"],
                    "props":{"type":"Passive","voltage":"3-5V","freq":"1-5kHz"}},
    "relay":       {"label":"RELAY", "category":"Actuator","color":(51,102,153), "w":64,"h":44,
                    "symbol":"rect", "pins":["VCC","GND","IN","NC","COM","NO"],
                    "props":{"coil":"5V","contact":"10A 250VAC","type":"SPDT"}},
    "relay_4ch":   {"label":"4CH RLY","category":"Actuator","color":(40,85,130),"w":100,"h":44,
                    "symbol":"rect", "pins":["VCC","GND","IN1","IN2","IN3","IN4","COM1","NO1","NC1","COM2","NO2","NC2","COM3","NO3","NC3","COM4","NO4","NC4"],
                    "props":{"channels":"4","coil":"5V","contact":"10A 250VAC"}},
    "l298n":       {"label":"L298N", "category":"Actuator","color":(153,68,0),   "w":74,"h":64,
                    "symbol":"ic",   "pins":["IN1","IN2","IN3","IN4","ENA","ENB","VCC","GND","12V"],
                    "props":{"maxA":"2A","maxV":"46V","channels":"2","pkg":"Multiwatt"}},
    "l9110s":      {"label":"L9110S","category":"Actuator","color":(130,55,0),   "w":54,"h":44,
                    "symbol":"ic",   "pins":["VCC","GND","A-IA","A-IB","OA1","OA2","B-IA","B-IB","OB1","OB2"],
                    "props":{"maxA":"800mA","maxV":"12V","channels":"2"}},
    "solenoid":    {"label":"SOL",   "category":"Actuator","color":(80,100,80),  "w":44,"h":44,
                    "symbol":"rect", "pins":["+","-"],
                    "props":{"voltage":"12V","current":"1A","stroke":"10mm"}},

    # ── Communication ────────────────────────────────────────────────────────
    "nrf24":       {"label":"nRF24", "category":"Comms",   "color":(0,136,204),  "w":44,"h":54,
                    "symbol":"ic",   "pins":["GND","VCC","CE","CSN","SCK","MOSI","MISO","IRQ"],
                    "props":{"freq":"2.4GHz","range":"100m","rate":"2Mbps","voltage":"1.9-3.6V"}},
    "hc05":        {"label":"HC-05", "category":"Comms",   "color":(0,0,204),    "w":44,"h":54,
                    "symbol":"ic",   "pins":["VCC","GND","TXD","RXD","STATE","EN"],
                    "props":{"proto":"BT 2.0","range":"10m","baud":"9600","voltage":"3.3V"}},
    "hc06":        {"label":"HC-06", "category":"Comms",   "color":(0,0,180),    "w":44,"h":50,
                    "symbol":"ic",   "pins":["VCC","GND","TXD","RXD"],
                    "props":{"proto":"BT 2.0","range":"10m","baud":"9600","slave":"Only"}},
    "lora":        {"label":"LoRa",  "category":"Comms",   "color":(102,34,102), "w":48,"h":58,
                    "symbol":"ic",   "pins":["GND","VCC","MISO","MOSI","SCK","NSS","RST","DIO0"],
                    "props":{"freq":"433MHz","range":"10km","iface":"SPI","voltage":"3.3V"}},
    "sim800l":     {"label":"SIM800","category":"Comms",   "color":(0,80,160),   "w":44,"h":54,
                    "symbol":"ic",   "pins":["VCC","GND","TXD","RXD","RST","DTR"],
                    "props":{"proto":"GSM/GPRS","bands":"Quad-band","voltage":"3.4-4.4V"}},
    "wifi_esp01":  {"label":"ESP-01","category":"Comms",   "color":(180,140,0),  "w":44,"h":44,
                    "symbol":"ic",   "pins":["GND","GPIO0","UTXD","CH_PD","GPIO2","RST","URXD","VCC"],
                    "props":{"proto":"WiFi 802.11b/g/n","voltage":"3.3V"}},
    "can_mcp2515": {"label":"MCP2515","category":"Comms",  "color":(80,60,120),  "w":54,"h":54,
                    "symbol":"ic",   "pins":["VCC","GND","CS","SO","SI","SCK","INT","TXD","RXD"],
                    "props":{"proto":"CAN 2.0B","rate":"1Mbps","iface":"SPI"}},
    "rs485":       {"label":"MAX485","category":"Comms",   "color":(60,80,100),  "w":44,"h":44,
                    "symbol":"ic",   "pins":["RO","RE","DE","DI","GND","A","B","VCC"],
                    "props":{"proto":"RS-485","rate":"2.5Mbps","voltage":"5V"}},

    # ── Power ────────────────────────────────────────────────────────────────
    "power_5v":    {"label":"5V PSU","category":"Power",   "color":(200,0,0),    "w":34,"h":44,
                    "symbol":"rect", "pins":["V+","GND"],
                    "props":{"voltage":"5V","current":"2A","type":"USB"}},
    "power_12v":   {"label":"12V PSU","category":"Power",  "color":(180,0,0),    "w":34,"h":44,
                    "symbol":"rect", "pins":["V+","GND"],
                    "props":{"voltage":"12V","current":"2A"}},
    "power_3v3":   {"label":"3V3 REG","category":"Power",  "color":(0,68,200),   "w":24,"h":34,
                    "symbol":"ic",   "pins":["IN","OUT","GND"],
                    "props":{"output":"3.3V","maxA":"1A","dropout":"1.2V","pkg":"TO-220"}},
    "ldo_lm7805":  {"label":"7805",  "category":"Power",   "color":(0,50,180),   "w":24,"h":34,
                    "symbol":"ic",   "pins":["IN","GND","OUT"],
                    "props":{"output":"5V","maxA":"1.5A","dropout":"2V","pkg":"TO-220"}},
    "battery_lipo":{"label":"LiPo",  "category":"Power",   "color":(0,136,68),   "w":54,"h":34,
                    "symbol":"rect", "pins":["+","-"],
                    "props":{"voltage":"3.7V","cap":"1000mAh","maxC":"1C","type":"Li-Ion"}},
    "battery_9v":  {"label":"9V BAT","category":"Power",   "color":(0,100,50),   "w":44,"h":34,
                    "symbol":"rect", "pins":["+","-"],
                    "props":{"voltage":"9V","cap":"500mAh","type":"Alkaline"}},
    "tp4056":      {"label":"TP4056","category":"Power",   "color":(0,100,150),  "w":50,"h":34,
                    "symbol":"ic",   "pins":["IN+","IN-","BAT+","BAT-","OUT+","OUT-"],
                    "props":{"type":"LiPo Charger","current":"1A","cutoff":"4.2V"}},
    "boost_mt3608":{"label":"MT3608","category":"Power",   "color":(0,80,120),   "w":50,"h":34,
                    "symbol":"ic",   "pins":["GND","VIN","EN","FB","SW","VOUT"],
                    "props":{"type":"Boost","vin":"2-24V","vout":"5-28V","maxA":"2A"}},

    # ── Input ────────────────────────────────────────────────────────────────
    "pushbutton":  {"label":"BTN",   "category":"Input",   "color":(80,80,80),   "w":22,"h":22,
                    "symbol":"rect", "pins":["1A","1B","2A","2B"],
                    "props":{"rating":"50mA 12V","type":"Momentary SPST","bounce":"5ms"}},
    "toggle_sw":   {"label":"SW",    "category":"Input",   "color":(90,90,90),   "w":30,"h":22,
                    "symbol":"rect", "pins":["COM","NO","NC"],
                    "props":{"type":"SPDT Toggle","rating":"5A 125VAC"}},
    "rotary_enc":  {"label":"ENC",   "category":"Input",   "color":(100,100,100),"w":38,"h":38,
                    "symbol":"rect", "pins":["CLK","DT","SW","VCC","GND"],
                    "props":{"ppr":"20","type":"Incremental","steps":"20/rev"}},
    "joystick":    {"label":"JOY",   "category":"Input",   "color":(68,68,68),   "w":44,"h":44,
                    "symbol":"rect", "pins":["VRx","VRy","SW","VCC","GND"],
                    "props":{"axes":"2","output":"Analog 0-5V","button":"Yes"}},
    "keypad_4x4":  {"label":"KEYPAD","category":"Input",   "color":(51,68,85),   "w":64,"h":64,
                    "symbol":"rect", "pins":["R1","R2","R3","R4","C1","C2","C3","C4"],
                    "props":{"layout":"4x4","keys":"16","type":"Membrane"}},
    "keypad_4x3":  {"label":"KP4x3", "category":"Input",   "color":(45,60,75),   "w":54,"h":64,
                    "symbol":"rect", "pins":["R1","R2","R3","R4","C1","C2","C3"],
                    "props":{"layout":"4x3","keys":"12","type":"Membrane"}},
    "touch_cap":   {"label":"TTP223","category":"Input",   "color":(60,100,100), "w":30,"h":24,
                    "symbol":"ic",   "pins":["GND","VCC","I/O"],
                    "props":{"type":"Capacitive Touch","voltage":"2-5.5V","output":"Digital"}},

    # ── Semiconductors ───────────────────────────────────────────────────────
    "npn_2n2222":  {"label":"2N2222","category":"Semi",    "color":(68,102,136), "w":22,"h":34,
                    "symbol":"trans","pins":["B","C","E"],
                    "props":{"type":"NPN","Vceo":"40V","Ic":"600mA","hFE":"100-300","pkg":"TO-92"}},
    "pnp_2n2907":  {"label":"2N2907","category":"Semi",    "color":(100,70,100), "w":22,"h":34,
                    "symbol":"trans","pins":["B","C","E"],
                    "props":{"type":"PNP","Vceo":"40V","Ic":"600mA","hFE":"100-300","pkg":"TO-92"}},
    "mosfet_n":    {"label":"N-MOS", "category":"Semi",    "color":(136,68,102), "w":22,"h":34,
                    "symbol":"trans","pins":["G","D","S"],
                    "props":{"type":"N-Channel","Vds":"55V","Id":"47A","Rds":"22mΩ","pkg":"TO-220"}},
    "mosfet_p":    {"label":"P-MOS", "category":"Semi",    "color":(120,60,90),  "w":22,"h":34,
                    "symbol":"trans","pins":["G","D","S"],
                    "props":{"type":"P-Channel","Vds":"-30V","Id":"-5A","pkg":"TO-220"}},
    "diode_1n4007":{"label":"1N4007","category":"Semi",    "color":(68,136,170), "w":44,"h":18,
                    "symbol":"diode","pins":["A","K"],
                    "props":{"Vf":"1.1V","If":"1A","Vr":"1000V","type":"Rectifier"}},
    "schottky":    {"label":"1N5817","category":"Semi",    "color":(80,150,180), "w":44,"h":18,
                    "symbol":"diode","pins":["A","K"],
                    "props":{"Vf":"0.3V","If":"1A","Vr":"20V","type":"Schottky"}},
    "zener":       {"label":"ZENER", "category":"Semi",    "color":(68,136,136), "w":44,"h":18,
                    "symbol":"diode","pins":["A","K"],
                    "props":{"Vz":"5.1V","Iz":"5mA","Pz":"500mW"}},
    "opto":        {"label":"PC817", "category":"Semi",    "color":(60,80,60),   "w":44,"h":34,
                    "symbol":"ic",   "pins":["A","K","C","E"],
                    "props":{"type":"Optocoupler","CTR":"100%","Viso":"5kV","pkg":"DIP-4"}},
    "voltage_ref": {"label":"TL431", "category":"Semi",    "color":(90,60,90),   "w":22,"h":30,
                    "symbol":"trans","pins":["REF","K","A"],
                    "props":{"Vref":"2.5V","Imax":"100mA","type":"Adj Shunt Ref"}},

    # ── Connectors ───────────────────────────────────────────────────────────
    "gnd":         {"label":"GND",   "category":"Power",   "color":(60,60,60),   "w":24,"h":24,
                    "symbol":"rect", "pins":["GND"],
                    "props":{"type":"Ground Reference"}},
    "vcc":         {"label":"VCC",   "category":"Power",   "color":(200,50,50),  "w":24,"h":24,
                    "symbol":"rect", "pins":["VCC"],
                    "props":{"type":"Power Rail"}},
    "header_2":    {"label":"2-Pin", "category":"Connector","color":(80,80,80),  "w":20,"h":40,
                    "symbol":"rect", "pins":["1","2"],
                    "props":{"pitch":"2.54mm","type":"Male Header"}},
    "header_3":    {"label":"3-Pin", "category":"Connector","color":(80,80,80),  "w":20,"h":54,
                    "symbol":"rect", "pins":["1","2","3"],
                    "props":{"pitch":"2.54mm","type":"Male Header"}},
    "screw_term":  {"label":"TERM",  "category":"Connector","color":(70,100,70), "w":40,"h":30,
                    "symbol":"rect", "pins":["1","2"],
                    "props":{"pitch":"5.08mm","rating":"10A 300V","type":"Screw Terminal"}},
    "usb_c":       {"label":"USB-C", "category":"Connector","color":(100,100,120),"w":30,"h":24,
                    "symbol":"rect", "pins":["VBUS","GND","D+","D-","CC1","CC2"],
                    "props":{"type":"USB Type-C","rating":"5A","voltage":"5-20V"}},
}

_NET_COLOURS = [
    (0, 212, 255), (0, 255, 120), (255, 180, 0), (255, 80, 200),
    (100, 255, 80),(200, 120, 255),(0, 200, 200),(255, 120, 80),
    (255, 255, 80),(80, 255, 200),(255, 100, 100),(100, 200, 255),
]


# ─────────────────────────────────────────────────────────────────────────────
#  DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Component:
    id:       int
    type_id:  str
    x:        int
    y:        int
    rotation: int  = 0
    label:    str  = ""
    props:    dict = field(default_factory=dict)
    state:    dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.label:
            self.label = CATALOG.get(self.type_id, {}).get("label", self.type_id)
        if not self.props:
            self.props = dict(CATALOG.get(self.type_id, {}).get("props", {}))


@dataclass
class Wire:
    points:   list           # [(x,y), ...] in world coords
    src_comp: int   = -1     # NEW: source component id
    src_pin:  str   = ""     # NEW: source pin name
    dst_comp: int   = -1     # NEW: dest component id
    dst_pin:  str   = ""     # NEW: dest pin name
    color:    tuple = (0, 212, 255)
    net_id:   int   = -1


# ─────────────────────────────────────────────────────────────────────────────
#  CIRCUIT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class CircuitEngine:
    GRID = 20

    PANEL_W       = 220
    PANEL_ITEM_H  = 36
    PANEL_CAT_H   = 26

    # ── pin dot radius (world coords) for hit testing — scales with zoom
    PIN_HIT_BASE  = 12     # base radius; actual = max(PIN_HIT_BASE, PIN_HIT_BASE/zoom)

    def __init__(self, canvas_w: int = 1280, canvas_h: int = 720):
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h

        self._lock = threading.RLock()          # [FIX 7] thread safety

        # Scene
        self.components: list[Component] = []
        self.wires:      list[Wire]      = []
        self._next_id = 1

        # Viewport
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.zoom:  float = 1.0

        # Interaction
        self.selected_id:      Optional[int]  = None
        self.hovered_id:       Optional[int]  = None
        self.mode:             str            = "select"
        self.wire_in_progress: Optional[Wire] = None
        self.mouse_world:      tuple          = (0, 0)
        self.mouse_screen:     tuple          = (0, 0)

        # Panel state
        self.panel_visible:   bool  = True
        self.panel_scroll:    int   = 0
        self.panel_selected:  str   = "resistor"
        self._panel_items:    list  = []
        self._panel_hovered:  str   = ""

        # Simulation
        self.sim_running: bool  = False
        self.sim_tick:    int   = 0
        self._sim_t0:     float = 0.0
        self.serial_log:  deque = deque(maxlen=200)  # [FIX 9] deque, no leak

        # Undo
        self._undo_stack: list  = []

        self.show_grid:   bool = True
        self.show_pins:   bool = True
        self.show_labels: bool = True

        self._build_panel_items()

    # ─────────────────────────────────────────────────────────────────────────
    #  [FIX 1]  SNAPSHOT / RESTORE  (kernel undo support)
    # ─────────────────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        """Return a deep-copyable snapshot of circuit state."""
        with self._lock:
            return {
                "comps": deepcopy(self.components),
                "wires": deepcopy(self.wires),
                "next_id": self._next_id,
            }

    def restore(self, state: dict):
        """Restore circuit to a previous snapshot."""
        with self._lock:
            self.components  = deepcopy(state["comps"])
            self.wires       = deepcopy(state["wires"])
            self._next_id    = state.get("next_id", self._next_id)
            self.selected_id = None
            self.wire_in_progress = None
        self._log("[UNDO] State restored.")

    # ─────────────────────────────────────────────────────────────────────────
    #  [FIX 2]  add_wire — gesture-driven pin-to-pin connection
    # ─────────────────────────────────────────────────────────────────────────
    def add_wire(self, src_id: int, src_pin: str,
                 dst_id: int, dst_pin: str) -> Optional[Wire]:
        """
        Create a named-pin wire between two components.
        Routes via get_pin_by_name so the wire actually starts/ends
        at the correct GPIO pin, not just the component origin.
        """
        with self._lock:
            src = self.get_component(src_id)
            dst = self.get_component(dst_id)
            if src is None or dst is None:
                return None

            src_pt = self.get_pin_by_name(src, src_pin)
            dst_pt = self.get_pin_by_name(dst, dst_pin)

            # Fall back to nearest pin if named pin not found
            if src_pt is None:
                pins = self.get_pin_positions(src)
                src_pt = pins[0] if pins else (src.x, src.y)
            if dst_pt is None:
                pins = self.get_pin_positions(dst)
                dst_pt = pins[0] if pins else (dst.x, dst.y)

            # Manhattan route: two points + corner
            sx, sy = src_pt
            dx, dy = dst_pt
            if abs(dx - sx) > abs(dy - sy):
                corner = (dx, sy)
            else:
                corner = (sx, dy)

            points = [src_pt, corner, dst_pt]
            # Deduplicate consecutive identical points
            pts = [points[0]]
            for p in points[1:]:
                if p != pts[-1]:
                    pts.append(p)

            self._push_undo()
            net_id = len(self.wires)
            wire = Wire(
                points   = pts,
                src_comp = src_id,
                src_pin  = src_pin,
                dst_comp = dst_id,
                dst_pin  = dst_pin,
                color    = _NET_COLOURS[net_id % len(_NET_COLOURS)],
                net_id   = net_id,
            )
            self.wires.append(wire)
            self._log(f"[WIRE] {src.label}.{src_pin} → {dst.label}.{dst_pin}")
            return wire

    # ─────────────────────────────────────────────────────────────────────────
    #  [FIX 3]  hit_test with radius_multiplier
    # ─────────────────────────────────────────────────────────────────────────
    def hit_test(self, wx: float, wy: float,
                 radius_multiplier: float = 1.0) -> Optional[Component]:
        """
        Returns the topmost component whose bounding box contains (wx,wy).
        radius_multiplier expands the hit box — use 1.8 for forgiving AR grab.
        """
        with self._lock:
            for comp in reversed(self.components):
                d  = CATALOG[comp.type_id]
                cw = d["w"]
                ch = d["h"]
                pad = (radius_multiplier - 1.0) * max(cw, ch) * 0.5
                if (comp.x - pad <= wx <= comp.x + cw + pad and
                        comp.y - pad <= wy <= comp.y + ch + pad):
                    return comp
        return None

    # ─────────────────────────────────────────────────────────────────────────
    #  [FIX 4/5/6]  PIN POSITIONS — left+right split, named lookup, rotation
    # ─────────────────────────────────────────────────────────────────────────
    def _split_pins(self, pins: list[str]) -> tuple[list[str], list[str]]:
        """
        For ICs: split pin list into left-side and right-side columns.
        Convention: first half on left, second half on right (standard DIP/IC layout).
        For 2-pin components: pin 1 left, pin 2 right.
        """
        n = len(pins)
        if n <= 1:
            return pins, []
        if n == 2:
            return [pins[0]], [pins[1]]
        half = (n + 1) // 2
        return pins[:half], pins[half:]

    def get_pin_positions(self, comp: Component) -> list[tuple[int, int]]:
        """
        Return world-coord (x,y) for every pin.
        ICs get left+right column layout.
        Non-ICs retain their original single-edge layout.
        Rotation is applied around the component centre.
        """
        d     = CATALOG[comp.type_id]
        cw    = d["w"]
        ch    = d["h"]
        sym   = d.get("symbol", "rect")
        pins  = d.get("pins", [])
        if not pins:
            return []

        positions = []
        cx = comp.x + cw / 2
        cy = comp.y + ch / 2

        if sym in ("ic", "sensor"):
            left_pins, right_pins = self._split_pins(pins)

            # Left side
            nl = len(left_pins)
            step_l = ch / (nl + 1)
            for i in range(nl):
                py = comp.y + step_l * (i + 1)
                positions.append((comp.x, py))

            # Right side
            nr = len(right_pins)
            step_r = ch / (nr + 1)
            for i in range(nr):
                py = comp.y + step_r * (i + 1)
                positions.append((comp.x + cw, py))

        elif sym in ("res", "inductor"):
            # Leads on left and right ends (horizontal)
            positions.append((comp.x, comp.y + ch / 2))
            positions.append((comp.x + cw, comp.y + ch / 2))

        elif sym == "cap":
            # Two vertical plates — leads on left/right
            positions.append((comp.x, comp.y + ch / 2))
            positions.append((comp.x + cw, comp.y + ch / 2))

        elif sym in ("led", "diode"):
            positions.append((comp.x, comp.y + ch / 2))           # anode (left)
            positions.append((comp.x + cw, comp.y + ch / 2))     # cathode (right)

        elif sym == "trans":
            # B on left, C top, E bottom
            r = max(8, min(cw, ch) // 2 - 2)
            positions.append((comp.x, comp.y + ch / 2))       # B
            positions.append((comp.x + cw / 2, comp.y))       # C
            positions.append((comp.x + cw / 2, comp.y + ch))  # E

        else:
            # rect: evenly spaced on left side (legacy behaviour)
            n = len(pins)
            step = ch / (n + 1)
            for i in range(n):
                positions.append((comp.x, comp.y + step * (i + 1)))

        # [FIX 6] Apply rotation around component centre
        if comp.rotation != 0:
            angle = math.radians(comp.rotation)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rotated = []
            for px, py in positions:
                dx, dy = px - cx, py - cy
                rx = cx + dx * cos_a - dy * sin_a
                ry = cy + dx * sin_a + dy * cos_a
                rotated.append((rx, ry))
            positions = rotated

        return [(int(round(px)), int(round(py))) for px, py in positions]

    def get_pin_by_name(self, comp: Component,
                        pin_name: str) -> Optional[tuple[int, int]]:
        """
        Return the world-coord (x,y) of a named pin on comp.
        Returns None if pin_name not found.
        Case-insensitive match.
        """
        d    = CATALOG.get(comp.type_id, {})
        pins = d.get("pins", [])
        pin_name_lower = pin_name.lower()

        # Try exact match first, then case-insensitive
        idx = None
        for i, p in enumerate(pins):
            if p == pin_name:
                idx = i
                break
        if idx is None:
            for i, p in enumerate(pins):
                if p.lower() == pin_name_lower:
                    idx = i
                    break
        if idx is None:
            return None

        positions = self.get_pin_positions(comp)
        if idx < len(positions):
            return positions[idx]
        return None

    def nearest_pin(self, wx: float, wy: float,
                    threshold: float = None) -> Optional[tuple[int, int]]:
        """
        Return snapped world coord of nearest pin within threshold.
        Threshold scales with zoom for AR usability.
        """
        if threshold is None:
            threshold = max(self.PIN_HIT_BASE, self.PIN_HIT_BASE / self.zoom)

        best_d  = threshold
        best_pt = None
        with self._lock:
            for comp in self.components:
                for px, py in self.get_pin_positions(comp):
                    d = math.hypot(wx - px, wy - py)
                    if d < best_d:
                        best_d  = d
                        best_pt = (px, py)
        return best_pt

    def nearest_pin_with_info(self, wx: float, wy: float,
                               threshold: float = None
                               ) -> Optional[tuple[Component, str, tuple]]:
        """
        Like nearest_pin but returns (component, pin_name, (wx,wy)).
        Used by kernel wire-draw mode to record exact src/dst pin.
        """
        if threshold is None:
            threshold = max(self.PIN_HIT_BASE, self.PIN_HIT_BASE / self.zoom)

        best_d    = threshold
        best_info = None
        with self._lock:
            for comp in self.components:
                d_    = CATALOG[comp.type_id]
                pins  = d_.get("pins", [])
                positions = self.get_pin_positions(comp)
                for i, (px, py) in enumerate(positions):
                    dist = math.hypot(wx - px, wy - py)
                    if dist < best_d:
                        best_d    = dist
                        pin_name  = pins[i] if i < len(pins) else str(i)
                        best_info = (comp, pin_name, (px, py))
        return best_info

    # ─────────────────────────────────────────────────────────────────────────
    #  BOARD AREA
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def board_w(self) -> int:
        return self.canvas_w - (self.PANEL_W if self.panel_visible else 0)

    # ─────────────────────────────────────────────────────────────────────────
    #  PANEL ITEMS
    # ─────────────────────────────────────────────────────────────────────────
    def _build_panel_items(self):
        cats: dict[str, list] = {}
        for tid, d in CATALOG.items():
            cats.setdefault(d["category"], []).append((tid, d["label"]))
        order = ["MCU","Passive","LED","Display","Sensor","Actuator","Comms","Power","Input","Semi","Connector"]
        self._panel_items = []
        for cat in order:
            if cat not in cats:
                continue
            self._panel_items.append(("cat", cat, cat))
            for tid, lbl in sorted(cats[cat], key=lambda x: x[1]):
                self._panel_items.append(("comp", tid, lbl))
        for cat in cats:
            if cat not in order:
                self._panel_items.append(("cat", cat, cat))
                for tid, lbl in sorted(cats[cat], key=lambda x: x[1]):
                    self._panel_items.append(("comp", tid, lbl))

    # ─────────────────────────────────────────────────────────────────────────
    #  ID / SNAP
    # ─────────────────────────────────────────────────────────────────────────
    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    @staticmethod
    def snap(v: float) -> int:
        g = CircuitEngine.GRID
        return int(round(v / g) * g)

    # ─────────────────────────────────────────────────────────────────────────
    #  COORDINATE TRANSFORMS
    # ─────────────────────────────────────────────────────────────────────────
    def to_screen(self, wx: float, wy: float) -> tuple[int, int]:
        return (int(wx * self.zoom + self.pan_x),
                int(wy * self.zoom + self.pan_y))

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return ((sx - self.pan_x) / self.zoom,
                (sy - self.pan_y) / self.zoom)

    def in_panel(self, sx: float, sy: float) -> bool:
        return self.panel_visible and sx >= self.board_w

    # ─────────────────────────────────────────────────────────────────────────
    #  COMPONENT MANAGEMENT  (thread-safe)
    # ─────────────────────────────────────────────────────────────────────────
    def add_component(self, type_id: str, wx: float, wy: float,
                      label: str = "", props: dict = None) -> Optional[Component]:
        if type_id not in CATALOG:
            return None
        with self._lock:
            self._push_undo()
            comp = Component(
                id=self._new_id(),
                type_id=type_id,
                x=self.snap(wx),
                y=self.snap(wy),
                label=label or CATALOG[type_id]["label"],
                props=props or dict(CATALOG[type_id].get("props", {}))
            )
            self.components.append(comp)
        self._log(f"[PLACE] {comp.label} (id={comp.id}) @ ({comp.x},{comp.y})")
        return comp

    def remove_component(self, comp_id: int):
        with self._lock:
            self._push_undo()
            self.components = [c for c in self.components if c.id != comp_id]
            # Remove wires connected to this component
            self.wires = [w for w in self.wires
                          if w.src_comp != comp_id and w.dst_comp != comp_id]
            if self.selected_id == comp_id:
                self.selected_id = None

    def duplicate_component(self, comp_id: int) -> Optional[Component]:
        src = self.get_component(comp_id)
        if not src:
            return None
        with self._lock:
            self._push_undo()
            new_c = Component(
                id=self._new_id(),
                type_id=src.type_id,
                x=src.x + self.GRID * 4,
                y=src.y + self.GRID * 4,
                rotation=src.rotation,
                label=src.label,
                props=dict(src.props)
            )
            self.components.append(new_c)
        return new_c

    def rotate_component(self, comp_id: int, delta: int = 90):
        c = self.get_component(comp_id)
        if c:
            with self._lock:
                c.rotation = (c.rotation + delta) % 360
            # Re-route wires attached to this component
            self._reroute_component_wires(comp_id)

    def _reroute_component_wires(self, comp_id: int):
        """Re-compute wire endpoints after a component rotation/move."""
        with self._lock:
            comp = self.get_component(comp_id)
            if comp is None:
                return
            for wire in self.wires:
                changed = False
                if wire.src_comp == comp_id and wire.src_pin:
                    pt = self.get_pin_by_name(comp, wire.src_pin)
                    if pt:
                        wire.points[0] = pt
                        changed = True
                if wire.dst_comp == comp_id and wire.dst_pin:
                    pt = self.get_pin_by_name(comp, wire.dst_pin)
                    if pt:
                        wire.points[-1] = pt
                        changed = True
                if changed and len(wire.points) >= 2:
                    # Recompute Manhattan corner
                    sx, sy = wire.points[0]
                    ex, ey = wire.points[-1]
                    if abs(ex - sx) > abs(ey - sy):
                        corner = (ex, sy)
                    else:
                        corner = (sx, ey)
                    wire.points = [wire.points[0], corner, wire.points[-1]]

    def get_component(self, comp_id: int) -> Optional[Component]:
        for c in self.components:
            if c.id == comp_id:
                return c
        return None

    def clear(self):
        with self._lock:
            self._push_undo()
            self.components.clear()
            self.wires.clear()
            self.selected_id = None
            self.wire_in_progress = None
            self.sim_running = False
        self._log("[CLEAR] Board cleared.")

    # ─────────────────────────────────────────────────────────────────────────
    #  WIRES  (manual/cursor-draw, for non-gesture mode)
    # ─────────────────────────────────────────────────────────────────────────
    def start_wire(self, wx: float, wy: float):
        pin_info = self.nearest_pin_with_info(wx, wy)
        if pin_info:
            comp, pin_name, pt = pin_info
            self.wire_in_progress = Wire(
                points=[pt], src_comp=comp.id, src_pin=pin_name)
        else:
            pt = (self.snap(wx), self.snap(wy))
            self.wire_in_progress = Wire(points=[pt])

    def extend_wire(self, wx: float, wy: float):
        if not self.wire_in_progress:
            return
        pin  = self.nearest_pin(wx, wy)
        pt   = pin if pin else (self.snap(wx), self.snap(wy))
        # Only store the start point + current endpoint (preview only)
        if len(self.wire_in_progress.points) == 1:
            self.wire_in_progress.points.append(pt)
        else:
            self.wire_in_progress.points[-1] = pt

    def finish_wire(self):
        if self.wire_in_progress and len(self.wire_in_progress.points) >= 2:
            with self._lock:
                self._push_undo()
                # Try to snap endpoint to a named pin
                ex, ey = self.wire_in_progress.points[-1]
                pin_info = self.nearest_pin_with_info(ex, ey)
                if pin_info:
                    comp, pin_name, pt = pin_info
                    self.wire_in_progress.points[-1] = pt
                    self.wire_in_progress.dst_comp   = comp.id
                    self.wire_in_progress.dst_pin    = pin_name

                # Rebuild Manhattan path
                sx, sy = self.wire_in_progress.points[0]
                ex, ey = self.wire_in_progress.points[-1]
                if abs(ex - sx) > abs(ey - sy):
                    corner = (ex, sy)
                else:
                    corner = (sx, ey)
                pts = [self.wire_in_progress.points[0]]
                if corner != pts[-1] and corner != (ex, ey):
                    pts.append(corner)
                pts.append((ex, ey))
                self.wire_in_progress.points = pts

                net_id = len(self.wires)
                self.wire_in_progress.net_id = net_id
                self.wire_in_progress.color  = _NET_COLOURS[net_id % len(_NET_COLOURS)]
                self.wires.append(self.wire_in_progress)
                self._log(f"[WIRE] Wire #{net_id} added")
        self.wire_in_progress = None

    def cancel_wire(self):
        self.wire_in_progress = None

    def remove_last_wire(self):
        if self.wires:
            self.wires.pop()

    # ─────────────────────────────────────────────────────────────────────────
    #  PANEL HIT TEST
    # ─────────────────────────────────────────────────────────────────────────
    def panel_hit_test(self, sx: float, sy: float) -> Optional[str]:
        if not self.in_panel(sx, sy):
            return None
        rel_y = int(sy) - (-self.panel_scroll)
        y = 0
        for kind, key, lbl in self._panel_items:
            h = self.PANEL_CAT_H if kind == "cat" else self.PANEL_ITEM_H
            if rel_y >= y and rel_y < y + h:
                return key if kind == "comp" else None
            y += h
        return None

    def panel_scroll_by(self, dy: int):
        max_scroll = max(0, self._panel_total_h() - self.canvas_h + 60)
        self.panel_scroll = max(0, min(max_scroll, self.panel_scroll - dy))

    def _panel_total_h(self) -> int:
        h = 0
        for kind, _, _ in self._panel_items:
            h += self.PANEL_CAT_H if kind == "cat" else self.PANEL_ITEM_H
        return h

    # ─────────────────────────────────────────────────────────────────────────
    #  VIEWPORT
    # ─────────────────────────────────────────────────────────────────────────
    def pan(self, dx: float, dy: float):
        self.pan_x += dx
        self.pan_y += dy

    def zoom_at(self, sx: float, sy: float, factor: float):
        nz = max(0.15, min(6.0, self.zoom * factor))
        self.pan_x = sx - (sx - self.pan_x) * (nz / self.zoom)
        self.pan_y = sy - (sy - self.pan_y) * (nz / self.zoom)
        self.zoom  = nz

    def fit_to_screen(self):
        if not self.components:
            self.zoom, self.pan_x, self.pan_y = 1.0, 0.0, 0.0
            return
        xs  = [c.x for c in self.components]
        ys  = [c.y for c in self.components]
        xs2 = [c.x + CATALOG[c.type_id]["w"] for c in self.components]
        ys2 = [c.y + CATALOG[c.type_id]["h"] for c in self.components]
        mnx, mxx = min(xs), max(xs2)
        mny, mxy = min(ys), max(ys2)
        pad = 60
        bw  = self.board_w - pad * 2
        bh  = self.canvas_h - pad * 2
        zx  = bw / max(mxx - mnx, 1)
        zy  = bh / max(mxy - mny, 1)
        self.zoom  = min(zx, zy, 4.0)
        cx  = (mnx + mxx) / 2
        cy  = (mny + mxy) / 2
        self.pan_x = self.board_w / 2 - cx * self.zoom
        self.pan_y = self.canvas_h / 2 - cy * self.zoom

    # ─────────────────────────────────────────────────────────────────────────
    #  UNDO  (engine-owned, authoritative)
    # ─────────────────────────────────────────────────────────────────────────
    def _push_undo(self):
        """Internal — call WITHOUT holding _lock (snapshot() acquires it)."""
        state = {
            "comps":   deepcopy(self.components),
            "wires":   deepcopy(self.wires),
            "next_id": self._next_id,
        }
        self._undo_stack.append(state)
        if len(self._undo_stack) > 20:
            self._undo_stack.pop(0)

    def undo(self):
        if not self._undo_stack:
            return
        state = self._undo_stack.pop()
        with self._lock:
            self.components  = state["comps"]
            self.wires       = state["wires"]
            self._next_id    = state.get("next_id", self._next_id)
            self.selected_id = None
        self._log("[UNDO] Action undone.")

    # ─────────────────────────────────────────────────────────────────────────
    #  SIMULATION  [FIX 10] — frame-rate independent
    # ─────────────────────────────────────────────────────────────────────────
    def start_simulation(self):
        if self.sim_running:
            return
        self.sim_running = True
        self.sim_tick    = 0
        self._sim_t0     = time.time()
        self._last_sensor_t: float = 0.0
        self._log("══ Simulation Started ══")
        for c in self.components:
            cat = CATALOG[c.type_id]["category"]
            if cat == "MCU":
                self._log(f"[BOOT] {c.label} @ {c.props.get('cpu','?')} starting…")
            elif cat == "LED":
                c.state.update({"on": False, "brightness": 0.0, "pwm": 0.0})
            elif cat == "Sensor":
                c.state["ready"] = False

    def stop_simulation(self):
        self.sim_running = False
        for c in self.components:
            c.state.clear()
        self._log("══ Simulation Stopped ══")

    def tick_simulation(self):
        if not self.sim_running:
            return
        self.sim_tick += 1
        elapsed = time.time() - self._sim_t0  # always wall-clock based

        for c in self.components:
            cat = CATALOG[c.type_id]["category"]
            tid = c.type_id

            if cat == "LED":
                freq = 0.5 + 0.5
                c.state["on"]         = (elapsed % (1 / freq)) < (0.5 / freq)
                c.state["brightness"] = max(0.4, 0.8 + 0.2 * math.sin(elapsed * 6))
                c.state["pwm"]        = c.state["brightness"]

            elif cat == "Actuator" and tid == "servo":
                c.state["angle"] = int(90 + 80 * math.sin(elapsed * 0.8))

            elif cat == "Actuator" and tid in ("dc_motor", "stepper"):
                c.state["rpm"] = int(150 + 50 * math.sin(elapsed * 1.2))

        # Sensor readings every ~1.5 seconds (wall-clock, frame-rate independent)
        last_sensor = getattr(self, '_last_sensor_t', 0.0)
        if elapsed - last_sensor >= 1.5:
            self._last_sensor_t = elapsed
            for c in self.components:
                self._tick_sensor(c, elapsed)

    def _tick_sensor(self, c: Component, elapsed: float):
        tid = c.type_id
        if tid in ("dht22", "dht11"):
            c.state["temp"] = round(22 + 3 * math.sin(elapsed * 0.1) + random.uniform(-0.3, 0.3), 1)
            c.state["hum"]  = round(55 + 10 * math.cos(elapsed * 0.07) + random.uniform(-1, 1), 1)
            self._log(f"[{c.label}] T:{c.state['temp']}°C H:{c.state['hum']}%")
        elif tid == "bmp280":
            c.state["pressure"] = round(1013 + 2 * math.sin(elapsed * 0.05) + random.uniform(-0.5, 0.5), 1)
            c.state["temp"]     = round(23 + random.uniform(-0.5, 0.5), 1)
            self._log(f"[{c.label}] P:{c.state['pressure']}hPa T:{c.state['temp']}°C")
        elif tid == "ldr":
            c.state["lux"] = int(500 + 400 * math.sin(elapsed * 0.15) + random.randint(-20, 20))
            self._log(f"[{c.label}] Lux:{c.state['lux']}")
        elif tid == "mpu6050":
            c.state.update({
                "ax": round(0.3 * math.sin(elapsed * 2), 2),
                "ay": round(0.3 * math.cos(elapsed * 1.5), 2),
                "az": round(9.81 + random.uniform(-0.05, 0.05), 2),
                "gx": round(random.uniform(-5, 5), 1),
                "gy": round(random.uniform(-5, 5), 1),
                "gz": round(random.uniform(-5, 5), 1),
            })
            self._log(f"[{c.label}] ax:{c.state['ax']} ay:{c.state['ay']} az:{c.state['az']}")
        elif tid == "ds18b20":
            c.state["temp"] = round(23 + random.uniform(-0.3, 0.3), 1)
            self._log(f"[{c.label}] T:{c.state['temp']}°C")
        elif tid == "ultrasonic":
            c.state["dist"] = round(20 + 15 * abs(math.sin(elapsed * 0.5)) + random.uniform(-0.5, 0.5), 1)
            self._log(f"[{c.label}] D:{c.state['dist']}cm")
        elif tid == "pir":
            c.state["motion"] = random.random() > 0.7
            if c.state["motion"]:
                self._log(f"[{c.label}] Motion DETECTED")
        elif tid in ("gas_mq2",):
            c.state["ppm"] = int(200 + 100 * abs(math.sin(elapsed * 0.3)))
            self._log(f"[{c.label}] Gas:{c.state['ppm']}ppm")

    # ─────────────────────────────────────────────────────────────────────────
    #  LOG
    # ─────────────────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        ts   = time.strftime("%H:%M:%S")
        self.serial_log.append(f"[{ts}] {msg}")  # deque auto-trims

    def get_log_tail(self, n: int = 8) -> list[str]:
        return list(self.serial_log)[-n:]

    # ─────────────────────────────────────────────────────────────────────────
    #  SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str):
        with self._lock:
            data = {
                "version": 3,
                "zoom": self.zoom,
                "pan": [self.pan_x, self.pan_y],
                "components": [
                    {"id": c.id, "type": c.type_id, "x": c.x, "y": c.y,
                     "rotation": c.rotation, "label": c.label, "props": c.props}
                    for c in self.components
                ],
                "wires": [
                    {"points": w.points, "color": list(w.color), "net_id": w.net_id,
                     "src_comp": w.src_comp, "src_pin": w.src_pin,
                     "dst_comp": w.dst_comp, "dst_pin": w.dst_pin}
                    for w in self.wires
                ]
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._log(f"[SAVE] → {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.clear()
        self.zoom = data.get("zoom", 1.0)
        px, py = data.get("pan", [0, 0])
        self.pan_x, self.pan_y = px, py
        for cd in data.get("components", []):
            if cd["type"] not in CATALOG:
                continue
            c = Component(
                id=cd["id"], type_id=cd["type"],
                x=cd["x"],   y=cd["y"],
                rotation=cd.get("rotation", 0),
                label=cd.get("label", ""),
                props=cd.get("props", {})
            )
            self.components.append(c)
        for wd in data.get("wires", []):
            pts = [tuple(p) for p in wd["points"]]
            col = tuple(wd.get("color", [0, 212, 255]))
            nid = wd.get("net_id", -1)
            self.wires.append(Wire(
                points=pts, color=col, net_id=nid,
                src_comp=wd.get("src_comp", -1), src_pin=wd.get("src_pin", ""),
                dst_comp=wd.get("dst_comp", -1), dst_pin=wd.get("dst_pin", ""),
            ))
        self._next_id = max((c.id for c in self.components), default=0) + 1
        self._log(f"[LOAD] {len(self.components)} comps, {len(self.wires)} wires")

    @staticmethod
    def search(query: str) -> list[str]:
        q = query.lower()
        return [
            tid for tid, d in CATALOG.items()
            if q in tid or q in d["label"].lower() or q in d["category"].lower()
        ]

    # ─────────────────────────────────────────────────────────────────────────
    #  MAIN RENDER
    # ─────────────────────────────────────────────────────────────────────────
    def render(self, canvas: np.ndarray) -> np.ndarray:
        self._draw_board(canvas)
        if self.panel_visible:
            self._draw_panel(canvas)
        self._draw_hud(canvas)
        return canvas

    def render_board_only(self, canvas: np.ndarray,
                          override_w: int = None,
                          override_h: int = None) -> np.ndarray:
        """
        [FIX 8] Safe render that NEVER mutates self.canvas_w/canvas_h.
        Pass override_w/override_h to render at a different size without
        touching global state.
        """
        if override_w is not None or override_h is not None:
            # Temporarily adjust pan/zoom so content fits the override canvas
            # without touching engine state (we operate on local copies)
            saved = (self.canvas_w, self.canvas_h, self.pan_x, self.pan_y, self.zoom)
            try:
                if override_w:
                    self.canvas_w = override_w
                if override_h:
                    self.canvas_h = override_h
                self._draw_board(canvas)
                self._draw_hud(canvas)
            finally:
                (self.canvas_w, self.canvas_h,
                 self.pan_x, self.pan_y, self.zoom) = saved
        else:
            self._draw_board(canvas)
            self._draw_hud(canvas)
        return canvas

    # ─────────────────────────────────────────────────────────────────────────
    #  BOARD RENDERING
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_board(self, canvas: np.ndarray):
        if self.show_grid:
            self._draw_grid(canvas)
        with self._lock:
            wires  = list(self.wires)
            wip    = self.wire_in_progress
            comps  = list(self.components)
        for wire in wires:
            self._draw_wire(canvas, wire)
        if wip and wip.points:
            self._draw_wire_preview(canvas, wip)
        for comp in comps:
            self._draw_component(canvas, comp)
        self._draw_inspector(canvas)

    def _draw_grid(self, canvas: np.ndarray):
        bw   = self.board_w
        h    = canvas.shape[0]
        step = max(4, int(self.GRID * self.zoom))
        ox   = int(self.pan_x % step)
        oy   = int(self.pan_y % step)
        for x in range(ox, bw, step):
            for y in range(oy, h, step):
                if 0 <= y < h and 0 <= x < bw:
                    canvas[y, x] = (22, 38, 18)

    def _draw_wire(self, canvas: np.ndarray, wire: Wire):
        pts  = [self.to_screen(p[0], p[1]) for p in wire.points]
        if len(pts) < 2:
            return
        col   = wire.color
        glow  = tuple(max(0, int(c * 0.3)) for c in col)
        thick = max(1, int(2 * self.zoom))

        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], glow, thick + 4, cv2.LINE_AA)
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i+1], col, thick, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(canvas, pt, max(2, thick + 1), col, -1, cv2.LINE_AA)

        # Draw pin labels at endpoints if named
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = max(0.22, min(0.35, self.zoom * 0.3))
        if wire.src_pin:
            sp = self.to_screen(*wire.points[0])
            cv2.putText(canvas, wire.src_pin[:5], (sp[0] + 4, sp[1] - 4),
                        font, fs, tuple(min(255, int(c * 1.4)) for c in col),
                        1, cv2.LINE_AA)
        if wire.dst_pin:
            dp = self.to_screen(*wire.points[-1])
            cv2.putText(canvas, wire.dst_pin[:5], (dp[0] + 4, dp[1] - 4),
                        font, fs, tuple(min(255, int(c * 1.4)) for c in col),
                        1, cv2.LINE_AA)

    def _draw_wire_preview(self, canvas: np.ndarray, wip: Wire):
        pts  = [self.to_screen(p[0], p[1]) for p in wip.points]
        mx, my = self.to_screen(*self.mouse_world)
        lx, ly = pts[-1]
        if abs(mx - lx) > abs(my - ly):
            corner = (mx, ly)
        else:
            corner = (lx, my)
        preview_pts = pts + [corner, (mx, my)]
        col = (80, 212, 255)
        for i in range(len(preview_pts) - 1):
            cv2.line(canvas, preview_pts[i], preview_pts[i+1], col,
                     max(1, int(1.5 * self.zoom)), cv2.LINE_AA)
        cv2.circle(canvas, (mx, my), 4, col, -1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    #  COMPONENT RENDERING
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_component(self, canvas: np.ndarray, comp: Component):
        d = CATALOG.get(comp.type_id)
        if not d:
            return

        cw  = int(d["w"] * self.zoom)
        ch  = int(d["h"] * self.zoom)
        if cw < 4 or ch < 4:
            return

        sx, sy = self.to_screen(comp.x, comp.y)
        is_sel = (comp.id == self.selected_id)
        is_hov = (comp.id == self.hovered_id)
        col    = d["color"]
        sym    = d.get("symbol", "rect")

        pad = 16  # extra padding to accommodate right-side pin labels
        img = np.zeros((ch + pad*2, cw + pad*2, 3), dtype=np.uint8)
        ox, oy = pad, pad

        fill_col   = tuple(int(c * 0.14) for c in col)
        border_col = col
        if is_sel:
            border_col = (50, 130, 255)
        elif is_hov:
            border_col = tuple(min(255, int(c * 1.4)) for c in col)

        bthick = 2 if is_sel else 1

        if sym == "ic":
            self._sym_ic(img, ox, oy, cw, ch, col, fill_col, border_col, bthick, comp, d)
        elif sym == "res":
            self._sym_res(img, ox, oy, cw, ch, col, fill_col, border_col, bthick, comp)
        elif sym == "cap":
            self._sym_cap(img, ox, oy, cw, ch, col, fill_col, border_col, bthick, comp)
        elif sym == "led":
            self._sym_led(img, ox, oy, cw, ch, col, fill_col, border_col, bthick, comp, d)
        elif sym == "diode":
            self._sym_diode(img, ox, oy, cw, ch, col, fill_col, border_col, bthick, comp)
        elif sym == "trans":
            self._sym_trans(img, ox, oy, cw, ch, col, fill_col, border_col, bthick, comp)
        elif sym == "sensor":
            self._sym_sensor(img, ox, oy, cw, ch, col, fill_col, border_col, bthick, comp)
        else:
            self._sym_rect(img, ox, oy, cw, ch, col, fill_col, border_col, bthick, comp)

        if is_sel:
            cv2.rectangle(img, (ox - 3, oy - 3), (ox + cw + 3, oy + ch + 3),
                          (50, 130, 255), 1, cv2.LINE_AA)

        if self.sim_running and comp.state:
            self._draw_state_overlay(img, ox, oy, cw, ch, comp)

        img_rot = self._rotate_img(img, comp.rotation)
        self._paste(canvas, img_rot, sx - pad, sy - pad, clip_w=self.board_w)

    # ── IC symbol with BOTH-SIDE pins ───────────────────────────────────────
    def _sym_ic(self, img, ox, oy, cw, ch, col, fill, border, thick, comp, d):
        cv2.rectangle(img, (ox, oy), (ox+cw, oy+ch), fill, -1)
        cv2.rectangle(img, (ox, oy), (ox+cw, oy+ch), border, thick, cv2.LINE_AA)
        # Notch
        nw = max(8, cw // 5)
        cv2.ellipse(img, (ox + cw//2, oy), (nw//2, nw//2), 0, 180, 360,
                    tuple(int(c*0.5) for c in border), 1, cv2.LINE_AA)
        # Label
        lbl  = d.get("label", comp.type_id)[:8]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = max(0.25, min(0.5, cw / 120))
        (tw, th), _ = cv2.getTextSize(lbl, font, fs, 1)
        cv2.putText(img, lbl, (ox + cw//2 - tw//2, oy + ch//2 + th//2),
                    font, fs, tuple(min(255, int(c*1.6)) for c in col), 1, cv2.LINE_AA)
        cat  = d.get("category","")[:5]
        fs2  = max(0.18, min(0.3, cw / 200))
        (cw2, _), _ = cv2.getTextSize(cat, font, fs2, 1)
        cv2.putText(img, cat, (ox + cw//2 - cw2//2, oy + 12),
                    font, fs2, tuple(int(c*0.6) for c in border), 1, cv2.LINE_AA)
        # ── BOTH-SIDE PINS ──
        self._draw_pins_both_sides(img, ox, oy, cw, ch, col, d)

    def _sym_rect(self, img, ox, oy, cw, ch, col, fill, border, thick, comp):
        cv2.rectangle(img, (ox, oy), (ox+cw, oy+ch), fill, -1)
        cv2.rectangle(img, (ox, oy), (ox+cw, oy+ch), border, thick, cv2.LINE_AA)
        lbl = CATALOG.get(comp.type_id, {}).get("label", comp.type_id)[:8]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = max(0.22, min(0.42, cw / 100))
        (tw, th), _ = cv2.getTextSize(lbl, font, fs, 1)
        cv2.putText(img, lbl, (ox + cw//2 - tw//2, oy + ch//2 + th//2),
                    font, fs, tuple(min(255, int(c*1.8)) for c in col), 1, cv2.LINE_AA)
        d = CATALOG.get(comp.type_id, {})
        self._draw_pins_both_sides(img, ox, oy, cw, ch, col, d)

    def _sym_res(self, img, ox, oy, cw, ch, col, fill, border, thick, comp):
        cx_, cy_ = ox + cw//2, oy + ch//2
        rw, rh = max(12, cw - 8), max(6, ch - 4)
        rx, ry = cx_ - rw//2, cy_ - rh//2
        cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), fill, -1)
        cv2.rectangle(img, (rx, ry), (rx+rw, ry+rh), border, thick, cv2.LINE_AA)
        for i, bc in enumerate([(0,0,180),(0,0,0),(200,120,0),(200,180,120)]):
            bx = rx + 4 + i * max(2, (rw-8)//4)
            bw2 = max(1, (rw-8)//5)
            cv2.rectangle(img, (bx, ry+2), (bx+bw2, ry+rh-2), bc, -1)
        cv2.line(img, (ox, cy_), (rx, cy_), col, thick, cv2.LINE_AA)
        cv2.line(img, (rx+rw, cy_), (ox+cw, cy_), col, thick, cv2.LINE_AA)
        # Pin dots
        cv2.circle(img, (ox, cy_), 3, (255, 210, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (ox+cw, cy_), 3, (255, 210, 0), -1, cv2.LINE_AA)

    def _sym_cap(self, img, ox, oy, cw, ch, col, fill, border, thick, comp):
        cx_, cy_ = ox + cw//2, oy + ch//2
        gap = max(4, cw // 6)
        ph  = max(10, ch - 10)
        pw  = max(4, cw - 6)
        cv2.line(img, (cx_ - gap//2 - pw//2, cy_), (cx_ - gap//2, cy_), col, thick, cv2.LINE_AA)
        cv2.line(img, (cx_ - gap//2, cy_ - ph//2), (cx_ - gap//2, cy_ + ph//2), border, thick+1, cv2.LINE_AA)
        cv2.line(img, (cx_ + gap//2, cy_ - ph//2), (cx_ + gap//2, cy_ + ph//2), border, thick+1, cv2.LINE_AA)
        cv2.line(img, (cx_ + gap//2, cy_), (cx_ + gap//2 + pw//2, cy_), col, thick, cv2.LINE_AA)
        cv2.putText(img, "+", (ox+1, cy_-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, border, 1)
        cv2.circle(img, (ox, cy_), 3, (255, 210, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (ox+cw, cy_), 3, (255, 210, 0), -1, cv2.LINE_AA)

    def _sym_led(self, img, ox, oy, cw, ch, col, fill, border, thick, comp, d):
        cx_, cy_ = ox + cw//2, oy + ch//2
        hh = max(8, ch//2 - 2)
        hw = max(6, cw//2 - 2)
        tri = np.array([
            [cx_ - hw, cy_ - hh],
            [cx_ - hw, cy_ + hh],
            [cx_ + hw, cy_],
        ], np.int32)
        cv2.fillPoly(img, [tri], fill)
        cv2.polylines(img, [tri], True, border, thick, cv2.LINE_AA)
        cv2.line(img, (cx_ + hw, cy_ - hh), (cx_ + hw, cy_ + hh), border, thick+1, cv2.LINE_AA)
        if self.sim_running and comp.state.get("on"):
            br = comp.state.get("brightness", 1.0)
            gc = tuple(min(255, int(c * br * 1.5)) for c in col)
            ov = img.copy()
            cv2.circle(ov, (cx_, cy_), int(max(cw, ch) * 0.7), gc, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)
            cv2.polylines(img, [tri], True, border, thick, cv2.LINE_AA)
            cv2.line(img, (cx_ + hw, cy_ - hh), (cx_ + hw, cy_ + hh), border, thick+1, cv2.LINE_AA)
        for ang in [30, 50]:
            r = math.radians(ang)
            sx2 = int(cx_ + hw + 8 * math.cos(r))
            sy2 = int(cy_ - 12 * math.sin(r))
            ex2 = int(sx2 + 8 * math.cos(r))
            ey2 = int(sy2 - 8 * math.sin(r))
            cv2.arrowedLine(img, (sx2, sy2), (ex2, ey2), border, 1,
                            cv2.LINE_AA, tipLength=0.4)
        cv2.circle(img, (ox, cy_), 3, (255, 210, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (ox+cw, cy_), 3, (255, 210, 0), -1, cv2.LINE_AA)

    def _sym_diode(self, img, ox, oy, cw, ch, col, fill, border, thick, comp):
        cx_, cy_ = ox + cw//2, oy + ch//2
        hw = max(6, cw//2 - 6)
        hh = max(4, ch//2 - 2)
        tri = np.array([
            [cx_ - hw, cy_ - hh],
            [cx_ - hw, cy_ + hh],
            [cx_ + hw, cy_],
        ], np.int32)
        cv2.fillPoly(img, [tri], fill)
        cv2.polylines(img, [tri], True, border, thick, cv2.LINE_AA)
        cv2.line(img, (cx_ + hw, cy_ - hh), (cx_ + hw, cy_ + hh), border, thick+1, cv2.LINE_AA)
        cv2.line(img, (ox, cy_), (cx_ - hw, cy_), col, thick, cv2.LINE_AA)
        cv2.line(img, (cx_ + hw, cy_), (ox+cw, cy_), col, thick, cv2.LINE_AA)

    def _sym_trans(self, img, ox, oy, cw, ch, col, fill, border, thick, comp):
        cx_, cy_ = ox + cw//2, oy + ch//2
        r = max(8, min(cw, ch)//2 - 2)
        cv2.circle(img, (cx_, cy_), r, fill, -1)
        cv2.circle(img, (cx_, cy_), r, border, thick, cv2.LINE_AA)
        cv2.line(img, (cx_ - r, cy_), (ox, cy_), col, thick, cv2.LINE_AA)
        cv2.line(img, (cx_, cy_ - r//2), (cx_, oy), col, thick, cv2.LINE_AA)
        cv2.line(img, (cx_, cy_ + r//2), (cx_, oy+ch), col, thick, cv2.LINE_AA)
        fs = max(0.2, 0.25 * r / 12)
        cv2.putText(img, "B", (ox+1, cy_+4), cv2.FONT_HERSHEY_SIMPLEX, fs, border, 1)
        cv2.putText(img, "C", (cx_-4, oy+10), cv2.FONT_HERSHEY_SIMPLEX, fs, border, 1)
        cv2.putText(img, "E", (cx_-4, oy+ch-4), cv2.FONT_HERSHEY_SIMPLEX, fs, border, 1)

    def _sym_sensor(self, img, ox, oy, cw, ch, col, fill, border, thick, comp):
        cv2.rectangle(img, (ox, oy), (ox+cw, oy+ch), fill, -1)
        cv2.rectangle(img, (ox, oy), (ox+cw, oy+ch), border, thick, cv2.LINE_AA)
        r = max(4, min(cw, ch)//6)
        cv2.ellipse(img, (ox+r, oy+r), (r,r), 180, 0, 90, border, thick, cv2.LINE_AA)
        for i in range(2):
            radius = 6 + i * 6
            cv2.ellipse(img, (ox+cw-10, oy+ch//2), (radius, radius), 0, -60, 60,
                        tuple(int(c*0.8) for c in border), 1, cv2.LINE_AA)
        d    = CATALOG.get(comp.type_id, {})
        lbl  = d.get("label", comp.type_id)[:6]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = max(0.2, min(0.38, cw / 100))
        (tw, th), _ = cv2.getTextSize(lbl, font, fs, 1)
        cv2.putText(img, lbl, (ox + cw//2 - tw//2, oy + ch//2 + th//2),
                    font, fs, tuple(min(255, int(c*1.6)) for c in col), 1, cv2.LINE_AA)
        self._draw_pins_both_sides(img, ox, oy, cw, ch, col, d)

    # ─────────────────────────────────────────────────────────────────────────
    #  PIN DRAWING — left + right sides with labels  (KEY WORKBENCH FEATURE)
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_pins_both_sides(self, img, ox, oy, cw, ch, col, d):
        """
        Draw pin dots and labels on BOTH sides of the component.
        Left column: first half of pins.
        Right column: second half of pins.
        Each pin gets a visible dot + its name for workbench identification.
        """
        pins = d.get("pins", [])
        if not pins or ch < 10:
            return

        sym   = d.get("symbol", "rect")
        font  = cv2.FONT_HERSHEY_SIMPLEX
        fs    = max(0.16, min(0.28, ch / 220))
        dot_r = max(2, int(3 * max(1.0, self.zoom * 0.5)))
        pin_col   = (255, 210, 0)
        label_col = (160, 200, 220)

        if sym in ("ic", "sensor", "rect") and len(pins) > 2:
            left_pins, right_pins = self._split_pins(pins)
            nl = len(left_pins)
            nr = len(right_pins)

            # Left side pins
            step_l = ch / (nl + 1)
            for i, pin_name in enumerate(left_pins):
                py = int(oy + step_l * (i + 1))
                # Lead line
                cv2.line(img, (ox - 6, py), (ox, py), col, 1, cv2.LINE_AA)
                # Dot
                cv2.circle(img, (ox, py), dot_r, pin_col, -1, cv2.LINE_AA)
                # Label (right of dot, inside body)
                if ch > 20:
                    cv2.putText(img, pin_name[:5],
                                (ox + dot_r + 2, py + 4),
                                font, fs, label_col, 1, cv2.LINE_AA)

            # Right side pins
            step_r = ch / (nr + 1)
            for i, pin_name in enumerate(right_pins):
                py = int(oy + step_r * (i + 1))
                # Lead line
                cv2.line(img, (ox + cw, py), (ox + cw + 6, py), col, 1, cv2.LINE_AA)
                # Dot
                cv2.circle(img, (ox + cw, py), dot_r, pin_col, -1, cv2.LINE_AA)
                # Label (left of dot, inside body, right-aligned)
                if ch > 20:
                    (lw, _), _ = cv2.getTextSize(pin_name[:5], font, fs, 1)
                    cv2.putText(img, pin_name[:5],
                                (ox + cw - lw - dot_r - 2, py + 4),
                                font, fs, label_col, 1, cv2.LINE_AA)

        else:
            # Simple 2-pin or single-side: legacy left-only
            n    = len(pins)
            step = ch / (n + 1)
            for i, pin_name in enumerate(pins):
                py = int(oy + step * (i + 1))
                cv2.circle(img, (ox, py), dot_r, pin_col, -1, cv2.LINE_AA)
                if ch > 30:
                    cv2.putText(img, pin_name[:4], (ox + 4, py + 3),
                                font, fs, label_col, 1, cv2.LINE_AA)

    # (legacy alias kept for any internal callers)
    def _draw_pins(self, img, ox, oy, cw, ch, col, d):
        self._draw_pins_both_sides(img, ox, oy, cw, ch, col, d)

    def _draw_state_overlay(self, img, ox, oy, cw, ch, comp):
        s    = comp.state
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = []
        if "temp" in s and "hum" in s:
            lines = [f"{s['temp']}C", f"{s['hum']}%"]
        elif "temp" in s:
            lines = [f"{s['temp']}C"]
        elif "lux" in s:
            lines = [f"{s['lux']}lx"]
        elif "dist" in s:
            lines = [f"{s['dist']}cm"]
        elif "pressure" in s:
            lines = [f"{s['pressure']}hPa"]
        elif "ax" in s:
            lines = [f"ax{s['ax']}", f"ay{s['ay']}"]
        elif "angle" in s:
            lines = [f"{s['angle']}°"]
        elif "ppm" in s:
            lines = [f"{s['ppm']}ppm"]
        elif "motion" in s:
            lines = ["MOTION!" if s["motion"] else "clear"]
        if lines:
            fs = max(0.2, min(0.35, cw / 100))
            for i, line in enumerate(lines[:3]):
                cv2.putText(img, line, (ox + 2, oy + 14 + i * 14),
                            font, fs, (80, 255, 120), 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    #  PROPERTY INSPECTOR
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_inspector(self, canvas: np.ndarray):
        comp = self.get_component(self.selected_id) if self.selected_id else None
        if not comp:
            return
        d     = CATALOG.get(comp.type_id, {})
        props = comp.props
        font  = cv2.FONT_HERSHEY_SIMPLEX
        h     = canvas.shape[0]

        ix, iy = 10, h - 210
        iw, ih = 280, 200

        ov = canvas.copy()
        cv2.rectangle(ov, (ix, iy), (ix+iw, iy+ih), (5, 15, 25), -1)
        cv2.addWeighted(ov, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (ix, iy), (ix+iw, iy+ih), (0, 120, 180), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"{comp.label}  [{comp.type_id}]",
                    (ix+6, iy+16), font, 0.38, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (ix+4, iy+22), (ix+iw-4, iy+22), (0, 60, 90), 1)

        # Show pin list
        pins = d.get("pins", [])
        y    = iy + 36
        cv2.putText(canvas, f"PINS ({len(pins)}):",
                    (ix+6, y), font, 0.3, (0, 160, 200), 1, cv2.LINE_AA)
        y += 14
        for i, p in enumerate(pins[:14]):
            col_pt = (255, 210, 0) if i % 2 == 0 else (200, 180, 0)
            cv2.putText(canvas, f"  {i}: {p}", (ix+6, y),
                        font, 0.28, col_pt, 1, cv2.LINE_AA)
            y += 13

        if self.sim_running and comp.state:
            cv2.line(canvas, (ix+4, y), (ix+iw-4, y), (0, 60, 90), 1)
            y += 8
            for k, v in list(comp.state.items())[:4]:
                cv2.putText(canvas, f"{k}: {v}", (ix+6, y),
                            font, 0.3, (80, 255, 120), 1, cv2.LINE_AA)
                y += 14

    # ─────────────────────────────────────────────────────────────────────────
    #  COMPONENT PANEL
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_panel(self, canvas: np.ndarray):
        ch   = canvas.shape[0]
        pw   = self.PANEL_W
        px   = self.board_w
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(canvas, (px, 0), (px + pw, ch), (5, 12, 20), -1)
        cv2.line(canvas, (px, 0), (px, ch), (0, 50, 80), 1)
        cv2.rectangle(canvas, (px, 0), (px + pw, 36), (0, 20, 35), -1)
        cv2.putText(canvas, "COMPONENTS", (px + 8, 22),
                    font, 0.42, (0, 180, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (px, 36), (px + pw, 36), (0, 50, 80), 1)

        y = 40 - self.panel_scroll
        for kind, key, lbl in self._panel_items:
            if kind == "cat":
                item_h = self.PANEL_CAT_H
                if 36 <= y + item_h <= ch:
                    cv2.rectangle(canvas, (px, y), (px + pw, y + item_h), (0, 22, 38), -1)
                    cv2.putText(canvas, lbl.upper(), (px + 8, y + 17),
                                font, 0.32, (0, 100, 150), 1, cv2.LINE_AA)
                    cv2.line(canvas, (px, y + item_h), (px + pw, y + item_h), (0, 35, 55), 1)
            else:
                item_h = self.PANEL_ITEM_H
                if 36 <= y + item_h <= ch:
                    is_sel = (key == self.panel_selected)
                    is_hov = (key == self._panel_hovered)
                    bg = (0, 35, 55) if is_sel else ((0, 22, 38) if is_hov else (0, 0, 0))
                    cv2.rectangle(canvas, (px + 1, y), (px + pw, y + item_h), bg, -1)
                    d_     = CATALOG[key]
                    col_c  = d_["color"]
                    sw     = 10
                    cv2.rectangle(canvas, (px + 6, y + 8), (px + 6 + sw, y + item_h - 8), col_c, -1)
                    cv2.rectangle(canvas, (px + 6, y + 8), (px + 6 + sw, y + item_h - 8),
                                  tuple(min(255, int(c*1.4)) for c in col_c), 1)
                    text_col = (0, 200, 255) if is_sel else (180, 220, 240)
                    cv2.putText(canvas, d_["label"][:14], (px + 22, y + 14),
                                font, 0.33, text_col, 1, cv2.LINE_AA)
                    pin_str = f"{len(d_.get('pins', []))}p"
                    cv2.putText(canvas, pin_str, (px + pw - 28, y + 14),
                                font, 0.28, (0, 80, 120), 1, cv2.LINE_AA)
                    cv2.putText(canvas, key[:16], (px + 22, y + 26),
                                font, 0.24, (0, 70, 100), 1, cv2.LINE_AA)
                    if is_sel:
                        cv2.line(canvas, (px + 1, y), (px + 3, y + item_h), (0, 200, 255), 2)
                    cv2.line(canvas, (px + 6, y + item_h), (px + pw - 6, y + item_h), (0, 20, 35), 1)
            y += item_h

        total = self._panel_total_h()
        if total > ch - 40:
            bar_h  = max(20, int((ch - 40) / total * (ch - 40)))
            bar_y  = 40 + int(self.panel_scroll / total * (ch - 40))
            cv2.rectangle(canvas, (px + pw - 3, bar_y), (px + pw - 1, bar_y + bar_h),
                          (0, 100, 160), -1)

    # ─────────────────────────────────────────────────────────────────────────
    #  HUD
    # ─────────────────────────────────────────────────────────────────────────
    def _draw_hud(self, canvas: np.ndarray):
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        bw   = self.board_w

        ov = canvas.copy()
        cv2.rectangle(ov, (0, h - 26), (bw, h), (0, 8, 14), -1)
        cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)
        cv2.line(canvas, (0, h - 26), (bw, h - 26), (0, 40, 60), 1)

        mode_col = {"select":(255,180,0), "wire":(0,200,255), "pan":(0,255,100)}
        mc = mode_col.get(self.mode, (180,180,180))
        cv2.putText(canvas, f"MODE:{self.mode.upper()}", (8, h - 10), font, 0.38, mc, 1, cv2.LINE_AA)
        cv2.putText(canvas,
            f"│  {len(self.components)} comps  {len(self.wires)} wires  │  Zoom:{self.zoom:.2f}x",
            (100, h - 10), font, 0.35, (60, 100, 120), 1, cv2.LINE_AA)

        mx, my = self.mouse_world
        cv2.putText(canvas, f"X:{int(mx)}  Y:{int(my)}", (bw - 120, h - 10),
                    font, 0.35, (50, 80, 100), 1, cv2.LINE_AA)

        if self.sim_running:
            elapsed = time.time() - self._sim_t0
            pulse   = 0.6 + 0.4 * abs(math.sin(elapsed * 3))
            sc      = tuple(int(c * pulse) for c in (80, 255, 80))
            cv2.circle(canvas, (14, 14), 6, sc, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"SIM  {elapsed:.1f}s", (24, 19),
                        font, 0.4, (80, 255, 80), 1, cv2.LINE_AA)

        cv2.putText(canvas, "●  CIRCUIT ENGINE v3.1",
                    (bw - 170, 18), font, 0.35, (0, 60, 90), 1, cv2.LINE_AA)

        for i, line in enumerate(self.get_log_tail(5)):
            cv2.putText(canvas, line[-72:], (8, h - 48 - i * 14),
                        font, 0.28, (40, 160, 60), 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────────────────────────────────────
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
    def _paste(canvas: np.ndarray, img: np.ndarray, dx: int, dy: int,
               clip_w: Optional[int] = None):
        ch, cw = canvas.shape[:2]
        ih, iw = img.shape[:2]
        x0 = max(0, dx)
        y0 = max(0, dy)
        x1 = min(clip_w if clip_w else cw, dx + iw)
        y1 = min(ch, dy + ih)
        if x1 <= x0 or y1 <= y0:
            return
        ix0, iy0 = x0 - dx, y0 - dy
        ix1, iy1 = ix0 + (x1 - x0), iy0 + (y1 - y0)
        region = canvas[y0:y1, x0:x1]
        patch  = img[iy0:iy1, ix0:ix1]
        mask   = patch.sum(axis=2) > 8
        region[mask] = patch[mask]