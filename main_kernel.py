import cv2
import numpy as np
import threading
import textwrap
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import ImageFont, ImageDraw, Image

# --- IMPORT OUR CUSTOM MODULES ---
from modules.vision_engine import scan_object
from modules.voice_engine import (
    listen_and_process_command, 
    WORKING_MIC_INDEX, WORKING_MIC_NAME, 
    SPEAKER_NAME, SPEAKER_INDEX,
    get_hardware_info
)
from modules.ui_hub import UIHub
from modules.circuit_engine import CircuitEngine

# --- FONT CONFIGURATION (FILES IN FONTS FOLDER) ---
FONT_BOLD = "fonts/arialbd.ttf" 
FONT_REGULAR = "fonts/consola.ttf"

# --- HELPER: MODERN TEXT RENDERING ---
def draw_modern_text(img, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    # PIL uses RGB, cv2 uses BGR
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_modern_paragraph(img, text, position, font_path, font_size, color, max_width):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    words = text.split(' ')
    lines = []
    current_line = []
    for word in words:
        test_line = ' '.join(current_line + [word])
        if draw.textlength(test_line, font=font) <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))

    x, y = position
    for i, line in enumerate(lines):
        draw.text((x, y + (i * (font_size + 5))), line, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 1. LOAD ASSETS ---
layer1_img = cv2.imread("database/nothing_1.png")
layer2_img = cv2.imread("database/nothing_2.png")
battery_img = cv2.imread("database/nothing_bat.png")
settings_btn_img = cv2.imread("database/settings_button.png") 

if layer1_img is not None: layer1_img = cv2.resize(layer1_img, (250, 500))
if layer2_img is not None: layer2_img = cv2.resize(layer2_img, (250, 500))
if battery_img is not None: battery_img = cv2.resize(battery_img, (200, 300))
if settings_btn_img is not None: settings_btn_img = cv2.resize(settings_btn_img, (40, 40))

# --- 2. GLOBAL STATE ---
app_state = {
    'active_category': None,
    'active_model': None,
    'current_layer_view': 1,
    'is_listening': False,
    'voice_feedback': "",
    'dynamic_ar_text': "",
    'selected_tool': 'resistor', 
    'is_pinching': False,
    'settings_open': False,
    'camera_index': 0, # Usually 0 for primary laptop camera
    'mic_index': WORKING_MIC_INDEX,
    'mic_name': WORKING_MIC_NAME,
    'speaker_index': SPEAKER_INDEX,
    'speaker_name': SPEAKER_NAME
}

hub = UIHub()
circuit_engine = CircuitEngine()
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 150 and 5 <= y <= 45:
            app_state['settings_open'] = not app_state['settings_open']

cap = cv2.VideoCapture(app_state['camera_index'])
window_name = "AIILA_INTEGRATED_OS"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1400, 700) 
cv2.setMouseCallback(window_name, mouse_handler)

prev_finger_x = 0.0
swipe_cooldown = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Create Base Frame
    projector_canvas = hub.create_base_frame()
    debug_h = 300 
    debug_w = int(300 * (frame.shape[1]/frame.shape[0]))
    debug_small = cv2.resize(frame, (debug_w, debug_h))

    # --- MEDIAPIPE LOGIC ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        index_tip, thumb_tip = hand[8], hand[4]
        if prev_finger_x != 0.0 and swipe_cooldown == 0:
            if abs(index_tip.x - prev_finger_x) > 0.15: 
                app_state['current_layer_view'] = 2 if app_state['current_layer_view'] == 1 else 1
                swipe_cooldown = 20 
        prev_finger_x = index_tip.x
        cv2.circle(debug_small, (int(index_tip.x * debug_w), int(index_tip.y * debug_h)), 10, (255, 0, 0), -1)
        cursor_x, cursor_y = int(index_tip.x * 1000), int(index_tip.y * 700)
        cv2.circle(projector_canvas, (cursor_x, cursor_y), 10, (255, 191, 0), 2)

        dist = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
        if dist < 0.05:
            if not app_state['is_pinching']:
                circuit_engine.add_component(app_state['selected_tool'], cursor_x, cursor_y)
                app_state['is_pinching'] = True
        else:
            app_state['is_pinching'] = False

    if swipe_cooldown > 0: swipe_cooldown -= 1

    # 3. Draw AR blueprint components
    circuit_engine.draw_components(projector_canvas)

    if app_state['active_model']:
        # MODERN UNIT TITLE
        projector_canvas = draw_modern_text(projector_canvas, f"UNIT: {app_state['active_model'].upper()}", 
                                            (350, 75), FONT_BOLD, 28, (255, 191, 0))
        
        if app_state['active_category'] == "smartphone":
            x_off, y_off = 250, 120 
            if app_state['current_layer_view'] == 1 and layer1_img is not None:
                projector_canvas[y_off:y_off+500, x_off:x_off+250] = layer1_img
            elif app_state['current_layer_view'] == 2 and layer2_img is not None:
                projector_canvas[y_off:y_off+500, x_off:x_off+250] = layer2_img
                if battery_img is not None: projector_canvas[200:500, 520:720] = battery_img

        # MODERN AI WRITING AREA
        if app_state['dynamic_ar_text'] != "":
            projector_canvas = draw_modern_paragraph(projector_canvas, app_state['dynamic_ar_text'], 
                                                     (550, 150), FONT_REGULAR, 18, (0, 255, 0), 400)
        else:
            projector_canvas = draw_modern_text(projector_canvas, "Waiting for Jarvis Analysis...", 
                                                (550, 150), FONT_REGULAR, 16, (0, 255, 0))

    # --- BUTTONS AND HUD ---
    cv2.rectangle(projector_canvas, (15, 10), (145, 40), (45, 45, 45), -1)
    cv2.rectangle(projector_canvas, (15, 10), (145, 40), (0, 255, 0), 1)
    projector_canvas = draw_modern_text(projector_canvas, "SETTINGS", (35, 18), FONT_BOLD, 14, (0, 255, 0))

    projector_canvas = draw_modern_text(projector_canvas, f"TOOL: {app_state['selected_tool'].upper()}", 
                                        (20, 125), FONT_BOLD, 18, (0, 255, 0))
    
    projector_canvas = hub.update_status(projector_canvas, app_state['voice_feedback'] or "SYSTEM IDLE")

    # --- SETTINGS OVERLAY ---
    if app_state['settings_open']:
        overlay = projector_canvas.copy()
        cv2.rectangle(overlay, (150, 80), (850, 620), (35, 35, 35), -1)
        cv2.addWeighted(overlay, 0.85, projector_canvas, 0.15, 0, projector_canvas)
        
        projector_canvas = draw_modern_text(projector_canvas, "HARDWARE CONTROL CENTER", 
                                            (250, 140), FONT_BOLD, 24, (255, 191, 0))
        
        projector_canvas = draw_modern_text(projector_canvas, f"MIC (M): {app_state['mic_name']} ({app_state['mic_index']})", 
                                            (180, 240), FONT_REGULAR, 18, (255, 255, 255))
        
        projector_canvas = draw_modern_text(projector_canvas, f"SPEAKER: {app_state['speaker_name']}", 
                                            (180, 310), FONT_REGULAR, 18, (255, 255, 255))

        projector_canvas = draw_modern_text(projector_canvas, f"CAMERA (C): Integrated ({app_state['camera_index']})", 
                                            (180, 380), FONT_REGULAR, 18, (255, 255, 255))

        projector_canvas = draw_modern_text(projector_canvas, "Controls: [C] Cam | [M] Change Mic | [ESC] Close", 
                                            (220, 580), FONT_REGULAR, 16, (0, 255, 0))

    # --- DISPLAY FINAL FRAME ---
    right_sidebar = np.zeros((700, debug_w+20, 3), dtype=np.uint8)
    right_sidebar[:] = (25, 25, 25) 
    right_sidebar[200:500, 10:10+debug_w] = debug_small
    cv2.imshow(window_name, np.hstack((projector_canvas, right_sidebar)))

    # --- KEYBOARD LOGIC ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == 27: # ESC Key
        app_state['settings_open'] = False
    
    # Logic while Settings is Open
    if app_state['settings_open']:
        if key == ord('c'):
            app_state['camera_index'] = (app_state['camera_index'] + 1) % 3
            cap.release()
            cap = cv2.VideoCapture(app_state['camera_index'])
            app_state['voice_feedback'] = f"CAMERA SWITCHED TO {app_state['camera_index']}"
        elif key == ord('m'):
            mics, _ = get_hardware_info()
            if mics:
                app_state['mic_index'] = (app_state['mic_index'] + 1) % len(mics)
                app_state['mic_name'] = mics[app_state['mic_index']]
                app_state['voice_feedback'] = f"MIC SWITCHED: {app_state['mic_name']}"
    
    # Logic while Settings is Closed
    else:
        if key == ord('r'): 
            circuit_engine.clear_components()
            app_state['voice_feedback'] = "Marks Cleared"
        elif key == ord('s'):
            app_state['voice_feedback'] = "SCANNING..."
            cv2.waitKey(1)
            app_state['active_category'], app_state['active_model'] = scan_object(frame)
            app_state['voice_feedback'] = f"DETECTED: {app_state['active_model']}"
        elif key == ord('v'):
            if not app_state['is_listening'] and app_state['active_model']:
                app_state['is_listening'] = True
                threading.Thread(target=listen_and_process_command, args=(app_state,)).start()

cap.release()
cv2.destroyAllWindows()