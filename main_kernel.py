import cv2
import numpy as np
import threading
import textwrap
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- IMPORT OUR CUSTOM MODULES ---
from vision_engine import scan_object
from voice_engine import listen_and_process_command

# --- [NEW] UI & CIRCUIT CLASSES ---
class UIHub:
    def __init__(self):
        self.accent_color = (255, 191, 0) # Cyber Blue/Orange
        
    def draw_hud_base(self, canvas):
        # Semi-transparent Sidebar
        cv2.rectangle(canvas, (0, 0), (220, 700), (25, 25, 25), -1)
        cv2.line(canvas, (220, 0), (220, 700), (50, 50, 50), 1)
        # Bottom Console Area
        cv2.rectangle(canvas, (220, 600), (1000, 700), (15, 15, 15), -1)
        cv2.line(canvas, (220, 600), (1000, 600), (50, 50, 50), 1)
        cv2.putText(canvas, "AIILA HOLOMAT v2.0", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.accent_color, 2)

class CircuitModule:
    def __init__(self):
        self.components = [] # List of {type, pos}
        
    def add_comp(self, c_type, pos):
        self.components.append({'type': c_type, 'pos': pos})
        
    def render(self, canvas):
        for comp in self.components:
            x, y = comp['pos']
            # Draw holographic glow for components
            cv2.circle(canvas, (x, y), 12, (0, 255, 0), 2)
            cv2.putText(canvas, comp['type'].upper(), (x+15, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# --- 1. LOAD LOCAL DATABASE IMAGES ---
print("Loading Database Images...")
layer1_img = cv2.imread("database/nothing_1.png")
layer2_img = cv2.imread("database/nothing_2.png")
battery_img = cv2.imread("database/nothing_bat.png")

if layer1_img is not None: layer1_img = cv2.resize(layer1_img, (250, 500))
if layer2_img is not None: layer2_img = cv2.resize(layer2_img, (250, 500))
if battery_img is not None: battery_img = cv2.resize(battery_img, (200, 300))

# --- 2. GLOBAL STATE DICTIONARY ---
app_state = {
    'active_category': None,
    'active_model': None,
    'current_layer_view': 1,
    'is_listening': False,
    'voice_feedback': "",
    'dynamic_ar_text': "",
    'selected_tool': 'resistor', # Default tool for circuit builder
    'is_pinching': False
}

# --- 3. INITIALIZE MEDIAPIPE ---
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Initialize new modules
hub = UIHub()
circuit_engine = CircuitModule()

prev_finger_x = 0.0          
swipe_cooldown = 0           

# --- AR TEXT ENGINE ---
def draw_ar_paragraph(img, text, position, font, font_scale, color, thickness, max_width_pixels):
    x, y0 = position
    char_width = 15 if font_scale == 0.6 else 20
    max_chars_per_line = max(10, int(max_width_pixels / char_width))
    wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
    for i, line in enumerate(wrapped_lines):
        y = y0 + i * 30 
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness)

# --- 4. CAMERA SETUP ---
cap = cv2.VideoCapture(1)
window_name = "AIILA_Projector_OS"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1000, 700) 

# --- THE MAIN KERNEL LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: break

    debug_canvas = frame.copy()
    h, w, _ = frame.shape
    
    # Create the Hub Canvas
    projector_canvas = np.zeros((700, 1000, 3), dtype=np.uint8)
    hub.draw_hud_base(projector_canvas)

    # --- MEDIAPIPE LOGIC ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        # 1. Existing Swipe Logic
        finger_x = result.hand_landmarks[0][8].x
        if prev_finger_x != 0.0 and swipe_cooldown == 0:
            delta_x = finger_x - prev_finger_x
            if abs(delta_x) > 0.15: 
                app_state['current_layer_view'] = 2 if app_state['current_layer_view'] == 1 else 1
                swipe_cooldown = 20 
        prev_finger_x = finger_x
        
        # 2. [NEW] Circuit Placement Logic (Pinch Gesture)
        index_tip = result.hand_landmarks[0][8]
        thumb_tip = result.hand_landmarks[0][4]
        dist = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
        
        # Map hand coords to projector canvas
        cursor_x, cursor_y = int(index_tip.x * 1000), int(index_tip.y * 700)
        cv2.circle(projector_canvas, (cursor_x, cursor_y), 10, (255, 191, 0), 2) # Cursor

        if dist < 0.05: # Pinch detected
            if not app_state['is_pinching']:
                circuit_engine.add_comp(app_state['selected_tool'], (cursor_x, cursor_y))
                app_state['is_pinching'] = True
        else:
            app_state['is_pinching'] = False

    if swipe_cooldown > 0: swipe_cooldown -= 1

    # --- RENDER CIRCUIT MODULE ---
    circuit_engine.render(projector_canvas)

    # --- EXISTING AR PROJECTION DRAWING ---
    if app_state['active_category'] == "smartphone":
        x_offset, y_offset = 250, 100 # Adjusted for new Sidebar
        if app_state['current_layer_view'] == 1 and layer1_img is not None:
            projector_canvas[y_offset:y_offset+500, x_offset:x_offset+250] = layer1_img
            cv2.putText(projector_canvas, f"{app_state['active_model']} - Layer 1", (x_offset, y_offset - 10), 2, 0.6, (255, 255, 255), 2)
        elif app_state['current_layer_view'] == 2 and layer2_img is not None:
            projector_canvas[y_offset:y_offset+500, x_offset:x_offset+250] = layer2_img
            if battery_img is not None:
                projector_canvas[200:500, 520:720] = battery_img

    # --- UI COMPONENTS ---
    # Sidebar Info
    cv2.putText(projector_canvas, "TOOLBOX", (20, 100), 2, 0.5, (150, 150, 150), 1)
    cv2.putText(projector_canvas, f"> {app_state['selected_tool'].upper()}", (20, 130), 2, 0.6, (0, 255, 0), 2)
    cv2.putText(projector_canvas, "1: Resistor  2: LED", (20, 160), 2, 0.4, (100, 100, 100), 1)

    # Console Logs
    log_text = app_state['voice_feedback'] if app_state['voice_feedback'] else "SYSTEM IDLE"
    cv2.putText(projector_canvas, f"TERMINAL: {log_text}", (240, 640), 2, 0.5, (0, 255, 255), 1)
    
    if app_state['dynamic_ar_text'] != "":
        draw_ar_paragraph(projector_canvas, app_state['dynamic_ar_text'], (550, 100), 2, 0.7, (0, 255, 0), 2, 400)

    cv2.imshow("Laptop Debug", debug_canvas)
    cv2.imshow(window_name, projector_canvas)

    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'):
        app_state['active_category'], app_state['active_model'] = scan_object(frame)
    elif key == ord('v'):
        if not app_state['is_listening'] and app_state['active_category'] is not None:
            app_state['is_listening'] = True
            threading.Thread(target=listen_and_process_command, args=(app_state,)).start()
    # Tool Selection
    elif key == ord('1'): app_state['selected_tool'] = 'resistor'
    elif key == ord('2'): app_state['selected_tool'] = 'led'

cap.release()
cv2.destroyAllWindows()