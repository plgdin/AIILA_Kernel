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

# --- 1. LOAD LOCAL DATABASE IMAGES ---
print("Loading Database Images...")
layer1_img = cv2.imread("database/nothing_1.png")
layer2_img = cv2.imread("database/nothing_2.png")
battery_img = cv2.imread("database/nothing_bat.png")

if layer1_img is not None: layer1_img = cv2.resize(layer1_img, (250, 500))
if layer2_img is not None: layer2_img = cv2.resize(layer2_img, (250, 500))
if battery_img is not None: battery_img = cv2.resize(battery_img, (200, 300))

# --- 2. GLOBAL STATE DICTIONARY ---
# We store our variables in a dictionary so we can pass them into our voice thread easily
app_state = {
    'active_category': None,
    'active_model': None,
    'current_layer_view': 1,
    'is_listening': False,
    'voice_feedback': "",
    'dynamic_ar_text': ""
}

# --- 3. INITIALIZE MEDIAPIPE (HAND TRACKING) ---
print("\n--- Initializing Gesture Engine ---")
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

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
print("\nHunting for camera...")
cap = cv2.VideoCapture(1)

window_name = "AIILA_Projector_OS"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1000, 700) 

print("System Active. Place object on mat and press 's' to scan.")

# --- THE MAIN KERNEL LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: break

    debug_canvas = frame.copy()
    h, w, _ = frame.shape
    
    proj_h, proj_w = 700, 1000
    projector_canvas = np.zeros((proj_h, proj_w, 3), dtype=np.uint8)

    # --- MEDIAPIPE SWIPE LOGIC ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        finger_x = result.hand_landmarks[0][8].x
        if prev_finger_x != 0.0 and swipe_cooldown == 0:
            delta_x = finger_x - prev_finger_x
            if abs(delta_x) > 0.15: 
                app_state['current_layer_view'] = 2 if app_state['current_layer_view'] == 1 else 1
                print(f"SWIPE DETECTED! Switched to Layer {app_state['current_layer_view']}")
                swipe_cooldown = 20 
                
        prev_finger_x = finger_x
        cx, cy = int(finger_x * w), int(result.hand_landmarks[0][8].y * h)
        cv2.circle(debug_canvas, (cx, cy), 15, (255, 0, 0), -1)
    else:
        prev_finger_x = 0.0 

    if swipe_cooldown > 0:
        swipe_cooldown -= 1

    # --- AR PROJECTION DRAWING ---
    if app_state['active_category'] == "smartphone":
        
        # 1. Draw the Layers
        if app_state['current_layer_view'] == 1 and layer1_img is not None:
            x_offset, y_offset = 50, 100
            projector_canvas[y_offset:y_offset+500, x_offset:x_offset+250] = layer1_img
            cv2.putText(projector_canvas, f"{app_state['active_model']} - Layer 1", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        elif app_state['current_layer_view'] == 2 and layer2_img is not None and battery_img is not None:
            x_offset, y_offset = 50, 100
            projector_canvas[y_offset:y_offset+500, x_offset:x_offset+250] = layer2_img
            cv2.putText(projector_canvas, "Layer 2 (Internal)", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            x_offset_bat, y_offset_bat = 320, 200
            projector_canvas[y_offset_bat:y_offset_bat+300, x_offset_bat:x_offset_bat+200] = battery_img
            cv2.putText(projector_canvas, "Battery Details", (x_offset_bat, y_offset_bat - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
    elif app_state['active_category'] == "unknown":
        pass 

    # 2. Draw the Dynamic AR Text
    if app_state['dynamic_ar_text'] != "":
        draw_ar_paragraph(
            img=projector_canvas, 
            text=app_state['dynamic_ar_text'], 
            position=(550, 100), 
            font=cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale=0.7, 
            color=(0, 255, 0), 
            thickness=2, 
            max_width_pixels=400
        )

    # 3. UI Voice Status
    if app_state['active_category'] is not None:
        cv2.putText(projector_canvas, "Press 'v' to speak a command or question", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if app_state['voice_feedback'] != "":
            cv2.putText(projector_canvas, f"Status: {app_state['voice_feedback']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(projector_canvas, "Awaiting Object. Press 's' to scan.", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Laptop Debug", debug_canvas)
    cv2.imshow(window_name, projector_canvas)

    # --- KEYBOARD CONTROLS ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('s'):
        cv2.putText(projector_canvas, "SCANNING...", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.imshow(window_name, projector_canvas)
        cv2.waitKey(1)
        
        # Unpack both the category and the specific model into our state dictionary!
        app_state['active_category'], app_state['active_model'] = scan_object(frame)
        
        if app_state['active_category'] != "error":
            app_state['dynamic_ar_text'] = f"Identified: {app_state['active_model'].upper()}. How can I assist?"
        else:
            app_state['dynamic_ar_text'] = "Scan failed. Please try again."
            app_state['active_category'] = None
        
    elif key == ord('v'):
        if not app_state['is_listening'] and app_state['active_category'] is not None:
            app_state['is_listening'] = True
            # Pass our dictionary to the thread so it can update our UI!
            threading.Thread(target=listen_and_process_command, args=(app_state,)).start()

cap.release()
cv2.destroyAllWindows()