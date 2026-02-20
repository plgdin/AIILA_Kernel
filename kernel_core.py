import cv2
import numpy as np
import os
import threading
import textwrap
import speech_recognition as sr
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. SETUP GOOGLE GEMINI API ---
from google import genai
client = genai.Client(api_key="AIzaSyB1f6Pe348ThuMGf4fOccBOUMdHPD4yQ1Y")

# --- 2. LOAD LOCAL DATABASE IMAGES ---
print("Loading Database Images...")
layer1_img = cv2.imread("database/nothing_1.png")
layer2_img = cv2.imread("database/nothing_2.png")
battery_img = cv2.imread("database/nothing_bat.png")

if layer1_img is not None: layer1_img = cv2.resize(layer1_img, (250, 500))
if layer2_img is not None: layer2_img = cv2.resize(layer2_img, (250, 500))
if battery_img is not None: battery_img = cv2.resize(battery_img, (200, 300))

# --- STATE VARIABLES ---
active_category = None       # Holds the broad type (e.g., "smartphone" or "calculator")
active_model = None          # Holds the exact name (e.g., "Nothing Phone 3a Pro")
current_layer_view = 1       
is_listening = False         
voice_feedback = ""          
dynamic_ar_text = ""         # Holds Gemini's answers to your questions!

# --- 3. AUTO MIC HUNTER ---
WORKING_MIC_INDEX = None
print("\n--- Initializing Audio System ---")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    try:
        m = sr.Microphone(device_index=index)
        with m as source:
            if m.stream is not None:
                print(f"SUCCESS: Audio channel unlocked at index {index} ({name})")
                WORKING_MIC_INDEX = index
                break
    except Exception:
        pass

# --- 4. INITIALIZE MEDIAPIPE (HAND TRACKING) ---
print("\n--- Initializing Gesture Engine ---")
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Gesture tracking variables
prev_finger_x = 0.0          
swipe_cooldown = 0           

# --- AR TEXT ENGINE (Word Wrapping for OpenCV) ---
def draw_ar_paragraph(img, text, position, font, font_scale, color, thickness, max_width_pixels):
    """Draws multi-line text dynamically so it doesn't fall off the screen."""
    x, y0 = position
    char_width = 15 if font_scale == 0.6 else 20
    max_chars_per_line = max(10, int(max_width_pixels / char_width))
    
    wrapped_lines = textwrap.wrap(text, width=max_chars_per_line)
    
    for i, line in enumerate(wrapped_lines):
        y = y0 + i * 30 # 30 pixels spacing between lines
        cv2.putText(img, line, (x, y), font, font_scale, color, thickness)

# --- THE AUDIO/VOICE ENGINE ---
def listen_and_process_command():
    global current_layer_view, is_listening, voice_feedback, dynamic_ar_text
    
    if WORKING_MIC_INDEX is None:
        voice_feedback = "Error: No Mic Found!"
        is_listening = False
        return

    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone(device_index=WORKING_MIC_INDEX) as source:
            voice_feedback = "Listening..."
            print("\nMicrophone active. Speak now...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                voice_feedback = "Processing speech..."
                user_text = recognizer.recognize_google(audio)
                print(f"You said: '{user_text}'")
                
                # --- THE DYNAMIC GEMINI PROMPT ---
                prompt = f"""
                You are JARVIS, an AR engineering assistant. The user is looking at a disassembled {active_model}. 
                The user asked: "{user_text}". 
                
                RULES:
                1. If they ask to see the battery or motherboard, start your response with exactly: [LAYER2]
                2. If they ask to see the front or first layer, start your response with exactly: [LAYER1]
                3. Answer their question concisely in 1 or 2 short sentences. Ensure your technical facts match the exact specs of the {active_model}.
                """
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                answer = response.text.strip()
                
                # Check if Gemini wants to change the layers via Voice
                if "[LAYER2]" in answer:
                    current_layer_view = 2
                    answer = answer.replace("[LAYER2]", "").strip()
                elif "[LAYER1]" in answer:
                    current_layer_view = 1
                    answer = answer.replace("[LAYER1]", "").strip()
                
                # Update the AR Text that gets projected to the table
                dynamic_ar_text = answer
                voice_feedback = "Response generated."
                print(f"JARVIS answered: {dynamic_ar_text}")
                    
            except sr.WaitTimeoutError:
                voice_feedback = "Error: Nobody spoke."
            except sr.UnknownValueError:
                voice_feedback = "Error: Could not understand audio."
                
    except Exception as e:
        print(f"\n[MIC ERROR]: {e}")
        voice_feedback = "Error: Mic disconnected."
            
    is_listening = False

# --- THE VISION SCANNER ---
def scan_object(frame):
    print("Scanning with Gemini Vision Core...")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    try:
        # The "Pipe Split" Prompt for exact model ID
        prompt = """
        Identify the object in the image. I need the broad category AND the exact specific model name.
        Reply strictly in this format: category | exact model name
        Valid categories are: 'smartphone', 'calculator', or 'unknown'. 
        Example 1: smartphone | Nothing Phone 3a Pro
        Example 2: calculator | Casio fx-991EX
        Example 3: unknown | Sony WH-1000XM4 Headphones
        Do not add any other text, markdown, or formatting.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pil_img]
        )
        ans = response.text.strip().replace("```", "").replace("\n", "")
        
        if "|" in ans:
            category, model = ans.split("|", 1)
            print(f"AI Identified Category: {category.strip()}")
            print(f"AI Identified Exact Model: {model.strip()}")
            return category.strip().lower(), model.strip().title()
        else:
            print(f"AI Raw Answer: {ans}")
            return "unknown", ans.strip()
            
    except Exception as e:
        print(f"API Error: {e}")
        return "error", "error"

# --- 5. CAMERA SETUP ---
print("\nHunting for camera...")
cap = cv2.VideoCapture(1)

# --- 6. WINDOW SETUP ---
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
    # Convert frame to format Mediapipe needs
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        # Get the X coordinate of the Index Finger Tip (Landmark 8)
        finger_x = result.hand_landmarks[0][8].x
        
        # Calculate how far the finger moved since the last frame
        if prev_finger_x != 0.0 and swipe_cooldown == 0:
            delta_x = finger_x - prev_finger_x
            
            # SWIPE TRIGGER: If finger jumps more than 15% of the screen horizontally
            if abs(delta_x) > 0.15: 
                # Toggle between Layer 1 and Layer 2 using HANDS
                current_layer_view = 2 if current_layer_view == 1 else 1
                print(f"SWIPE DETECTED! Switched to Layer {current_layer_view}")
                
                # Cooldown so it doesn't trigger repeatedly in one wave
                swipe_cooldown = 20 
                
        prev_finger_x = finger_x
        
        # Draw a blue tracking dot on the Laptop Debug screen so you can see it working
        cx, cy = int(finger_x * w), int(result.hand_landmarks[0][8].y * h)
        cv2.circle(debug_canvas, (cx, cy), 15, (255, 0, 0), -1)
    else:
        # Reset the swipe tracker if hand leaves the frame
        prev_finger_x = 0.0 

    # Decrease cooldown timer
    if swipe_cooldown > 0:
        swipe_cooldown -= 1

    # --- AR PROJECTION DRAWING ---
    if active_category == "smartphone":
        
        # 1. Draw the Layers
        if current_layer_view == 1 and layer1_img is not None:
            x_offset = 50
            y_offset = 100
            projector_canvas[y_offset:y_offset+500, x_offset:x_offset+250] = layer1_img
            cv2.putText(projector_canvas, f"{active_model} - Layer 1", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        elif current_layer_view == 2 and layer2_img is not None and battery_img is not None:
            x_offset = 50
            y_offset = 100
            projector_canvas[y_offset:y_offset+500, x_offset:x_offset+250] = layer2_img
            cv2.putText(projector_canvas, "Layer 2 (Internal)", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            x_offset_bat = 320
            y_offset_bat = 200
            projector_canvas[y_offset_bat:y_offset_bat+300, x_offset_bat:x_offset_bat+200] = battery_img
            cv2.putText(projector_canvas, "Battery Details", (x_offset_bat, y_offset_bat - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
    elif active_category == "unknown":
        pass # It just skips drawing the blueprints

    # 2. Draw the Dynamic AR Text on the right side of the screen
    if dynamic_ar_text != "":
        draw_ar_paragraph(
            img=projector_canvas, 
            text=dynamic_ar_text, 
            position=(550, 100), 
            font=cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale=0.7, 
            color=(0, 255, 0), 
            thickness=2, 
            max_width_pixels=400
        )

    # 3. UI Voice Status
    if active_category is not None:
        cv2.putText(projector_canvas, "Press 'v' to speak a command or question", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        if voice_feedback != "":
            cv2.putText(projector_canvas, f"Status: {voice_feedback}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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
        
        # Unpack both the category and the specific model!
        active_category, active_model = scan_object(frame)
        
        if active_category != "error":
            dynamic_ar_text = f"Identified: {active_model.upper()}. How can I assist?"
        else:
            dynamic_ar_text = "Scan failed. Please try again."
            active_category = None
        
    elif key == ord('v'):
        if not is_listening and active_category is not None:
            is_listening = True
            threading.Thread(target=listen_and_process_command).start()

cap.release()
cv2.destroyAllWindows()