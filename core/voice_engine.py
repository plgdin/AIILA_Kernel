import os
import speech_recognition as sr
from google import genai
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
# ElevenLabs key kept for future integration, though not used in this snippet
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

# Setup Gemini API (Gemini 2.0 Flash)
client = genai.Client(api_key=GEMINI_KEY)

def get_hardware_info():
    """Returns available audio hardware. Fallback to sounddevice if sr fails."""
    try:
        mics = sr.Microphone.list_microphone_names()
    except Exception:
        # Fallback for Python 3.14 compatibility issues with PyAudio
        mics = [d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0]
    
    speakers = sd.query_devices()
    return mics, speakers

def initialize_mic():
    """Safely checks for a working microphone index."""
    print("\n--- Initializing AIILA Audio System ---")
    try:
        mic_list = sr.Microphone.list_microphone_names()
        for index, name in enumerate(mic_list):
            try:
                # Attempt to open the mic to verify it's actually accessible
                with sr.Microphone(device_index=index) as source:
                    return index, name
            except Exception:
                continue
    except Exception as e:
        print(f"Hardware Warning: {e}")
    
    # Check sounddevice as a secondary check
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            return i, dev['name']

    return None, "No Microphone Detected"

def get_speaker_info():
    """Identifies the default output device."""
    try:
        device_info = sd.query_devices(kind='output')
        return device_info['name'], device_info['index']
    except Exception:
        return "Default Speaker", 0

# Initial Hardware Setup for A.E.G.I.S. Kernel
WORKING_MIC_INDEX, WORKING_MIC_NAME = initialize_mic()
SPEAKER_NAME, SPEAKER_INDEX = get_speaker_info()

def listen_and_process_command(app_state):
    """Processes voice input and updates AR state using Gemini 2.0."""
    if app_state.get('mic_index') is None and WORKING_MIC_INDEX is None:
        app_state['voice_feedback'] = "System: No Microphone Hardware!"
        app_state['is_listening'] = False
        return

    # Use the discovered index if not already in state
    mic_idx = app_state.get('mic_index') if app_state.get('mic_index') is not None else WORKING_MIC_INDEX

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(device_index=mic_idx) as source:
            app_state['voice_feedback'] = "Listening..."
            # Shortened duration for faster response in AR context
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            app_state['voice_feedback'] = "Processing Speech..."
            user_text = recognizer.recognize_google(audio)
            
            # Contextual Prompt for A.E.G.I.S. / AIILA
            model_context = app_state.get('active_model', 'General System')
            prompt = f"You are AIILA, an AR Security Assistant for A.E.G.I.S. Target: {model_context}. User: {user_text}"
            
            response = client.models.generate_content(
                model='gemini-2.0-flash', 
                contents=prompt
            )
            answer = response.text.strip()
            
            # Layer Logic Integration
            if "[LAYER2]" in answer: 
                app_state['current_layer_view'] = 2
            elif "[LAYER1]" in answer: 
                app_state['current_layer_view'] = 1
            
            # Clean text for AR Overlay
            display_text = answer.replace("[LAYER2]", "").replace("[LAYER1]", "").strip()
            app_state['dynamic_ar_text'] = display_text
            app_state['voice_feedback'] = "AIILA: Online."

    except sr.UnknownValueError:
        app_state['voice_feedback'] = "AIILA: Speech not understood."
    except sr.RequestError:
        app_state['voice_feedback'] = "AIILA: API Connection Error."
    except Exception as e:
        app_state['voice_feedback'] = f"System Error: {str(e)}"
    
    app_state['is_listening'] = False