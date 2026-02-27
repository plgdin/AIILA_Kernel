import os
import speech_recognition as sr
from google import genai
import sounddevice as sd
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

# Setup Gemini API
client = genai.Client(api_key=GEMINI_KEY)

def get_hardware_info():
    mics = sr.Microphone.list_microphone_names()
    speakers = sd.query_devices()
    return mics, speakers

def initialize_mic():
    print("\n--- Initializing Audio System ---")
    mic_list = sr.Microphone.list_microphone_names()
    for index, name in enumerate(mic_list):
        try:
            m = sr.Microphone(device_index=index)
            with m as source:
                if m.stream is not None:
                    return index, name
        except Exception:
            pass
    return 0, "Default Microphone"

def get_speaker_info():
    try:
        device_info = sd.query_devices(kind='output')
        return device_info['name'], device_info['index']
    except Exception:
        return "Default Speaker", 0

# Initial Hardware Setup
WORKING_MIC_INDEX, WORKING_MIC_NAME = initialize_mic()
SPEAKER_NAME, SPEAKER_INDEX = get_speaker_info()

def listen_and_process_command(app_state):
    if app_state['mic_index'] is None:
        app_state['voice_feedback'] = "Error: No Mic Found!"
        app_state['is_listening'] = False
        return

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone(device_index=app_state['mic_index']) as source:
            app_state['voice_feedback'] = "Listening..."
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            user_text = recognizer.recognize_google(audio)
            
            prompt = f"AR Assistant JARVIS. User at disassembled {app_state['active_model']}. Question: {user_text}"
            response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
            answer = response.text.strip()
            
            if "[LAYER2]" in answer: app_state['current_layer_view'] = 2
            elif "[LAYER1]" in answer: app_state['current_layer_view'] = 1
            
            app_state['dynamic_ar_text'] = answer.replace("[LAYER2]", "").replace("[LAYER1]", "").strip()
            app_state['voice_feedback'] = "Response generated."
    except Exception as e:
        app_state['voice_feedback'] = f"Error: {str(e)}"
    
    app_state['is_listening'] = False