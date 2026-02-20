import speech_recognition as sr
from google import genai

# Setup Gemini API
client = genai.Client(api_key="AIzaSyB1f6Pe348ThuMGf4fOccBOUMdHPD4yQ1Y")

def initialize_mic():
    print("\n--- Initializing Audio System ---")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        try:
            m = sr.Microphone(device_index=index)
            with m as source:
                if m.stream is not None:
                    print(f"SUCCESS: Audio channel unlocked at index {index} ({name})")
                    return index
        except Exception:
            pass
    return None

# Find the mic once when this module is imported
WORKING_MIC_INDEX = initialize_mic()

def listen_and_process_command(app_state):
    """
    Takes the app_state dictionary and modifies it based on voice commands.
    """
    if WORKING_MIC_INDEX is None:
        app_state['voice_feedback'] = "Error: No Mic Found!"
        app_state['is_listening'] = False
        return

    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone(device_index=WORKING_MIC_INDEX) as source:
            app_state['voice_feedback'] = "Listening..."
            print("\nMicrophone active. Speak now...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                app_state['voice_feedback'] = "Processing speech..."
                user_text = recognizer.recognize_google(audio)
                print(f"You said: '{user_text}'")
                
                # Dynamic Gemini Prompt using the active model from the app state
                prompt = f"""
                You are JARVIS, an AR engineering assistant. The user is looking at a disassembled {app_state['active_model']}. 
                The user asked: "{user_text}". 
                
                RULES:
                1. If they ask to see the battery or motherboard, start your response with exactly: [LAYER2]
                2. If they ask to see the front or first layer, start your response with exactly: [LAYER1]
                3. Answer their question concisely in 1 or 2 short sentences. Ensure your technical facts match the exact specs of the {app_state['active_model']}.
                """
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                answer = response.text.strip()
                
                # Check if Gemini wants to change the layers, update the app_state
                if "[LAYER2]" in answer:
                    app_state['current_layer_view'] = 2
                    answer = answer.replace("[LAYER2]", "").strip()
                elif "[LAYER1]" in answer:
                    app_state['current_layer_view'] = 1
                    answer = answer.replace("[LAYER1]", "").strip()
                
                app_state['dynamic_ar_text'] = answer
                app_state['voice_feedback'] = "Response generated."
                print(f"JARVIS answered: {app_state['dynamic_ar_text']}")
                    
            except sr.WaitTimeoutError:
                app_state['voice_feedback'] = "Error: Nobody spoke."
            except sr.UnknownValueError:
                app_state['voice_feedback'] = "Error: Could not understand audio."
                
    except Exception as e:
        print(f"\n[MIC ERROR]: {e}")
        app_state['voice_feedback'] = "Error: Mic disconnected."
            
    # Always unlock the listening state when done
    app_state['is_listening'] = False