import os
import re
import time
import json
from pathlib import Path

import numpy as np
import speech_recognition as sr
import sounddevice as sd
from dotenv import load_dotenv

from core.local_aiila_engine import generate_offline_response

load_dotenv()
_VOSK_MODEL = None
_VOSK_MODEL_PATH = None


def _normalize_device_name(name: str) -> str:
    cleaned = (name or "").lower()
    cleaned = cleaned.replace("(r)", " ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())


def _safe_query_devices() -> list[dict]:
    try:
        return list(sd.query_devices())
    except Exception:
        return []


def _default_sounddevice(kind: str) -> dict | None:
    try:
        device = sd.query_devices(kind=kind)
        return dict(device)
    except Exception:
        return None


def _can_open_microphone(index: int | None) -> bool:
    if index is None:
        return False
    try:
        with sr.Microphone(device_index=index):
            return True
    except Exception:
        return False


def _match_sounddevice_index(name: str, *, want_input: bool = False) -> int | None:
    target = _normalize_device_name(name)
    if not target:
        return None

    best_match = None
    best_score = -1
    target_tokens = set(target.split())
    for index, device in enumerate(_safe_query_devices()):
        if want_input and device.get('max_input_channels', 0) <= 0:
            continue
        if not want_input and device.get('max_output_channels', 0) <= 0:
            continue

        device_name = _normalize_device_name(device.get('name', ''))
        if not device_name:
            continue

        score = 0
        if device_name == target:
            score = 100
        elif target in device_name or device_name in target:
            score = 80
        else:
            overlap = len(target_tokens & set(device_name.split()))
            score = overlap * 10

        if score > best_score:
            best_score = score
            best_match = index

    return best_match if best_score > 0 else None


def list_working_microphones() -> list[dict]:
    devices = []
    seen_names = set()
    default_input = _default_sounddevice('input')
    default_name = _normalize_device_name((default_input or {}).get('name', ''))

    try:
        microphone_names = sr.Microphone.list_microphone_names()
    except Exception as exc:
        print(f"Hardware Warning: {exc}")
        microphone_names = []

    for index, name in enumerate(microphone_names):
        clean_name = (name or "").strip()
        normalized = _normalize_device_name(clean_name)
        if not clean_name or not normalized or normalized in seen_names:
            continue
        if not _can_open_microphone(index):
            continue

        seen_names.add(normalized)
        sd_index = _match_sounddevice_index(clean_name, want_input=True)
        devices.append({
            'index': index,
            'name': clean_name,
            'sd_index': sd_index,
            'is_default': bool(default_name and normalized == default_name),
        })

    devices.sort(key=lambda item: (not item['is_default'], item['name'].lower()))
    return devices


def list_working_speakers() -> list[dict]:
    devices = []
    seen_names = set()
    default_output = _default_sounddevice('output')
    default_name = _normalize_device_name((default_output or {}).get('name', ''))

    for index, device in enumerate(_safe_query_devices()):
        if device.get('max_output_channels', 0) <= 0:
            continue

        clean_name = (device.get('name') or '').strip()
        normalized = _normalize_device_name(clean_name)
        if not clean_name or not normalized or normalized in seen_names:
            continue

        try:
            sd.check_output_settings(device=index)
        except Exception:
            continue

        seen_names.add(normalized)
        devices.append({
            'index': index,
            'name': clean_name,
            'is_default': bool(default_name and normalized == default_name),
        })

    devices.sort(key=lambda item: (not item['is_default'], item['name'].lower()))
    return devices


def get_hardware_info():
    return list_working_microphones(), list_working_speakers()


def initialize_mic():
    print("\n--- Initializing AIILA Audio System ---")
    microphones = list_working_microphones()
    if microphones:
        return microphones[0]['index'], microphones[0]['name']
    return None, "No Microphone Detected"


def get_speaker_info():
    speakers = list_working_speakers()
    if speakers:
        return speakers[0]['name'], speakers[0]['index']
    return "Default Speaker", 0


def _resolve_mic_index(app_state) -> int | None:
    requested_index = app_state.get('mic_index')
    requested_name = app_state.get('mic_name', '')

    if _can_open_microphone(requested_index):
        return requested_index

    normalized_name = _normalize_device_name(requested_name)
    for device in list_working_microphones():
        if normalized_name and _normalize_device_name(device['name']) == normalized_name:
            app_state['mic_index'] = device['index']
            app_state['mic_name'] = device['name']
            return device['index']

    if _can_open_microphone(WORKING_MIC_INDEX):
        return WORKING_MIC_INDEX
    return None


def _configure_recognizer(recognizer: sr.Recognizer):
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.85
    recognizer.phrase_threshold = 0.25
    recognizer.non_speaking_duration = 0.45
    recognizer.operation_timeout = 10


def _estimate_audio_level(audio: sr.AudioData) -> float:
    try:
        raw_data = audio.get_raw_data(convert_width=2)
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples))) / 32768.0)
    except Exception:
        return 0.0


def _find_vosk_model_path() -> Path | None:
    candidate_paths = []
    env_path = os.getenv("VOSK_MODEL_PATH", "").strip()
    if env_path:
        candidate_paths.append(Path(env_path))

    candidate_paths.extend([
        Path("assets/models/vosk"),
        Path("models/vosk"),
        Path("assets/models/vosk-model-small-en-us-0.15"),
        Path("models/vosk-model-small-en-us-0.15"),
    ])

    for path in candidate_paths:
        if path.exists() and path.is_dir():
            return path
    return None


def _get_vosk_model():
    global _VOSK_MODEL, _VOSK_MODEL_PATH

    model_path = _find_vosk_model_path()
    if model_path is None:
        return None

    if _VOSK_MODEL is not None and _VOSK_MODEL_PATH == str(model_path):
        return _VOSK_MODEL

    try:
        from vosk import Model, SetLogLevel
    except Exception:
        return None

    try:
        SetLogLevel(-1)
    except Exception:
        pass

    try:
        _VOSK_MODEL = Model(str(model_path))
        _VOSK_MODEL_PATH = str(model_path)
        print(f"[VOICE ENGINE] Loaded Vosk model from {model_path}")
        return _VOSK_MODEL
    except Exception as exc:
        print(f"[VOICE ENGINE] Failed to load Vosk model: {exc}")
        return None


def _recognize_audio_offline(audio: sr.AudioData) -> str:
    model = _get_vosk_model()
    if model is None:
        return ""

    try:
        from vosk import KaldiRecognizer
    except Exception:
        return ""

    try:
        sample_rate = 16_000
        recognizer = KaldiRecognizer(model, sample_rate)
        recognizer.AcceptWaveform(
            audio.get_raw_data(convert_rate=sample_rate, convert_width=2)
        )
        result = json.loads(recognizer.FinalResult())
        text = (result.get("text") or "").strip()
        if text:
            print(f"[VOICE ENGINE] Vosk transcription: {text}")
        return text
    except Exception as exc:
        print(f"[VOICE ENGINE] Offline Vosk recognition failed: {exc}")
        return ""


def _recognize_audio(recognizer: sr.Recognizer, audio: sr.AudioData) -> str:
    offline_text = _recognize_audio_offline(audio)
    if offline_text:
        return offline_text
    raise sr.UnknownValueError()


def _listen_for_speech(recognizer: sr.Recognizer, source, app_state) -> str:
    recognizer.adjust_for_ambient_noise(source, duration=0.9)
    ambient_threshold = int(recognizer.energy_threshold)
    app_state['voice_feedback'] = f"Listening... [{ambient_threshold}]"

    last_unknown_value_error = None
    quiet_threshold = 0.008

    for attempt in range(2):
        audio = recognizer.listen(source, timeout=6, phrase_time_limit=7)
        level = _estimate_audio_level(audio)
        print(f"[VOICE ENGINE] Captured audio level={level:.4f} attempt={attempt + 1}")

        if level < quiet_threshold:
            if attempt == 0:
                app_state['voice_feedback'] = "Voice too quiet, try speaking again..."
                continue
            raise sr.UnknownValueError()

        app_state['voice_feedback'] = "Processing Speech..."
        try:
            return _recognize_audio(recognizer, audio)
        except sr.UnknownValueError as exc:
            last_unknown_value_error = exc
            if attempt == 0:
                app_state['voice_feedback'] = "Didn't catch that. Listening once more..."
                recognizer.adjust_for_ambient_noise(source, duration=0.35)
                continue
            raise

    raise last_unknown_value_error or sr.UnknownValueError()

# Initial Hardware Setup for A.E.G.I.S. Kernel
WORKING_MIC_INDEX, WORKING_MIC_NAME = initialize_mic()
SPEAKER_NAME, SPEAKER_INDEX = get_speaker_info()

def listen_and_process_command(app_state, command_handler=None):
    """Processes voice input using offline speech recognition and local responses."""
    mic_idx = _resolve_mic_index(app_state)
    if mic_idx is None:
        app_state['voice_feedback'] = "System: No Microphone Hardware!"
        app_state['is_listening'] = False
        return

    app_state['mic_index'] = mic_idx

    recognizer = sr.Recognizer()
    _configure_recognizer(recognizer)
    try:
        with sr.Microphone(device_index=mic_idx) as source:
            mic_name = app_state.get('mic_name') or WORKING_MIC_NAME or f"Mic {mic_idx}"
            print(f"[VOICE ENGINE] Listening on mic index={mic_idx} name={mic_name}")
            user_text = _listen_for_speech(recognizer, source, app_state)
            app_state['dynamic_ar_text'] = user_text

            if command_handler is not None and command_handler(user_text):
                return

            answer = generate_offline_response(app_state, user_text)
            app_state['dynamic_ar_text'] = answer
            app_state['voice_feedback'] = f"AIILA: {answer}"

    except sr.UnknownValueError as uve:
        print(f"[VOICE ENGINE] UnknownValueError: No audio recognized or speech unintelligible. Details: {uve}")
        model_path = _find_vosk_model_path()
        if model_path is None:
            app_state['voice_feedback'] = "AIILA: Offline speech model missing."
        else:
            app_state['voice_feedback'] = "AIILA: Speech not understood."
    except sr.RequestError as re:
        print(f"[VOICE ENGINE] RequestError: Could not request results from service. Details: {re}")
        app_state['voice_feedback'] = "AIILA: API Connection Error."
    except sr.WaitTimeoutError:
        print(f"[VOICE ENGINE] WaitTimeoutError: Listening timed out before speech started.")
        app_state['voice_feedback'] = "AIILA: Listening timed out."
    except Exception as e:
        print(f"[VOICE ENGINE] General Exception: {type(e).__name__} - {str(e)}")
        import traceback
        traceback.print_exc()
        app_state['voice_feedback'] = f"System Error: {str(e)}"
    finally:
        app_state['is_listening'] = False
