"""
Alisa_full_fixed.py
Full Jarvis-like assistant (face memory, hand detection, STT, TTS, LLM, persistent memory)
Compatible with Python 3.12.x

Controls in camera window:
    q -> quit
    t -> teach (save face for the largest detected face)
    r -> recall last memories (spoken)
    s -> toggle hand detection
    p -> toggle pause/resume face detection display
    l -> list known faces
    d -> delete a face (type name in console)
    n -> rename a face (type oldname,newname in console)
    h -> print help
"""

import os
import io
import uuid
import time
import json
import tempfile
import sqlite3
import threading
from datetime import datetime

# core libs
import cv2
import numpy as np

# face recognition
import face_recognition

# audio & STT/TTS
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import pyttsx3
import requests
from pydub import AudioSegment
import simpleaudio as sa

# LLM
import openai

# ElevenLabs (SDK has changed over time; we attempt to use common helpers)
# Many SDKs provide `generate` / `play` functions; if not present we fallback to HTTP route.
try:
    # try modern import helpers
    from elevenlabs import ElevenLabs, generate, play
    ELEVENLABS_SDK_OK = True
except Exception:
    try:
        from elevenlabs import ElevenLabs
        ELEVENLABS_SDK_OK = True
    except Exception:
        ELEVENLABS_SDK_OK = False

# optional hand detection via Mediapipe
try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
except Exception:
    HAVE_MEDIAPIPE = False

# ---------------- CONFIG ----------------
DATA_DIR = "jarvis_data"
IMAGES_DIR = os.path.join(DATA_DIR, "known_faces")
DB_PATH = os.path.join(DATA_DIR, "memory.db")
SIMILARITY_THRESHOLD = 0.45   # lower = stricter matching (face_distance threshold)
CAMERA_INDEX = 0            # default webcam index

USE_ELEVENLABS = True         # set False to use pyttsx3 fallback
USE_OPENAI = True             # set False to skip OpenAI LLM calls

# Put keys here or set environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "sk-REPLACE_WITH_YOURS"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or "REPLACE_ELEVEN_KEY"
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID") or None  # if None, Eleven default voice will be used

# audio params for STT recording
STT_FS = 16000
STT_SECONDS_DEFAULT = 3.5

# create directories
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------------- DATABASE ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id TEXT PRIMARY KEY,
    name TEXT,
    encoding BLOB,
    img_path TEXT,
    created_at REAL
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    category TEXT,
    content TEXT
)
""")
conn.commit()

# ---------------- HELPERS ----------------
def encoding_to_bytes(enc: np.ndarray) -> bytes:
    return enc.tobytes()

def bytes_to_encoding(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float64)

def save_face_to_db(name: str, encoding: np.ndarray, img_np: np.ndarray):
    fid = str(uuid.uuid4())
    img_path = os.path.join(IMAGES_DIR, f"{fid}.jpg")
    cv2.imwrite(img_path, img_np)
    enc_bytes = encoding_to_bytes(encoding)
    cur.execute("INSERT INTO faces (id, name, encoding, img_path, created_at) VALUES (?, ?, ?, ?, ?)",
                (fid, name, enc_bytes, img_path, time.time()))
    conn.commit()
    return fid

# This is the only load_known_faces function
def load_known_faces():
    global known_ids, known_names, known_encodings
    cur.execute("SELECT id, name, encoding, img_path FROM faces")
    rows = cur.fetchall()
    encodings, names, ids = [], [], []
    for rid, name, enc_blob, img_path in rows:
        try:
            enc = bytes_to_encoding(enc_blob)
            encodings.append(enc)
            names.append(name)
            ids.append(rid)
        except Exception as e:
            print(f"Failed to load encoding for {name} (ID {rid}):", e)
    known_ids, known_names, known_encodings = ids, names, encodings


# ✅ Load faces at startup
load_known_faces()


# ✅ Memory system
def store_memory(category: str, content: str):
    cur.execute(
        "INSERT INTO memories (timestamp, category, content) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), category, content)
    )
    conn.commit()


def recall_memories(limit=5):
    cur.execute(
        "SELECT timestamp, category, content FROM memories ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    return cur.fetchall()

# ---------------- TTS: local fallback engine ----------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def tts_local(text: str):
    """Synchronous local TTS (pyttsx3)."""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Local TTS error:", e)

# ---------------- ElevenLabs TTS wrapper (robust) ----------------
eleven_client = None
if USE_ELEVENLABS and ELEVENLABS_API_KEY and ELEVENLABS_SDK_OK:
    try:
        eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except Exception as e:
        print("ElevenLabs SDK init failed:", e)
        eleven_client = None
else:
    eleven_client = None

def play_audio_bytes_with_sounddevice(audio_bytes: bytes):
    """
    Utility: attempt to decode audio bytes (mp3/wav) and play via sounddevice.
    """
    try:
        # pydub can read mp3/wav from bytes
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        samples = np.array(seg.get_array_of_samples())
        # convert depending on channels
        if seg.channels == 2:
            samples = samples.reshape((-1, 2))
        # normalize samples to float32 in range [-1, 1]
        max_val = float(1 << (8 * seg.sample_width - 1))
        samples = samples.astype(np.float32) / max_val
        sd.play(samples, seg.frame_rate)
        sd.wait()
        return True
    except Exception as e:
        # last resort: write tmp file and try simpleaudio/pydub playback
        try:
            tmpfn = os.path.join(tempfile.gettempdir(), f"alisa_tts_{int(time.time()*1000)}.mp3")
            with open(tmpfn, "wb") as f:
                f.write(audio_bytes)
            seg = AudioSegment.from_file(tmpfn)
            raw = seg.raw_data
            wave_obj = sa.WaveObject(raw, num_channels=seg.channels, bytes_per_sample=seg.sample_width, sample_rate=seg.frame_rate)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            os.remove(tmpfn)
            return True
        except Exception as e2:
            print("Failed to play audio bytes:", e, e2)
            return False

def tts_elevenlabs(text: str):
    """
    Robust ElevenLabs TTS: tries SDK generate/play, falls back to HTTP convert endpoint,
    then falls back to local pyttsx3.
    """
    if not eleven_client:
        # no client -> fallback
        tts_local(text)
        return

    print("ElevenLabs TTS:", text)
    # Primary attempt: SDK generate + play (if present)
    try:
        if 'generate' in globals():
            # If SDK provides generate/play helpers
            audio = generate(text=text, voice=ELEVEN_VOICE_ID, model="eleven_turbo_v2")
            # play accepts object or bytes; try play()
            try:
                play(audio)
                return
            except Exception:
                # Some SDKs return bytes — handle below
                if isinstance(audio, bytes):
                    if play_audio_bytes_with_sounddevice(audio):
                        return
        # Next attempt: use client.text_to_speech.generate or convert (different SDK names)
        try:
            # Many SDKs: client.text_to_speech.generate or client.text_to_speech.convert
            if hasattr(eleven_client, "text_to_speech") and hasattr(eleven_client.text_to_speech, "generate"):
                stream_gen = eleven_client.text_to_speech.generate(text=text, voice=ELEVEN_VOICE_ID)
                # SDK may return bytes or file-like; try to play bytes
                if isinstance(stream_gen, bytes):
                    if play_audio_bytes_with_sounddevice(stream_gen):
                        return
                # If generator of chunks:
                if hasattr(stream_gen, "__iter__"):
                    audio_bytes = b"".join([chunk for chunk in stream_gen if isinstance(chunk, (bytes, bytearray))])
                    if audio_bytes and play_audio_bytes_with_sounddevice(audio_bytes):
                        return
            # alternative API name: convert / synthesize
            if hasattr(eleven_client, "text_to_speech") and hasattr(eleven_client.text_to_speech, "convert"):
                out = eleven_client.text_to_speech.convert(voice_id=ELEVEN_VOICE_ID, text=text)
                # convert may return bytes or a path; try to play bytes
                if isinstance(out, (bytes, bytearray)):
                    if play_audio_bytes_with_sounddevice(out):
                        return
        except Exception as e:
            # try raw HTTP fallback next
            print("SDK generate/convert attempt failed:", e)
    except Exception as e:
        print("ElevenLabs SDK top-level attempt failed:", e)

    # HTTP fallback: direct ElevenLabs v1 text-to-speech endpoint (MP3 output)
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}" if ELEVEN_VOICE_ID else "https://api.elevenlabs.io/v1/text-to-speech/default"
        headers = {
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {"text": text, "voice_settings": {"stability": 0.7, "similarity_boost": 0.75}}
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        if r.status_code == 200:
            audio_bytes = r.content
            if play_audio_bytes_with_sounddevice(audio_bytes):
                return
        else:
            print("ElevenLabs HTTP TTS failed:", r.status_code, r.text[:200])
    except Exception as e:
        print("ElevenLabs HTTP fallback failed:", e)

    # final fallback
    tts_local(text)

def speak(text: str, async_play=True):
    if USE_ELEVENLABS and eleven_client:
        if async_play:
            threading.Thread(target=tts_elevenlabs, args=(text,), daemon=True).start()
        else:
            tts_elevenlabs(text)
    else:
        if async_play:
            threading.Thread(target=tts_local, args=(text,), daemon=True).start()
        else:
            tts_local(text)

# ---------------- STT helpers using sounddevice and speech_recognition ----------------
recognizer = sr.Recognizer()

def record_wav_to_file(seconds=STT_SECONDS_DEFAULT, fs=STT_FS):
    """Record with sounddevice and write to a temporary WAV file; returns filepath."""
    try:
        print(f"[STT] recording {seconds}s at {fs}Hz...")
        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, recording, fs, subtype='PCM_16')
        return tmp.name
    except Exception as e:
        print("Recording error:", e)
        return None

def listen_with_timeout(timeout=4, phrase_time_limit=3.5):
    """
    Record short audio using sounddevice, then run speech_recognition on it.
    Returns recognized text or empty string.
    """
    wavfile = record_wav_to_file(seconds=phrase_time_limit, fs=STT_FS)
    if not wavfile:
        return ""
    try:
        with sr.AudioFile(wavfile) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                print("[STT] Heard:", text)
                return text
            except sr.UnknownValueError:
                print("[STT] UnknownValue")
                return ""
            except sr.RequestError as e:
                print("[STT] RequestError:", e)
                return ""
    except Exception as e:
        print("STT other error:", e)
        return ""
    finally:
        try:
            os.remove(wavfile)
        except Exception:
            pass

# ---------------- OpenAI LLM wrapper ----------------
if USE_OPENAI and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

def ask_llm(prompt: str, system="You are a helpful assistant named Alisa."):
    if not (USE_OPENAI and OPENAI_API_KEY):
        return "LLM not enabled"
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:", e)
        return "Sorry, I couldn't process that."

# ---------------- Face matching helper ----------------
def match_face(face_encoding, known_encodings, known_names, threshold=SIMILARITY_THRESHOLD):
    if not known_encodings:
        return None, None
    dists = face_recognition.face_distance(known_encodings, face_encoding)
    best_idx = np.argmin(dists)
    best_dist = float(dists[best_idx])
    if best_dist <= threshold:
        return known_names[best_idx], best_dist
    return None, best_dist

# ---------------- Mediapipe hand detection (optional) ----------------
if HAVE_MEDIAPIPE:
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
else:
    hands_detector = None

def detect_hands(frame):
    if not HAVE_MEDIAPIPE or not hands_detector:
        return []
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img)
    if not results or not results.multi_hand_landmarks:
        return []
    return results.multi_hand_landmarks

# ---------------- Camera & UI main loop ----------------
def camera_loop():
    global known_ids, known_names, known_encodings
    video = cv2.VideoCapture(CAMERA_INDEX)
    if not video.isOpened():
        print("Cannot open webcam")
        return
    print("Camera started. Controls: q=quit t=teach r=recall s=toggle hands p=pause l=list d=delete n=rename h=help")
    pause = False
    show_hands = False
    greeted = set()
    last_face_locations = []
    last_face_encodings = []

    def print_help():
        print("""
Controls:
  q -> quit
  t -> teach largest face
  r -> recall spoken memories
  s -> toggle hand detection
  p -> pause/resume face detection
  l -> list known faces
  d -> delete face (type name in console)
  n -> rename face (enter oldname,newname)
  h -> help
""")
    print_help()

    while True:
        ret, frame = video.read()
        if not ret:
            continue
        display_frame = frame.copy()

        if not pause:
            small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            last_face_locations = face_locations
            last_face_encodings = face_encodings

            # draw faces + labels
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                top *= 2; right *= 2; bottom *= 2; left *= 2
                name, score = match_face(face_encoding, known_encodings, known_names)
                if name:
                    label = f"{name} ({score:.2f})"
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0,255,0), 2)
                    cv2.putText(display_frame, label, (left, top-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    if name not in greeted:
                        greeted.add(name)
                        speak(f"Hey {name}, welcome back!")
                else:
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0,0,255), 2)
                    cv2.putText(display_frame, "Unknown - press t to teach", (left, top-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # hand detection
            if show_hands and hands_detector:
                hand_landmarks = detect_hands(frame)
                if hand_landmarks:
                    for hl in hand_landmarks:
                        for lm in hl.landmark:
                            h, w, _ = display_frame.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(display_frame, (cx, cy), 3, (255,200,0), -1)

        # show UI
        cv2.imshow("Alisa (q=quit, h=help)", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('p'):
            pause = not pause
            print("Paused:" , pause)
        if key == ord('s'):
            show_hands = not show_hands
            print("Hand detection:", show_hands)
        if key == ord('h'):
            print_help()
        if key == ord('r'):
            mems = recall_memories(limit=5)
            if mems:
                txt = "I remember: " + "; ".join([m[2] for m in mems])
            else:
                txt = "I don't have any memories yet."
            print("Recalling:", txt)
            speak(txt)
        if key == ord('l'):
            print("Known faces:")
            for i, name in enumerate(known_names):
                print(i, name)
        if key == ord('d'):
            # delete a face by name (console prompt)
            name_to_delete = input("Type name to delete: ").strip()
            if not name_to_delete:
                print("No name entered.")
            else:
                cur.execute("SELECT id, name FROM faces WHERE name=?", (name_to_delete,))
                rows = cur.fetchall()
                if not rows:
                    print("No matching name found.")
                else:
                    # delete all matches
                    for rid, rname in rows:
                        cur.execute("DELETE FROM faces WHERE id=?", (rid,))
                    conn.commit()
                    # reload lists
                   
                    # See [Bug 2] - The function updates globals, it doesn't return them
                    load_known_faces()
                    print("Deleted", len(rows), "entries named", name_to_delete)
                    speak(f"Deleted {len(rows)} records named {name_to_delete}")
        if key == ord('n'):
            inp = input("Enter rename as oldname,newname : ").strip()
            if ',' in inp:
                old, new = [x.strip() for x in inp.split(',',1)]
                cur.execute("UPDATE faces SET name=? WHERE name=?", (new, old))
                conn.commit()
                # See [Bug 2] - The function updates globals, it doesn't return them
                load_known_faces()
                print(f"Renamed {old} to {new}")
                speak(f"Renamed {old} to {new}")
            else:
                print("Invalid format.")
        if key == ord('t'):
            # teach largest face
            if pause:
                print("Unpause to teach.")
                continue
            if not last_face_locations:
                print("No faces to teach.")
                speak("I can't see a face to save. Move closer.")
                continue
            sizes = [(b - t) * (r - l) for (t, r, b, l) in last_face_locations]
            idx = int(np.argmax(sizes))
            top, right, bottom, left = last_face_locations[idx]
            top *= 2; right *= 2; bottom *= 2; left *= 2
            face_img = frame[top:bottom, left:right].copy()
            speak("Who is this? Say the name now or press enter to type.")
            name = listen_with_timeout(timeout=4, phrase_time_limit=3.5)
            if not name:
                name = input("Type name: ").strip()
            if name:
                # compute high-res encodings
                full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                full_locs = face_recognition.face_locations(full_rgb)
                full_encs = face_recognition.face_encodings(full_rgb, full_locs)
                chosen_enc = None
                if full_encs:
                    # crude overlap heuristic
                    for (ftop, fright, fbottom, fleft), enc in zip(full_locs, full_encs):
                        if abs(ftop*1 - top//1) < 120 and abs(fleft*1 - left//1) < 120:
                            chosen_enc = enc
                            break
                    if chosen_enc is None:
                        chosen_enc = full_encs[0]
                    fid = save_face_to_db(name, chosen_enc, face_img)
                    known_ids.append(fid)
                    known_names.append(name)
                    known_encodings.append(chosen_enc)
                    speak(f"Saved {name}. I will remember them.")
                    store_memory("face_added", f"Saved {name}")
                    print("Saved", name)
                else:
                    speak("Sorry, I couldn't capture a good encoding. Try again.")
                    print("No high-res encoding available.")

    video.release()
    cv2.destroyAllWindows()

# ---------------- Assistant console loop ----------------
def assistant_console_loop():
    print("Alisa console: type 'help' for commands, 'exit' to quit.")
    # See [Bug 1] - Global declaration must be at the top of the function
    global known_ids, known_names, known_encodings
    while True:
        try:
            cmd = input("You: ").strip()
        except EOFError:
            break
        if not cmd:
            continue
        if cmd.lower() in ("exit","quit","bye"):
            speak("Goodbye.")
            break
        if cmd.lower() == "help":
            print("""
Console commands:
  help -> this text
  exit -> quit assistant
  memories -> show last 10 memories
  say <text> -> speak text
  ask <text> -> ask LLM and speak answer
  listfaces -> print known faces
  delete <name> -> delete face(s) by name
  rename <old>,<new> -> rename face
""")
            continue
        if cmd.lower() == "memories":
            mems = recall_memories(10)
            for m in mems:
                print(m)
            continue
        if cmd.lower().startswith("say "):
            speak(cmd[4:])
            continue
        if cmd.lower().startswith("ask "):
            q = cmd[4:]
            if USE_OPENAI and OPENAI_API_KEY:
                ans = ask_llm(q)
                print("Alisa:", ans)
                speak(ans)
                store_memory("chat", f"Q:{q} | A:{ans}")
            else:
                print("LLM disabled.")
            continue
        if cmd.lower() == "listfaces":
            # This read is now valid because of the global declaration at the top
            print("Known faces:", known_names)
            continue
        if cmd.lower().startswith("delete "):
            nm = cmd[7:].strip()
            if nm:
                cur.execute("DELETE FROM faces WHERE name=?", (nm,))
                conn.commit()
                # See [Bug 2] - The function updates globals, it doesn't return them
                load_known_faces()
                print("Deleted faces named", nm)
            continue
        if cmd.lower().startswith("rename "):
            rest = cmd[7:].strip()
            if ',' in rest:
                old,new = [x.strip() for x in rest.split(',',1)]
                cur.execute("UPDATE faces SET name=? WHERE name=?", (new, old))
                conn.commit()
                # See [Bug 2] - The function updates globals, it doesn't return them
                load_known_faces()
                print("Renamed", old, "to", new)
            else:
                print("Format: rename oldname,newname")
            continue
        # default: treat as chat
        if USE_OPENAI and OPENAI_API_KEY:
            ans = ask_llm(cmd)
            print("Alisa:", ans)
            speak(ans)
            store_memory("chat", f"Q:{cmd} | A:{ans}")
        else:
            speak("You said: " + cmd)
            store_memory("chat", cmd)

# ---------------- Run ----------------
if __name__ == "__main__":
    try:
        # Start assistant console thread (optional interactive console)
        console_thread = threading.Thread(target=assistant_console_loop, daemon=True)
        console_thread.start()

        # Start camera UI (blocks until window closed)
        camera_loop()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        conn.close()
        print("Shutdown complete.")