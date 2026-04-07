# app/services/asr_service.py
from vosk import Model, KaldiRecognizer
import wave
import json
import tempfile
import os
import subprocess

# -----------------------------
# Model setup
# -----------------------------
MODEL_PATH = "models/vosk-model-small-en-us-0.15"

if not os.path.exists(MODEL_PATH):
    raise Exception(f"Vosk model not found at {MODEL_PATH}")

model = Model(MODEL_PATH)

# -----------------------------
# FFmpeg conversion
# -----------------------------
def convert_to_wav(input_path):
    output_path = input_path + "_converted.wav"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", input_path,
                "-ar", "16000",  # sample rate
                "-ac", "1",      # mono
                "-f", "wav",
                output_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15  # critical to prevent hanging
        )
    except subprocess.TimeoutExpired:
        raise Exception("FFmpeg conversion timed out")

    if result.returncode != 0:
        raise Exception("FFmpeg conversion failed")

    return output_path

# -----------------------------
# Audio transcription
# -----------------------------
def transcribe_audio(file):
    wav_path = None

    # Save temp uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        for chunk in iter(lambda: file.file.read(1024 * 1024), b""):
            tmp.write(chunk)
        temp_path = tmp.name

    try:
        if not os.path.exists(temp_path):
            raise Exception("File not saved properly")

        wav_path = convert_to_wav(temp_path)

        results = []

        # Open converted WAV safely
        try:
            with wave.open(wav_path, "rb") as wf:
                rate = wf.getframerate()
                if rate == 0:
                    raise Exception("Invalid audio file")

                duration = wf.getnframes() / rate
                if duration > 30:
                    raise Exception("Audio too long (max 30 sec)")

                rec = KaldiRecognizer(model, rate)

                while True:
                    data = wf.readframes(8000)  # larger chunk for performance
                    if len(data) == 0:
                        break
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        results.append(res.get("text", ""))

                final_res = json.loads(rec.FinalResult())
                results.append(final_res.get("text", ""))

        except wave.Error:
            raise Exception("Invalid or corrupt WAV file")

        return " ".join(results).strip()

    finally:
        # Cleanup temp files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

        try:
            file.file.close()
        except:
            pass
