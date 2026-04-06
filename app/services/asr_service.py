import whisper
from app.config import WHISPER_MODEL
import tempfile
import os

model = None

def get_model():
    global model
    if model is None:
        model = whisper.load_model(WHISPER_MODEL)
    return model


def save_temp_file(upload_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
        return tmp.name


def transcribe_audio(file):
    temp_path = save_temp_file(file)

    try:
        model = get_model()
        result = model.transcribe(temp_path)
    finally:
        os.remove(temp_path)

    return result["text"]
