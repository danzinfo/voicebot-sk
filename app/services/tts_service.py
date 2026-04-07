from gtts import gTTS
from app.config import TTS_LANG, TTS_SLOW
import os
import uuid

# Output folder
OUTPUT_DIR = "outputs/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Max text length for TTS to avoid memory issues
MAX_TTS_CHARS = 300  # adjust as needed

def synthesize_speech(text: str) -> str:
    """
    Converts text to speech using gTTS and returns path to audio file.
    Limits text length to avoid memory overload on free-tier instances.
    """
    if not text or len(text.strip()) == 0:
        raise Exception("No text provided for TTS")

    # Truncate text if too long
    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS]
    
    filename = f"{OUTPUT_DIR}/{uuid.uuid4().hex}.mp3"

    try:
        tts = gTTS(text=text, lang=TTS_LANG, slow=TTS_SLOW)
        tts.save(filename)
    except Exception as e:
        raise Exception(f"TTS failed: {str(e)}")

    return f"/audio/{os.path.basename(filename)}"
