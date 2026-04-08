AI Voice bot is a FastAPI-based voice assistant that can:

* Transcribe audio into text
* Detect user intent from speech
* Generate appropriate responses
* Convert responses back into speech (Text-to-Speech)

This project demonstrates a complete pipeline for building a conversational voice bot using Python, machine learning, and open-source libraries.

## Features

1. **Automatic Speech Recognition (ASR)** – Converts uploaded audio into text using [Vosk](https://alphacephei.com/vosk/).
2. **Intent Recognition** – Classifies user input into predefined intents (e.g., `order_status`, `greeting`) using a trained sklearn model.
3. **Response Generation** – Returns predefined responses based on detected intent.
4. **Text-to-Speech (TTS)** – Converts response text into speech using [gTTS](https://pypi.org/project/gTTS/).
5. **Web Interface & API** – Includes a simple web interface and REST API for programmatic access.
6. **Docker & Render Ready** – Easy deployment using Docker or Render cloud services.

## Architecture

Audio Input -> ASR (Vosk) -> Text -> Intent Detection -> Response Generation -> TTS (gTTS) -> Audio Output

* **ASR Service**: Transcribes audio files into text (`app/services/asr_service.py`).
* **Intent Service**: Predicts intent using a trained sklearn pipeline (`app/services/intent_service.py`).
* **Response Service**: Generates a textual response for the detected intent (`app/services/response_service.py`).
* **TTS Service**: Converts the response into an audio file (`app/services/tts_service.py`).
* **FastAPI Server**: Exposes API endpoints for voice processing (`main.py`).

---

## Installation

### Requirements

* Python 3.11+
* ffmpeg installed and accessible in PATH
* Internet connection for gTTS

### Local Setup

1. Clone the repository:

```bash
git clone https://github.com/danzinfo/voicebot-sk.git
cd voicebot-sk
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set environment variables:

```bash
cp .env.example .env
# Edit .env if needed
```

5. Download the Vosk ASR model:

```bash
# Example: small English model
mkdir -p models
# Download from https://alphacephei.com/vosk/models
# Unzip into models/vosk-model-small-en-us-0.15
```

---

## Usage

### Run locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Access

* Web Interface: [http://localhost:8000](http://localhost:8000)
* API Base     : [http://localhost:8000/api](http://localhost:8000/api)

---

## API Endpoints

| Endpoint             | Method | Description                                                       |
| -------------------- | ------ | ----------------------------------------------------------------- |
| `/`                  | GET    | Serves the main web page                                          |
| `/api`               | GET    | Basic API status                                                  |
| `/health`            | GET    | Health check                                                      |
| `/voicebot`          | POST   | Send audio file → returns transcript, intent, response, TTS audio |
| `/transcribe`        | POST   | Transcribe audio file only                                        |
| `/predict-intent`    | POST   | Predict intent from text                                          |
| `/generate-response` | POST   | Generate response text from intent                                |
| `/synthesize`        | POST   | Convert text to speech                                            |

**Example: `/voicebot`** (using `curl`):

```bash
curl -X POST "http://localhost:8000/voicebot" -F "file=@your_audio.wav"
```

## Intent Model Training

The intent detection model is a **sklearn pipeline** with TF-IDF features and logistic regression.

Train your own model:

```bash
python train_intent_model.py
```

* Input dataset: `data/train_expanded_1000.csv`
* Output model: `models/intent_model.pkl`

---

## Deployment

### Docker

```bash
docker build -t voicebot .
docker run -p 8000:8000 voicebot
```

### Render

Configured via `.render.yaml`. The service will automatically deploy with persistent storage for audio outputs.

---

## Folder Structure

```
.
├── app/
│   ├── main.py
│   ├── config.py
│   ├── services/
│   │   ├── asr_service.py
│   │   ├── intent_service.py
│   │   ├── response_service.py
│   │   ├── tts_service.py
│   │   └── evaluate_asr.py
│   ├── utils/
│   │   └── logger.py
├── models/intent_model.pkl  # ASR & intent models
├── outputs/audio/         # Generated audio files
├── templates/index.html   # HTML/CSS for web UI
├── Dockerfile
├── requirements.txt
├── train_intent_model.py
├── .env
└── .render.yaml
```

## License

This project is licensed under the **MIT License**.

Project Live Link:
https://voicebot-sk.onrender.com/
