# app/services/intent_service.py
import joblib
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_model.pkl")

print("Loading model from:", MODEL_PATH)
print("Exists:", os.path.exists(MODEL_PATH))

pipeline = joblib.load(MODEL_PATH)

print("TF-IDF fitted:", hasattr(pipeline.named_steps['tfidf'], 'idf_'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_intent(text):
    text = preprocess_text(text)

    if not text or len(text.strip()) < 2:
        return "unknown", 0.0

    pred = pipeline.predict([text])[0]

    if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
        proba = pipeline.predict_proba([text])
        confidence = max(proba[0])
    else:
        confidence = 1.0

    return pred, confidence
