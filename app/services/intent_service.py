# app/services/intent_service.py
import joblib
import os

MODEL_PATH = "models/intent_model.pkl"

pipeline = joblib.load(MODEL_PATH)

def predict_intent(text):
    # handle empty or noisy text from ASR
    if not text or len(text.strip()) < 2:
        return "unknown", 0.0

    pred = pipeline.predict([text])[0]
    # sklearn does not give confidence natively, estimate with predict_proba
    if hasattr(pipeline.named_steps['clf'], 'predict_proba'):
        proba = pipeline.predict_proba([text])
        confidence = max(proba[0])
    else:
        confidence = 1.0  # assume high confidence if predict_proba not available
    return pred, confidence
