import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import re
import os

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/train_expanded_1000.csv")
df['text'] = df['text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['intent'],
    test_size=0.2,
    random_state=42,
    stratify=df['intent']
)

# -----------------------------
# Sklearn pipeline
# -----------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(3,5), lowercase=True)),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=500)))
])

# -----------------------------
# Train
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = pipeline.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/intent_model.pkl")
print("\n✅ Model saved to models/intent_model.pkl")
