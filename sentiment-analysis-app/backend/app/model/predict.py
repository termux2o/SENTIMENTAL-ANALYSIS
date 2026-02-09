import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

def get_prediction(text: str):
    text = text.lower()
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    if pred == 0:
        return "Negative"
    elif pred == 1:
        return "Neutral"
    else:
        return "Positive"
