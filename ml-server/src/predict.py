import numpy as np
from scipy.sparse import hstack
from nltk.sentiment import SentimentIntensityAnalyzer

from src.preprocessing import clean_text
from src.model_loader import ModelService

sia = SentimentIntensityAnalyzer()

LABEL_MAP = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

def predict_risk(text: str):
    service = ModelService.get_instance()

    cleaned = clean_text(text)

    word_features = service.word_vectorizer.transform([cleaned])
    char_features = service.char_vectorizer.transform([cleaned])

    sentiment_score = sia.polarity_scores(cleaned)["compound"]
    sentiment_feature = np.array([[sentiment_score]])

    X = hstack([word_features, char_features, sentiment_feature])

    probabilities = service.model.predict_proba(X)[0]
    predicted_class = int(np.argmax(probabilities))

    return {
        "label": LABEL_MAP[predicted_class],
        "confidence": float(round(probabilities[predicted_class], 4)),
        "probabilities": {
            "Low Risk": float(round(probabilities[0], 4)),
            "Medium Risk": float(round(probabilities[1], 4)),
            "High Risk": float(round(probabilities[2], 4)),
        }
    }