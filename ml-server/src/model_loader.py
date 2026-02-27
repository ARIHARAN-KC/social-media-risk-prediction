import os
import joblib
from config import Config

class ModelService:
    _instance = None

    def __init__(self):
        self.model = joblib.load(
            os.path.join(Config.MODEL_DIR, "xgb_risk_model.pkl")
        )

        self.word_vectorizer = joblib.load(
            os.path.join(Config.MODEL_DIR, "word_vectorizer.pkl")
        )

        self.char_vectorizer = joblib.load(
            os.path.join(Config.MODEL_DIR, "char_vectorizer.pkl")
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance