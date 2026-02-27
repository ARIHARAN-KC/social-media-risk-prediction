import os

class Config:
    DEBUG = os.getenv("DEBUG", "False") == "True"
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    MAX_TEXT_LENGTH = 2000