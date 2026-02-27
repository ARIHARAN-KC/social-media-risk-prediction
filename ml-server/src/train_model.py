import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import nltk
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import clean_text
from feature_engineering import extract_features

# Download VADER lexicon (first time only)
nltk.download("vader_lexicon")

# LOGGING SETUP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# PATH CONFIG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

os.makedirs(MODEL_DIR, exist_ok=True)

logging.info(f"Loading dataset from: {DATA_PATH}")

# LOAD DATA

df = pd.read_csv(DATA_PATH)

toxicity_columns = [
    "toxic","severe_toxic","obscene",
    "threat","insult","identity_hate"
]

# CREATE RISK LABEL

def create_risk_label(row):
    score = row[toxicity_columns].sum()
    if score == 0:
        return 0
    elif row["severe_toxic"] == 1 or row["threat"] == 1 or score >= 2:
        return 2
    else:
        return 1

df["risk_label"] = df.apply(create_risk_label, axis=1)

logging.info("Class Distribution:")
logging.info(df["risk_label"].value_counts())

# CLEAN TEXT

logging.info("Cleaning text...")
df["clean_text"] = df["comment_text"].apply(clean_text)

# FEATURE EXTRACTION

logging.info("Extracting features...")
X, y, word_vectorizer, char_vectorizer = extract_features(df)

# TRAIN / VALIDATION SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# CLASS WEIGHTS

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

weights = np.array([class_weights[label] for label in y_train])

logging.info(f"Class Weights: {class_weights}")

# TRAIN XGBOOST (WITH EARLY STOPPING)

logging.info("Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist"
)

model.fit(
    X_train,
    y_train,
    sample_weight=weights,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=50
)

# EVALUATION

logging.info("Evaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

print(f"\nValidation Accuracy: {accuracy:.4f}")

# SAVE MODEL + VECTORIZERS

joblib.dump(model, os.path.join(MODEL_DIR, "xgb_risk_model_v2.pkl"))
joblib.dump(word_vectorizer, os.path.join(MODEL_DIR, "word_vectorizer_v2.pkl"))
joblib.dump(char_vectorizer, os.path.join(MODEL_DIR, "char_vectorizer_v2.pkl"))

logging.info("Model & vectorizers saved successfully!")