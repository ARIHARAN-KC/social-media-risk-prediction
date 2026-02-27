import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

# SENTIMENT EXTRACTION

sia = SentimentIntensityAnalyzer()

def extract_sentiment(text_series):
    sentiments = text_series.apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )
    return np.array(sentiments).reshape(-1, 1)

# WORD TF-IDF

def build_word_vectorizer():
    return TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english"
    )

# CHARACTER TF-IDF

def build_char_vectorizer():
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=5000
    )

# COMBINE FEATURES
def combine_features(X_word, X_char, sentiment_feature):
    return hstack([X_word, X_char, sentiment_feature])