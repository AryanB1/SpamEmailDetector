import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def load_and_preprocess_data(path):
    df = pd.read_csv(path, encoding="ISO-8859-1")[["Category", "Message"]]
    df["Message"] = df["Message"].apply(clean_text)
    df["Label"] = df["Category"].map({"ham": 0, "spam": 1})
    
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(df["Message"]).toarray()
    y = df["Label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
