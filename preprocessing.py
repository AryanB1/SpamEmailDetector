import re
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return "", {}
    
    features = {
        'text_length': len(text),
        'contains_url': int(bool(re.search(r'http\S+', text))),
        'contains_email': int(bool(re.search(r'\S+@\S+', text))),
        'contains_dollar': int('$' in text),
        'contains_exclamation': int('!' in text),
        'contains_question': int('?' in text),
        'capital_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1)
    }
    
    text_lower = text.lower()
    cleaned_text = re.sub(r'http\S+', ' url ', text_lower)
    cleaned_text = re.sub(r'\S+@\S+', ' email ', cleaned_text)
    cleaned_text = re.sub(r'<.*?>', ' ', cleaned_text)
    cleaned_text = re.sub(r'[^a-z0-9\s]', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in cleaned_text.split() if word not in stop_words and len(word) > 1]
    
    return " ".join(tokens), features
    
def load_and_preprocess_data(path, feature_count):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    
    text_col = "text"
    label_col = "spam"
    
    cleaned_texts = []
    feature_dicts = []
    
    for text in df[text_col]:
        cleaned_text, features = clean_text(text)
        cleaned_texts.append(cleaned_text)
        feature_dicts.append(features)
    
    feature_df = pd.DataFrame(feature_dicts)
    
    tf_idf = TfidfVectorizer(max_features=feature_count)
    text_features = tf_idf.fit_transform(cleaned_texts).toarray()
    
    x = np.hstack((text_features, feature_df.values))
    y = df[label_col].values
    
    return train_test_split(x, y, test_size=0.2, random_state=0)
