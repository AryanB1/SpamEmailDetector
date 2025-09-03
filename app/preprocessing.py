import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import cupy as cp
import cudf
from cuml.feature_extraction.text import TfidfVectorizer

_FORCE_CPU_ENV = os.getenv("SPAM_FORCE_CPU", "").lower() in ("1", "true", "yes")

def _gpu_is_available() -> bool:
    if _FORCE_CPU_ENV:
        return False
    try:
        test_arr = cp.array([1, 2, 3])
        _ = cp.sum(test_arr)
        return True
    except Exception:
        return False

_GPU_READY = _gpu_is_available()

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SK_EN_STOP
    _EN_STOP_LIST = list(_SK_EN_STOP)
except Exception:
    _EN_STOP_LIST = None

try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    _NLTK_STOPS = set(stopwords.words("english"))
except Exception:
    _NLTK_STOPS = set()


def clean_text(text: str):
    if pd.isna(text) or not isinstance(text, str):
        return "", {
            'text_length': 0, 'contains_url': 0, 'contains_email': 0,
            'contains_dollar': 0, 'contains_exclamation': 0,
            'contains_question': 0, 'capital_ratio': 0.0
        }

    features = {
        'text_length': len(text),
        'contains_url': int(bool(re.search(r'http\S+', text))),
        'contains_email': int(bool(re.search(r'\S+@\S+', text))),
        'contains_dollar': int('$' in text),
        'contains_exclamation': int('!' in text),
        'contains_question': int('?' in text),
        'capital_ratio': (sum(1 for c in text if c.isupper()) / max(len(text), 1))
    }

    t = text.lower()
    t = re.sub(r'http\S+', ' url ', t)
    t = re.sub(r'\S+@\S+', ' email ', t)
    t = re.sub(r'<.*?>', ' ', t)
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()

    if _NLTK_STOPS:
        tokens = [w for w in t.split() if w not in _NLTK_STOPS and len(w) > 1]
        t = " ".join(tokens)

    return t, features



def load_and_preprocess_data(path, feature_count, as_gpu=False, test_size=0.2, random_state=0):
    text_col = "text"
    label_col = "spam"

    if _GPU_READY:
        try:
            print("[preprocessing] Using CUDA-accelerated preprocessing...")
            
            try:
                df = cudf.read_csv(path)
            except Exception:
                df = cudf.from_pandas(pd.read_csv(path, encoding="ISO-8859-1"))

            if text_col not in df.columns or label_col not in df.columns:
                raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns.")

            s = df[text_col].astype('str').fillna('')

            text_length = s.str.len().fillna(0)
            contains_url = s.str.contains(r'http\S+', regex=True).fillna(False).astype('int8')
            contains_email = s.str.contains(r'\S+@\S+', regex=True).fillna(False).astype('int8')
            contains_dollar = s.str.contains(r'\$', regex=True).fillna(False).astype('int8')
            contains_exclamation = s.str.contains(r'!', regex=True).fillna(False).astype('int8')
            contains_question = s.str.contains(r'\?', regex=True).fillna(False).astype('int8')

            caps_only = s.str.replace(r'[^A-Z]', '', regex=True)
            caps = caps_only.str.len().fillna(0)
            capital_ratio = (caps.astype('float32') / text_length.clip(lower=1)).astype('float32')

            feature_df = cudf.DataFrame({
                'text_length': text_length.astype('int32'),
                'contains_url': contains_url,
                'contains_email': contains_email,
                'contains_dollar': contains_dollar,
                'contains_exclamation': contains_exclamation,
                'contains_question': contains_question,
                'capital_ratio': capital_ratio
            })

            print("[preprocessing] Cleaning text on GPU with cuDF...")
            cleaned = (
                s.str.lower()
                 .str.replace(r'http\S+', ' url ', regex=True)
                 .str.replace(r'\S+@\S+', ' email ', regex=True)
                 .str.replace(r'<.*?>', ' ', regex=True)
                 .str.replace(r'[^a-z0-9\s]', ' ', regex=True)
                 .str.replace(r'\s+', ' ', regex=True)
                 .str.strip()
            )

            print("[preprocessing] Converting to CPU for TF-IDF (cuML compatibility)...")
            cleaned_cpu = cleaned.to_arrow().to_pylist()
            
            tfidf = TfidfVectorizer(
                max_features=int(feature_count),
                lowercase=False
            )
            text_features_cpu = tfidf.fit_transform(cleaned_cpu).toarray().astype(np.float32)
            
            meta_features_cpu = feature_df.to_pandas().values.astype(np.float32)
            
            text_features_gpu = cp.asarray(text_features_cpu)
            meta_features_gpu = cp.asarray(meta_features_cpu)
            X_gpu = cp.hstack([text_features_gpu, meta_features_gpu])
            y_gpu = cp.asarray(df[label_col].astype('int32').to_arrow().to_pylist())

            np.random.seed(random_state)
            n_samples = X_gpu.shape[0]
            indices = cp.arange(n_samples)
            cp.random.shuffle(indices)
            
            test_size_abs = int(test_size * n_samples)
            test_indices = indices[:test_size_abs]
            train_indices = indices[test_size_abs:]
            
            X_tr = X_gpu[train_indices]
            X_te = X_gpu[test_indices]
            y_tr = y_gpu[train_indices]
            y_te = y_gpu[test_indices]
            
            if as_gpu:
                return X_tr, X_te, y_tr, y_te
            else:
                return (cp.asnumpy(X_tr), cp.asnumpy(X_te),
                        cp.asnumpy(y_tr), cp.asnumpy(y_te))

        except Exception as e:
            print(f"GPU path failed, falling back to CPU: {type(e).__name__}: {e}")

    df = pd.read_csv(path, encoding="ISO-8859-1")
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns.")

    cleaned_texts = []
    feature_dicts = []

    for text in df[text_col]:
        cleaned_text, features = clean_text(text)
        cleaned_texts.append(cleaned_text)
        feature_dicts.append(features)

    feature_df = pd.DataFrame(feature_dicts).astype({
        'text_length': 'int32',
        'contains_url': 'int8',
        'contains_email': 'int8',
        'contains_dollar': 'int8',
        'contains_exclamation': 'int8',
        'contains_question': 'int8',
        'capital_ratio': 'float32'
    })

    tfidf = TfidfVectorizer(
        max_features=int(feature_count),
        lowercase=False
    )
    text_features = tfidf.fit_transform(cleaned_texts).toarray().astype(np.float32)

    X = np.hstack((text_features, feature_df.values.astype(np.float32)))
    y = df[label_col].values.astype(np.int32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    return X_tr, X_te, y_tr, y_te
