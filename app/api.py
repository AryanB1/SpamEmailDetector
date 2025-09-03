from flask import Flask, request, jsonify
import os
import onnxruntime as ort
import pandas as pd
import numpy as np
from .preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
MAX_FEATURES = 1500
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.onnx')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'emails.csv')


def _load_model():
    session = ort.InferenceSession(MODEL_PATH)
    return session

def _load_vectorizer():
    df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
    cleaned = [clean_text(text)[0] for text in df['text']]
    vect = TfidfVectorizer(max_features=MAX_FEATURES)
    vect.fit(cleaned)
    return vect

def create_app(test_config=None):
    app = Flask(__name__)
    # Apply test settings if provided
    if test_config:
        app.config.update(test_config)
    
    model = _load_model()
    vectorizer = _load_vectorizer()
    
    @app.route('/health', methods=['GET'])
    def health():
        return jsonify(status='ok'), 200

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json(force=True)
        if not data or 'email' not in data:
            return jsonify(error='Missing email in request'), 400
        text = data['email']
        cleaned, feats = clean_text(text)
        text_vec = vectorizer.transform([cleaned]).toarray()
        extra = np.array([list(feats.values())])
        X = np.hstack((text_vec, extra)).astype(np.float32)
        
        outputs = model.run(None, {'input': X})
        prob = outputs[0][0][0]
        label = 'spam' if prob > 0.5 else 'legitimate'
        return jsonify(label=label, probability=float(prob)), 200
        
    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
