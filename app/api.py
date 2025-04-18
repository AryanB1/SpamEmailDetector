from flask import Flask, request, jsonify
import os
import torch
import pandas as pd
import numpy as np
from preprocessing import clean_text
from model import LogisticRegressionModel
from sklearn.feature_extraction.text import TfidfVectorizer

# Constants
MAX_FEATURES = 1500
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model.pth')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'emails.csv')


def _load_model():
    state = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model = LogisticRegressionModel(input_dim=state['input_dim'])
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model

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
        X = np.hstack((text_vec, extra))
        tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            prob = model(tensor).item()
        label = 'spam' if prob > 0.5 else 'legitimate'
        return jsonify(label=label, probability=prob), 200
        
    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
