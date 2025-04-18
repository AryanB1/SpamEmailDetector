import pytest
from app.api import create_app

@ pytest.fixture
def client():
    app = create_app(test_config={'TESTING': True})
    return app.test_client()


def test_health(client):
    resp = client.get('/health')

    assert resp.status_code == 200
    assert resp.get_json() == {'status': 'ok'}


def test_predict_missing_email(client):
    resp = client.post('/predict', json={})
    data = resp.get_json()

    assert resp.status_code == 400
    assert 'error' in data


def test_predict_legitimate_email(client):
    email = "Hello, I hope you are doing well. Let's catch up soon."
    resp = client.post('/predict', json={'email': email})
    data = resp.get_json()

    assert resp.status_code == 200
    assert 'label' in data and 'probability' in data
    assert data['label'] == 'legitimate'
    assert 0.0 <= data['probability'] <= 0.5


def test_predict_spam_email(client):
    email = "Congratulations!!! You've won a free lottery. Click here: http://spam.link"
    resp = client.post('/predict', json={'email': email})
    data = resp.get_json()

    assert resp.status_code == 200
    assert 'label' in data and 'probability' in data
    assert data['label'] == 'spam'
    assert 0.5 <= data['probability'] <= 1
