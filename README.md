# Spam Email Classifier

This project is a machine learning model that uses logistic regression to classify emails as spam or legitimate. The classifier has 98.6% accuracy, and a 97% F1 score.

## Key Features

- Text preprocessing with NLTK for cleaning and normalization
- TF-IDF vectorization for text representation
- Multi-layer neural network using PyTorch
- REST API for email classification

## Model Architecture

The model uses a multi-layer neural network with:

- Input layer matching feature dimensions
- Three hidden layers with ReLU activation and Batch Normalization
- Dropout regularization to prevent overfitting
- Output layer with sigmoid activation for binary classification

## Evaluation Results

The model's performance can be seen through the following visualizations:

### Confusion Matrix

![Confusion Matrix](visualizations/confusion_matrix.png)

The confusion matrix shows:

- True Negatives (top left): Correctly identified non-spam emails
- False Positives (top right): Non-spam emails incorrectly flagged as spam
- False Negatives (bottom left): Spam emails that were missed
- True Positives (bottom right): Correctly identified spam emails

### Probability Distribution

![Probability Distribution](visualizations/probability_distribution.png)

This histogram shows how the model distributes probability scores for spam and non-spam emails.

## API

To run the project's REST API use the following command:

```bash
python app/api.py
```

Emails that need to be classified can be routed in a POST request to the `/predict` endpoint with the email provided in the request body.
