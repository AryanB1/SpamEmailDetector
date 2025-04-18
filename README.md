# Spam Email Classifier

A machine learning model built with PyTorch that classifies emails as spam or legitimate using natural language processing techniques.

## Project Overview

This project implements a neural network-based email classifier that can accurately detect spam emails with 98.6% accuracy, and a 97% F1 score. The model uses a combination of TF-IDF text features and engineered features to make predictions.

### Key Features

- Text preprocessing with NLTK for cleaning and normalization
- Feature engineering to extract meaningful patterns from emails
- TF-IDF vectorization for text representation
- Multi-layer neural network using PyTorch
- Balanced training to handle class imbalance
- Comprehensive evaluation metrics and visualizations
- REST API for real-time email classification

## Model Architecture

The model uses a multi-layer neural network with:

- Input layer matching feature dimensions
- Three hidden layers with ReLU activation and Batch Normalization
- Dropout regularization to prevent overfitting
- Output layer with sigmoid activation for binary classification

## Evaluation Results

The model's performance can be understood through the following visualizations:

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
