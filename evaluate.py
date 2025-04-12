import torch
import numpy as np
import os
from model import LogisticRegressionModel
from preprocessing import load_and_preprocess_data
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Function to load a saved model
def load_model(path):
    model_state = torch.load(path)
    model = LogisticRegressionModel(input_dim=model_state['input_dim'])
    model.load_state_dict(model_state['model_state_dict'])
    return model

# Load and preprocess data with the same max_features as in training
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/emails.csv", 1500)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Load the model using the load function
model = load_model("model.pth")
model.eval()

# Make predictions
with torch.no_grad():
    pred_probs = model(X_test_tensor).numpy()
    predicted_labels = (pred_probs > 0.5).astype(int)

# Calculate and print metrics
print("Classification Report:")
print(classification_report(y_test, predicted_labels))

# Add confusion matrix for better understanding
conf_matrix = confusion_matrix(y_test, predicted_labels)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nMatrix Interpretation:")
print(f"True Negatives: {conf_matrix[0,0]} (correctly classified as non-spam)")
print(f"False Positives: {conf_matrix[0,1]} (non-spam incorrectly classified as spam)")
print(f"False Negatives: {conf_matrix[1,0]} (spam incorrectly classified as non-spam)")
print(f"True Positives: {conf_matrix[1,1]} (correctly classified as spam)")

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, pred_probs)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, pred_probs)
average_precision = average_precision_score(y_test, pred_probs)

# Plot ROC curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Plot Precision-Recall curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {average_precision:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.savefig('model_evaluation.png')
plt.close()

print("\nModel architecture details:")
print(f"Input dimension: {model.input_dim}")
print(f"Hidden layers: {model.hidden_dims}")
print(f"Dropout rate: {model.dropout_rate}")
print("\nEvaluation plots saved to 'model_evaluation.png'")
