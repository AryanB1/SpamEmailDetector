import torch
import os
from app.model import LogisticRegressionModel
from app.preprocessing import load_and_preprocess_data
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def evaluate_model(
    model_path="./app/model/model.pth",
    data_path="./data/emails.csv",
    max_features=1500
):
    vis_dir = "visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    _, X_test, _, y_test = load_and_preprocess_data(data_path, max_features)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    model_state = torch.load(model_path)
    input_dim = model_state['input_dim']
    model = LogisticRegressionModel(input_dim=input_dim)
    model.load_state_dict(model_state['model_state_dict'])
    model.eval()

    # Evaluate
    with torch.no_grad():
        pred_probs = model(X_test_tensor).numpy().flatten()
        predicted_labels = (pred_probs > 0.5).astype(int)

    # Confusion Matrix with annotations
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, predicted_labels)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    labels = ['Non-Spam', 'Spam']
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)

    plt.text(0, 0, format(cm[0, 0], 'd'), ha="center", va="center",
             color="white" if cm[0, 0] > cm.max()/2 else "black", fontsize=14)
    plt.text(1, 0, format(cm[0, 1], 'd'), ha="center", va="center",
             color="white" if cm[0, 1] > cm.max()/2 else "black", fontsize=14)
    plt.text(0, 1, format(cm[1, 0], 'd'), ha="center", va="center",
             color="white" if cm[1, 0] > cm.max()/2 else "black", fontsize=14)
    plt.text(1, 1, format(cm[1, 1], 'd'), ha="center", va="center",
             color="white" if cm[1, 1] > cm.max()/2 else "black", fontsize=14)

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    confusion_matrix_path = os.path.join(vis_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Probability Distribution Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(
        pred_probs[y_test == 0],
        bins=20,
        alpha=0.6,
        label='Non-spam',
        color='green'
    )
    plt.hist(
        pred_probs[y_test == 1],
        bins=20,
        alpha=0.6,
        label='Spam',
        color='red'
    )
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Prediction Probability Distribution', fontsize=14)
    plt.legend(loc="upper center", fontsize=12)
    prob_dist_path = os.path.join(vis_dir, "probability_distribution.png")
    plt.tight_layout()
    plt.savefig(prob_dist_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Summary metrics
    print("\n===== EMAIL CLASSIFIER EVALUATION =====")
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_labels))

    print("\nConfusion Matrix Interpretation:")
    print(f"True Negatives: {cm[0, 0]} (correctly identified as non-spam)")
    print(f"False Positives: {cm[0, 1]} (incorrectly flagged as spam)")
    print(f"False Negatives: {cm[1, 0]} (spam missed)")
    print(f"True Positives: {cm[1, 1]} (correctly identified as spam)")

    print(f"\nModel input dimension: {input_dim}")
    print(f"\nVisualizations saved to '{vis_dir}' directory")

    metrics = {
        'confusion_matrix': cm,
        'classification_report': classification_report(
            y_test, predicted_labels, output_dict=True
        )
    }

    vis_paths = {
        'confusion_matrix': confusion_matrix_path,
        'prob_dist': prob_dist_path
    }

    return metrics, vis_paths


if __name__ == "__main__":
    evaluate_model()
