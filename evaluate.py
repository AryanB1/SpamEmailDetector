import torch
from model import LogisticRegressionModel
from preprocessing import load_and_preprocess_data
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = load_and_preprocess_data("data/emails.csv")
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

model = LogisticRegressionModel(input_dim=X_test.shape[1])
model.load_state_dict(torch.load("model.pth"))
model.eval()

with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_labels = (predictions > 0.5).int().numpy()

print(classification_report(y_test, predicted_labels))
