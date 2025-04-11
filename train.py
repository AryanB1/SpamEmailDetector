import torch
import torch.nn as nn
import torch.optim as optim
from model import LogisticRegressionModel
from preprocessing import load_and_preprocess_data

X_train, X_test, y_train, y_test = load_and_preprocess_data("data/emails.csv")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

model = LogisticRegressionModel(input_dim=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pth")
