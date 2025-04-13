import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model import LogisticRegressionModel
from preprocessing import load_and_preprocess_data
from sklearn.utils.class_weight import compute_class_weight

# Get data
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/emails.csv", 1500)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

sample_weights = torch.tensor([class_weights[int(label)] for label in y_train], dtype=torch.float32).view(-1, 1)

model = LogisticRegressionModel(input_dim=X_train.shape[1])

criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

# Training configuration
num_epochs = 100
batch_size = 64
num_samples = X_train.shape[0]
best_loss = float('inf')
best_model_state = None
input_dim = X_train.shape[1]

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    indices = torch.randperm(num_samples)
    
    # Process mini-batches
    for i in range((num_samples - 1) // batch_size + 1):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Get batch data
        X_batch = X_train_tensor[batch_indices]
        y_batch = y_train_tensor[batch_indices]
        weights_batch = sample_weights[batch_indices]
        
        # Forward pass and compute weighted loss
        outputs = model(X_batch)
        batch_loss = criterion(outputs, y_batch)
        weighted_loss = (batch_loss * weights_batch).mean()
        
        # Backward pass
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        
        epoch_loss += weighted_loss.item()
    
    # Track progress
    avg_loss = epoch_loss / ((num_samples - 1) // batch_size + 1)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_model_state = model.state_dict().copy()
        print(f"New best model found at epoch {epoch+1} with loss: {best_loss:.4f}")
    
    # Update learning rate
    scheduler.step(avg_loss)

model_path = "model.pth"
# Only create directory if path contains a directory component
dirname = os.path.dirname(model_path)
if dirname:
    os.makedirs(dirname, exist_ok=True)

model.load_state_dict(best_model_state or model.state_dict())
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': input_dim  # Use the stored input dimension instead of accessing model.input_dim
}, model_path)
