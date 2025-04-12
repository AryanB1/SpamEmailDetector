import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from model import LogisticRegressionModel
from preprocessing import load_and_preprocess_data
from sklearn.utils.class_weight import compute_class_weight

# Load and preprocess the data
X_train, X_test, y_train, y_test = load_and_preprocess_data("data/emails.csv", 1500)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Calculate class weights to address class imbalance
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train), 
    y=y_train
)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

sample_weights = torch.tensor(
    [class_weights[int(label)] for label in y_train],
    dtype=torch.float32
).view(-1, 1)

# Create model - simplified since configuration is now hardcoded
model = LogisticRegressionModel(input_dim=X_train.shape[1])

criterion = nn.BCELoss(reduction='none')
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Add learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

num_epochs = 50
batch_size = 64
num_samples = X_train.shape[0]
num_batches = (num_samples - 1) // batch_size + 1

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    indices = torch.randperm(num_samples)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        X_batch = X_train_tensor[batch_indices]
        y_batch = y_train_tensor[batch_indices]
        weights_batch = sample_weights[batch_indices]
        
        outputs = model(X_batch)
        
        batch_loss = criterion(outputs, y_batch)
        weighted_loss = (batch_loss * weights_batch).mean()
        
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        
        epoch_loss += weighted_loss.item()
    
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
    
    # Update learning rate based on performance
    scheduler.step(avg_epoch_loss)

# Save model with its parameters
def save_model(model, path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
            
    # Save model parameters along with hyperparameters
    model_state = {
        'model_state_dict': model.state_dict(),
        'input_dim': model.input_dim,
        'hidden_dims': model.hidden_dims,
        'dropout_rate': model.dropout_rate
    }
    torch.save(model_state, path)

# Save the model
save_model(model, "model.pth")

# Evaluate model on test set
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_binary = (y_pred > 0.5).float()
    accuracy = (y_pred_binary == y_test_tensor).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")
