import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle
import time
import numpy as np
import torch.nn.functional as F

# Load dataset from a pickle file
def load_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    return data

# Assuming the data is a dictionary with 'inputs' and 'targets' keys
data = load_data('./human_data/inferno/pickle/inferno_10l_speed_formatted.pickle')
data = data[0]
inputs = torch.tensor(np.array(data[0]), dtype=torch.float32)
targets = torch.tensor(np.array(data[2]), dtype=torch.float32)

# Create a dataset
dataset = TensorDataset(inputs, targets)

# Split the dataset into training (80%), validation (10%), and test (10%) sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoader for each dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=46, shuffle=False)

# Define the neural network
class RegressionNet(nn.Module):
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.input_layer = nn.Linear(24, 512)
        self.hidden_layer1 = nn.Linear(512, 256)
        self.hidden_layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 2)

        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)
        self.drop3 = nn.Dropout(p=0.1)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.drop1(x)
        x = F.relu(self.hidden_layer1(x))
        x = self.drop2(x)
        x = F.relu(self.hidden_layer2(x))
        x = self.drop3(x)
        x = F.tanh(self.output_layer(x))
        return x

# Define a custom weighted MSE loss function
class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, output, target):
        # Calculate the absolute differences
        diff = torch.abs(output - target)
        # Apply a weighting function to emphasize smaller differences
        weights = torch.exp(-diff)
        # Compute the weighted MSE loss
        loss = (weights * (output - target) ** 2).mean()
        return loss

# Check if GPU is available and move tensors and model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to GPU
model = RegressionNet().to(device)

# Move data to GPU
inputs = inputs.to(device)
targets = targets.to(device)

# Update dataset with data moved to GPU
dataset = TensorDataset(inputs, targets)
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Update DataLoader with datasets moved to GPU
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Define the loss function and the optimizer
criterion = WeightedMSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Training loop with validation and time tracking
num_epochs = 300
start_time = time.time()
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    train_loss = 0.0
    for batch_inputs, batch_targets in train_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Step the scheduler
    scheduler.step(val_loss)
    
    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - start_time
    epoch_time = epoch_end_time - epoch_start_time
    remaining_time = epoch_time * (num_epochs - epoch - 1)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, '
          f'Time: {elapsed_time:.2f}s, ETA: {remaining_time:.2f}s')

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_inputs, batch_targets in test_loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f'Test Loss: {test_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'results/AAAI/BC_actor.pth')
