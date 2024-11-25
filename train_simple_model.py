import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dinosaur import Dinosaur
from obstacle_new import Obstacle
from batsymbol import Batsymb
from pygame import mixer
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class DinoModel(nn.Module):
#     def __init__(self):
#         super(DinoModel, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(6, 128),  # Increased size for initial layer
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 3)  # Output: [jump, duck, do nothing]
#         )

#     def forward(self, x):
#         return self.fc(x)

class DinoModel(nn.Module):
    def __init__(self):
        super(DinoModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 256),  # Increased input size and more neurons
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output: [jump, duck, do nothing]
        )
        # Softmax will be applied outside this model during action selection.

    def forward(self, x):
        return self.fc(x)
    
import pandas as pd

# Load the CSV into a DataFrame for easier manipulation
# Directory containing the training data files
train_data_folder = "D:/IITB/Second Year/Sem 3/PH227/ChromeDinoGame/Train_Data"

# List to hold dataframes
dataframes = []

# Iterate over all files in the training data folder
for filename in os.listdir(train_data_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(train_data_folder, filename)
        df = pd.read_csv(file_path, header=0)  # Read the CSV file, assuming the first row is the header
        dataframes.append(df)

# Concatenate all dataframes into one
df = pd.concat(dataframes, ignore_index=True)

# Drop any non-numeric columns (or handle them appropriately)
df = df.apply(pd.to_numeric, errors='coerce')  # Converts everything to numeric, NaN for errors

# Replace NaNs (optional, depending on your data)
df.fillna(0, inplace=True)

X = df.iloc[:, 1:7].values  # Features (assuming columns 1 to 5 are features)
y = df.iloc[:, 7].values  # Labels (assuming the action column is column 6)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.long).to(device)

# Split the dataset into training and testing
X_train, X_test = X[:int(0.8 * len(X))], X[int(0.2 * len(X)):]
y_train, y_test = y[:int(0.8 * len(y))], y[int(0.2 * len(y)):]

# train_dataset = TensorDataset(X_train, y_train)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model, loss function, and optimizer
model = DinoModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10000):
    model.train()  # Ensure model is in training mode
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch} Loss: {loss.item()}')

from torch.utils.data import DataLoader, TensorDataset


# Wrap the testing data into a DataLoader
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluate the model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:  # Loop through mini-batches
        outputs = model(X_batch)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        total += y_batch.size(0)  # Accumulate total samples
        correct += (predicted == y_batch).sum().item()  # Compare predictions to true labels

accuracy = 100 * correct / total
print(f'Accuracy: {accuracy}%')


# Save the trained model
torch.save(model.state_dict(), 'model.pth')
