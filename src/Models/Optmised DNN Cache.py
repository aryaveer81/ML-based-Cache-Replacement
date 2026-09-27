import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
import time
import json
from pathlib import Path

pd.set_option('display.float_format', '{:.13f}'.format)
project_root = Path(__file__).resolve().parent.parent.parent
file_path = project_root / "NN_training_processed_dataset_Final.csv"
data = pd.read_csv(file_path)

X_features = data[['block_number', 'spatial_locality_score', 'mean_time_interval', 'last_access_time']].values
y_labels = data['hit_miss'].values
raw_blocks = data['block_number'].values

split_idx = int(len(data) * 0.7)
X_train_raw = X_features[:split_idx]
X_test_raw = X_features[split_idx:]
y_train_np = y_labels[:split_idx]
y_test_np = y_labels[split_idx:]
blocks_test = raw_blocks[split_idx:]

# Scaler correctly fitted ONLY on training data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

# 1. OPTIMIZATION: Mini-Batching for better convergence
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)

class DeepNeuralNet(nn.Module):
    def __init__(self, input_size):
        super(DeepNeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = DeepNeuralNet(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

model.eval()
with torch.no_grad():
    y_pred = (model(X_test) >= 0.5).float()
accuracy = accuracy_score(y_test_np, y_pred.numpy())
print(f"Test Accuracy: {accuracy * 100:.2f}%")

cache_size = 1000
cache = {}  # Format: {block_no: {'score': ml_score, 'timestamp': t}}
hits, misses = 0, 0
start_time = time.time()
decay_alpha = 0.0001 # 2. OPTIMIZATION: Priority decay factor

with torch.no_grad():
    for t in range(len(blocks_test)):
        block_no = blocks_test[t]
        if block_no in cache:
            hits += 1
            cache[block_no]['timestamp'] = t # Update access time on hit
        else:
            misses += 1
            if len(cache) >= cache_size:
                # 3. OPTIMIZATION: Dynamic Eviction (Score - Age Penalty)
                evict_block = min(cache, key=lambda k: cache[k]['score'] - (decay_alpha * (t - cache[k]['timestamp'])))
                del cache[evict_block]
            
            block_tensor = X_test[t].unsqueeze(0)
            new_score = model(block_tensor).item()
            cache[block_no] = {'score': new_score, 'timestamp': t}

execution_time = time.time() - start_time
hit_ratio = hits / (hits + misses) if (hits + misses) > 0 else 0

print(f"Algorithm: Optimized_Deep_NN | Hit Ratio: {hit_ratio:.4f} | Exec Time: {execution_time:.4f}s")