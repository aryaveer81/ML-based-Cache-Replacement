import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import numpy as np
import time
from pathlib import Path

pd.set_option('display.float_format', '{:.13f}'.format)
project_root = Path(__file__).resolve().parent.parent.parent
file_path = project_root / "NN_training_processed_dataset_Final.csv"
data = pd.read_csv(file_path)

X_features = data[['block_number', 'spatial_locality_score', 'mean_time_interval', 'last_access_time']].values
y_labels = data['hit_miss'].values
raw_blocks = data['block_number'].values

# 1. OPTIMIZATION: Prevent Data Leakage (Split THEN Scale)
split_idx_raw = int(len(data) * 0.7)
X_train_raw = X_features[:split_idx_raw]
X_test_raw = X_features[split_idx_raw:]

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Recombine temporarily for sequence generation to maintain the sliding window across the split border
X_scaled = np.vstack((X_train_scaled, X_test_scaled))

window_size = 10
def create_sequences(X, y, blocks, time_steps):
    Xs, ys, block_seq = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
        block_seq.append(blocks[i + time_steps])
    return np.array(Xs), np.array(ys), np.array(block_seq)

X_seq, y_seq, blocks_seq = create_sequences(X_scaled, y_labels, raw_blocks, window_size)

# 2. Adjust split index for sequence offset
split_idx = split_idx_raw - window_size
X_train = torch.tensor(X_seq[:split_idx], dtype=torch.float32)
X_test = torch.tensor(X_seq[split_idx:], dtype=torch.float32)
y_train = torch.tensor(y_seq[:split_idx], dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_seq[split_idx:], dtype=torch.float32).unsqueeze(1)
blocks_test = blocks_seq[split_idx:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.sigmoid(self.fc(out[:, -1, :]))
        return out

model = LSTMNet(X_train.shape[2], hidden_size=32, num_layers=2)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
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
cache_size = 1000
cache = {} 
hits, misses = 0, 0
start_time = time.time()
decay_alpha = 0.0001 

with torch.no_grad():
    for t in range(len(blocks_test)):
        block_no = blocks_test[t]
        if block_no in cache:
            hits += 1
            cache[block_no]['timestamp'] = t
        else:
            misses += 1
            if len(cache) >= cache_size:
                # 3. OPTIMIZATION: Priority decay
                evict_block = min(cache, key=lambda k: cache[k]['score'] - (decay_alpha * (t - cache[k]['timestamp'])))
                del cache[evict_block]
            
            block_tensor = X_test[t].unsqueeze(0)
            new_score = model(block_tensor).item()
            cache[block_no] = {'score': new_score, 'timestamp': t}

execution_time = time.time() - start_time
hit_ratio = hits / (hits + misses) if (hits + misses) > 0 else 0

print(f"Algorithm: Optimized_LSTM | Hit Ratio: {hit_ratio:.4f} | Exec Time: {execution_time:.4f}s")