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

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_features)

window_size = 10
def create_sequences(X, y, blocks, time_steps):
    Xs, ys, block_seq = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
        block_seq.append(blocks[i + time_steps])
    return np.array(Xs), np.array(ys), np.array(block_seq)

X_seq, y_seq, blocks_seq = create_sequences(X_scaled, y_labels, raw_blocks, window_size)

split_idx = int(len(X_seq) * 0.7)
X_train_np = X_seq[:split_idx]
X_test_np = X_seq[split_idx:]
y_train_np = y_seq[:split_idx]
y_test_np = y_seq[split_idx:]
blocks_test = blocks_seq[split_idx:]

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

# PyTorch DataLoader for memory efficiency
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# 2. LSTM Architecture
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
        out = out[:, -1, :]  
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

input_size = X_train.shape[2]
hidden_size = 32
num_layers = 2
model = LSTMNet(input_size, hidden_size, num_layers)

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
    y_pred_probs = model(X_test)
    y_pred = (y_pred_probs >= 0.5).float()
accuracy = accuracy_score(y_test_np, y_pred.numpy())
print(f"Test Accuracy: {accuracy * 100:.2f}%")

cache_size = 1000
cache = {}
hits = 0
misses = 0
start_time = time.time()

with torch.no_grad():
    for i in range(len(blocks_test)):
        block_no = blocks_test[i]
        if block_no in cache:
            hits += 1
        else:
            misses += 1
            if len(cache) >= cache_size:
                evict_block = min(cache, key=cache.get)
                del cache[evict_block]
            block_tensor = X_test[i].unsqueeze(0)
            new_score = model(block_tensor).item()
            cache[block_no] = new_score

execution_time = time.time() - start_time
hit_ratio = hits / (hits + misses) if (hits + misses) > 0 else 0

results = {
    "Algorithm": "LSTM_Sequence",
    "Cache_Size": cache_size,
    "Test_Accuracy": round(accuracy, 4),
    "Hit_Ratio": round(hit_ratio, 4),
    "Execution_Time_Sec": round(execution_time, 4)
}

results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)
results_file = results_dir / "baseline_metrics.json"

with open(results_file, "a") as f:
    json.dump(results, f)
    f.write("\n")

print(json.dumps(results, indent=2))