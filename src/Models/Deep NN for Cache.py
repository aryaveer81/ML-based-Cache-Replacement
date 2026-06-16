import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)
X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)
class DeepNeuralNet(nn.Module):
    def __init__(self, input_size):
        super(DeepNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.sigmoid(self.out(x))
        return x
input_size = X_train.shape[1]
model = DeepNeuralNet(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
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
belady_hits = int(sum(y_test_np))
belady_misses = len(y_test_np) - belady_hits
belady_hit_ratio = belady_hits / len(y_test_np)
results = {
    "Algorithm": "Deep_NN",
    "Cache_Size": cache_size,
    "Test_Accuracy": round(accuracy, 4),
    "Hit_Ratio": round(hit_ratio, 4),
    "Belady_Hit_Ratio": round(belady_hit_ratio, 4),
    "Execution_Time_Sec": round(execution_time, 4)
}
results_dir = project_root / "results"
results_dir.mkdir(exist_ok=True)
results_file = results_dir / "baseline_metrics.json"
with open(results_file, "a") as f:
    json.dump(results, f)
    f.write("\n")
print(json.dumps(results, indent=2))