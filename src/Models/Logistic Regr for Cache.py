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

project_root = Path(__file__).resolve().parent.parent.parent
file_path = project_root / "NN_training_processed_dataset_Final.csv"
data = pd.read_csv(file_path)

X_features = data[['block_number', 'spatial_locality_score', 'mean_time_interval', 'last_access_time']].values
y_labels = data['hit_miss'].values
raw_blocks = data['block_number'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_features)

split_idx = int(len(data) * 0.7)
X_train_np = X_scaled[:split_idx]
X_test_np = X_scaled[split_idx:]
y_train_np = y_labels[:split_idx]
y_test_np = y_labels[split_idx:]
blocks_test = raw_blocks[split_idx:]

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

input_size = X_train.shape[1]
model = LogisticRegressionModel(input_size)
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

results = {
    "Algorithm": "Logistic_Regression",
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