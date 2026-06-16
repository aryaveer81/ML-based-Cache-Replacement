import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

pd.set_option('display.float_format', '{:.13f}'.format)
project_root = Path(__file__).resolve().parent.parent.parent
file_path = project_root / "updated_data.csv"
data = pd.read_csv(file_path, nrows=100000)
useful_columns = data[['timestamp', 'pname', 'block_number', 'read_write', 'last_access_time', 'mean_time_interval']].copy()
useful_columns = useful_columns.sort_values(by='timestamp').reset_index(drop=True)

def calculate_spatial_locality_fast(block_numbers):
    n = len(block_numbers)
    scores = np.zeros(n)
    blocks = block_numbers.values
    for i in range(1, n):
        start_idx = max(0, i - 5000)
        past_blocks = blocks[start_idx:i]
        distances = np.abs(past_blocks - blocks[i])
        score = np.sum(distances <= 64) * 5 + np.sum((distances > 64) & (distances <= 256)) * 2.5
        scores[i] = score / len(past_blocks) if len(past_blocks) > 0 else 0
    return scores

useful_columns['spatial_locality_score'] = calculate_spatial_locality_fast(useful_columns['block_number'])
useful_columns['mean_time_interval'] = useful_columns['mean_time_interval'].replace([np.inf, -np.inf], 10**6).fillna(10**6)
useful_columns['last_access_time'] = useful_columns['last_access_time'].replace([np.inf, -np.inf], 10**6).fillna(10**6)

cache_size = 1000
cache = set()
hits = 0
misses = 0
block_access_sequence = useful_columns['block_number'].astype(int).tolist()
hit_miss_labels = []

for current_index in range(len(block_access_sequence)):
    block_number = block_access_sequence[current_index]
    if block_number in cache:
        hits += 1
        hit_miss_labels.append(1)
    else:
        misses += 1
        hit_miss_labels.append(0)
        if len(cache) == cache_size:
            future_uses = {block: float('inf') for block in cache}
            max_lookahead = min(10000, len(block_access_sequence) - current_index - 1)
            for i in range(1, max_lookahead + 1):
                future_block = block_access_sequence[current_index + i]
                if future_block in future_uses and future_uses[future_block] == float('inf'):
                    future_uses[future_block] = i
            block_to_evict = max(future_uses, key=future_uses.get)
            cache.remove(block_to_evict)
        cache.add(block_number)

useful_columns['hit_miss'] = hit_miss_labels
scaler = MinMaxScaler()
useful_columns[['block_number', 'spatial_locality_score']] = scaler.fit_transform(useful_columns[['block_number', 'spatial_locality_score']])

output_file_path = project_root / "NN_training_processed_dataset_Final.csv"
useful_columns.to_csv(output_file_path, index=False)
print(f"Total Hits: {hits}, Total Misses: {misses}")