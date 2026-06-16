import pandas as pd
from collections import defaultdict
import time
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
file_path = project_root / "NN_training_processed_dataset_Final.csv"
df = pd.read_csv(file_path)
cache_size = 1000
cache = set()
hits = 0
misses = 0
block_access_frequency = defaultdict(int)
block_insertion_order = defaultdict(int)
block_numbers = df['block_number'].tolist()
start_time = time.time()
for current_index, block_number in enumerate(block_numbers):
    block_access_frequency[block_number] += 1
    if block_number in cache:
        hits += 1
    else:
        misses += 1
        if len(cache) >= cache_size:
            lfu_block = min(cache, key=lambda b: (block_access_frequency[b], block_insertion_order[b]))
            cache.remove(lfu_block)
            del block_access_frequency[lfu_block]
            del block_insertion_order[lfu_block]
        cache.add(block_number)
        block_insertion_order[block_number] = current_index
execution_time = time.time() - start_time
hit_ratio = hits / (hits + misses) if (hits + misses) > 0 else 0
results = {
    "Algorithm": "LFU",
    "Cache_Size": cache_size,
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