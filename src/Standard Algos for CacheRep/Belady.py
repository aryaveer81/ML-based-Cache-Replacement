import pandas as pd
from collections import defaultdict, deque
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
block_numbers = df['block_number'].tolist()

# Precompute future positions to avoid O(N) lookahead penalty
positions = defaultdict(deque)
for i, block in enumerate(block_numbers):
    positions[block].append(i)

start_time = time.time()

for i, block in enumerate(block_numbers):
    # Consume the current access
    positions[block].popleft()
    
    if block in cache:
        hits += 1
    else:
        misses += 1
        if len(cache) >= cache_size:
            furthest_block = None
            max_future_index = -1
            
            for c_block in cache:
                # If a block is never accessed again, it is the perfect eviction candidate
                if not positions[c_block]:
                    furthest_block = c_block
                    break
                
                # Otherwise, find the block whose next access is furthest away
                next_use = positions[c_block][0]
                if next_use > max_future_index:
                    max_future_index = next_use
                    furthest_block = c_block
            
            cache.remove(furthest_block)
        cache.add(block)

execution_time = time.time() - start_time
hit_ratio = hits / (hits + misses) if (hits + misses) > 0 else 0

results = {
    "Algorithm": "Belady_MIN",
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