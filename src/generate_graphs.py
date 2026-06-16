import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "results" / "baseline_metrics.json"


data = []
with open(file_path, 'r') as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

df = pd.DataFrame(data)


df = df.sort_values('Hit_Ratio', ascending=False).drop_duplicates('Algorithm').reset_index(drop=True)

standard_algos = ['LRU', 'LFU', 'FIFO']
ml_algos = ['Logistic_Regression', 'Deep_NN', 'LSTM_Sequence']

def assign_category(algo):
    if algo in standard_algos:
        return 'Standard Baseline'
    elif algo in ml_algos:
        return 'ML Model'
    elif algo == 'Belady_MIN':
        return 'Theoretical Optimal (MIN)'
    return 'Other'

df['Category'] = df['Algorithm'].apply(assign_category)


sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12})

output_dir = project_root / "results" / "graphs"
output_dir.mkdir(exist_ok=True)



plt.figure(figsize=(12, 7))
order = df.sort_values('Hit_Ratio')['Algorithm']
ax = sns.barplot(x='Algorithm', y='Hit_Ratio', hue='Category', data=df, order=order, dodge=False)

plt.title('Cache Hit Ratio by Algorithm (Cache Size: 1000)', fontsize=16, pad=15)
plt.ylabel('Hit Ratio', fontsize=14)
plt.xlabel('Algorithm', fontsize=14)
plt.xticks(rotation=45, ha='right')


belady_val = df[df['Algorithm'] == 'Belady_MIN']['Hit_Ratio'].values
if len(belady_val) > 0:
    plt.axhline(y=belady_val[0], color='black', linestyle='--', alpha=0.7, label='Optimal Ceiling')

plt.legend(title='Algorithm Type')
plt.tight_layout()
plt.savefig(output_dir / 'hit_ratio_comparison.png', dpi=300)
plt.close()



plt.figure(figsize=(12, 7))
order_time = df.sort_values('Execution_Time_Sec')['Algorithm']
sns.barplot(x='Algorithm', y='Execution_Time_Sec', hue='Category', data=df, order=order_time, dodge=False)

plt.title('Simulation Execution Time (Latency Overhead)', fontsize=16, pad=15)
plt.ylabel('Execution Time (Seconds)', fontsize=14)
plt.xlabel('Algorithm', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yscale('log') 

plt.legend(title='Algorithm Type')
plt.tight_layout()
plt.savefig(output_dir / 'execution_time_comparison.png', dpi=300)
plt.close()


plt.figure(figsize=(10, 7))
sns.scatterplot(x='Execution_Time_Sec', y='Hit_Ratio', hue='Category', style='Category', s=200, data=df)


for i in range(df.shape[0]):
    plt.text(x=df.Execution_Time_Sec[i] * 1.1, 
             y=df.Hit_Ratio[i], 
             s=df.Algorithm[i], 
             fontsize=10)

plt.title('Trade-off: Execution Time vs. Hit Ratio', fontsize=16, pad=15)
plt.ylabel('Hit Ratio (Higher is better)', fontsize=14)
plt.xlabel('Execution Time in Seconds (Lower is better)', fontsize=14)
plt.xscale('log') 

plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(output_dir / 'latency_vs_hit_ratio.png', dpi=300)
plt.close()

print(f"Graphs successfully generated and saved to: {output_dir}")