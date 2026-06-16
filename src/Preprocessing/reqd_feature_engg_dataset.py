import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

pd.set_option('display.float_format', '{:.13f}'.format)

project_root = Path(__file__).resolve().parent.parent.parent

file_path = project_root / "cleaned_data.csv"

useful_columns = pd.read_csv(file_path)

useful_columns['timestamp'] = useful_columns['timestamp'].astype('int64') / (10**13)

useful_columns['read_write'] = useful_columns['read_write'].map({'R': 0, 'W': 1})

label_encoder = LabelEncoder()

useful_columns['pname'] = label_encoder.fit_transform(useful_columns['pname']) + 1

useful_columns = useful_columns.sort_values(by='timestamp').reset_index(drop=True)

useful_columns['last_access'] = useful_columns.groupby('block_number')['timestamp'].shift(1)

useful_columns['last_access_time'] = useful_columns['timestamp'] - useful_columns['last_access']

useful_columns['last_access_time'] = useful_columns['last_access_time'].fillna(10**6)

useful_columns['mean_time_interval'] = useful_columns.groupby('block_number')['last_access_time'].expanding().mean().reset_index(level=0, drop=True)

output_file_path = project_root / "updated_data.csv"
useful_columns.to_csv(output_file_path, index=False)