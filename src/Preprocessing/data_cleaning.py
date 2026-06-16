import pandas as pd
from pathlib import Path

pd.set_option('display.float_format', '{:.15f}'.format)

project_root = Path(__file__).resolve().parent.parent.parent

file_path = project_root / "cheetah.cs.fiu.edu-110108-113008.1.blkparse"

data = pd.read_csv(file_path, header=None, sep='\s+')

print("Original Data Shape:", data.shape)

print(data.head())

useful_columns = data[[0, 2, 3, 5]].copy()

useful_columns.columns = ['timestamp', 'pname', 'block_number', 'read_write']

useful_columns.sort_values('timestamp', inplace=True)

cleaned_data = useful_columns.drop_duplicates().copy()

print("Cleaned Data Shape:", cleaned_data.shape)

print(cleaned_data.head())

output_file_path = project_root / "cleaned_data.csv"

cleaned_data.to_csv(output_file_path, index=False)

print(f"Cleaned data saved to: {output_file_path}")