import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd

all_files = []
for folder in os.listdir('data'):
    folder_path = os.path.join('data', folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.csv') or file.endswith('.CSV'):
                all_files.append(os.path.join(folder_path, file))
print(f"Found {len(all_files)} CSV files.")
if len(all_files) == 0:
    raise ValueError("No CSV files found! Check if the 'data' folder contains subfolders with CSVs.")

df_list = [pd.read_csv(file, low_memory=False) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

print(df.shape)
print(df.columns)
print(df.head())
print(df.dtypes)
burglary_df = df[df['Crime type'].str.contains('burglary', case=False, na=False)]

burglary_df = burglary_df[burglary_df['Falls within'].str.contains('Metropolitan Police Service', case=False, na=False)]
burglary_df.to_csv("metropolitan_police_data.csv", index=False)
print("Total Burglary Cases:", burglary_df.shape[0])
print("Date Range:", burglary_df['Month'].min(), "to", burglary_df['Month'].max())
print("Unique Locations:", burglary_df['LSOA name'].nunique())