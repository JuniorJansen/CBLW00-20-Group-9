import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import gc  # Garbage collector
from tqdm import tqdm  # For progress bars (install with: pip install tqdm)


# Define function to safely read CSV files with optimized memory usage
def safe_read_csv(file_path):
    try:
        # First read just the header to check columns
        header = pd.read_csv(file_path, nrows=0)

        # Define column types to optimize memory usage
        dtypes = {}
        for col in header.columns:
            dtypes[col] = 'category' if col in ['Crime type', 'Falls within', 'LSOA name', 'Month'] else 'object'

        return pd.read_csv(file_path, dtype=dtypes, low_memory=True)
    except pd.errors.EmptyDataError:
        print(f"Warning: Empty file found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return None


# Find all CSV files
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

# Process files in batches and filter for burglary data directly
chunk_size = 500  # Process this many files at a time
total_burglary_records = 0
metropolitan_burglary_records = 0
months = set()
locations = set()

# Create empty output file
with open("data/metropolitan_police_data.csv", "w") as f:
    pass  # Just create an empty file

for i in range(0, len(all_files), chunk_size):
    print(f"\nProcessing batch {i // chunk_size + 1} of {len(all_files) // chunk_size + 1}...")
    batch_files = all_files[i:i + chunk_size]

    batch_burglary_data = []
    empty_files = 0
    error_files = 0

    for file in tqdm(batch_files, desc="Reading files"):
        df = safe_read_csv(file)

        if df is None:
            error_files += 1
            continue

        if df.empty:
            empty_files += 1
            continue

        # Only keep burglary records to save memory
        if 'Crime type' in df.columns:
            burglary_records = df[df['Crime type'].str.contains('burglary', case=False, na=False)]
            total_burglary_records += len(burglary_records)

            # Filter for Metropolitan Police Service
            if 'Falls within' in df.columns:
                metro_burglary = burglary_records[
                    burglary_records['Falls within'].str.contains('Metropolitan Police Service', case=False, na=False)
                ]

                if not metro_burglary.empty:
                    batch_burglary_data.append(metro_burglary)
                    metropolitan_burglary_records += len(metro_burglary)

                    # Collect unique months and locations
                    if 'Month' in metro_burglary.columns:
                        months.update(metro_burglary['Month'].unique())

                    if 'LSOA name' in metro_burglary.columns:
                        locations.update(metro_burglary['LSOA name'].unique())

        # Explicitly delete dataframe to free memory
        del df

    print(f"Batch summary: {empty_files} empty files, {error_files} error files")

    # Save this batch's results to file
    if batch_burglary_data:
        batch_df = pd.concat(batch_burglary_data, ignore_index=True)
        # Append to the output file
        batch_df.to_csv("metropolitan_police_data.csv", mode='a',
                        header=not os.path.getsize("metropolitan_police_data.csv"), index=False)
        print(f"Added {len(batch_df)} Metropolitan Police burglary records to output file")

        # Clear memory
        del batch_df
        del batch_burglary_data
        gc.collect()

# Final summary
print("\nProcessing complete!")
print(f"Total Burglary Cases (all forces): {total_burglary_records}")
print(f"Metropolitan Police Burglary Cases: {metropolitan_burglary_records}")

if months:
    print(f"Date Range: {min(months)} to {max(months)}")

if locations:
    print(f"Unique Locations: {len(locations)}")

print(f"Results saved to metropolitan_police_data.csv")