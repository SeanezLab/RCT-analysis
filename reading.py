import csv
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def read_csv_and_filter_floats(file_path):
    filtered_data = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        csv_reader = csv.reader((line.replace('\0', '') for line in file))
        for row in csv_reader:
            filtered_row = [float(value) if is_float(value) else None for value in row]

            # Check if the filtered row has the expected number of columns (e.g., 13)
            if len(filtered_row) == 13:
                filtered_data.append(filtered_row)

    return filtered_data


# Example usage:
file_path = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT008\RCT008_20240208\Error\RCT008_20240208_RC_E00_08_5p_DF.csv'
filtered_data = read_csv_and_filter_floats(file_path)

# Create a DataFrame with 13 columns
df = pd.DataFrame(filtered_data)

# Specify the output CSV file path
output_file_path = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT008\RCT008_20240208\Error\FixedRCT008_20240208_RC_E00_08_5p_DF.csv'

# Save the DataFrame to the specified CSV file
df.to_csv(output_file_path, index=False)

print(f"DataFrame saved to: {output_file_path}")