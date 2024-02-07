#%%
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re

def read_processed_csv(root_folder):
    """
    Returns:
    - List of DataFrames: A list containing DataFrames read from CSV files in 'PeaktoPeakValues' folders.
    """
    processed_dict = {}

    # Look for CSV files
    csv_files = [file for file in os.listdir(root_folder) if file.endswith('.csv')]
    general_pattern = re.compile(r'RCT\d+_(\d+p)?_(DF|PF)\.csv')
    trials = []
    # Read CSV files into DataFrames
    for csv_file in csv_files:
        match = general_pattern.match(csv_file)
        if match:
            # Extract the components from the pattern
            percentage, category = match.groups()

            # Check if percentage and category are present
            if percentage and category:
                key_string = f"{percentage}_{category}"
                print(key_string)
                trials.append(key_string)
        else:
            key_string = "rest"
            print(key_string)
            trials.append(key_string)
        csv_file_path = os.path.join(root_folder, csv_file)
        print(csv_file_path)

        df = pd.read_csv(csv_file_path)
        processed_dict[key_string] = df
    
    return processed_dict, trials

#userPath = os.path.expanduser('~') #gets path to user on any computer
#boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT001\RCT001_20240116\peak_to_peak_values' #puts user into box

#%%
keys_DF = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF']
keys_PF = ['rest', '5p_PF', '15p_PF', '30p_PF', '45p_PF']
trial_DF = 'Dorsi'
trial_PF = 'Plantar'

import pickle


def NormalizedPlots(values, save_path, keys, SID, trial_type):

    SID_rest = values[keys[0]].pivot(index='Amplitude', columns='Muscle', values='PeakToPeakValue')
    max_rest = SID_rest.max()
    SID_5p = values[keys[1]].pivot(index='Amplitude', columns='Muscle', values='PeakToPeakValue')
    max_5p = SID_5p.max()
    SID_15p = values[keys[2]].pivot(index='Amplitude', columns='Muscle', values='PeakToPeakValue')
    max_15p = SID_15p.max()
    SID_30p = values[keys[3]].pivot(index='Amplitude', columns='Muscle', values='PeakToPeakValue')
    max_30p = SID_30p.max()
    SID_45p = values[keys[4]].pivot(index='Amplitude', columns='Muscle', values='PeakToPeakValue')
    max_45p = SID_45p.max()

    muscles_org = ['L Gastrocs', 'L Rectus Femoris', 'L Semitendinosus','L Soleus','L Tibialis Anterior','L Vastus Lateralis',
        'R Gastrocs', 'R Rectus Femoris', 'R Semitendinosus', 'R Soleus', 'R Tibialis Anterior',  'R Vastus Lateralis']

    maximum_values = [max_rest, max_5p, max_15p, max_30p, max_45p]
    ToSearchMax = pd.DataFrame(data=dict(zip(keys, maximum_values)), index=muscles_org)
    NormalMax = ToSearchMax.max(axis=1)

    for i, muscle in enumerate(muscles_org):
        print("MUSCLE: ", muscle)
        SID_rest[muscle] = SID_rest[muscle] / NormalMax[muscle]
        SID_5p[muscle] = SID_5p[muscle] / NormalMax[muscle]
        SID_15p[muscle] = SID_15p[muscle] / NormalMax[muscle]
        SID_30p[muscle] = SID_30p[muscle] / NormalMax[muscle]
        SID_45p[muscle] = SID_45p[muscle] / NormalMax[muscle]

    dfs = [SID_rest, SID_5p, SID_15p,
        SID_30p, SID_45p]  

    colors = ['gold', 'skyblue', 'olivedrab', 'teal', 'indigo']

    # Plot each column together on the same plot across all tables
    for col in SID_rest.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, df in enumerate(dfs):
            ax.plot(df.index, df[col], color = colors[i], label = keys[i], alpha = 0.8)

        ax.set_title(f'{col} {SID} {trial_type}')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Normalized Values')
        ax.legend()
        save_title = f'\{col} {SID}_{trial_type}.png'

        # Save the figure
        fig.savefig(boxPath + save_title)

    fileName = f'\{SID}_{trial_type}.pkl'
    with open(boxPath + fileName, 'wb') as file:
        pickle.dump(dfs, file)
    


#%%
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT001\RCT001_20240116\peak_to_peak_values' #puts user into box
values, RCT001_trials = read_processed_csv(boxPath)
NormalizedPlots(values, boxPath, keys_DF, 'RCT001', trial_DF)
NormalizedPlots(values, boxPath, keys_PF, 'RCT001', trial_PF)


# %% #NEED TO EDIT CODE TO ACCOUNT FOR MISSING TRIALS
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT002\RCT002_20240118\peak_to_peak_values' #puts user into box
values, RCT002_trials = read_processed_csv(boxPath)
dorsi_keys = [item for item in keys_DF if item in RCT002_trials]
plantar_keys = [item for item in keys_PF if item in RCT002_trials]
NormalizedPlots(values, boxPath, dorsi_keys, 'RCT002', trial_DF)
NormalizedPlots(values, boxPath, plantar_keys, 'RCT002', trial_PF)


# %%
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT003\RCT003_20240119\peak_to_peak_values' #puts user into box
values, RCT003_trials = read_processed_csv(boxPath)
NormalizedPlots(values, boxPath, keys_DF, 'RCT003', trial_DF)
NormalizedPlots(values, boxPath, keys_PF, 'RCT003', trial_PF)


# %%
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT004\RCT004_20240125\peak_to_peak_values' #puts user into box
values, RCT004_trials = read_processed_csv(boxPath)
NormalizedPlots(values, boxPath, keys_DF, 'RCT004', trial_DF)
NormalizedPlots(values, boxPath, keys_PF, 'RCT004', trial_PF)


# %%
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT005\RCT005_20240126\peak_to_peak_values' #puts user into box
values, RCT005_trials = read_processed_csv(boxPath)
NormalizedPlots(values, boxPath, keys_DF, 'RCT005', trial_DF)
NormalizedPlots(values, boxPath, keys_PF, 'RCT005', trial_PF)


# %%  #NEED TO EDIT CODE TO ACCOUNT FOR MISSING TRIALS
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT006\RCT006_20240126\peak_to_peak_values' #puts user into box
values, RCT006_trials = read_processed_csv(boxPath)
NormalizedPlots(values, boxPath, keys_DF, 'RCT006', trial_DF)
NormalizedPlots(values, boxPath, keys_PF, 'RCT006', trial_PF)


# %% 
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT007\RCT007_20240129\peak_to_peak_values' #puts user into box
values, RCT007_trials = read_processed_csv(boxPath)
NormalizedPlots(values, boxPath, keys_DF, 'RCT007', trial_DF)
NormalizedPlots(values, boxPath, keys_PF, 'RCT007', trial_PF)

