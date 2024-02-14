#%%
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

#%%
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
                trials.append(key_string)
        else:
            key_string = "rest"
            trials.append(key_string)
        csv_file_path = os.path.join(root_folder, csv_file)

        df = pd.read_csv(csv_file_path)
        processed_dict[key_string] = df

    print("trials: ", trials)
    return processed_dict, trials


def NormalizedPlots(values, save_path, keys, SID, trial_type):
    # Combine data from all trials into a single dataframe
    dfs = [values[key].pivot(index='Amplitude', columns='Muscle', values='PeakToPeakValue') for key in keys]
    all_trials_df = pd.concat(dfs, axis=0)

    # Calculate the maximum value for each muscle across all trials
    NormalMax = all_trials_df.max()
    print("normal max vals: ", NormalMax)

    # Normalize each trial against the maximum value of each muscle
    for muscle in all_trials_df.columns:
        all_trials_df[muscle] = all_trials_df[muscle]/NormalMax[muscle]
    
    # Add extra cloumn to dataframe with trial info
    trial_list = ['15p_DF', '15p_PF', '30p_DF', '45p_DF', '45p_PF', '5p_DF', '5p_PF', 'rest']
    new_trial_list = [item for item in trial_list for _ in range(6)]
    all_trials_df = all_trials_df.assign(TrialName = new_trial_list)
    print("new all_trials_df with trial labels: ", all_trials_df)
    
    # Group by 'TrialName' and iterate through each group
    # Specify the trial names you want to plot
    selected_trials_DF = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF']
    selected_trials_PF = ['rest', '5p_DF', '15p_PF', '30p_PF', '45p_PF']
    selected_trials_PF_RCT006 = ['rest', '5p_DF', '15p_PF', '45p_PF']

    for muscle in all_trials_df.columns[:-1]: # Exclude the last column (Trial name)
        # Filter the DataFrame based on selected trial names
        selected_df = all_trials_df[all_trials_df['TrialName'].isin(selected_trials_PF)]

        # Group by 'TrialName' and plot each group separately
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['pink', 'hotpink', 'deeppink', 'darkorchid', 
                  'mediumpurple'
                  ]

        for name, group, color in zip(reversed(selected_df['TrialName'].unique()), selected_df.groupby('TrialName'), colors):
            # Plot the values of the target muscle for each group
            ax.plot(group[1].index, group[1][muscle], label=name, color=color)

        #ax.set_title(f'{SID}_{muscle} Dorsiflexion')
        ax.set_title(f'{SID}_{muscle} Plantarflexion')
        ax.set_xlabel('Amplitude')
        ax.set_ylabel(f'{muscle} Recruitment')
        ax.legend()
        plt.show()
        save_title = f'\{SID}_{muscle}_{trial_type}.png'
        fig.savefig(save_path+save_title)

#%% RCT002 - MISSING TRIALS
# Adjust the keys based on the available trials
all_keys = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF','15p_PF', '30p_PF', '45p_PF'] # Used for normalization across all trials
trial_type = ['Dorsi', 'Plantar']
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT002\RCT002_20240118\peak_to_peak_values' #puts user into box
values, RCT002_trials = read_processed_csv(boxPath)

# Process the data for each trial type
NormalizedPlots(values, boxPath, all_keys, 'RCT002', trial_type[1]) # Uncomment to run plantar plots
#NormalizedPlots(values, boxPath, all_keys, 'RCT002', trial_type[0]) # Uncomment to run dorsi plots

# %% RCT001
all_keys = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF','5p_PF', '15p_PF', '30p_PF', '45p_PF'] # Used for normalization across all trials
trial_type = ['Dorsi', 'Plantar']
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT001\RCT001_20240116\new_plots'
values, RCT001_trials = read_processed_csv(boxPath)
#NormalizedPlots(values, boxPath, all_keys, 'RCT001', trial_type[0]) # Uncomment to run dorsi plots
NormalizedPlots(values, boxPath, all_keys, 'RCT001', trial_type[1]) # Uncomment to run plantar plots

# %% RCT003
all_keys = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF','5p_PF', '15p_PF', '30p_PF', '45p_PF']
trial_type = ['Dorsi', 'Plantar']
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT003\RCT003_20240119\new_plots'
values, RCT003_trials = read_processed_csv(boxPath)
NormalizedPlots(values, boxPath, all_keys, 'RCT003', trial_type[0]) # Uncomment to run dorsi plots
#NormalizedPlots(values, boxPath, all_keys, 'RCT003', trial_type[1]) # Uncomment to run plantar plots

#%% RCT004
all_keys = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF','5p_PF', '15p_PF', '30p_PF', '45p_PF']
trial_type = ['Dorsi', 'Plantar']
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT004\RCT004_20240125\new_plots'
values, RCT004_trials = read_processed_csv(boxPath)
#NormalizedPlots(values, boxPath, all_keys, 'RCT004', trial_type[0]) # Uncomment to run dorsi plots
NormalizedPlots(values, boxPath, all_keys, 'RCT004', trial_type[1]) # Uncomment to run plantar plots

# %% RCT005
all_keys = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF','5p_PF', '15p_PF', '30p_PF', '45p_PF']
trial_type = ['Dorsi', 'Plantar']
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT005\RCT005_20240126\new_plots'
values, RCT005_trials = read_processed_csv(boxPath)
#NormalizedPlots(values, boxPath, all_keys, 'RCT005', trial_type[0]) # Uncomment to run dorsi plots
NormalizedPlots(values, boxPath, all_keys, 'RCT005', trial_type[1]) # Uncomment to run plantar plots

#%% RCT006 - MISSING TRIALS
all_keys = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF','5p_PF', '15p_PF', '45p_PF']
trial_type = ['Dorsi', 'Plantar']
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT006\RCT006_20240126\new_plots'
values, RCT006_trials = read_processed_csv(boxPath)
#NormalizedPlots(values, boxPath, all_keys, 'RCT006', trial_type[0]) # Uncomment to run dorsi plots
NormalizedPlots(values, boxPath, all_keys, 'RCT006', trial_type[1]) # Uncomment to run plantar plots

#%% RCT007
all_keys = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF','5p_PF', '15p_PF', '30p_PF', '45p_PF']
trial_type = ['Dorsi', 'Plantar']
boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT007\RCT007_20240129\new_plots'
values, RCT007_trials = read_processed_csv(boxPath)
#NormalizedPlots(values, boxPath, all_keys, 'RCT007', trial_type[0]) # Uncomment to run dorsi plots
NormalizedPlots(values, boxPath, all_keys, 'RCT007', trial_type[1]) # Uncomment to run plantar plots

#%% RCT008