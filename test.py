import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    # Combine data from all trials
    dfs = [values[key].pivot(index='Amplitude', columns='Muscle', values='PeakToPeakValue') for key in keys]
    all_trials_df = pd.concat(dfs, axis=0)
    #print("all trials DF: ", all_trials_df)

    # Calculate the maximum value for each muscle across all trials
    NormalMax = all_trials_df.max()
    print("normal max vals: ", NormalMax)

    # Normalize each trial against the maximum value of each muscle
    for muscle in all_trials_df.columns:
        all_trials_df[muscle] = all_trials_df[muscle]/NormalMax[muscle]
    
    # Add extra cloumn to dataframe with trial info
    all_trials_df = all_trials_df.assign(TrialName = ['15p_DF', '15p_DF', '15p_DF', '15p_DF', '15p_DF', '15p_DF',
                                                      '15p_PF', '15p_PF', '15p_PF', '15p_PF', '15p_PF', '15p_PF',
                                                      '30p_DF', '30p_DF', '30p_DF', '30p_DF', '30p_DF', '30p_DF',
                                                      '30p_PF', '30p_PF', '30p_PF', '30p_PF', '30p_PF', '30p_PF',
                                                      '45p_DF', '45p_DF', '45p_DF', '45p_DF', '45p_DF', '45p_DF',
                                                      '45p_PF', '45p_PF', '45p_PF', '45p_PF', '45p_PF', '45p_PF',
                                                      '5p_DF', '5p_DF', '5p_DF', '5p_DF', '5p_DF', '5p_DF',
                                                      'rest','rest','rest','rest','rest','rest',
                                                      ])
    print("new all_trials_df with trial labels: ", all_trials_df)
    
# Group by 'TrialName' and iterate through each group
    target_muscle = 'L Soleus'
    # Specify the trial names you want to plot
    selected_trials_DF = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF']
    selected_trials_PF = ['rest', '15p_PF', '30p_PF', '45p_PF']

# Filter the DataFrame based on selected trial names
    # Filter the DataFrame based on selected trial names
    selected_df = all_trials_df[all_trials_df['TrialName'].isin(selected_trials_PF)]

# Group by 'TrialName' and plot each group separately
    fig, ax = plt.subplots(figsize=(10, 6))

    #colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_df['TrialName'].unique())))
    colors = ['pink', 'hotpink', 'deeppink', 'darkorchid']

    for name, group, color in zip(selected_df['TrialName'].unique(), selected_df.groupby('TrialName'), colors):
        # Plot the values of the target muscle for each group
        ax.plot(group[1].index, group[1][target_muscle], label=name, color=color)

    ax.set_title(f'{target_muscle} Plantarflexion')
    ax.set_xlabel('Amplitude')
    ax.set_ylabel(f'{target_muscle} Values')
    ax.legend()
    plt.show()

#%%
# Adjust the keys based on the available trials
all_keys = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF','15p_PF', '30p_PF', '45p_PF'] # Used for normalization across all trials

keys_DF = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF']
keys_PF = ['rest', '15p_PF', '30p_PF', '45p_PF']

trial_type = ['Dorsi', 'Plantar']

boxPath = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT\RCT002\RCT002_20240118\peak_to_peak_values' #puts user into box
values, RCT002_trials = read_processed_csv(boxPath)
dorsi_keys = [item for item in keys_DF if item in RCT002_trials]
plantar_keys = [item for item in keys_PF if item in RCT002_trials]

# Process the data for each trial type
NormalizedPlots(values, boxPath, all_keys, 'RCT002', trial_type[0])

#NormalizedPlots(values, boxPath, plantar_keys, 'RCT002', trial_type[1])

# %%
