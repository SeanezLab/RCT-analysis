#Group analysis ideas
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import re
import pickle

# %%
# # Reading back in the list of DataFrames from Pickle file
# fileName = f'\{SID}_{trial_type}.pkl'
# with open(boxPath + fileName, 'wb') as file:
#     pickle.dump(dfs, file)
# with open(boxPath + fileName, 'rb') as file:
#     read_df_list = pickle.load(file)

# %%
#
#
#
#
#

#%%
#Read and plot all participants
# Specify the directory path to search
directory_path = r'C:\Users\Lab\Box\Seanez_Lab\SharedFolders\RAW DATA\RCT'  # Replace with the actual directory path

# List to store the paths of .pkl files with 'Dorsi' in the name
pkl_paths = []

# Search through the directory
for root, dirs, files in os.walk(directory_path):
    for file in files:
        if file.endswith('.pkl') and 'Dorsi' in file:
            pkl_path = os.path.join(root, file)
            pkl_paths.append(pkl_path)

# Print the list of paths
print("Paths of .pkl files with 'Dorsi' in the name:")
for path in pkl_paths:
    print(path)

# %%
with open(pkl_paths[0], 'rb') as file:
    RCT1 = pickle.load(file)
with open(pkl_paths[1], 'rb') as file:
    RCT3 = pickle.load(file)
with open(pkl_paths[2], 'rb') as file:
    RCT4 = pickle.load(file)
with open(pkl_paths[3], 'rb') as file:
    RCT5 = pickle.load(file)
with open(pkl_paths[4], 'rb') as file:
    RCT7 = pickle.load(file)

df_participants = [RCT1, RCT3, RCT4, RCT5, RCT7]
#%%
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10))
alpha = [0.1, 0.3, 0.5, 0.7, 1]
participants = ['RCT1', 'RCT3', 'RCT4', 'RCT5', 'RCT7']
keys_DF = ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF']
colors = ['gold', 'skyblue', 'olivedrab', 'teal', 'indigo']

RTA_group = {}
RTA_index = {}

for i in range(len(df_participants)):
    # dataframe = df_participants[i] #pulls SID RCT1, RCT3, RCT4, RCT5, RCT7
    # print(len(dataframe[i]))
    # print(dataframe[i])
    participant = participants[i]
    print(i, participant)
    color = colors[i]
    for j in range(len(df_participants[i])): #pulls trial  ['rest', '5p_DF', '15p_DF', '30p_DF', '45p_DF']


        print("TRIAL", keys_DF[j])
        label = participant +'_'+ keys_DF[j]
        data = df_participants[i][j]

        RTA_group[label] = np.array(data['R Tibialis Anterior'] / max(data.index))
        #RTA_index[label] = np.array(data.index / 230)
        RTA_index[label] = ['.5MT','1','2','3','4','ST']


        ax.plot(data.index, data['R Tibialis Anterior'], color = color, alpha = alpha[j], label = label)
        ax.set_title('R TA for Dorsiflexion Tasks')
        ax.set_xlabel('Amplitude (mA)')
        ax.set_ylabel('Normalized Values')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


boxPath = r'C:\Users\Lab\Desktop\RCTorque'
save_title = f'\RTA_Dorsi.png'
fig.savefig(boxPath + save_title)
# %%

#print(RTA_group)
#print(RTA_index)
df = pd.DataFrame({'Trial': list(RTA_group.keys()),
                   'RCValues': list(RTA_group.values()),
                   'Amplitudes': list(RTA_index.values())})
df['Trial'] = df['Trial'].str.replace('_', ' ', 1)

df[['SID', 'TrialType']] = df['Trial'].str.split(' ', expand=True)
df = df.drop(columns=['Trial'])

# Display the DataFrame

#%%
df_flat = pd.DataFrame({
    'SID': df['SID'].repeat(df['RCValues'].apply(len)),
    'TrialType': df['TrialType'].repeat(df['RCValues'].apply(len)),
    'x': df['Amplitudes'].explode(),
    'y': df['RCValues'].explode()
})

rest = df_flat[df_flat['TrialType'].str.contains('rest')]
_5p_DF = df_flat[df_flat['TrialType'] == '5p_DF']
_15p_DF = df_flat[df_flat['TrialType'] == '15p_DF']
_30p_DF = df_flat[df_flat['TrialType'] == '30p_DF']
_45p_DF = df_flat[df_flat['TrialType'] == '45p_DF']

sns.set_style("ticks")

sns.lineplot(data=rest, x="x", y="y", hue="TrialType", palette = ['gray'])
plt.show()

sns.lineplot(data=_5p_DF, x="x", y="y", hue="TrialType",  palette = ['green'])
plt.show()

sns.lineplot(data=_15p_DF, x="x", y="y", hue="TrialType", palette = ['#4CC9F0'])
plt.show()

sns.lineplot(data=_30p_DF, x="x", y="y", hue="TrialType",  palette = ['blue'])
plt.show()

sns.lineplot(data=_45p_DF, x="x", y="y", hue="TrialType",  palette = ['#3A0CA3'])

plt.show()

#%%
import seaborn as sns

rest = df[df['Trial'].str.contains('_rest')]
_5p_DF = df[df['Trial'].str.contains('_5p_DF')]
_15p_DF = df[df['Trial'].str.contains('_15p_DF')]
_30p_DF = df[df['Trial'].str.contains('_30p_DF')]
_45p_DF = df[df['Trial'].str.contains('_45p_DF')]


# %%
rest = rest.drop(columns=['Trial'])

# Flatten the 'RCValues' and 'Amplitudes' columns
rest_flat = pd.DataFrame({
    'x': rest['Amplitudes'].explode(),
    'y': rest['RCValues'].explode()
})

# %%
fmri = sns.load_dataset("fmri")
sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal", col="region",
    hue="event", style="event",
)
# %%