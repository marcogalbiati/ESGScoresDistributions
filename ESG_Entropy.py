# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:22:00 2024

@author: marco
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:42:41 2024
@author: marco
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:45:12 2024

@author: marco
"""

import pandas as pd
import numpy as np
import os

directory = r"C:\Users\marco\Documents\ESGScores"
os.chdir(directory)


#%% Load from original csv from Florian and save
file_path = r'merged_all_data_esg_data_ffill.csv'
df = pd.read_csv(file_path)
df.to_pickle("df.pkl")


#%% Load from pickle
df = pd.read_pickle("df.pkl")


#%% Display cols

print("Below: original cols")
for i, col_name in enumerate(df.columns):
    print(f"{i+1}. {col_name}")
print("Above: original cols")


#%% Drop columns that I don't need (find idxs using the display in above block)

columns_to_drop = [45] + list(range(50, 63)) + list(range(69, 76))
df = df.drop(df.columns[columns_to_drop], axis=1)

#%% Display column numbers and names after dropping
print("Below: cols after dropping cols")
for i, col_name in enumerate(df.columns):
    print(f"{i+1}. {col_name}")
print("Above: cols after dropping cols")


#%% Define relevant scores. Count unique Firms & Dates for each score

Scorers     = ['MSCI: ESG', 'SPGlobal: ESG', 'ISS: ESG', 'Vigeo-Eiris: ESG', 'Reprisk: ESG', 'TVL: ESG', 'SustainalyticsNEW: ESG', 'SustainalyticsOLD: ESG']

data = []  # Initialize an empty list
for score in Scorers:
    filtered_df = df[df[score].notna()]      # filter df rows where score is not nan
    unique_firms = filtered_df['Firm'].nunique()  # count unique Firms
    unique_dates = filtered_df['Date'].nunique()  # count unique Dates
    data.append({'score': score, 'firms': unique_firms, 'dates': unique_dates})
scorecounts = pd.DataFrame(data)
print("Number of unique firms & dates for each score:")
print(scorecounts)
print("Number of unique firms & dates for each score:")

df['Date'].unique()

#%% Extract rows with no nan scores --> df_full

# List of cols with scores
scorecols = ['Firm', 'Date', 'MSCI: ESG', 'SPGlobal: ESG', 'ISS: ESG', 'Vigeo-Eiris: ESG', 'Reprisk: ESG', 'TVL: ESG', 'SustainalyticsNEW: ESG']

# Filter rows where we have scores for ALL providers (all columns non-nan)
df_full = df[df[scorecols].notnull().all(axis=1)]

unique_fulldates          = df_full['Date'].unique()  # Unique dates in df_full
unique_firms_per_fulldate = df_full.groupby('Date')['Firm'].nunique()

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(unique_firms_per_fulldate.index, unique_firms_per_fulldate.values, marker='o', linestyle='-')
plt.xlabel('Date')
plt.ylabel('Number of Unique Firms')
plt.title('Number of Unique Firms per Full Date')
plt.xticks(unique_firms_per_fulldate.index[::5], rotation=45)


#%% One firm's score over time

# Select a Firm and a Scorer
firm          = df_full['Firm'].iloc[1230]      # Select firm
scorer        = Scorers[1]                      # Select a scorer
filtered_rows = df_full[(df['Firm'] == firm) & df_full[scorer].notnull()]


# Plot the firm's score by the selected scorer
plt.plot(filtered_rows['Date'], filtered_rows[scorer], marker='o', linestyle='-')
plt.title(firm)
plt.xlabel('Date')
plt.ylabel(scorer)
date_changes   = filtered_rows[scorer].diff()!= 0
date_positions = filtered_rows.index[date_changes]
plt.xticks(filtered_rows['Date'][date_positions], rotation=45)
plt.grid(True)
plt.show()

# Plot the firm's scores by all scorers
plt.figure(figsize=(10, 6))
plt.figure(dpi=400)
for scorer in Scorers:
    filtered_rows = df_full[(df['Firm'] == firm) & df_full[scorer].notnull()]
    plt.plot(filtered_rows['Date'], filtered_rows[scorer], marker='.', linestyle='-', label=scorer)
    plt.title(firm)
    plt.xlabel('Date')
    plt.ylabel(scorer)
    plt.title(firm)
plt.legend(loc='lower right', fontsize='xx-small')
plt.grid(True)
plt.show()


#%% Entropy - functions

from sklearn.neighbors import KernelDensity

def shannon_entropy_kernel(sample, bandwidth):
   """
   Calculate entropy (use kernel estimate of the distr. at unique sample values)
   """
   kde          = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
   valid_sample = sample[~np.isnan(sample)]                 # pick only non-nans
   kde.fit(valid_sample.reshape(-1, 1))
   # x_grid      = np.linspace(valid_sample.min(), valid_sample.max(), 101).reshape(-1, 1)
   uniques      = np.unique(sample[~np.isnan(sample)]).reshape(-1, 1)
   # Estimate density at points
   # log_dens    = kde.score_samples(x_grid)  # Log-density for numerical stability
   log_dens     = kde.score_samples(uniques)
   freq_kernel  = np.exp(log_dens)
   entropy      = -np.sum(freq_kernel * np.log2(freq_kernel))
   return entropy

   
def shannon_entropy_freqs(sample):
   """
   Calculate the Shannon entropy using sample frequencies
   """
   unique_values, counts = np.unique(sample[~np.isnan(sample)], return_counts = True)
   frequencies           = counts / len(sample)
   entropy               = -np.sum(frequencies * np.log2(frequencies))
   return entropy
    
# # Compare frequencies vs estimated density - part run funcs content
# plt.scatter(frequencies, freq_kernel)
# min_val = min(min(x), min(frequencies))
# max_val = max(max(x), max(frequencies))
# plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='45-degree line')
# plt.ylabel('freq_kernel')
# plt.xlabel('frequencies')
# plt.show()


#%% Distributions & their entropy - one date at a time

#% --------- Pick date and its scores
datesfull           = df_full['Date'].unique()
datepicked          = datesfull[-40]
filtered_rows       = df_full[df_full['Date'] == datepicked]
samples             = filtered_rows[Scorers]
samples['ISS: ESG'] = pd.to_numeric(samples['ISS: ESG'], errors='coerce')  # Convert 'ISS: ESG' from object to float64 (??)
print(datepicked)
# print(samples.dtypes)


#% --------- Discretize (some scores seem more continuous than discreet!?!)
# unique_counts = samples.nunique()
# print(unique_counts)
# nan_counts = samples.isna().sum()
# print(nan_counts)
def discretize_column(column):
    return pd.cut(column, bins=100, labels=False)
samples_backup = samples
samples        = samples.apply(discretize_column)


# #% --------- Scores distributions
# bnumber = 50
# plt.figure(figsize=(10, 6))
# plt.figure(dpi=400)
# for score in Scorers:
#     plt.hist(samples[score], bins=bnumber, label= score, alpha = 0.5, density = False)
# plt.xlabel("Score")
# plt.ylabel("Count")
# plt.legend(loc='center right', fontsize='xx-small')
# plt.yticks([])
# plt.show()


#% --------- Compute entropy
# Optimal kernel bandwidth for each scorer - by eyeballing in block below
values            = [6, 7, 4, 6, 4, 6, 6, 6]
bandwidth_optimal = pd.DataFrame([values], columns=Scorers)
s = samples
data = []
E = pd.DataFrame()
for i, score in enumerate(Scorers):
    sample        = s[score].values.astype(float)
    sample_norm   = (sample - np.nanmax(sample)) / (np.nanmax(sample)-np.nanmin(sample))
    entro_f       = shannon_entropy_freqs(sample)
    entro_k       = shannon_entropy_kernel(sample, bandwidth_optimal.loc[0, score])
    data.append({'Score': score, 'Entro K': entro_k, 'Entro F': entro_f})
    E = pd.DataFrame(data)
    E = E.sort_values(by='Entro K', ascending=False)
print(E)

# ratio_entropies = E['Entro K'] / E['Entro F']


#%% Plot histograms and estimated densities, scorer by scorer

for s in range(7):
    score = Scorers[s]
    # print(score)
    
    # Sample
    sample        = samples[score].values.astype(float)
    
    # Estimate distribution from sample
    bandwidth = bandwidth_optimal.loc[0, score]
    kde       = KernelDensity(bandwidth=bandwidth, kernel='epanechnikov')
    kde.fit(sample.reshape(-1, 1))
    # Estimate density at points
    uniques   = np.unique(sample[~np.isnan(sample)]).reshape(-1, 1)
    log_dens  = kde.score_samples(uniques.reshape(-1, 1))
    
    # Histogram of the sample
    plt.figure(figsize=(10, 6))
    plt.figure(dpi=400)
    plt.hist(sample, bins=len(uniques), density=True, alpha=0.5, label='Histogram')
    
    # Estimated distribution from KDE
    plt.plot(uniques, np.exp(log_dens), label='KDE', color='red')
    plt.xlabel('Score bin')
    plt.ylabel('Density')
    plt.title(score)
    plt.legend()
    
