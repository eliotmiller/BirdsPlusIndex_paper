#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
# hashed out because I don't think I need this
# import urllib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import glob

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
plt.rcParams.update({'font.size': 14})
import warnings
warnings.filterwarnings('ignore')
import os
import time
import joblib

from collections import Counter

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import PartialDependenceDisplay
from sklearn import metrics


# In[2]:


# set some variable names. this defines which dataset we will process. the options are BirdNET 0.25, 0.5, and 0.9,
# and the same for Merlin plus also custom
theModel = "Merlin"
theThreshold = "custom"


# In[4]:


# load checklist data. pre-processed in R and Python.
all_data=pd.read_csv("../data/acoustic_checklist_metadata.csv")


# In[7]:


# loop over your detections folder, loading each file and stacking the detections together
# into a frame along with their asset number

# get your path together.
lil_string = "NE_" + theModel + "_detections_" + theThreshold
base_dir = os.path.expanduser("../data")
data_dir = os.path.join(base_dir, lil_string)

# Now use it with glob to find CSV files
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

# List to store the small dataframes
dfs = []

for file in csv_files:
    # Load only the species_code column
    df = pd.read_csv(file, usecols=['species_code'])
    
    # Add a column with the filename (minus .csv)
    df['asset'] = os.path.splitext(os.path.basename(file))[0]
    
    # Store in the list
    dfs.append(df)

# Combine all into a single DataFrame (equivalent to rbind)
result = pd.concat(dfs, ignore_index=True)


# In[5]:


# load your BirdsPlus species scores and merge with the detections.
# we call these bpi scores here, but this is really a BirdsPlus species score.
bpi_scores = pd.read_csv("../data/indexScores_v1-4.csv")

# cube the scores for better spread first
bpi_scores['bpi.score'] = bpi_scores['bpi.score'] ** 3

# select only those columns you need for the merge
cols_to_keep = ['SPECIES_CODE', 'bpi.score']
bpi_scores = bpi_scores[cols_to_keep]

# now do the merge (inner join)
merged = pd.merge(result, bpi_scores, left_on='species_code', right_on='SPECIES_CODE', how='inner')


# In[6]:


# loop through assets in all_data, and if there are values in merged, sum and plug
# into all_data, otherwise fill in a 0 for now. need to be a little careful here, as
# it is possible some of these zeros will be from processing errors (and there will
# be a bunch for non-June, July, August months), but mostly this should work out
# with a little cleaning later

# Ensure both asset columns are of string type
all_data['asset'] = all_data['asset'].astype(str)
merged['asset'] = merged['asset'].astype(str)

# Step 1: Sum bpi.score by asset in `result`
bpi_sums = merged.groupby('asset')['bpi.score'].sum().reset_index()
bpi_sums.rename(columns={'bpi.score': 'bpi.sum'}, inplace=True)

# Step 2: Merge into all_data (left join so all rows in all_data are preserved)
all_data = all_data.merge(bpi_sums, on='asset', how='left')

# Step 3: Fill NaNs with 0 where there were no matching assets in `result`
all_data['bpi.sum'] = all_data['bpi.sum'].fillna(0)


# In[7]:


# let's have a look at distribution of bpi.sum before running model
plt.hist(all_data['bpi.sum'], bins=30, edgecolor='black')
plt.xlabel('BPI Sum')
plt.ylabel('Count')
plt.title('Histogram of BirdsPlus Site Sums in Train Set')
plt.show()


# In[8]:


# ok, there's a very strong skew there. let's transform for
# model fitting purposes.
all_data['bpi.score.log'] = np.log1p(all_data['bpi.sum'])

plt.hist(all_data['bpi.score.log'], bins=30, edgecolor='black')
plt.xlabel('BPI Score')
plt.ylabel('Count')
plt.title('Histogram of Log BPI Score in Train Set')
plt.show()


# In[9]:


# now load your acoustic covariates and merge the ones you want in. remember there are others here
# you might want to explore at some point
covs=pd.read_csv('../data/NE_acoustic_covariates.csv')

# cast this as the correct type
covs['asset_id'] = covs['asset_id'].astype(str)

# drop covs to just what you want 
cols_to_keep = ['asset_id', 'EAS', 'ACI_sum']
covs = covs[cols_to_keep]

# now do the merge (inner join)
all_data = pd.merge(all_data, covs, left_on='asset', right_on='asset_id', how='inner')

# get rid of the redundant col
all_data.drop(columns='asset_id', inplace=True)


# In[10]:


# create your training and test set here
train_data = all_data

# this will differ from previous SDMs you fit, where you did this separately for each
# of presences and absences. here we will simply stratify across space and time.
# overlay a grid and bin observations by year by grid
x_grid = np.linspace(np.min(train_data['longitude']), np.max(train_data['longitude']), 100)
y_grid = np.linspace(np.min(train_data['latitude']), np.max(train_data['latitude']), 100)

# figure out which x and y index each asset would be in
x_cell = np.searchsorted(x_grid, train_data['longitude'])
y_cell = np.searchsorted(y_grid, train_data['latitude'])

# create a new index by combining the x,y cells and the year
new_index = []
for row in range(0, len(x_cell)):
    new_index.append(str(x_cell[row]) + '-' + str(y_cell[row]) + '-' +
                     str(train_data['year'].iloc[row]))

# count the number of instances of these new indices
counts = Counter(new_index)

# go through the counts and replace with their reciprocals
for key in counts:
    counts[key] = 1 / counts[key]

# set number of points you want in the test data
test_n = round(train_data.shape[0] * 0.20)

# pull out the asset ids, combine them with the new_index, merge with
# the reciprocals above, and treat those as a probability of sampling
sampling_df = pd.DataFrame({
    'asset': train_data['asset'],
    'key': new_index})

# create a vector of sampling probabilities and bind in
sampling_df['prob'] = sampling_df['key'].map(counts)

# now sample asset ids according to these sampling probabilities for test set
test_assets = sampling_df['asset'].sample(n=test_n,
                                          replace=False,
                                          weights=sampling_df['prob'],
                                          random_state=0)

# extract the test data set
test_set = train_data[train_data['asset'].isin(test_assets)]

# the training data is what remains
train_set = train_data[~train_data['asset'].isin(test_assets)]

# reset the row indices on these objects
train_set = train_set.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)


# In[11]:


# this file will give you the asset IDs of all recordings that come from small parks 
# with at least 20 recordings from the focal time period. R comments below
# this number comes from assuming that the park sizes are in meters squared. then
# finding the area of a 3km diameter circle in meters squared, as follows:
# pi*1.5^2*10^6 and then taking half that value to be conservative
exclusions = pd.read_csv('../data/holdout_parks_8Apr2025.csv')

# extract the data for those parks that meet these requirements. recast variable
exclude_assets = exclusions['asset'].astype(str)
exclude_set = train_data[train_data['asset'].isin(exclude_assets)]

# next, exclude this data from train and test
test_set = test_set[~test_set['asset'].isin(exclude_assets)]
train_set = train_set[~train_set['asset'].isin(exclude_assets)]

# reset indices and save out
train_set = train_set.reset_index(drop=True)
test_set = test_set.reset_index(drop=True)
exclude_set = exclude_set.reset_index(drop=True)

# create filenames and save out
train_filename = os.path.join('../data/', 'train_set_' + theModel + "_" + theThreshold + '.' + 'csv')
train_set.to_csv(train_filename, index=False)  

test_filename = os.path.join('../data/', 'test_set_' + theModel + "_" + theThreshold + '.' + 'csv')
test_set.to_csv(test_filename, index=False)  

exclude_filename = os.path.join('../data/', 'exclude_set_' + theModel + "_" + theThreshold + '.' + 'csv')
exclude_set.to_csv(exclude_filename, index=False)  

