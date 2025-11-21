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
import joblib
import json
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
plt.rcParams.update({'font.size': 14})
import warnings
warnings.filterwarnings('ignore')
import os
import time

from collections import Counter

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics


# In[2]:


# set some variable names here for loading and saving below
theModel = "ebird"
theThreshold = "NA"


# In[3]:


# load the training data, including the mapping data
train_filename = os.path.join('../data/', 'train_set_' + theModel + "_" + theThreshold + '.' + 'csv')
train_set = pd.read_csv(train_filename)

map_data=pd.read_csv('../data/srd_subset_acoustic+ebird_17Oct2023.csv')


# In[4]:


# define your feature list
# found the columns of the predictors by doing this
column_names = train_set.columns
column_df = pd.DataFrame({'col_name': column_names})

# print all rows here by changing the options
pd.set_option('display.max_rows', None)
#print(column_df)
# change options back to default
pd.reset_option('display.max_rows')

# define the names of the desired predictors
ideal_feature_list = list(train_set.columns[list(range(162, 163)) + list(range(11, 158))])

# later discovered a few columns can have missing values. remove the rows with missing data.
# do the same for the test data too
for col in range(0, len(ideal_feature_list)):
    train_set = train_set.dropna(subset=[ideal_feature_list[col]])

# subset this list to column names that overlap with those in the map data (you need this
# for plotting later).
feature_list = list(set(ideal_feature_list).intersection(map_data.columns))

# append a few columns you are going to add later
other_features = list(['cci', 'year', 'day_of_year',
                       'hours_of_day', 'solar_noon_diff', 'effort_hrs',
                       'effort_distance_km', 'num_observers'])

feature_list = feature_list + other_features


# In[5]:


# do some grid searching and then fit random forest model with
# best parameters
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestRegressor(n_jobs=8, oob_score=True, random_state=42)

grid_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    scoring='r2',
    random_state=0,
    n_jobs=-1
)

grid_search.fit(train_set[feature_list], train_set['bpi.score.log'])
best_rf = grid_search.best_estimator_


# In[6]:


# Increase grid resolution for smoother plots
fig, ax = plt.subplots(figsize=(20, 25))
ax.set_title("Partial Dependences")
mlp_disp = PartialDependenceDisplay.from_estimator(
    best_rf, 
    train_set[feature_list], 
    other_features, 
    ax=ax, 
    line_kw={"color": "red"}, 
    grid_resolution=200  # Default is 50; increase for smoother plots
)

# Save the figure
temp_filename = os.path.join('../outputs/', 'PDs_' + theModel + "_" + theThreshold + '.' + 'png')
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig(temp_filename, dpi=200)

# For each target feature, find the feature value that maximizes the predicted target
max_feature = []
for feat in range(len(other_features)):
    # Grid values for the feature
    grid_values = mlp_disp.pd_results[feat].grid_values[0]
    # Average predictions for each grid value
    average_values = mlp_disp.pd_results[feat].average
    # Find the index of the maximum average value
    max_detect_index = np.argmax(average_values)
    # Get the corresponding feature value
    max_feature.append(grid_values[max_detect_index])

print("Feature values that maximize the predicted target:")
print(max_feature)


# In[7]:


# prep the map data now. use the optimal scores from above to seed the maps
for feat_index in range(0, len(other_features)):
    column = other_features[feat_index]
    map_data[column] = max_feature[feat_index]


# In[8]:


# plot predicted site total BirdsPlus score
# subset map_data to the same columns as were used in training
map_data_subset = map_data[feature_list]

# those same columns can have missing data here. originally tried dropping rows.
# errored out. replace NAs with 0s
map_data_subset = map_data_subset.fillna(0)

# predict to the map surface. exponentiate or not based on your input.
pred_map_rf = np.expm1(best_rf.predict(map_data_subset))

# create an array of the occurrence rate
nrow=450
ncol=450
occurrence_array=np.zeros((nrow,ncol))

# compute the boundaries of the map
grid_size_h=(map_data.longitude.max()-map_data.longitude.min())/(ncol-1)
grid_size_v=(map_data.latitude.max()-map_data.latitude.min())/(nrow-1)
BBox = ((map_data.longitude.min()-grid_size_h/2,   map_data.longitude.max()+grid_size_h/2,      
         map_data.latitude.min()-grid_size_v/2, map_data.latitude.max()+grid_size_v/2))

# compute the column and row number of each grid
col_idx=np.array((map_data.longitude-map_data.longitude.min())/(map_data.longitude.max()-map_data.longitude.min())*(ncol-1))
row_idx=np.array((map_data.latitude.max()-map_data.latitude)/(map_data.latitude.max()-map_data.latitude.min())*(nrow-1))
for i in range(len(row_idx)):
    occurrence_array[int(row_idx[i]+0.01)][int(col_idx[i]+0.01)]=pred_map_rf[i]

# Reshape the array into 1D
arr_1d = occurrence_array.flatten()

# Define the number of quantiles
num_quantiles = 256

# Calculate the quantiles
quantiles = np.quantile(arr_1d, np.linspace(0, 1, num_quantiles + 1))

# Bin the values
bins_1d = np.digitize(arr_1d, quantiles)

# Reshape the bins array back to the original shape
bins = bins_1d.reshape(occurrence_array.shape)

# subtract one and divide through by num_quantiles
bins = (bins-1)/num_quantiles

# create a filename for saving this png
temp_filename = os.path.join('../outputs/', 'bpi_map_' + theModel + "_" + theThreshold + '.' + 'png')
temp_title = 'Estimated bpi score'

# save out the bins so you can compare to the other form (ebird or acoustic) later and calculate map mSE
filename_raw_score = os.path.join('../outputs/', 'raw_map_scores_' + theModel + "_" + theThreshold + '.' + 'png')
np.savetxt(filename_raw_score, bins, delimiter=',')

# also save out the predicted surface so you can load into R and make nicer maps. note that
# this line of code makes the assumption that map_data and map_data_subset are in the same
# order/shape, which seems to be true
map_predictions = pd.DataFrame(
    {'srd_id': map_data.srd_id,
     'prediction': pred_map_rf
    })

predictions_filename = os.path.join('../outputs/', 'map_predictions_' + theModel + '_' + theThreshold + '.' + 'csv')
map_predictions.to_csv(predictions_filename, index=False)  

# plot the quilt map of predictions
plt.figure(figsize = (15,8))
plt.imshow(bins, cmap='viridis', extent = BBox, aspect = 'auto')
plt.clim((0,1))
plt.colorbar()
plt.title(temp_title)
plt.savefig(temp_filename)
plt.show()


# In[9]:


# take the training data and predict the bpi score for each site.
# exponentiate here.
training_predictions = np.expm1(best_rf.predict(train_set[feature_list]))

# bind into the training data and save out
train_set['predictions'] = training_predictions
train_pred_filename = os.path.join('../data/', 'training_predictions_' + theModel + '_' + theThreshold + '.' + 'csv')
train_set.to_csv(train_pred_filename, index=False)  


# In[10]:


# save out the model and associated hyperparameters for downstream use
# Set names

model_filename = "random_forest_model_" + theModel + "_" + theThreshold + ".pkl"
metadata_filename = "random_forest_metadata" + theModel + "_" + theThreshold + ".json"

# save out the model
joblib.dump(best_rf, model_filename)

# Save metadata (feature list & hyperparameters)
metadata = {
    "feature_list": feature_list,  # List of features used for training
    "hyperparameters": best_rf.get_params()  # Save model parameters
}

# Save metadata as a JSON file
with open(metadata_filename, "w") as f:
    json.dump(metadata, f)

