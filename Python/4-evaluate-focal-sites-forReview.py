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
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
plt.rcParams.update({'font.size': 14})
import warnings
warnings.filterwarnings('ignore')
import os
import time

from collections import Counter

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[2]:


# set some variable names. this defines which dataset we will process. the options are BirdNET 0.25, 0.5, and 0.9,
# and the same for Merlin plus also custom. ebird and NA is an eighth option here
theModel = "ebird"
theThreshold = "NA"


# In[3]:


# Load the model
model_filename = "random_forest_model_" + theModel + "_" + theThreshold + ".pkl"
rf_loaded = joblib.load(model_filename)

# Load metadata
metadata_filename = "random_forest_metadata" + theModel + "_" + theThreshold + ".json"
with open(metadata_filename, "r") as f:
    metadata = json.load(f)

# Extract feature list & hyperparameters
feature_list = metadata["feature_list"]
hyperparameters = metadata["hyperparameters"]

print("Loaded feature list:", feature_list)
print("Loaded hyperparameters:", hyperparameters)


# In[4]:


# load the eval data
exclude_filename = os.path.join('../data/', 'exclude_set_' + theModel + "_" + theThreshold + '.' + 'csv')
exclude_set = pd.read_csv(exclude_filename)


# In[5]:


# use the trained model to make predictions
y_pred = np.expm1(rf_loaded.predict(exclude_set[feature_list]))  # Predictions  
y_true = exclude_set['bpi.sum']  # True values


# In[6]:


mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5  # Root Mean Squared Error
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)  # R-squared

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R²): {r2:.4f}')


# In[7]:


# Scatter Plot: Predictions vs Actual Values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')  # Perfect fit line
plt.xlabel("Actual BPI Score")
plt.ylabel("Predicted BPI Score")
plt.title("Actual vs Predicted BPI Score")
plt.show()

# Residual Plot: Residuals vs Actual Values
residuals = y_true - y_pred

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_true, y=residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Actual BPI Score")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()


# In[8]:


# Or another view of bias

preds = rf_loaded.predict(exclude_set[feature_list])
residuals = exclude_set['bpi.sum'] - preds

plt.scatter(preds, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()


# In[9]:


# calculate the residuals (index) for each point
# Reshape for sklearn
X = y_pred.reshape(-1, 1)  # predictor is predicted values
y = y_true.values  # response is true values

# Fit linear model: predicted as a function of true values
lm = LinearRegression()
lm.fit(X, y)

# Get fitted (expected) values
y_fit = lm.predict(X)

# Calculate residuals: observed - expected
residuals = y - y_fit

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=y_true, hue=residuals, palette="coolwarm", edgecolor='k')
plt.plot(y_pred, y_fit, color='black', linewidth=2, label='Best fit line')
plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='1:1 line')
plt.xlabel('Predicted values')
plt.ylabel('True values')
plt.legend()
plt.title('True vs Predicted with Residuals')
plt.show()


# In[10]:


# bind in the predicted scores and save out
exclude_set['predictions'] = y_pred

exclude_output_filename = os.path.join('../data/', 'exclude_predictions_' + theModel + '_' + theThreshold + '.' + 'csv')
exclude_set.to_csv(exclude_output_filename, index=False)  

