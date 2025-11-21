#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
import numpy as np
import shap
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

from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import PartialDependenceDisplay
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


# load the test data
test_filename = os.path.join('../data/', 'test_set_' + theModel + "_" + theThreshold + '.' + 'csv')
test_set = pd.read_csv(test_filename)


# In[5]:


# use the trained model to make predictions. exponentiate here
y_pred = np.expm1(rf_loaded.predict(test_set[feature_list]))
y_true = test_set['bpi.sum']  # True values


# In[6]:


mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5  # Root Mean Squared Error
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)  # R-squared

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared (R²): {r2:.4f}')

# Write to CSV
metrics = {
    'MSE': [mse],
    'RMSE': [rmse],
    'MAE': [mae],
    'R2': [r2]
}

df_metrics = pd.DataFrame(metrics)
metrics_filename = os.path.join('../data/', 'model_performance_metrics_' + theModel + "_" + theThreshold + '.' + 'csv')
df_metrics.to_csv(metrics_filename, index=False)


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

preds = rf_loaded.predict(test_set[feature_list])
residuals = test_set['bpi.sum'] - preds

plt.scatter(preds, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()


# In[9]:


# calculate variable importance, riffing off the example from 
# https://scikit-learn.org/stable/modules/permutation_importance.html
# see also (https://christophm.github.io/interpretable-ml-book/feature-importance.html)
# run the permutation feature importance on the test set
r = permutation_importance(rf_loaded, test_set[feature_list], test_set['bpi.sum'],
                           n_repeats=8, n_jobs=4)

# print the sorted importances
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{feature_list[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")

# create a PI results file and save it out
pi_results = pd.DataFrame(
    {'feature': feature_list,
     'mean': r.importances_mean,
     'sd': r.importances_std
    }
)

# create a file name
temp_filename = os.path.join('../outputs/', 'RF_PIs_' + theModel + '_' + theThreshold + '.' + 'csv')
pi_results.to_csv(temp_filename, index=False)  


# In[10]:


# 1. Compute SHAP values using TreeExplainer
# see here too: https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability
explainer = shap.TreeExplainer(rf_loaded)
shap_values = explainer.shap_values(test_set[feature_list])

# 2. Plot SHAP summary (global feature importance)
shap.summary_plot(shap_values, test_set[feature_list], show=False)
summary_plot_filename = os.path.join('../outputs/', 'shap_summary_plot_' + theModel + '_' + theThreshold + '.' + 'png')
plt.savefig(summary_plot_filename)  # Optional: save plot

# 3. Create a dataframe for mean absolute SHAP values (global importances)
shap_df = pd.DataFrame({
    'feature': feature_list,
    'shap_mean_abs': np.abs(shap_values).mean(axis=0)
}).sort_values(by='shap_mean_abs', ascending=False)

# 4. Save SHAP results
shap_df_filename = os.path.join('../data/', 'shap_importance_' + theModel + '_' + theThreshold + '.' + 'csv')
shap_df.to_csv(shap_df_filename, index=False)

# Optional: merge with your permutation importances
merged_results = pi_results.merge(shap_df, on='feature')
merged_results_filename = os.path.join('../data/', 'merged_importances_' + theModel + '_' + theThreshold + '.' + 'csv')
merged_results.to_csv(merged_results_filename, index=False)


# In[11]:


# Plot PDP for the most important features.
top_9_features = shap_df['feature'].head(9).tolist()

# Load the training set and plot these over that
train_filename = os.path.join('../data/', 'train_set_' + theModel + "_" + theThreshold + '.' + 'csv')
train_set = pd.read_csv(train_filename)

# set up a figure name
fig_filename = os.path.join('../outputs/', 'pdp_top9_shap_' + theModel + "_" + theThreshold + '.' + 'png')

# Create 3x3 PDP grid
fig, axs = plt.subplots(3, 3, figsize=(18, 12))
axs = axs.flatten()

for i, feature in enumerate(top_9_features):
    PartialDependenceDisplay.from_estimator(
        rf_loaded,
        train_set[feature_list],
        features=[feature],
        kind='both',
        grid_resolution=200,
        ax=axs[i]
    )
    axs[i].set_title(f"{feature}")

# Hide any unused subplots (in case there are fewer than 9)
for j in range(i + 1, len(axs)):
    axs[j].axis('off')

plt.suptitle("Partial Dependence Plots for Top 9 SHAP Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for subtitle
plt.savefig(fig_filename, dpi=300)
plt.show()


# In[12]:


# bind in the predicted scores and save out
test_set['predictions'] = y_pred

test_output_filename = os.path.join('../data/', 'test_predictions_' + theModel + "_" + theThreshold + '.' + 'csv')
test_set.to_csv(test_output_filename, index=False)  

