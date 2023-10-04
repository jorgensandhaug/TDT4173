# %%
!pip install pandas matplotlib numpy statsforecast prophet scikit-learn scipy pyarrow seaborn xgboost h2o sktime tsfresh

# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
%matplotlib inline

# %%
y_train_a = pd.read_parquet('A/train_targets.parquet')
y_train_b = pd.read_parquet('B/train_targets.parquet')
y_train_c = pd.read_parquet('C/train_targets.parquet')

# %%
X_train_estimated_a = pd.read_parquet('A/X_train_estimated.parquet')
X_train_estimated_b = pd.read_parquet('B/X_train_estimated.parquet')
X_train_estimated_c = pd.read_parquet('C/X_train_estimated.parquet')


# %%
X_train_observed_a = pd.read_parquet('A/X_train_observed.parquet')
X_train_observed_b = pd.read_parquet('B/X_train_observed.parquet')
X_train_observed_c = pd.read_parquet('C/X_train_observed.parquet')

# %%
X_test_estimated_a = pd.read_parquet('A/X_test_estimated.parquet')
X_test_estimated_b = pd.read_parquet('B/X_test_estimated.parquet')
X_test_estimated_c = pd.read_parquet('C/X_test_estimated.parquet')


# %% [markdown]
# # Exploratory analysis of the data

# %%
# add location features
X_train_estimated_a['location'] = 0
X_train_estimated_b['location'] = 1
X_train_estimated_c['location'] = 2

X_train_observed_a['location'] = 0
X_train_observed_b['location'] = 1
X_train_observed_c['location'] = 2

X_test_estimated_a['location'] = 0
X_test_estimated_b['location'] = 1
X_test_estimated_c['location'] = 2

# concat observed and estimated, then merge with y_train
X_train_a = pd.concat([X_train_estimated_a, X_train_observed_a])
X_train_b = pd.concat([X_train_estimated_b, X_train_observed_b])
X_train_c = pd.concat([X_train_estimated_c, X_train_observed_c])

# %%
import warnings

import numpy as np
import pandas as pd

# hide warnings
warnings.filterwarnings("ignore")
#warnings.resetwarnings()

# %%


# %%
# from sktime.registry import all_estimators

# for forecaster in all_estimators(filter_tags={"scitype:y": ["multivariate", "both"]}):
#     print(forecaster[0])

# %%
# import numpy as np

# from sktime.datasets import load_airline
# from sktime.forecasting.theta import ThetaForecaster

# # until fit, identical with the simple workflow
# y = load_airline()

# fh = np.arange(1, 13)

# forecaster = ThetaForecaster(sp=12)
# forecaster.fit(y, fh=fh)

# %%
# coverage = 0.9
# y_pred_ints = forecaster.predict_interval(coverage=coverage)
# y_pred_ints

# %%
# from sktime.utils import plotting

# # also requires predictions
# y_pred = forecaster.predict()

# fig, ax = plotting.plot_series(
#     y, y_pred, labels=["y", "y_pred"], pred_interval=y_pred_ints
# )

# %%
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import make_pipeline

# from sktime.datasets import load_arrow_head, load_basic_motions
# from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
# X, y = load_basic_motions(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# #  multivariate input data
# X_train.head()

# %%
# t = TSFreshFeatureExtractor(default_fc_parameters="efficient", show_warnings=False)
# Xt = t.fit_transform(X_train)

# %%
data_by_location = {
    'A': {'X_train': X_train_a, 'y_train': y_train_a},
    'B': {'X_train': X_train_b, 'y_train': y_train_b},
    'C': {'X_train': X_train_c, 'y_train': y_train_c},
}

# convert all datetime columns to datetime type
for location in data_by_location:
    X_train = data_by_location[location]['X_train']
    X_train['ds'] = pd.to_datetime(X_train['date_forecast'])
    X_train.drop(columns=['date_forecast'], inplace=True)
    X_train.drop(columns=['date_calc'], inplace=True)
    # y_train has "time" instead
    y_train = data_by_location[location]['y_train']
    y_train['ds'] = pd.to_datetime(y_train['time'])
    y_train.drop(columns=['time'], inplace=True)




# %%
y_train = data_by_location['A']['y_train'].copy()
y_train = y_train.squeeze()
X_train = data_by_location['A']['X_train'].copy()

# set index to ds
y_train.set_index('ds', inplace=True)
X_train.set_index('ds', inplace=True)
# sort by index
X_train.sort_index(inplace=True)
y_train.sort_index(inplace=True)




# %%
# date is index now
start_date = y_train.index[0]
end_date = y_train.index[-1]
expected_hourly_timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

print(f"Expected hourly timestamps: {expected_hourly_timestamps}")

missing_timestamps = expected_hourly_timestamps[~expected_hourly_timestamps.isin(y_train.index)]

missing_timestamps

# %%
from sktime.transformations.compose import TransformerPipeline
from sktime.transformations.series.impute import Imputer
from sklearn.impute import SimpleImputer as SklearnSimpleImputer



# # For exogenous variables (X_train)
# exog_imputer = SklearnSimpleImputer(strategy="mean")
# # transform the data
# X_train = exog_imputer.fit_transform(X_train)
# X_train
# check if X_train has missing values
# drop snow_density:kgm3, cloud_base_agl:m, ceiling_height_agl:m
X_train.drop(columns=['snow_density:kgm3', 'cloud_base_agl:m', 'ceiling_height_agl:m'], inplace=True, errors='ignore')
X_train.isna().sum()


# %%
# missing values in y_train
start_date = y_train.index[0]
end_date = y_train.index[-1]
expected_hourly_timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
missing_timestamps = expected_hourly_timestamps[~expected_hourly_timestamps.isin(y_train.index)]
missing_timestamps

# %%
# make missing values explicit nan
y_train = y_train.reindex(expected_hourly_timestamps)
#num of missing values
y_train.isna().sum()

# %%
imputer = Imputer(method="drift")


y_train_imputed = imputer.fit_transform(X=X_train, y=y_train)
y_train_imputed.shape

# %%

# y_train is series
missing_timestamps_after = expected_hourly_timestamps[~expected_hourly_timestamps.isin(y_train_imputed.index)]
missing_timestamps_after
#y_train_a_imputed



# for location, data in data_by_location.items():
#     X_train = data['X_train'].copy()  # To ensure original dataframes are not changed
#     y_train = data['y_train'].copy()

#     # Apply target imputation transformations for y_train
#     y_train_imputed = target_imputer.fit_transform(y_train)
    
#     # Apply exogenous variable imputation transformations for X_train
#     X_train_imputed = exog_imputer.fit_transform(X_train)
    
#     # Update the data dictionary with imputed data
#     data['y_train_imputed'] = y_train_imputed
#     data['X_train_imputed'] = X_train_imputed

# %%



