import pandas as pd
import numpy as np
from sktime.transformations.series.impute import Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer



import warnings
warnings.filterwarnings("ignore")

def fix_datetime(X):
    X['ds'] = pd.to_datetime(X['date_forecast'])
    X.drop(columns=['date_forecast'], inplace=True, errors='ignore')

    # if exists, convert date_calc to datetime, else set date_calc to ds
    X["date_calc"] = pd.to_datetime(X["date_calc"], errors='coerce')
    X["date_calc"] = X["date_calc"].fillna(X["ds"])

    # calculate time difference between date_forecast and date_calc and store in new column
    X['time_diff'] = X['ds'] - X['date_calc']
    X.drop(columns=['date_calc'], inplace=True, errors='ignore')

    # convert time_diff to float 32
    X['time_diff'] = X['time_diff'].dt.total_seconds().astype('float32')

    # Sort by 'date_forecast' and reset index
    X.sort_values(by='ds', inplace=True)

    # Set 'date_forecast' as the index for easier resampling
    X.set_index('ds', inplace=True)



    # Resample to hourly and aggregate using mean
    X.fillna(method='ffill', inplace=True)
    X_hourly = X.resample('H').mean()
    X_hourly.reset_index(inplace=True)
    X_hourly.set_index('ds', inplace=True)

    return X_hourly


def convert_to_datetime(X_train, X_test, y_train):
    X_train = fix_datetime(X_train)
    X_test = fix_datetime(X_test)

    y_train['ds'] = pd.to_datetime(y_train['time'])
    y_train.drop(columns=['time'], inplace=True)

    y_train_series = y_train.squeeze()

    # set index to ds
    y_train_series.set_index('ds', inplace=True)
    y_train_series.sort_index(inplace=True)

    return X_train, X_test, y_train_series




location_map = {
    "A": 0,
    "B": 1,
    "C": 2
}

def preprocess_data(X_train, X_test, y_train, location):
    # convert to datetime
    X_train, X_test, y_train = convert_to_datetime(X_train, X_test, y_train)


    # cast all columns to float64
    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')


    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    X_train["location"] = location_map[location]
    X_test["location"] = location_map[location]

    # set index to 1, 2, 3, ...
    X_train.reset_index(inplace=True)
    X_test.reset_index(inplace=True)


    y_train["y"] = y_train["pv_measurement"].astype('float64')
    # drop unnecessary columns
    y_train.drop(columns=['pv_measurement'], inplace=True)
    
    return X_train, X_test, y_train
    


def load_all_and_save():
    # Define locations
    locations = ['A', 'B', 'C']

    X_trains = []
    X_tests = []
    y_trains = []
    # Loop through locations
    for loc in locations:
        # Read target training data
        y_train = pd.read_parquet(f'{loc}/train_targets.parquet')
        
        # Read estimated training data and add location feature
        X_train_estimated = pd.read_parquet(f'{loc}/X_train_estimated.parquet')
        
        # Read observed training data and add location feature
        X_train_observed= pd.read_parquet(f'{loc}/X_train_observed.parquet')

        # Read estimated test data and add location feature
        X_test_estimated = pd.read_parquet(f'{loc}/X_test_estimated.parquet')
        
        # Concatenate observed and estimated datasets for each location
        X_train = pd.concat([X_train_estimated, X_train_observed])
        



        # Preprocess data
        X_train, X_test, y_train = preprocess_data(X_train, X_test_estimated, y_train, location=loc)

        # print(y_train.head(), y_train.shape)
        # print(X_train.head(), X_train.shape)
        X_train = pd.merge(X_train, y_train, how="outer", on="ds")
        # print(X_train.head(), X_train.shape)
        # print(type(X_train['y']))

        # Save data to csv
        X_train.to_csv(f'{loc}/X_train.csv', index=False)
        X_test.to_csv(f'{loc}/X_test.csv', index=False)
        y_train.to_csv(f'{loc}/y_train.csv', index=False)


        X_trains.append(X_train)
        X_tests.append(X_test)

    # Concatenate all data and save to csv
    X_train = pd.concat(X_trains)
    X_test = pd.concat(X_tests)

    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)


    



if __name__ == "__main__":
    load_all_and_save()




        
