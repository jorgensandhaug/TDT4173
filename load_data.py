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

    


def add_features(X):
    X['day_of_week'] = X.index.weekday
    X['hour_of_day'] = X.index.hour
    X['month_of_year'] = X.index.month
    X['quarter_of_year'] = X.index.quarter
    X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
    return X



def features_with_correlation_below_threshold(X_train, y_train, threshold=0.3):
    # Calculate correlation for all features, with the Y value.
    # If correlation is below threshold, add to list of features to drop
    features_to_drop = []
    for feature in X_train.columns:
        correlation = X_train[feature].corr(y_train)
        if abs(correlation) < threshold:
            features_to_drop.append(feature)
    return features_to_drop

def impute_missing_values_y(X_train, y_train):
    # clip y_train negative values to 0
    y_train = y_train.clip(lower=0)

    return X_train, y_train
    # impute y_train
    df = X_train.copy()
    df['y'] = y_train
    imp_mean = IterativeImputer(random_state=0)
    print("Fitting imputer for y_train...")
    df_y_imputed = imp_mean.fit_transform(df)
    print("Finished fitting imputer for y_train.")

    df = pd.DataFrame(df_y_imputed)
    # get last column
    df = df.iloc[:, -1]
    # set y_train series to imputed values, and turn into series again
    #old_y_train = y_train.copy()
    y_train = pd.Series(df.values, index=X_train.index)

    y_train = y_train.clip(lower=0)

    return X_train, y_train


def impute(X_train, X_test, y_train):
    return X_train, X_test, y_train
    imputer = Imputer(method="drift")
    imputer.fit(X_train, y=y_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    X_train, y_train = impute_missing_values_y(X_train, y_train) 

    return X_train, X_test, y_train
    

def fill_missing_timestamps_with_nan(X_train, y_train):
    start_date = y_train.index[0]
    end_date = y_train.index[-1]
    expected_hourly_timestamps = pd.date_range(start=start_date, end=end_date, freq='H')

    #print(f"Expected hourly timestamps: {expected_hourly_timestamps}")

    missing_timestamps = expected_hourly_timestamps[~expected_hourly_timestamps.isin(y_train.index)]

    # do the same for X_train, add nan values where missing timestamps are
    X_train = X_train.reindex(expected_hourly_timestamps, fill_value=np.nan)

    
    #........
    # missing values in y_train
    # make missing values explicit nan
    y_train = y_train.reindex(expected_hourly_timestamps)
    return X_train, y_train

def create_one_hot_encoding(X_train, X_test, location):
    locations = ['A', 'B', 'C']
    for loc in locations:
        X_train[f'location_{loc}'] = 0
        X_test[f'location_{loc}'] = 0
    
    X_train[f'location_{location}'] = 1
    X_test[f'location_{location}'] = 1
    
    return X_train, X_test


def preprocess_data(X_train, X_test, y_train, location):
    # convert to datetime
    X_train, X_test, y_train = convert_to_datetime(X_train, X_test, y_train)


    # cast all columns to float64
    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')


    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # fill missing timestamps with nan
    X_train, y_train = fill_missing_timestamps_with_nan(X_train, y_train)

    # impute missing values
    X_train, X_test, y_train = impute(X_train, X_test, y_train)

    # add features
    X_train = add_features(X_train)
    X_test = add_features(X_test)

    # create one-hot encoding of location
    X_train, X_test = create_one_hot_encoding(X_train, X_test, location)

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

        X_train['y'] = y_train

        # Save data to csv
        X_train.to_csv(f'{loc}/X_train.csv')
        X_test.to_csv(f'{loc}/X_test.csv')
        y_train.to_csv(f'{loc}/y_train.csv')


        X_trains.append(X_train)
        X_tests.append(X_test)

    # Concatenate all data and save to csv
    X_train = pd.concat(X_trains)
    X_test = pd.concat(X_tests)

    X_train.to_csv('X_train.csv')
    X_test.to_csv('X_test.csv')


    



if __name__ == "__main__":
    load_all_and_save()




        
