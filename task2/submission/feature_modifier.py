""" Fill the NaN values. """

import numpy as np
import pandas as pd
from sklearn import model_selection

def feature_tuning(train_or_test):
    """
    train_or_test: String, either 'train' or 'test'
    """
    #import features information
    feature_info = pd.read_csv("./data/filled/feature_information_2.csv", sep=';')

    #import and short by pid and time
    if train_or_test == 'train':
        train_features = pd.read_csv("./data/provided/train_features.csv").sort_values(["pid", "Time"])
    else:
        train_features = pd.read_csv("./data/provided/test_features.csv").sort_values(["pid", "Time"])

    # get the list of patients
    pid_unique = train_features['pid'].unique()

    n_row = pid_unique.shape[0]

    # calculate median of each column (features) whilst ignoring the nan values
    col_median = np.nanmedian(train_features, axis=0)

    processed_df = []

    for idx in range(len(pid_unique)):
        idx_pid = train_features.loc[:, 'pid'] == pid_unique[idx]
        patient = train_features.loc[idx_pid, :]

        for i_col, column in enumerate(patient.columns):
            n_notnull = patient.loc[:, column].notnull().sum()

            # if there are no values for the measurement, than place all the idx_data columns with the median of ALL measured values for that measurement.
            if n_notnull == 0:
                if column in feature_info.columns:
                    if feature_info[column][0] is int:
                        patient.loc[:, column] = feature_info[column][0]
                    else:
                        patient.loc[:, column] = col_median[i_col]
                else:
                    patient.loc[:, column] = col_median[i_col]
            # if there is only one value for the measurement, fill the remaining values with that single value.
            elif n_notnull == 1:
                patient.loc[:, column] = np.ones((12,)) * patient.loc[patient.loc[:, column].notnull(), column].values
            #if there are more values, just interpolate in between to fill the missing values.
            else:
                patient.loc[:, column] = patient.loc[:, column].interpolate(limit_direction='both')

        processed_df.append(patient)

    processed_set= pd.concat(processed_df)

    if train_or_test == 'train':
        processed_set.to_csv('./data/filled/train/train_features_filled_submit.csv')
    else:
        processed_set.to_csv('./data/filled/test/test_features_filled_submit.csv')
