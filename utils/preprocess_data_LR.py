"""Preprocessing of the data before send it to logstic regression."""

from pandas.api.types import is_numeric_dtype
from utils.transform_df_nan_to_mean import transform_nan_to_mean
from sklearn.feature_selection import chi2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def preprocessing_data(
    dataFrame,
    features_to_drop=[
        "Arithmancy",
        "Care of Magical Creatures",
    ],
):
    """Preprocess the data before training or predicting.

    X is for the data and y for the labels
    """
    f_drop = ["Index"] + features_to_drop
    df = drop_all_none_required_feature(dataFrame, f_drop)
    X_df = transform_nan_to_mean(df)
    X = X_df.values
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y = list(dataFrame["Hogwarts House"])
    return X, y, X_df


def drop_all_none_required_feature(data_csv, features_to_excludes=[]):
    """Drop all the none numerical features and return a copy of the df."""
    none_num_column = (
        list([x for x in data_csv.columns[1:] if not is_numeric_dtype(data_csv[x])])
        + features_to_excludes
    )
    df = data_csv.drop(columns=none_num_column)
    return df


def chi_square(X, y):
    


def choose_features(dataFrame):
    """Use of chi2 algorithm to define which features use."""
    X, y, x_df = preprocessing_data(dataFrame, [])
    unique_y = np.unique(y)

    y = [np.where(unique_y == class_y)[0][0] for class_y in y]
    chi_scores = chi2(X, y)
    p_values = pd.Series(chi_scores[1], index=x_df.columns)
    p_values.sort_values(ascending=False, inplace=True)
    p_values.plot.bar()
    plt.show()
