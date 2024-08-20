"""Preprocessing of the data before send it to logstic regression."""

from pandas.api.types import is_numeric_dtype
from utils.statistic import statistic


def preprocessing_data(
    dataFrame,
    features_to_drop=[
        "Arithmancy",
        "Care of Magical Creatures",
        "Defense Against the Dark Arts",
        "Hogwarts House"
    ],
):
    """Preprocess the data before training or predicting.

    X is for the data and y for the labels
    """
    f_drop = ["Index"] + features_to_drop
    df = drop_all_none_required_feature(dataFrame, f_drop)
    X_df = transform_nan_to_mean(df)
    X = X_df.values
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y = list(dataFrame["Hogwarts House"])
    return X, y


def drop_all_none_required_feature(data_csv, features_to_excludes=[]):
    """Drop all the none numerical features and return a copy of the df."""
    none_num_column = (
        list([x for x in data_csv.columns[1:] if not is_numeric_dtype(data_csv[x])])
        + features_to_excludes
    )
    df = data_csv.drop(columns=none_num_column)
    return df


def transform_nan_to_mean(data_csv):
    """Transform all NaN / NA / None in column to the mean of the column.

    That permit keep stability in our data
    """
    data_cpy = data_csv.copy()
    column = data_cpy.columns
    for name in column:
        mean_column = statistic.mean(list(data_cpy[name]))
        data_cpy.fillna({name: mean_column}, inplace=True)
    return data_cpy
