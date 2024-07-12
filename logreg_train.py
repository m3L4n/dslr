"""Train module of the LogisticRegression program."""

import math

import pandas as pd
import numpy as np

from model.LogisticRegression import LogisticRegression
from sklearn.metrics import accuracy_score


def mean(data) -> float:
    """Return the mean of data.

    mean(data: list[float]) -> float
    """
    data = [x for x in data if not math.isnan(x)]

    if len(data) == 0:
        raise ValueError("List empty")
    return sum(data) / len(data)


def transform_nan_to_mean(data_csv):
    """Transform all NaN / NA / None in column to the mean of the column.

    That permit keep stability in our data
    """
    data_cpy = data_csv.copy()

    for column_name in data_cpy.columns:
        mean_column = mean(list(data_cpy[column_name]))
        data_cpy.fillna({column_name: mean_column}, inplace=True)
    return data_cpy


def standardization(x):
    """Standardize x based on z-score algorithm.

    standardization(x)
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x_norm = x.copy()
    for i in range(x.shape[1]):
        x_norm.iloc[:i] = (x.iloc[:i] - mean.iloc[i]) / std.iloc[i]
    return x_norm


def train() -> None:
    """Create a LogisticRegression instance and use it to train our model.

    train() -> None
    """
    df = pd.read_csv("datasets/dataset_train.csv")
    y_train = df["Hogwarts House"].values
    x_train = df.drop(
        [
            "Index",
            "Hogwarts House",
            "First Name",
            "Last Name",
            "Birthday",
            "Best Hand",
            "Arithmancy",
            "Care of Magical Creatures",
            "Defense Against the Dark Arts",
        ],
        axis=1,
    )

    x_train = transform_nan_to_mean(x_train)
    x_norm = standardization(x_train)
    logreg = LogisticRegression()

    logreg.fit(x_norm, y_train)

    logreg.save(x_train.columns.values)
    pred_y = logreg.predict(x_train)

    true_pred = np.count_nonzero(pred_y == y_train)
    print((true_pred / 1600) * 100)
    print(accuracy_score(pred_y, y_train) * 100)


if __name__ == "__main__":
    train()
