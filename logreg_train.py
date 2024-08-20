"""Train module of the LogisticRegression program."""

import math
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from model.LogisticRegression import LogisticRegression


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


def standardize(x):
    """Standardize data using min-max algorithm.

    standardize(x)
    """
    min = np.min(x, axis=0)
    max = np.max(x, axis=0)

    x_norm = x.copy()
    for i in range(x.shape[1]):
        x_norm.iloc[:, i] = (x_norm.iloc[:, i] - min.iloc[i]) / (max.iloc[i] - min.iloc[i])
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
    x_norm = standardize(x_train)

    logreg = None
    if len(sys.argv) > 1:
        if "--mini_batch" in sys.argv[1::]:
            logreg = LogisticRegression(optimizer="mini_batch")
        elif "--gd" in sys.argv[1::]:
            logreg = LogisticRegression(optimizer="gd")
    else:
        logreg = LogisticRegression()

    if logreg:
        logreg.fit(x_norm, y_train)
        logreg.save(x_train.columns.values)

        pred_y = logreg.predict(x_norm)

        print(f"Model accuracy: {accuracy_score(pred_y, y_train) * 100}%")


if __name__ == "__main__":
    train()
    df = pd.read_csv("datasets/weights.csv")
    print(df)
