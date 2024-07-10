"""Train module of the LogisticRegression program."""

import math

import pandas as pd

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
        ],
        axis=1,
    )

    x_train = transform_nan_to_mean(x_train)
    logreg = LogisticRegression()

    logreg.fit(x_train, y_train)
    logreg.save(x_train.columns.values)
