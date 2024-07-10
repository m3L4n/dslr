"""Implementation of Logistic Regression model."""

import math

import numpy as np
import pandas as pd


class LogisticRegression:
    """LogisticRegression class."""

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1_000) -> None:
        """LogisticRegression constructor."""
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.classes = None

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """Train the model using gradient descent algorithm.

        fit(self, X, y)
        """
        self.classes = np.unique(y)

        n_features = X.shape[1]
        n_classes = len(self.classes)

        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)

        for i in range(n_classes):
            class_name = self.classes[i]
            y_one_vs_all = [1 if name == class_name else 0 for name in y]
            self.gradient_descent(X, y, i, y_one_vs_all)

    def gradient_descent(self, X, y, i, y_one_vs_all):
        """Perform a gradient_descent to find the optimal weights and bias.

        gradient_descent(self, X, y, i, y_one_vs_all)
        """
        n_samples = X.shape[0]
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights[i]) + self.bias[i]
            predictions = self._sigmoid(linear_pred)

            dj_w = (1 / n_samples) * np.dot(X.T, (predictions - y_one_vs_all))
            dj_b = (1 / n_samples) * np.sum(predictions - y_one_vs_all)

            self.weights[i] -= self.learning_rate * dj_w
            self.bias[i] -= self.learning_rate * dj_b


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


if __name__ == "__main__":
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
