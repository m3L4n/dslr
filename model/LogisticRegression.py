"""Implementation of Logistic Regression model."""

import numpy as np
import pandas as pd


class LogisticRegression:
    """LogisticRegression class."""

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 10_000) -> None:
        """LogisticRegression constructor."""
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.classes = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        """Train the model using gradient descent algorithm.

        fit(self, X, y)
        """
        self.classes = np.unique(y)

        n_features = X.shape[1]
        n_classes = len(self.classes)

        self.weights = np.zeros((n_classes, n_features))

        for i in range(n_classes):
            class_name = self.classes[i]
            y_one_vs_all = [1 if name == class_name else 0 for name in y]
            self._gradient_descent(X, y, i, y_one_vs_all)

    def _gradient_descent(self, X, y, i, y_one_vs_all):
        """Perform a gradient_descent to find the optimal weights and bias.

        gradient_descent(self, X, y, i, y_one_vs_all)
        """
        n_samples = X.shape[0]
        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights[i])
            predictions = self._sigmoid(linear_pred)

            dj_w = (1 / n_samples) * np.dot(X.T, (predictions - y_one_vs_all))

            self.weights[i] -= self.learning_rate * dj_w

    def predict(self, X):
        """Predict classes of the given X datas.

        predict(self, X)
        """
        y_predictions = []
        for i in range(len(self.classes)):
            y_predictions.append(self._sigmoid(np.dot(self.weights[i], X.T)))
        y_best_scores = np.argmax(y_predictions, axis=0)
        return [self.classes[x] for x in y_best_scores]

    def save(self, index_name=[]):
        """Save weights as csv.

        save(self, column_name)
        """
        d = {}
        for i in range(len(self.classes)):
            d[self.classes[i]] = self.weights[i]
        df = pd.DataFrame(data=d, index=index_name)
        df.to_csv("datasets/weights.csv")
