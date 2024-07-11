"""Implementation of Logistic Regression model."""

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

    def predict(self, X):
        """Predict classes of the given X datas.

        predict(self, X)
        """
        y_predictions = []
        for i in range(len(self.classes)):
            y_predictions.append(self._sigmoid(np.dot(self.weights[i], X.T)))
        y_best_prediction = np.argmax(y_predictions, axis=0)
        return [self.classes[x] for x in y_best_prediction]

    def save(self, column_name):
        """Save weights in csv files, one for each classes.

        save(self, column_name)
        """
        for c_i in range(len(self.classes)):
            d = {}
            for w_i in range(len(self.weights[c_i])):
                d[column_name[w_i]] = self.weights[c_i]
            d["bias"] = self.bias[c_i]
            df = pd.DataFrame(data=d, dtype=np.float64)
            df.to_csv(f"datasets/{self.classes[c_i]}_weights.csv")
