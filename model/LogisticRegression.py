"""Implementation of Logistic Regression model."""

import numpy as np
import pandas as pd

# from sklearn import datasets
# from sklearn.model_selection import train_test_split


class LogisticRegression:
    """LogisticRegression class."""

    def __init__(self, learning_rate: float = 0.01, n_iters: int = 1000) -> None:
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

        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        self.weights = np.zeros((n_classes, n_features))
        self.bias = np.zeros(n_classes)

        for i in range(n_classes):
            class_name = self.classes[i]
            y_norm = [1 if name == class_name else 0 for name in y]

            for _ in range(self.n_iters):
                linear_pred = np.dot(X, self.weights[i]) + self.bias[i]
                predictions = self._sigmoid(linear_pred)

                dj_w = (1 / n_samples) * np.dot(X.T, (predictions - y_norm))
                dj_b = (1 / n_samples) * np.sum(predictions - y_norm)

                self.weights[i] -= self.learning_rate * dj_w
                self.bias[i] -= self.learning_rate * dj_b

        print(f"weights: {self.weights} | bias: {self.bias}")

    def predict(self, X):
        """Placeholder."""
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_predictions = self._sigmoid(linear_predictions)
        class_predictions = [0 if y <= 0.5 else 1 for y in y_predictions]
        return class_predictions


# def test_1():
#     bc = datasets.load_breast_cancer()
#     X, y = bc.data, bc.target
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#
#     print(f"X_train {X_train}")
#     print(f"y_train: {y_train}")
#
#     clf = LogisticRegression()
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#
#     def accuracy(y_pred, y_test):
#         """Placeholder."""
#         return np.sum(y_pred == y_test) / len(y_test)
#
#     acc = accuracy(y_pred, y_test)
#     print(acc)


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
    ).values

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
