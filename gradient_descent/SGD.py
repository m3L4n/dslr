from logisticRegression import LogisticRegression
import numpy as np
from sklearn.utils import shuffle


class SGD(LogisticRegression):

    def __init__(self, lr=0.001, n_iters=000, weight=[], class_=[]):
        super().__init__(lr=lr, n_iters=n_iters, weight=weight, class_=class_)

    def fit(self, X, y):
        """Fit method that permit to save weight matrix to predict after the class."""
        self.processing_fit(X, y)
        for idx, name_house in enumerate(self._class):
            self._fit_class(X, y, idx, name_house)
        self._save_weight_bias_class(name_weight="weight_SGD.csv")

    def gradient_descent(self, X, y, weight):
        m = len(y)
        linear_pred = np.dot(X, weight)
        pred = self.sigmoid(linear_pred)
        dw = 2 * np.dot(X.T, (pred - y))
        weight = weight - self.lrn * dw[0]
        loss = self.compute_loss(y, pred)
        return weight, loss

    def _fit_class(self, X, Y, idx_class, name_house):
        loss_ = []
        y_transformed = self.transform_y_binary(Y, name_house)
        for epoch in range(self.n_iters):
            indices = np.random.permutation(len(X))
            X = X[indices]
            y_transformed = y_transformed[indices]
            idx_rand = np.random.choice(len(X))
            self.weights[idx_class], loss = self.gradient_descent(
                X[idx_rand].reshape(1, -1),
                y_transformed[idx_rand].reshape(1, -1),
                self.weights[idx_class],
            )
            loss_.append(loss)
        self.loss_.append(loss_)
