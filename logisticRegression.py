"""Implementation of Logistic regressions.

We Have to implement a mutliclassifier (predict between A or B or C or D) with one vs rest algorithm
  -> To do this we have to implement a binary logistic regression( predict between a or B )
   ( we have to compare A vs BC , B vs AC , C vs AB) and take the maximum of probability of 
   the binary logistic regression.
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import random


class LogisticRegression:
    """Logistic Regression class.

    Based on sklearn class OneVsRestClassifier.
    """

    def __init__(self, lr=0.001, n_iters=1000, weight=[], class_=[], batch_size=None):
        """Constructor of the class.

        Define Learning Rate and the number of iteration for the gradient descent.
        """
        self.lrn = lr
        self.n_iters = n_iters
        self.weights = weight
        self._class = class_
        self.loss_ = []
        self.batch_size = batch_size

    def fit(self, X, y):
        """Fit method that permit to save weight matrix to predict after the class."""
        self._class = np.unique(y)
        _, n_features = X.shape

        self.weights = np.zeros((len(self._class), n_features))

        for idx, name_house in enumerate(self._class):
            x_sliced = X
            y_sliced = y

            if self.batch_size is not None:
                indices = list(range(len(X)))

                random.shuffle(indices)

                x_shuffled = [X[i] for i in indices]
                y_shuffled = [y[i] for i in indices]
                x_sliced = np.array(x_shuffled[: self.batch_size])
                y_sliced = np.array(y_shuffled[: self.batch_size])

            self._fit_class(x_sliced, y_sliced, idx, name_house)
        self._save_weight_class()

    def _fit_class(self, X, Y, idx_class, name_house):
        """Binary logistic regression."""
        loss_ = []
        y_transformed = self.transform_y_binary(Y, name_house)
        for _ in range(self.n_iters):

            self.weights[idx_class], loss = self.gradient_descent(
                X, y_transformed, self.weights[idx_class]
            )
            loss_.append(loss)
        self.loss_.append(loss_)

    def sigmoid(self, x):
        """Sigmoid function.

        BE CAREFUL BECAUSE  the right forumla is sig = 1 / (1  + np.exp(-x))
        but we have to put boundary to value x at 500 and -500
        because otherwise it will overflow ( because dtype of x is float64)
        """
        x = np.clip(x, -500, 500)
        sig = 1 / (1 + np.exp(-x))
        return sig

    def gradient_descent(self, X, y, weight):
        """Compute gradient descent."""
        loss = []
        linear_pred = np.dot(X, weight.T)
        pred = self.sigmoid(linear_pred)
        n_sample, _ = X.shape
        dw = (1 / n_sample) * np.dot(X.T, (pred - y))
        weight = weight - self.lrn * dw
        loss.append(self.compute_loss(y, pred))
        return weight, loss

    def compute_loss(self, y_true, y_pred):
        """Compute loss of the function."""
        return (-1 / (len(y_pred))) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

    def transform_y_binary(self, y, name_truth):
        """Take name of class and pass all label( Y aka the truth) to 1  if y == class else 0."""
        return np.array([1 if label == name_truth else 0 for label in y])

    def plot_loss(self):
        """Plot loss of the n class."""
        if len(self.loss_) == 0:
            print("You have to fit the class to plot the loss")
        for x in self.loss_:
            plt.plot(x)
            plt.show()

    def _save_weight_class(
        self, name_weight="weight_lr.csv", name_class="class_lr.csv"
    ):
        """Save weight and name of class in model directory."""
        curr_path = os.getcwd()
        directory = "model"
        path = os.path.join(curr_path, directory)
        os.makedirs(path, exist_ok=True)
        np.savetxt(f"{path}/{name_weight}", self.weights, delimiter=",")
        np.savetxt(
            f"{path}/{name_class}", np.array(self._class), delimiter=",", fmt="%s"
        )

    def argmax(self, classPred):
        """Reproduction of np argmax.

        Function is important for one vs rest because its help to choose which
        probability is the max and return an array of prediction ( with name of class)
        """
        n_class, n_epoch = classPred.shape
        best_pred = []
        for idx_epoch in range(n_epoch):
            best_res = 0
            best_class = 0
            for idx_class in range(n_class):
                if classPred[idx_class][idx_epoch] > best_res:
                    best_res = classPred[idx_class][idx_epoch]
                    best_class = self._class[idx_class]
            best_pred.append(best_class)
        return np.array(best_pred)

    def predict(self, X):
        """Predict method of linear regression.

        It take the saved matrix of weight ( for every class) and use it to predict
        (for all class) and return the best probability of classification
        """
        res = []
        res_arg_max = []
        for i in range(len(self._class)):
            linear_pred = np.dot(
                X,
                self.weights[i].T,
            )
            pred = self.sigmoid(linear_pred)

            res.append(pred)
        res = np.array(res)
        res_arg_max = self.argmax(res)

        return np.array(res_arg_max)
