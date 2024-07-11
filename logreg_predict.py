"""Script for predict the logistic regression."""

from logisticRegression import LogisticRegression
from utils.ft_accuracy_score import ft_accuracy_score
from utils.load_csv import load
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocess_data_LR import preprocessing_data
import sys
import os


def logreg_predict(data_csv, weight, bias, class_):
    """ "Function that instanciate a LR with weight and bias and predict."""
    X, y = preprocessing_data(data_csv)

    log_model = LogisticRegression(weight=weight, class_=class_)
    y_pred = log_model.predict(X)
    print(y_pred)
    a = ft_accuracy_score(y_pred, y)
    print(a)


def main(argv):
    """Main Function."""
    try:
        assert (
            len(argv) == 5
        ), "Error\nUsage: python logreg_predict.py path_csv path_weight path_bias path_class"
        data_csv = load(argv[1])
        curr_path = os.getcwd()
        directory = "model"
        path = os.path.join(curr_path, directory)
        weight = np.loadtxt(
            argv[2],
            delimiter=",",
        )
        bias = np.loadtxt(
            argv[3],
            delimiter=",",
        )
        class_ = np.loadtxt(argv[4], delimiter=",", dtype=str)
        logreg_predict(data_csv, weight, bias, class_)
    except Exception as e:
        print(f"error{e.__class__}", e)


if __name__ == "__main__":
    main(sys.argv)
