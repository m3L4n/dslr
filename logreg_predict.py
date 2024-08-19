"""Script for predict the logistic regression."""

from logisticRegression import LogisticRegression
from utils.ft_accuracy_score import ft_accuracy_score
from utils.load_csv import load
import numpy as np
import pandas as pd
from utils.preprocess_data_LR import preprocessing_data
import sys


def logreg_predict(data_csv, weight, class_):
    """Function that instanciate a LR with weight and bias and predict."""
    X, y = preprocessing_data(data_csv)
    log_model = LogisticRegression(weight=weight, class_=class_)
    y_pred = log_model.predict(X)
    a = ft_accuracy_score(y_pred, y)
    df = pd.DataFrame(y_pred, columns=["Hogwarts House"])
    df.to_csv("houses.csv", index_label="Index")
    print(a)


def main(argv):
    """Main Function."""
    try:
        assert (
            len(argv) == 4
        ), "Error\nUsage: python logreg_predict.py path_csv path_weight path_class"
        data_csv = load(argv[1])
        weight = np.loadtxt(
            argv[2],
            delimiter=",",
        )

        class_ = np.loadtxt(argv[3], delimiter=",", dtype=str)
        logreg_predict(data_csv, weight, class_)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main(sys.argv)
