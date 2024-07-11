"""Script for training the logistic regression."""

from logisticRegression import LogisticRegression
from utils.ft_accuracy_score import ft_accuracy_score
from utils.load_csv import load

from utils.preprocess_data_LR import choose_features, preprocessing_data
import sys


def logreg_train(data_csv):
    """ "Function that instanciate a LR object and train and save weight and bias."""
    X, y, X_df = preprocessing_data(data_csv)
    choose_features(data_csv)
    log_model = LogisticRegression(lr=0.12, n_iters=2_500)
    log_model.fit(X, y)
    y_pred = log_model.predict(X)
    a = ft_accuracy_score(y_pred, y)
    print(a)
    log_model.plot_loss()


def main(argv):
    """Main Function."""
    try:
        assert len(argv) == 2, "Error\nUsage: python logreg_train.py path csv"
        data_csv = load(argv[1])
        logreg_train(data_csv)
    except Exception as e:
        print("error", e)


if __name__ == "__main__":
    main(sys.argv)
