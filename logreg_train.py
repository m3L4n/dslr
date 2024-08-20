"""Script for training the logistic regression."""

from logisticRegression import LogisticRegression
from utils.ft_accuracy_score import ft_accuracy_score
from utils.load_csv import load

from utils.preprocess_data_LR import preprocessing_data
import argparse


def logreg_train(data_csv, action, batch_size, learning_rate, iters):
    """Function that instantiate a LR object and train and save weight."""
    X, y = preprocessing_data(data_csv)
    log_model = LogisticRegression(
        lr=learning_rate, n_iters=iters, batch_size=batch_size
    )
    log_model.fit(X, y)
    y_pred = log_model.predict(X)

    a = ft_accuracy_score(y_pred, y)
    print(a)
    if action == "plot":
        log_model.plot_loss()


def main():
    """Main Function."""
    try:
        parser = argparse.ArgumentParser(description="Train your logistic regression")
        parser.add_argument(
            "dataset",
            type=str,
            help="select the dataset to train with",
        )
        parser.add_argument(
            "--action",
            choices=["plot"],
            help="plot the loss function",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            help="number of students use in each iteration",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.12,
            help="define the learning between 0 and 1 ",
        )
        parser.add_argument(
            "--iters",
            type=int,
            default=2_000,
            help="define the number of iterations for the training phase",
        )
        args = parser.parse_args()
        data_csv = load(args.dataset)
        logreg_train(
            data_csv, args.action, args.batch_size, args.learning_rate, args.iters
        )
    except Exception as e:
        print("error", e)


if __name__ == "__main__":
    main()
