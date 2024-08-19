"""Pair plot our data.

This visualization permit to respond at what features are you going to use for your logistic regression?
"""

import seaborn as sns
from utils.load_csv import load
import matplotlib.pyplot as plt


def pairplot():
    """Function that print pair plot with seaborn.

    Permit to respond to the question :
    what features are you going to use for your logistic regression?
    """
    data_csv = load("datasets/dataset_train.csv")
    df = data_csv.drop(
        columns=["Index", "First Name", "Last Name", "Birthday", "Best Hand"]
    )
    colors = {
        "Ravenclaw": "#1b546c",
        "Slytherin": "#31AF56",
        "Gryffindor": "#B81F24",
        "Hufflepuff": "#DEB720",
    }
    sns.pairplot(df, hue="Hogwarts House", palette=colors, dropna=True)
    plt.show()


def main():
    """Main function."""
    try:
        pairplot()
    except Exception as e:
        print(type(e).__name__, ":", str(e))


if __name__ == "__main__":
    main()
