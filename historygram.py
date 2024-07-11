"""Plot the result of this questions ?

Which Hogwarts course has a homogeneous score distribution between all four houses?
"""

import matplotlib.pyplot as plt

from utils.load_csv import load
from utils.separate_data_per_feature_per_house import (
    separate_data_per_feature_per_house,
)


def plot_all_features(dict_csv_per_features):
    """Plot all the features histogram of all student per houses."""
    dict_copy = dict_csv_per_features.copy()
    for feature in dict_csv_per_features.keys():
        plot_one_features(dict_copy, feature)


def plot_one_features(dict_csv_per_features, name):
    """Plot one feature histogram of all student per houses."""
    feature_dict = dict_csv_per_features[name]
    color = ["green", "red", "blue", "yellow"]
    plt.title(name)
    for index_house, (name_f, value_f) in enumerate(feature_dict.items()):
        plt.hist(value_f, color=color[index_house], alpha=0.5, label=name_f)
        plt.legend()
    plt.show()


def histogram():
    """Histogram function that calculate and plot the result of the questions."""
    try:
        data_csv = load("datasets/dataset_train.csv")
        dict_csv_per_feature = separate_data_per_feature_per_house(data_csv)
        plot_all_features(dict_csv_per_feature)
        plot_one_features(dict_csv_per_feature, "Arithmancy")

    except Exception as e:
        print(type(e).__name__, ":", str(e))


def main():
    """Main functions."""
    histogram()


if __name__ == "__main__":
    main()
