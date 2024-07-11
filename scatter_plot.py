"""Plot the result of this questions ?

What are the two features that are similar ?
"""

import sys

from utils.load_csv import load
from utils.separate_data_per_feature_per_house import (
    separate_data_per_feature_per_house,
)
import matplotlib.pyplot as plt


def plot_all_features():
    """Plot all features. in scatter plot."""
    data_csv = load("datasets/dataset_train.csv")
    dict_csv_per_feature = separate_data_per_feature_per_house(data_csv)
    for idx_f1, (feature_name_f1, feature_value_f1) in enumerate(
        dict_csv_per_feature.items()
    ):
        for idx_f2, (feature_name_f2, feature_value_2) in enumerate(
            dict_csv_per_feature.items()
        ):
            if idx_f1 != idx_f2:
                plot_two_features(
                    dict_csv_per_feature, feature_name_f1, feature_name_f2
                )


def plot_two_features(data_dict_per_features, name_feature1, name_feature2):
    """Plot two feature in scatter pot."""
    feature_1 = data_dict_per_features[name_feature1]
    feature_2 = data_dict_per_features[name_feature2]
    for (key, value), (key1, value1) in zip(feature_1.items(), feature_2.items()):
        plt.scatter(value, value1, marker="o", label=key)
        plt.xlabel(name_feature1)
        plt.ylabel(name_feature2)
        plt.legend()
    plt.show()


def scatter_plot():
    """Plot a scatter plot that respond to the question."""
    data_csv = load("datasets/dataset_train.csv")
    dict_csv_per_feature = separate_data_per_feature_per_house(data_csv)
    plot_two_features(dict_csv_per_feature, "Arithmancy", "Care of Magical Creatures")


def main(argv):
    """Main.py."""
    if len(argv) == 1:
        scatter_plot()
    elif len(argv) == 2:
        if argv[1] == "all":
            plot_all_features()


if __name__ == "__main__":
    main(sys.argv)
