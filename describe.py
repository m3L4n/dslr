"""descripe.py."""

import sys

from utils.count import count
from utils.get_data_per_feature import define_numbers_features
from utils.load_csv import load
from utils.mean import mean
from utils.percentiles import percentile
from utils.quartile import f_quartile, s_quartile, t_quartile
from utils.std import std
from utils.sum import sum
from utils.max import max
from utils.min import min


# To Do improve the describe by use a dataFrame directly


def print_pandas_describe(numeric_features: dict):
    """Print like the method describe of Pandas."""
    key = numeric_features.keys()
    key_padded = [x[0:15].ljust(15, " ") for x in numeric_features.keys()]
    count_list = [str(count(numeric_features[x])).ljust(15, " ") for x in key]
    mean_list = [str(mean(numeric_features[x])).ljust(15, " ") for x in key]
    std_list = [str(std(numeric_features[x])).ljust(15, " ") for x in key]
    min_list = [str(min(numeric_features[x])).ljust(15, " ") for x in key]
    fp_list = [str(percentile(numeric_features[x], 0.25)).ljust(15, " ") for x in key]
    sp_list = [str(percentile(numeric_features[x], 0.50)).ljust(15, " ") for x in key]
    tp_list = [str(percentile(numeric_features[x], 0.75)).ljust(15, " ") for x in key]
    max_list = [str(max(numeric_features[x])).ljust(15, " ") for x in key]
    sum_list = [str(sum(numeric_features[x])).ljust(15, " ") for x in key]
    fq_list = [str(f_quartile(numeric_features[x])).ljust(15, " ") for x in key]
    sq_list = [str(s_quartile(numeric_features[x])).ljust(15, " ") for x in key]
    tq_list = [str(t_quartile(numeric_features[x])).ljust(15, " ") for x in key]
    print(" .    ", "\t".join(key_padded))
    print("Count ", "\t".join(count_list))
    print("Mean  ", "\t".join(mean_list))
    print("Std   ", "\t".join(std_list))
    print("Min   ", "\t".join(min_list))
    print("25%   ", "\t".join(fp_list))
    print("50%   ", "\t".join(sp_list))
    print("75%   ", "\t".join(tp_list))
    print("Max%  ", "\t".join(max_list))
    print("Sum   ", "\t".join(sum_list))
    print("0.25 Q", "\t".join(fq_list))
    print("0.50 Q", "\t".join(sq_list))
    print("0.75 Q", "\t".join(tq_list))


def describe(path):
    """Overload of the pandas method describe to show stat of each features."""
    data_csv = load(path)
    numeric_features = define_numbers_features(data_csv)
    print_pandas_describe(numeric_features)


def main(argv):
    """Main Function assert error and launch describe method."""
    try:
        assert len(argv) == 2, "Error\nUsage : python describe.py path_csv"
        describe(argv[1])
    except Exception as e:
        print(type(e).__name__, ":", str(e))

    pass


if __name__ == "__main__":
    main(sys.argv)
