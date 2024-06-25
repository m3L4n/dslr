"""descripe.py."""

import sys
import pandas as pd
from pandas.api.types import is_numeric_dtype

from utils.count import count
from utils.mean import mean
from utils.percentiles import percentile
from utils.quartile import f_quartile, s_quartile, t_quartile
from utils.std import std
from utils.sum import sum
from utils.max import max
from utils.min import min

# TO DO DELETE NULL SEGMENT
# TO DO IMPLEMENT THE correct print

def load(path: str) -> pd.DataFrame | None:
    """Take in parameter a csv and return its dataframe."""
    try:
        df = pd.read_csv(
            path,
        )
        return pd.DataFrame(df, index=None)
    except Exception as e:
        raise e


def define_numbers_features(data_csv):
    """Define all the numeric column.
    
    return a dict for key name of column and value a list of the value of this column.
    """
    column = data_csv.columns[1:]  # delete "index because its not important"
    row_numeric = dict([(x, list(data_csv[x])) for x in column if is_numeric_dtype(data_csv[x])])
    return row_numeric


def main(argv):
    """Main function."""
    try:
        assert len(argv) == 2, "Error\nUsage : python describe.py path_csv"
        path = argv[1]
        data_csv = load(path)
        numeric_features = define_numbers_features(data_csv)
        for key,value in numeric_features.items():
          print("KEY :", key)
          print (f"count {count(value)}")
          print (f"Mean {mean(value)}")
          print (f"Std {std(value)}")
          print (f"25Q {f_quartile(value)}")
          print (f"50Q {s_quartile(value)}")
          print (f"75Q{t_quartile(value)}")
          print (f"25% {percentile(value, 0.25)}")
          print (f"50% {percentile(value, 0.5)}")
          print (f"75% {percentile(value, 0.75)}")
          print (f"max {max(value)}")
          print (f"min {min(value)}")
          print (f"Sum {sum(value)}\n")

    except Exception as e:
        print(type(e).__name__, ":", str(e))

    pass


if __name__ == "__main__":
    main(sys.argv)
