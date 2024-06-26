"""Describe module read a csv file and display some stats on it numeric columns."""

import sys

import pandas as pd
from stats import StatsContainer
from utils import load


def get_df_numerics_columns() -> pd.DataFrame:
    """Open and load a csv file from path then filter only numerics columns.

    load_csv(path: str) -> pd.DataFrame
    """
    if len(sys.argv) == 1:
        print("Please provide the csv file you want to desribe")
        exit(1)
    df = load(sys.argv[1])
    numerics = ["float16", "float32", "float64"]
    return df.select_dtypes(numerics)


def describe():
    """Reimplementation of pandas describe method."""
    data_frame = get_df_numerics_columns()

    described = StatsContainer(data_frame)
    described.compute_stats()
    print(described.to_dataframe())


if __name__ == "__main__":
    describe()
