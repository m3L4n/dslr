"""Describe module read a csv file and display some stats on it numeric columns."""

import sys
from dataclasses import dataclass, field

import pandas as pd
import stats
from utils import load
from numpy import float64


def init_list() -> list:
    """List initiator."""
    return []


@dataclass
class StatsContainer:
    """Container class for stats data."""

    df: pd.DataFrame
    ROWS_NAME = list[str]
    COLUMNS_NAME = list[str]
    count: list[float64] = field(init=False, default_factory=init_list)
    mean: list[float64] = field(init=False, default_factory=init_list)
    std: list[float64] = field(init=False, default_factory=init_list)
    min: list[float64] = field(init=False, default_factory=init_list)
    q1: list[float64] = field(init=False, default_factory=init_list)
    q2: list[float64] = field(init=False, default_factory=init_list)
    q3: list[float64] = field(init=False, default_factory=init_list)
    max: list[float64] = field(init=False, default_factory=init_list)
    median: list[float64] = field(init=False, default_factory=init_list)

    def __post_init__(self) -> None:
        """Post construction method."""
        self.ROWS_NAME = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Median"]
        self.COLUMNS_NAME = [column_name for column_name in self.df][2::]

    def compute_stats(self) -> None:
        """Fill all stats attributes with data from DataFrame.

        compute_stats(self) -> None
        """
        for name in self.COLUMNS_NAME:
            data = self.df[name].tolist()
            self.count.append(stats.count(data))
            self.mean.append(stats.mean(data))
            self.std.append(stats.std(data))
            self.min.append(stats.min(data))
            self.q1.append(stats.lower_quartile(data))
            self.q2.append(stats.median_quartile(data))
            self.q3.append(stats.upper_quartile(data))
            self.max.append(stats.max(data))
            self.median.append(stats.median(data))

    def to_dataframe(self) -> pd.DataFrame:
        """Convert this class to a pandas DataFrame.

        to_dataframe(self) -> pd.DataFrame
        """
        return pd.DataFrame(
            [self.count, self.mean, self.std, self.min, self.q1, self.q2, self.q3, self.max, self.median],
            columns=self.COLUMNS_NAME,
            index=self.ROWS_NAME,
            dtype="float64",
        )


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
    """Main function."""
    data_frame = get_df_numerics_columns()

    described = StatsContainer(data_frame)
    described.compute_stats()
    print(described.to_dataframe())


if __name__ == "__main__":
    describe()
