"""Load module, all you need to open securely a csv file."""

import pandas as pd


def load(path: str) -> pd.DataFrame:
    """Open a dataset and return it.

    load(path: str) -> Dataset
    """
    dataframe = None
    try:
        if not path.lower().endswith(".csv"):
            raise AssertionError("path isn't a csv file")
        dataframe = pd.read_csv(path)
    except AssertionError as e:
        print(f"Error: {e}")
        exit(1)
    except FileNotFoundError:
        print("Error: file not found")
        exit(1)
    except PermissionError:
        print("Error: you don't have permission to open this file")
        exit(1)
    except Exception:
        print("An error occured...")
        exit(1)
    return dataframe
