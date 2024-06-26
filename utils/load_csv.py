"""Load_csv.py."""
import pandas as pd

def load(path: str) -> pd.DataFrame | None:
    """Take in parameter a csv and return its dataframe."""
    try:
        df = pd.read_csv(
            path,
        )
        return pd.DataFrame(df, index=None)
    except Exception as e:
        raise e