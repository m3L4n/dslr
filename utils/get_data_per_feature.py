"""Get data per feature .py."""

from pandas.api.types import is_numeric_dtype

def define_numbers_features(data_csv):
    """Define all the numeric column.
    
    return a dict for key name of column and value a list of the value of this column.
    """
    column = data_csv.columns[1:]  # delete "index because its not important"
    row_numeric = dict([(x, list(data_csv[x])) for x in column if is_numeric_dtype(data_csv[x])])
    return row_numeric