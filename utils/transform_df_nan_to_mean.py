""""Transform all numerical column with NA  in the mean of the column."""
from utils.mean import mean


def transform_nan_to_mean(data_csv):
  """Transform all NaN / NA / None in column to the mean of the column.
  
  That permit keep stability in our data
  """
  data_cpy = data_csv.copy()
  column = data_cpy.columns
  
  for name in column:
    mean_column = mean(list(data_cpy[name]))
    data_cpy.fillna({name: mean_column}, inplace=True)
  return data_cpy
   