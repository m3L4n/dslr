"""Sum.py."""
import numpy as np
def sum(X:list):
  """Return Sum of a list."""
  sum_lst =  0
  for x in X:
    if not np.isnan(x):
      sum_lst += x
  return sum_lst