"""Quartile.py."""
import numpy as np

def sort_without_nan(X: list):
   """Return a sorted list without nan."""
   res = [ x for x in X if not np.isnan(x)]
   return sorted(res)


def f_quartile(X:list):
  """Return the first quartile (25) of a list."""
  X = sort_without_nan(X)
  len_x = len(X) + 1
  index = (len_x * (1/4))
  return X[int(index)]

def s_quartile(X:list):
  """Return the second quartile (50) of a list."""
  X = sort_without_nan(X)
  len_x = len(X) + 1
  index = (len_x * (2/4))
  return X[int(index)]
def t_quartile(X:list):
  """Return the third quartile (75) of a list."""
  X = sort_without_nan(X)
  len_x = len(X) + 1
  index = (len_x * (3/4))
  return X[int(index)]