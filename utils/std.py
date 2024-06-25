"""Std.py."""
from utils.mean import mean
def std(X:list):
  """Return std of a list."""
  mean_lst = mean(X)
  variance = sum([((x - mean_lst) ** 2) for x in X]) / len(X) 
  return variance ** 0.5