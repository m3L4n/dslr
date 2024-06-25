"""Quartile.py."""


def f_quartile(X:list):
  """Return the first quartile (25) of a list."""
  X = sorted(X)
  len_x = len(X) + 1
  index = (len_x * (1/4))
  return X[int(index)]

def s_quartile(X:list):
  """Return the second quartile (50) of a list."""
  X = sorted(X)
  len_x = len(X) + 1
  index = (len_x * (2/4))
  return X[int(index)]
def t_quartile(X:list):
  """Return the third quartile (75) of a list."""
  X = sorted(X)
  len_x = len(X) + 1
  index = (len_x * (3/4))
  return X[int(index)]