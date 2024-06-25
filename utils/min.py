"""Min.py."""


def min(X:list):
  """Return min of the list."""
  if len(X) > 0:
    min_tmp = X[0]
    for x in X:
      if min_tmp > x:
        min_tmp = x
    return min_tmp
  return 0

