"""Max.py."""


def max(X:list):
  """Return max of the list."""
  if len(X) > 0:
    max_tmp = X[0]
    for x in X:
      if max_tmp < x:
        max_tmp = x
    return max_tmp
  return 0

