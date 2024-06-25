"""Mean.py."""
from utils.sum import sum
def mean(X : list):
  """Retuen Mean of a list."""
  return sum(X) / len(X)