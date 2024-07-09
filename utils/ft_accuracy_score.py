""""Wrapper of accuracy score."""

from sklearn.metrics import accuracy_score

def ft_accuracy_score(y_pred, y_true):
  """Wrapper for sklearn accuracy score."""
  return accuracy_score(y_true, y_pred)