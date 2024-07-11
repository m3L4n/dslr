"""Script for training the logistic regression."""

from logisticRegression import LogisticRegression
from utils.ft_accuracy_score import ft_accuracy_score
from utils.load_csv import load
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.multiclass import OneVsRestClassifier
from utils.preprocess_data_LR import preprocessing_data
import sys
from sklearn.svm import LinearSVC




def logreg_train(data_csv):
  
  log = []
  X, y = preprocessing_data(data_csv)
  test_lr = np.arange(0.1, 1, 0.01)
  # for lr in test_lr:
  # log_model = OneVsRestClassifier(LinearSVC())
  
  log_model = LogisticRegression(lr=0.12 , n_iters=2_500)
  log_model.fit(X, y)
  y_pred = log_model.predict(X)
  a = ft_accuracy_score(y_pred, y)
  print(a)
  log_model.plot_loss()
    # log.append(a)
  # np.savetxt("res.csv", log)
  """"Function that instanciate a LR object and train and save weight and bias."""
  # X, y = preprocessing_data(data_csv)
  # log_model = LogisticRegression(lr=0.0001 , n_iters=100_000)
  # log_model = LogisticRegression(lr=0.0125 , n_iters=35_000)
  # log_model.fit(X, y)
  # log_model.plot_loss()
  # y_pred = log_model.predict(X)
  # a = ft_accuracy_score(y_pred, y)
  # print(a)



def main(argv):
  """Main Function."""
  try:
    assert len(argv) == 2, "Error\nUsage: python logreg_train.py path csv"
    data_csv = load(argv[1])
    logreg_train(data_csv)
  except Exception as e:
    print("error", e)
    
  
  
  
if __name__ == "__main__":
  main(sys.argv)