"""Script for training the logistic regression."""

from logisticRegression import LogisticRegression
from utils.ft_accuracy_score import ft_accuracy_score
from utils.load_csv import load
from sklearn.model_selection import train_test_split
from utils.preprocess_data_LR import preprocessing_data
import sys




def logreg_train(data_csv):
  """"Function that instanciate a LR object and train and save weight and bias."""
  X, y = preprocessing_data(data_csv)
  log_model = LogisticRegression(lr=0.0001 , n_iters=100000)
  log_model.fit(X, y)
  log_model.plot_loss()
  y_pred = log_model.predict(X)
  a = ft_accuracy_score(y_pred, y)
  print(a)



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