import numpy as np
import pandas as pd

from utils.load_csv import load
from utils.mean import mean

class LogisticRegression():
  
  
  def __init__(self, lr=0.001, n_iters=1000):
    self.lrn = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None
    self._class = []
    
      
    
  def sigmoid(self, x):
    """"Sigmoid function."""
    # return np.tanh(x * 0.5) * 0.5 + 0.5
    x = np.clip(x, -500, 500)
    sig = 1 / (1  + np.exp(-x)) 
    return sig

    
  def fit(self, X,y):
    """Fit method that permit to save weight matrix to predict after the class."""
    self._class = np.unique(y)
    n_class = len(self._class)
    _, n_features = X.shape
    self.weights= np.zeros((n_class, n_features))
    self.bias = np.zeros(n_class)
    for idx, name_house in enumerate(self._class):
      self._fit_class(X, y, idx, name_house)
  
  def _fit_class(self, X, Y, idx_class, name_house):
        y_transformed = [1 if label == name_house else 0  for label in Y ]
        n_sample, _ = X.shape
        for _  in range(self.n_iters):
          
          linear_pred = np.dot(X, self.weights[idx_class]) + self.bias[idx_class]
          pred = self.sigmoid(linear_pred)
          
          # gradient descent
          dw = (1/n_sample) * np.dot(X.T, (pred - y_transformed))
          db = (1/n_sample) * np.sum(pred - y_transformed)
          self.weights[idx_class] = self.weights[idx_class] - self.lrn * dw
          self.bias[idx_class] = self.bias[idx_class] -  self.lrn * db
          
      
  def argmax(self, classPred):
    n_class, n_epoch = classPred.shape
    best_pred = []
    for idx_epoch in range(n_epoch):
      best_res = 0
      best_class =0
      for idx_class in range(n_class):
        if classPred[idx_class][idx_epoch] > best_res:
          best_res = classPred[idx_class][idx_epoch]
          best_class = self._class[idx_class]
      best_pred.append(best_class)
    return np.array(best_pred)      
      
  def predict(self, X):
      res = []
      res_arg_max = []
      for i in range(len(self._class)):
        linear_pred = np.dot(X, self.weights[i]) + self.bias[i]
        pred = self.sigmoid(linear_pred)
        res.append(pred)
      res = np.array(res)
      res_arg_max = self.argmax(res)
      
      return np.array(res_arg_max)
    
      

def accuracy(y_pred, y_test):
  for x, y in zip(y_pred,y_test):
    print("pred", x, 'truth', y)
    print("res",x == y )
  return np.sum(y_pred == y_test) / len(y_test)

def transform_nan_to_mean(data_csv):
  """Transform all NaN / NA / None in column to the mean of the column.
  
  That permit keep stability in our data
  """
  data_cpy = data_csv.copy()
  column = data_cpy.columns
  
  for name in column:
    mean_column = mean(list(data_cpy[name]))
    data_cpy[name].fillna(mean_column, inplace=True)
  return data_cpy
   
  
    
     

def main():
  data_csv = load('datasets/dataset_train.csv')
  X = data_csv.drop(columns=['Hogwarts House','Index', 'First Name', 'Last Name', 'Birthday','Best Hand','Arithmancy'],)
  X = transform_nan_to_mean(X)
  X = X.values
  house = "Hogwarts House"
  y = list(data_csv[house])
  
  log_model = LogisticRegression()
  log_model.fit(X, y)
  y_pred = log_model.predict(X)
  a = accuracy(y_pred, y)
  print(a)


if __name__ == "__main__":
    main()