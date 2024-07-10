"""Implementation of Logistic regressions.

We Have to implement a mutliclassifier (predict between A or B or C or D) with one vs rest algorithm
  -> To do this we have to implement a binary logistic regression( predict between a or B )
   ( we have to compare A vs BC , B vs AC , C vs AB) and take the maximum of probability of 
   the binary logistic regression.
"""
import numpy as np
import matplotlib.pyplot as plt



import os



class LogisticRegression:
  """Logistic Regression class.
  
  Based on sklearn class OneVsRestClassifier.
  There is a fit for train the data and a predict function
  """
  
  
  def __init__(self, lr=0.001, n_iters=1000, weight=[], bias=[], class_=[]):
    """Constructor of the class.
    
    Define Learning Rate and the number of iteration for the gradient descent.
    """
    self.lrn = lr
    self.n_iters = n_iters
    self.weights = weight
    self.bias = bias
    self._class = class_
    self.loss_ = []
  
   
  
  def sigmoid(self, x):
    """"Sigmoid function.
    
    BE CAREFUL BECAUSE  the right forumla is sig = 1 / (1  + np.exp(-x))  
    but we have to put boundary to value's x at 500 and -500 
    because otherwise it will overflow ( because dtype of x is float64)
    """
    x = np.clip(x, -500, 500)
    sig = 1 / (1  + np.exp(-x)) 
    return sig

  def compute_loss(self, y_true, y_pred):
    """Compute loss of the function."""
    epsilon = 1e-9
    y1 = y_true * np.log(y_pred + epsilon)
        
    y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
    return -np.mean(y1 + y2)
  
  def fit(self, X,y):
    """Fit method that permit to save weight matrix to predict after the class."""
    self._class = np.unique(y)
    n_class = len(self._class)
    _, n_features = X.shape
    self.weights= np.zeros((n_class, n_features))
    self.bias = np.zeros(n_class)
    for idx, name_house in enumerate(self._class):
      self._fit_class(X, y, idx, name_house)
    self._save_weight_bias_class()
    
  def plot_loss(self):
    """Plot loss of the n class."""
    if len(self.loss_) == 0:
      print("You have to fit the class to plot the loss")
    for x in self.loss_:
      plt.plot(x)
      plt.show()

  def _fit_class(self, X, Y, idx_class, name_house):
        """Binary logistic regression.
        
        take name of class and pass all label( Y aka the truth) to 1  if y == class else 0
        """
        loss = []
        y_transformed = np.array([1 if label == name_house else 0  for label in Y ])
        n_sample, _ = X.shape
        for _  in range(self.n_iters):
          
          linear_pred = np.dot(X, self.weights[idx_class]) + self.bias[idx_class]
          pred = self.sigmoid(linear_pred)
          
          # gradient descent
          dw = (1/n_sample) * np.dot(X.T, (pred - y_transformed))
          db = (1/n_sample) * np.sum(pred - y_transformed)
          self.weights[idx_class] = self.weights[idx_class] - self.lrn * dw
          self.bias[idx_class] = self.bias[idx_class] -  self.lrn * db
          loss.append(self.compute_loss(y_transformed, pred))
        self.loss_.append(loss)
        
          
  def _save_weight_bias_class(self,):
    """Save weight and bias in model directory."""
    curr_path = os.getcwd()
    directory = "model"
    path = os.path.join(curr_path, directory)  
    os.makedirs(path, exist_ok = True) 
    np.savetxt(f'{path}/weight_lr.csv', self.weights, delimiter=',')
    np.savetxt(f'{path}/bias_lr.csv', self.bias, delimiter=',')
    np.savetxt(f'{path}/class_lr.csv', np.array(self._class), delimiter=',',  fmt='%s')
    
  def argmax(self, classPred):
    """Reproduction of np argmax.
    
    Function Important for one vs rest because its help to choose which probability is the max
    """
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
      """Predict method of linear regression.
      
      It take the saved matrix of weight ( for every class) and use it to predict  
      (for all class) and return the best probability of classification
      """
      res = []
      res_arg_max = []
      for i in range(len(self._class)):
        linear_pred = np.dot(X, self.weights[i]) + self.bias[i]
        pred = self.sigmoid(linear_pred)
        res.append(pred)
      res = np.array(res)
      res_arg_max = self.argmax(res)
      
      return np.array(res_arg_max)
    

