"""Implementation of Logistic regressions

  We Have to implement a mutliclassifier (predict between A or B or C or D) with one vs rest algorithm
  -> To do this we have to implement a binary logistic regression( predict between a or B )
   ( we have to compare A vs BC , B vs AC , C vs AB) and take the maximum of probability of 
   the binary logistic regression.
"""
import numpy as np

from utils.load_csv import load
from utils.mean import mean
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



class LogisticRegression:
  """Logistic Regression class.
  
  Based on sklearn class OneVsRestClassifier.
  There is a fit for train the data and a predict function
  """
  
  
  def __init__(self, lr=0.001, n_iters=1000):
    """Constructor of the class.
    
    Define Learning Rate and the number of iteration for the gradient descent.
    """
    self.lrn = lr
    self.n_iters = n_iters
    self.weights = None
    self.bias = None
    self._class = []
    
      
    
  def sigmoid(self, x):
    """"Sigmoid function.
    
    BE CAREFUL BECAUSE  the right forumla is sig = 1 / (1  + np.exp(-x))  
    but we have to put boundary to value's x at 500 and -500 
    because otherwise it will overflow ( because dtype of x is float64)
    """
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
        """Binary logistic regression.
        
        take name of class and pass all label( Y aka the truth) to 1  if y == class else 0
        """
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
    
      

# def accuracy(y_pred, y_test):
#   for x, y in zip(y_pred,y_test):
#     print("pred", x, 'truth', y)
#     print("res",x == y )
#   return np.sum(y_pred == y_test) / len(y_test)

def ft_accuracy_score(y_pred, y_true):
  return accuracy_score(y_true, y_pred)

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
   
  
    
#  FEATURE THE MOST HETEROGEN
# ASTRONOMY
# Defense Against the Dark Arts
# DIVINATION
# Muggle Studies
# Ancient Runes

# Transfiguration
# Charms
# Flying

def main():
  data_csv = load('datasets/dataset_train.csv')
  # X = data_csv.drop(columns=['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday','Best Hand', 'Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying'],)
  # X = data_csv.drop(columns=['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday','Best Hand', 'Arithmancy','Care of Magical Creatures',],)  0.971875
  # X = data_csv.drop(columns=['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday','Best Hand', 'Arithmancy','Care of Magical Creatures','Potions'],)  #0.974375 
  X = data_csv.drop(columns=['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday','Best Hand', 'Arithmancy','Care of Magical Creatures'],)  #0.974375 
  # X = data_csv.drop(columns=['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday','Best Hand', 'Arithmancy','Care of Magical Creatures','Potions', 'Muggle Studies'],)  0.774375
  # X = data_csv.drop(columns=['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday','Best Hand', 'Arithmancy','Care of Magical Creatures','Potions'],)  
  # X = data_csv.drop(columns=['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday','Best Hand', 'Arithmancy','Care of Magical Creatures','Potions', 'Ancient Runes'],)  0.96625
  # X = data_csv.drop(columns=['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday','Best Hand', 'Arithmancy','Care of Magical Creatures','Potions', 'Transfiguration'],)  0.9575
  # X = data_csv.drop(columns=['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday','Best Hand', 'Arithmancy','Care of Magical Creatures','Potions'],)  
  X = transform_nan_to_mean(X)
  X = X.values
  house = "Hogwarts House"
  y = list(data_csv[house])
  X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.20, random_state=42)
  
  log_model = LogisticRegression(lr=0.01 , n_iters=10000)
  log_model.fit(X_train, y_train)
  y_pred = log_model.predict(X_test)
  a = ft_accuracy_score(y_pred, y_test)
  print(a)


if __name__ == "__main__":
    main()