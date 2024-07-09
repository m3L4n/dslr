"""Preprocessing of the data before send it to logstic regression."""

from pandas.api.types import is_numeric_dtype
from utils.transform_df_nan_to_mean import transform_nan_to_mean
import numpy as np


def preprocessing_data(dataFrame):
  """Preprocess the data before training or predicting.
  
  X is for the data and y for the labels
  """
  df = drop_all_none_required_feature(dataFrame, ['Index','Arithmancy','Care of Magical Creatures',  'Flying', 'Muggle Studies', 'Transfiguration', 'History of Magic', 'Defense Against the Dark Arts', 'Potions', 'Divination'])
  X = transform_nan_to_mean(df)
  X = X.values
  # X = (X - np.mean(X))/np.std(X)
  # X = (X - np.min(X))/(np.max(X) - np.min(X))
  y = list(dataFrame["Hogwarts House"])
  return X, y


def drop_all_none_required_feature(data_csv, features_to_excludes=[]):
  """Drop all the none numerical features and return a copy of the df."""
  none_num_column = list([x for x in data_csv.columns[1:] if not is_numeric_dtype(data_csv[x])]) + features_to_excludes
  df = data_csv.drop(columns=none_num_column)
  return df
 