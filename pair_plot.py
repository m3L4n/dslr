"""Pair plot our data.

This visualization permit to respond at what features are you going to use for your logistic regression?
"""

import seaborn as sns
from utils.load_csv import load
import matplotlib.pyplot as plt
def pairplot():
   """Function that print pair plot with seaborn.
   
   Permit to respond to the question :
   what features are you going to use for your logistic regression?
   """
   data_csv = load('datasets/dataset_train.csv')
   colors = {"Ravenclaw": "#1b546c", "Slytherin": "#31AF56", "Gryffindor": "#B81F24", "Hufflepuff": "#DEB720"}
   sns.pairplot(data_csv, hue="Hogwarts House", palette=colors, dropna=True)
   plt.show()
 


def main():
   """Main function."""
   data_csv = load('datasets/dataset_train.csv')
   colors = {"Ravenclaw": "#1b546c", "Slytherin": "#31AF56", "Gryffindor": "#B81F24", "Hufflepuff": "#DEB720"}
   sns.pairplot(data_csv, hue="Hogwarts House", palette=colors, dropna=True)
   plt.show()

if __name__ == "__main__":
  main()