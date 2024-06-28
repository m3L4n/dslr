import seaborn as sns
from utils.load_csv import load
import matplotlib.pyplot as plt

def main():
   data_csv = load('datasets/dataset_train.csv')
   sns.pairplot(data_csv)
   plt.show()

if __name__ == "__main__":
  main()