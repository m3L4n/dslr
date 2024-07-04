import seaborn as sns
from utils.load_csv import load
import matplotlib.pyplot as plt

def main():
   data_csv = load('datasets/dataset_train.csv')
   colors = {"Ravenclaw": "#1b546c", "Slytherin": "#31AF56", "Gryffindor": "#B81F24", "Hufflepuff": "#DEB720"}
   sns.pairplot(data_csv, hue="Hogwarts House", palette=colors, dropna=True)
   plt.show()

if __name__ == "__main__":
  main()