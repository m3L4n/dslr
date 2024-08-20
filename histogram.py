"""Display an histogram of the most homogeneous class."""

import sys

import matplotlib.pyplot as plt

from utils import load


def histogram(subject: str = "Arithmancy") -> None:
    """Make an histogram of Hogwarts students by grade in classes.

    histogram(subject: str = "Arithmancy") -> None
    """
    df = load("datasets/dataset_train.csv")
    ravenclaw = df.loc[df["Hogwarts House"] == "Ravenclaw"]
    slytherin = df.loc[df["Hogwarts House"] == "Slytherin"]
    gryffindor = df.loc[df["Hogwarts House"] == "Gryffindor"]
    hufflepuff = df.loc[df["Hogwarts House"] == "Hufflepuff"]

    columns_name = [x for x in df.columns][6::]

    class_index = 0
    try:
        class_index = columns_name.index(subject)
    except ValueError:
        print("Error: incorrect class name, the availables class names are:")
        for x in columns_name:
            print(f"- {x}")
        exit(1)

    plt.title(f"Histogram of Hogwarts students in {columns_name[class_index]} class")

    plt.hist(ravenclaw[columns_name[class_index]], label="Ravenclaw", color="#1B546C", alpha=0.75)
    plt.hist(slytherin[columns_name[class_index]], label="Slytherin", color="#31AF56", alpha=0.75)
    plt.hist(gryffindor[columns_name[class_index]], label="Gryffindor", color="#B81F24", alpha=0.75)
    plt.hist(hufflepuff[columns_name[class_index]], label="Hufflepuff", color="#DEB720", alpha=0.75)

    plt.xlabel("Grades")
    plt.ylabel("Students")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 2:
        histogram(sys.argv[1])
    else:
        histogram()
