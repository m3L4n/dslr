"""Display an histogram of the most homogeneous class."""

from utils import load


def main():
    """Main lol."""
    df = load("datasets/dataset_train.csv")
    ravenclaw = df.loc[df["Hogwarts House"] == "Ravenclaw"]
    slytherin = df.loc[df["Hogwarts House"] == "Slytherin"]
    gryffindor = df.loc[df["Hogwarts House"] == "Gryffindor"]
    hufflepuff = df.loc[df["Hogwarts House"] == "Hufflepuff"]

    print(ravenclaw)


if __name__ == "__main__":
    main()
