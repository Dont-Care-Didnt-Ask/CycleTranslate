import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Creating argument parser
    parser = argparse.ArgumentParser(
        description="Script for data preparation",
    )

    # Defining and parsing command line arguments
    parser.add_argument("--data", default="rus.txt", help="Path to the data file.")
    parser.add_argument("--seed", default=42, help="Seed for the RNG.")
    parser.add_argument("--subset_size", default=10_000, help="Size of subset of train, used to emulate low-resourse mode.")

    args = parser.parse_args()

    data = pd.read_csv("rus.txt", sep="\t", names=["en", "ru", "attribution"]).drop("attribution", axis=1)
    trainval, test = train_test_split(data, test_size=0.2, random_state=args.seed)
    train, val = train_test_split(trainval, test_size=0.2, random_state=args.seed)
    low_resource_train, remaining_train = train_test_split(train, train_size=args.subset_size / float(len(train)), random_state=args.seed)

    test.to_csv("test.tsv", sep="\t")
    val.to_csv("val.tsv", sep="\t")
    low_resource_train.to_csv("low_resource_train.tsv", sep="\t")
    remaining_train.to_csv("remaining_train.tsv", sep="\t")


if __name__ == "__main__":
    main()