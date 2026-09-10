import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "split"
    / "X_train.csv"
)

Y_TRAIN_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "split"
    / "y_train.csv"
)

FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)


def load_training_labels():

    print("\nLoading training labels...")

    y_train = pd.read_csv(Y_TRAIN_DATASET)

    print("Training labels loaded successfully!")
    print(f"Training samples: {len(y_train):,}")

    return y_train


def class_distribution(y_train):

    print("\n" + "=" * 60)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 60)

    label_counts = y_train["Label"].value_counts()

    print("\nClass Distribution")
    print("-" * 60)
    print(label_counts.to_string())

    print(f"\nTotal Classes: {len(label_counts)}")


def main():

    y_train = load_training_labels()

    class_distribution(y_train)


if __name__ == "__main__":
    main()