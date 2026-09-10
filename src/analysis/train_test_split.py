import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "selected_features_dataset.csv"
)

SPLIT_FOLDER = (
    PROJECT_ROOT
    / "dataset"
    / "split"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("\nLoading selected-feature dataset...")

    df = pd.read_csv(INPUT_DATASET)

    print("Dataset loaded successfully!")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_dataset(df):

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X = df.drop(columns=["Label"])
    y = df["Label"]

    print(f"\nFeatures : {X.shape[1]}")
    print(f"Records  : {len(df):,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nSplit completed successfully!")

    print(f"Training samples : {len(X_train):,}")
    print(f"Testing samples  : {len(X_test):,}")

    return X_train, X_test, y_train, y_test


# ============================================================
# SAVE SPLIT DATA
# ============================================================

def save_split_data(X_train, X_test, y_train, y_test):

    print("\nSaving train/test datasets...")

    SPLIT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    X_train.to_csv(
        SPLIT_FOLDER / "X_train.csv",
        index=False
    )

    X_test.to_csv(
        SPLIT_FOLDER / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        SPLIT_FOLDER / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        SPLIT_FOLDER / "y_test.csv",
        index=False
    )

    print("Train/test datasets saved successfully!")

    print(f"Location: {SPLIT_FOLDER}")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    X_train, X_test, y_train, y_test = split_dataset(df)

    save_split_data(
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()