import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "processed_dataset.csv"
)

OUTPUT_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "selected_features_dataset.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("\nLoading processed dataset...")

    df = pd.read_csv(INPUT_DATASET)

    print("Dataset loaded successfully!")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_features(df):

    print("\n" + "=" * 60)
    print("FEATURE SELECTION")
    print("=" * 60)

    # Separate target variable
    target = df["Label"]

    # Select numerical features
    numerical_features = df.drop(columns=["Label"])

    correlation_matrix = numerical_features.corr().abs()

    # Select upper triangle of correlation matrix
    upper_triangle = correlation_matrix.where(
    ~np.tril(
        np.ones(correlation_matrix.shape),
        k=0
    ).astype(bool)
)

    # Find columns with exact duplicate correlation
    features_to_remove = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] == 1.0)
    ]

    print("\nHighly Redundant Features:")
    print("-" * 60)

    for feature in features_to_remove:
        print(feature)

    print(f"\nTotal Features Removed: {len(features_to_remove)}")

    # Remove redundant features
    selected_df = df.drop(columns=features_to_remove)

    print(f"\nOriginal Columns : {df.shape[1]}")
    print(f"Selected Columns : {selected_df.shape[1]}")

    return selected_df


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df):

    print("\nSaving selected-feature dataset...")

    df.to_csv(OUTPUT_DATASET, index=False)

    print("Dataset saved successfully!")
    print(f"Location: {OUTPUT_DATASET}")


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    selected_df = select_features(df)

    save_dataset(selected_df)


if __name__ == "__main__":
    main()