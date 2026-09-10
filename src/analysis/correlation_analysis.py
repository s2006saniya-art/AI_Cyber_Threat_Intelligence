import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "processed_dataset.csv"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    print("\nLoading processed dataset...")

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully!")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# HIGHLY CORRELATED FEATURES
# ============================================================

def find_high_correlations(df, threshold=0.90):

    print("\n" + "=" * 60)
    print("HIGHLY CORRELATED FEATURE ANALYSIS")
    print("=" * 60)

    # Select numerical features only
    numerical_df = df.select_dtypes(include=["int64", "float64"])

    print(f"\nNumerical Features: {numerical_df.shape[1]}")

    # Calculate correlation matrix
    correlation_matrix = numerical_df.corr()

    # Find feature pairs with correlation >= threshold
    high_correlations = []

    for i in range(len(correlation_matrix.columns)):

        for j in range(i + 1, len(correlation_matrix.columns)):

            correlation = correlation_matrix.iloc[i, j]

            if abs(correlation) >= threshold:

                feature_1 = correlation_matrix.columns[i]
                feature_2 = correlation_matrix.columns[j]

                high_correlations.append(
                    (feature_1, feature_2, correlation)
                )

    # Convert to DataFrame
    high_corr_df = pd.DataFrame(
        high_correlations,
        columns=[
            "Feature 1",
            "Feature 2",
            "Correlation"
        ]
    )

    # Sort by strongest correlation
    high_corr_df = high_corr_df.sort_values(
        by="Correlation",
        key=lambda x: x.abs(),
        ascending=False
    )

    print("\nHighly Correlated Feature Pairs")
    print("-" * 60)

    if high_corr_df.empty:

        print("No highly correlated feature pairs found.")

    else:

        print(high_corr_df.to_string(index=False))

    print(f"\nTotal Highly Correlated Pairs: {len(high_corr_df)}")

    return high_corr_df


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    high_corr_df = find_high_correlations(
        df,
        threshold=0.90
    )


if __name__ == "__main__":
    main()