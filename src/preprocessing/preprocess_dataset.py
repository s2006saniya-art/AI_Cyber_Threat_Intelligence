import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATASET_FOLDER = PROJECT_ROOT / "dataset" / "raw"

PROCESSED_DATASET_FOLDER = PROJECT_ROOT / "dataset" / "processed"

def load_all_csv_files():
    """
    Load all CSV files from the raw dataset folder
    and combine them into a single DataFrame.
    """

    csv_files = sorted(RAW_DATASET_FOLDER.glob("*.csv"))

    print(f"Found {len(csv_files)} CSV files.\n")

    dataframes = []

    for file in csv_files:
        print(f"Reading {file.name}...")

        df = pd.read_csv(file)

        dataframes.append(df)

    print("\nAll files loaded successfully!")

    combined_df = pd.concat(dataframes, ignore_index=True)

    print(f"\nCombined Dataset Shape: {combined_df.shape}")

    return combined_df

def clean_column_names(df):
    """
    Remove leading and trailing spaces from column names.
    """

    print("\nCleaning column names...")

    df.columns = df.columns.str.strip()

    print("Column names cleaned successfully.")

    return df

def normalize_labels(df):
    """
    Standardize attack labels.
    """

    print("\nNormalizing attack labels...")

    df["Label"] = (
    df["Label"]
    .str.strip()
    .str.replace("–", "-", regex=False)
    .str.replace("�", "-", regex=False)
    .str.replace("Web Attack - Sql Injection", "Web Attack - SQL Injection", regex=False)
    .str.replace("Web Attack  -", "Web Attack -", regex=False)
    .str.replace("Web Attack - Brute Force", "Web Attack - Brute Force", regex=False)
    .str.replace("Web Attack - XSS", "Web Attack - XSS", regex=False)
    .str.replace("Web Attack - Sql Injection", "Web Attack - SQL Injection", regex=False)
)
    print(df["Label"].unique())

    print("Attack labels normalized successfully.")

    return df

def remove_infinite_values(df):
    """
    Replace infinite values with NaN.
    """

    print("\nReplacing infinite values...")

    numeric_columns = df.select_dtypes(include=[np.number]).columns

    df[numeric_columns] = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan
    )

    print("Infinite values replaced successfully.")

    return df

def remove_missing_values(df):
    """
    Remove rows containing missing values.
    """

    print("\nChecking for missing values...")

    missing_values = df.isnull().sum().sum()

    print(f"Total Missing Values: {missing_values}")

    rows_before = len(df)

    df = df.dropna()

    rows_after = len(df)

    print(f"Rows Removed: {rows_before - rows_after}")

    print("Missing values removed successfully.")

    return df

def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    """
    print("\nChecking for duplicate rows...")

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate Rows Found: {duplicate_rows}")

    rows_before = len(df)

    df = df.drop_duplicates()

    rows_after = len(df)

    print(f"Rows Removed: {rows_before - rows_after}")

    print("Duplicate rows removed successfully.")

    return df

def remove_negative_flow_duration(df):
    """
    Remove rows where Flow Duration is negative.
    """

    print("\nChecking negative Flow Duration values...")

    negative_rows = (df["Flow Duration"] < 0).sum()

    print(f"Negative Rows Found: {negative_rows}")

    df = df[df["Flow Duration"] >= 0]

    print("Negative Flow Duration rows removed successfully.")

    return df

def validate_dataset(df):
    """
    Validate the cleaned dataset.
    """

    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)

    print(f"Total Rows          : {len(df):,}")
    print(f"Total Columns       : {df.shape[1]}")

    missing_values = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    print(f"Missing Values      : {missing_values}")
    print(f"Duplicate Rows      : {duplicate_rows}")

    print("\nLabel Distribution")
    print("-" * 60)

    label_counts = df["Label"].value_counts()

    print(label_counts.to_string().encode("ascii", errors="replace").decode())

def save_processed_dataset(df):
    """
    Save the Cleaned dataset.
    """

    output_file = PROCESSED_DATASET_FOLDER / "processed_dataset.csv"

    print("\nSaving the cleaned dataset...")

    df.to_csv(output_file, index=False)

    print(f"Dataset saved successfully!")

    print(f"Location: {output_file}")

def main():
    """
    Main function to run the preprocessing pipeline.
    """

    combined_df = load_all_csv_files()

    combined_df = clean_column_names(combined_df)

    combined_df = normalize_labels(combined_df)

    combined_df = remove_infinite_values(combined_df)

    combined_df = remove_missing_values(combined_df)

    combined_df = remove_duplicates(combined_df)

    combined_df = remove_negative_flow_duration(combined_df)

    validate_dataset(combined_df)

    save_processed_dataset(combined_df)

if __name__ == "__main__":
    main()