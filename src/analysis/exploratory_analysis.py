import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "processed_dataset.csv"
)

RESULTS_FOLDER = PROJECT_ROOT / "results"
FIGURES_FOLDER = RESULTS_FOLDER / "figures"
STATISTICS_FOLDER = RESULTS_FOLDER / "statistics"

def load_dataset():
    print("\nLoading processed dataset...")

    df = pd.read_csv(PROCESSED_DATASET)

    print("\nDataset loaded successfully!")
    
    return df

def dataset_overview(df):

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("\n" + "=" * 60)

    print(f"Rows     : {df.shape[0]:,}")
    print(f"Columns  : {df.shape[1]}")

    print("\nColumns Name:\n")

    for column in df.columns:
        print(column)

def dataset_information(df):
    
    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("\n" + "=" * 60)

    df.info()

def statistical_summary(df):
    
    print("\n" + "=" * 60)
    print("STATISTICAL SUMMARY")
    print("\n" + "=" * 60)

    print(df.describe())

def check_infinite_values(df):

    print("\n" + "=" * 60)
    print("CHECK FOR INFINITE VALUES")
    print("=" * 60)

    numeric_df = df.select_dtypes(include=[np.number])

    inf_count = np.isinf(numeric_df).sum().sum()

    print(f"Total Infinite Values : {inf_count}")

    if inf_count > 0:

        print("\nColumns containing infinite values:\n")

        inf_columns = np.isinf(numeric_df).sum()

        print(inf_columns[inf_columns > 0])

    else:

        print("No infinite values found.")

def check_negative_values(df):

    print("\n" + "=" * 60)
    print("NEGATIVE VALUES CHECK")
    print("=" * 60)

    negative_flow_duration = df[df["Flow Duration"] < 0]

    print(f"Negative Flow Duration Records: {len(negative_flow_duration)}")

    if len(negative_flow_duration) > 0:
        print("\nFirst 5 Negative Records:\n")
        print(
            negative_flow_duration[
                ["Flow Duration", "Label"]
            ].head()
        )

def label_distribution(df):

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION")
    print("=" * 60)

    label_counts = df["Label"].value_counts()
    print(label_counts)

    plt.figure(figsize=(18 ,8))

    ax = sns.barplot(
    x=label_counts.index,
    y=label_counts.values,
    hue=label_counts.index,
    legend=False
)
    for i, value in enumerate(label_counts.values):
        ax.text(
        i,
        value,
        f"{value:,}",
        ha="center",
        va="bottom",
        fontsize=8,
        rotation=90
    )

    plt.xticks(rotation=45, ha="right")
    plt.title("Distribution of Attack Classes in CICIDS2017 Dataset", fontsize=16, fontweight="bold")
    plt.xlabel("Attack Classes", fontsize=12)
    plt.ylabel("Number of Network Flows (Log Scale)", fontsize=12)
    
    plt.yscale("log")

    plt.tight_layout()

    output_path = FIGURES_FOLDER / "label_distribution.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nGraph Saved at: \n{output_path}")

def main():

    df = load_dataset()

    dataset_overview(df)

    dataset_information(df)

    statistical_summary(df)

    check_infinite_values(df)

    check_negative_values(df)

    label_distribution(df)

if __name__ == "__main__":
    main()