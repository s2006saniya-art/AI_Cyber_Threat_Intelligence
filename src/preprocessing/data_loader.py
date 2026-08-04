import pandas as pd
from pathlib import Path

DATASET_FOLDER = Path("dataset/raw")

print(DATASET_FOLDER)
print(DATASET_FOLDER.exists())
print(list(DATASET_FOLDER.iterdir()))

monday_file = DATASET_FOLDER / "Monday-WorkingHours.pcap_ISCX.csv"

df = pd.read_csv(monday_file)

print(df.head())
print(df.shape)
print(df.columns)

df.columns = df.columns.str.strip()
print(df.columns)

print(df.info())

print(df.isnull().sum())

print(df[df.isnull().any(axis=1)])

df = df.dropna()

print(df.shape)

print(df.duplicated().sum())

df = df.drop_duplicates()
print(df.shape)

print(df["Label"].value_counts())