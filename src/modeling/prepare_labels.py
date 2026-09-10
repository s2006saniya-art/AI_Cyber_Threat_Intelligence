import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]

Y_TRAIN_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "split"
    / "y_train.csv"
)

Y_TEST_DATASET = (
    PROJECT_ROOT
    / "dataset"
    / "split"
    / "y_test.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "dataset" / "split"


def main():

    print("\nLoading training and testing labels...")

    y_train = pd.read_csv(Y_TRAIN_DATASET)
    y_test = pd.read_csv(Y_TEST_DATASET)

    print("Labels loaded successfully!")

    print(f"Training samples : {len(y_train):,}")
    print(f"Testing samples  : {len(y_test):,}")

    # Encode attack labels into numerical values
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train["Label"])
    y_test_encoded = label_encoder.transform(y_test["Label"])

    # Save encoded labels
    pd.DataFrame({
        "Label": y_train_encoded
    }).to_csv(
        OUTPUT_FOLDER / "y_train_encoded.csv",
        index=False
    )

    pd.DataFrame({
        "Label": y_test_encoded
    }).to_csv(
        OUTPUT_FOLDER / "y_test_encoded.csv",
        index=False
    )

    # Save class mapping
    class_mapping = pd.DataFrame({
        "Class_ID": range(len(label_encoder.classes_)),
        "Attack_Type": label_encoder.classes_
    })

    class_mapping.to_csv(
        OUTPUT_FOLDER / "class_mapping.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("LABEL ENCODING COMPLETED")
    print("=" * 60)

    print("\nClass Mapping:")
    print(class_mapping.to_string(index=False))

    print("\nEncoded labels saved successfully!")


if __name__ == "__main__":
    main()