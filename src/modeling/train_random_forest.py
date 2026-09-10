import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]

X_TRAIN = PROJECT_ROOT / "dataset" / "split" / "X_train.csv"
Y_TRAIN = PROJECT_ROOT / "dataset" / "split" / "y_train_encoded.csv"

X_TEST = PROJECT_ROOT / "dataset" / "split" / "X_test.csv"
Y_TEST = PROJECT_ROOT / "dataset" / "split" / "y_test_encoded.csv"

MODEL_FOLDER = PROJECT_ROOT / "models"
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)


def main():

    print("\n" + "=" * 60)
    print("RANDOM FOREST TRAINING")
    print("=" * 60)

    print("\nLoading training data...")

    X_train = pd.read_csv(X_TRAIN)
    y_train = pd.read_csv(Y_TRAIN)["Label"]

    print(f"Training features : {X_train.shape}")
    print(f"Training labels   : {y_train.shape}")

    print("\nLoading testing data...")

    X_test = pd.read_csv(X_TEST)
    y_test = pd.read_csv(Y_TEST)["Label"]

    print(f"Testing features  : {X_test.shape}")
    print(f"Testing labels    : {y_test.shape}")

    print("\nCreating Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
        random_state=42
    )

    print("\nModel configuration:")
    print("Trees             : 50")
    print("Maximum depth     : 20")
    print("Class weighting   : balanced")
    print("Maximum features  : sqrt")
    print("Parallel workers   : all available CPU cores")

    print("\nStarting training...")
    print("This may take some time on the current hardware.\n")

    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    print("\nTraining completed successfully!")
    print(f"Training time: {training_time / 60:.2f} minutes")

    print("\nEvaluating model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    model_path = MODEL_FOLDER / "random_forest_model.pkl"

    joblib.dump(model, model_path)

    print("\nModel saved successfully!")
    print(f"Location: {model_path}")


if __name__ == "__main__":
    main()