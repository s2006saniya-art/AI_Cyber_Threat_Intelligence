import pandas as pd
import joblib
import time

from pathlib import Path
from xgboost import XGBClassifier


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

X_TRAIN = PROJECT_ROOT / "dataset" / "split" / "X_train.csv"
Y_TRAIN = PROJECT_ROOT / "dataset" / "split" / "y_train_encoded.csv"

MODEL_FOLDER = PROJECT_ROOT / "models"
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_FOLDER / "xgboost_model.pkl"


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("XGBOOST TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\nLoading training data...")

    X_train = pd.read_csv(X_TRAIN)
    y_train = pd.read_csv(Y_TRAIN)["Label"]

    print(f"Training features : {X_train.shape}")
    print(f"Training labels   : {y_train.shape}")

    # --------------------------------------------------------
    # Create XGBoost model
    # --------------------------------------------------------

    print("\nCreating XGBoost model...")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=15,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        random_state=42
    )

    print("XGBoost model created successfully!")

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\nStarting XGBoost training...")
    print("This may take some time because the dataset is large.")

    start_time = time.time()

    model.fit(X_train, y_train)

    end_time = time.time()

    training_time = end_time - start_time

    # --------------------------------------------------------
    # Training information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"\nTraining time: {training_time / 60:.2f} minutes")

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(model, MODEL_PATH)

    print("\nXGBoost model saved at:")
    print(MODEL_PATH)

    print("\n" + "=" * 60)
    print("XGBOOST TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()