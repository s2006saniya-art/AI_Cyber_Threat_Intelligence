import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

X_TEST = PROJECT_ROOT / "dataset" / "split" / "X_test.csv"
Y_TEST = PROJECT_ROOT / "dataset" / "split" / "y_test_encoded.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"

REPORTS_FOLDER = PROJECT_ROOT / "results" / "reports"
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"

REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack - Brute Force",
    "Web Attack - SQL Injection",
    "Web Attack - XSS"
]


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("XGBOOST EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading test data...")

    X_test = pd.read_csv(X_TEST)
    y_test = pd.read_csv(Y_TEST)["Label"]

    print(f"Testing features : {X_test.shape}")
    print(f"Testing labels   : {y_test.shape}")

    # --------------------------------------------------------
    # Load XGBoost model
    # --------------------------------------------------------

    print("\nLoading XGBoost model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")

    # --------------------------------------------------------
    # Make predictions
    # --------------------------------------------------------

    print("\nMaking predictions...")

    y_pred = model.predict(X_test)

    print("Predictions completed!")

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    print("\nClassification Report:")
    print(report_df.round(2).to_string())

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    report_path = REPORTS_FOLDER / "xgboost_classification_report.csv"

    report_df.to_csv(report_path)

    print("\nClassification report saved at:")
    print(report_path)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\nGenerating confusion matrix...")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    plt.figure(figsize=(15, 12))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True
    )

    plt.title(
        "XGBoost Confusion Matrix",
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    plt.xlabel(
        "Predicted Attack Type",
        fontsize=13,
        fontweight="bold"
    )

    plt.ylabel(
        "Actual Attack Type",
        fontsize=13,
        fontweight="bold"
    )

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=9
    )

    plt.yticks(
        rotation=0,
        fontsize=9
    )

    plt.tight_layout()

    confusion_path = FIGURES_FOLDER / "xgboost_confusion_matrix.png"

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confusion matrix saved at:")
    print(confusion_path)

    # --------------------------------------------------------
    # F1-score chart
    # --------------------------------------------------------

    print("\nGenerating F1-score chart...")

    f1_scores = report_df.loc[
        CLASS_NAMES,
        "f1-score"
    ]

    plt.figure(figsize=(13, 8))

    sns.barplot(
        x=f1_scores.values,
        y=f1_scores.index,
        hue=f1_scores.index,
        legend=False
    )

    plt.title(
        "XGBoost F1-Score by Attack Type",
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    plt.xlabel(
        "F1-Score",
        fontsize=13,
        fontweight="bold"
    )

    plt.ylabel(
        "Attack Type",
        fontsize=13,
        fontweight="bold"
    )

    plt.xlim(0, 1.05)

    plt.tight_layout()

    f1_path = FIGURES_FOLDER / "xgboost_f1_scores.png"

    plt.savefig(
        f1_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("F1-score chart saved at:")
    print(f1_path)

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("XGBOOST EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()