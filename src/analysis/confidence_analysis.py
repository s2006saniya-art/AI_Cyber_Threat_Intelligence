"""
============================================================
CONFIDENCE ANALYSIS FOR XGBOOST
AI-Based Cyber Threat Intelligence Platform
============================================================

Purpose
-------
This module evaluates the prediction confidence of the trained
XGBoost multiclass classifier on the existing test dataset.

It calculates:
    1. Prediction confidence for every test sample
    2. Prediction correctness
    3. Class-wise average confidence
    4. Class-wise accuracy / empirical correctness
    5. Class-wise F1-score
    6. Confidence distribution
    7. Confidence vs correctness analysis

The existing model and test data are reused.
No retraining is performed.

Outputs
-------
results/reports/xgboost_confidence_analysis.csv
results/reports/xgboost_confidence_summary.csv
results/figures/xgboost_confidence_by_class.png
results/figures/xgboost_confidence_distribution.png
results/figures/xgboost_confidence_vs_reliability.png
============================================================
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score


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
# HELPER FUNCTIONS
# ============================================================

def validate_paths():
    """
    Verify that all required input files exist.
    """

    required_paths = {
        "X_test": X_TEST,
        "y_test": Y_TEST,
        "XGBoost model": MODEL_PATH
    }

    missing_paths = []

    for name, path in required_paths.items():
        if not path.exists():
            missing_paths.append(f"{name}: {path}")

    if missing_paths:
        raise FileNotFoundError(
            "\nRequired file(s) not found:\n"
            + "\n".join(missing_paths)
        )


def load_data_and_model():
    """
    Load test data and trained XGBoost model.
    """

    print("\nLoading test data...")

    X_test = pd.read_csv(X_TEST)
    y_test = pd.read_csv(Y_TEST)["Label"]

    print(f"Testing features : {X_test.shape}")
    print(f"Testing labels   : {y_test.shape}")

    print("\nLoading XGBoost model...")

    model = joblib.load(MODEL_PATH)

    print("XGBoost model loaded successfully!")

    return X_test, y_test, model


def validate_labels(y_test):
    """
    Validate that encoded labels are within the expected range.
    """

    unique_labels = sorted(y_test.unique())

    expected_labels = list(range(len(CLASS_NAMES)))

    if not set(unique_labels).issubset(set(expected_labels)):
        raise ValueError(
            "Unexpected encoded labels found in y_test: "
            f"{unique_labels}"
        )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("XGBOOST CONFIDENCE ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate required files
    # --------------------------------------------------------

    print("\nValidating project files...")

    validate_paths()

    print("All required files found!")

    # --------------------------------------------------------
    # Load data and model
    # --------------------------------------------------------

    X_test, y_test, model = load_data_and_model()

    validate_labels(y_test)

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)

    print("Predictions generated successfully!")

    # --------------------------------------------------------
    # Generate prediction probabilities
    # --------------------------------------------------------

    print("\nCalculating prediction probabilities...")

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The loaded XGBoost model does not support "
            "predict_proba()."
        )

    probabilities = model.predict_proba(X_test)

    print(
        f"Probability matrix shape: {probabilities.shape}"
    )

    # --------------------------------------------------------
    # Calculate confidence
    # --------------------------------------------------------

    print("\nCalculating prediction confidence...")

    confidence = probabilities.max(axis=1)

    predicted_class_probability = probabilities[
        range(len(probabilities)),
        y_pred.astype(int)
    ]

    # Sanity check
    if not (confidence >= 0).all() or not (confidence <= 1).all():
        raise ValueError(
            "Prediction confidence contains values outside "
            "the valid range [0, 1]."
        )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    correct = (y_test.to_numpy() == y_pred)

    # --------------------------------------------------------
    # Create sample-level result dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame({
        "actual_label": y_test.to_numpy(),
        "predicted_label": y_pred,
        "actual_class": [
            CLASS_NAMES[int(label)]
            for label in y_test
        ],
        "predicted_class": [
            CLASS_NAMES[int(label)]
            for label in y_pred
        ],
        "confidence": confidence,
        "predicted_class_probability": predicted_class_probability,
        "correct_prediction": correct
    })

    results_df["confidence_percent"] = (
        results_df["confidence"] * 100
    )

    # --------------------------------------------------------
    # Save sample-level analysis
    # --------------------------------------------------------

    sample_report_path = (
        REPORTS_FOLDER
        / "xgboost_confidence_analysis.csv"
    )

    results_df.to_csv(
        sample_report_path,
        index=False
    )

    print("\nSample-level confidence analysis saved at:")
    print(sample_report_path)

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    overall_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    average_confidence = results_df[
        "confidence"
    ].mean()

    correct_average_confidence = results_df.loc[
        results_df["correct_prediction"],
        "confidence"
    ].mean()

    incorrect_average_confidence = results_df.loc[
        ~results_df["correct_prediction"],
        "confidence"
    ].mean()

    overall_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("OVERALL CONFIDENCE RESULTS")
    print("=" * 60)

    print(
        f"\nAccuracy                 : "
        f"{overall_accuracy:.4f}"
    )

    print(
        f"Macro F1                 : "
        f"{overall_f1:.4f}"
    )

    print(
        f"Average Confidence      : "
        f"{average_confidence:.4f}"
    )

    print(
        f"Correct Prediction Conf.: "
        f"{correct_average_confidence:.4f}"
    )

    print(
        f"Incorrect Prediction Conf.: "
        f"{incorrect_average_confidence:.4f}"
    )

    # --------------------------------------------------------
    # Class-wise analysis
    # --------------------------------------------------------

    print("\nGenerating class-wise confidence analysis...")

    class_records = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        class_mask = (
            results_df["actual_label"] == class_id
        )

        class_data = results_df.loc[class_mask]

        support = len(class_data)

        if support == 0:
            class_accuracy = 0.0
            average_class_confidence = 0.0
        else:
            class_accuracy = (
                class_data["correct_prediction"].mean()
            )

            average_class_confidence = (
                class_data["confidence"].mean()
            )

        class_f1 = f1_score(
            y_test,
            y_pred,
            labels=[class_id],
            average="macro",
            zero_division=0
        )

        class_records.append({
            "class_id": class_id,
            "attack_class": class_name,
            "support": support,
            "accuracy_reliability": class_accuracy,
            "f1_score": class_f1,
            "average_confidence": average_class_confidence,
            "average_confidence_percent":
                average_class_confidence * 100
        })

    summary_df = pd.DataFrame(class_records)

    # --------------------------------------------------------
    # Confidence-reliability difference
    # --------------------------------------------------------

    summary_df["confidence_reliability_gap"] = (
        summary_df["average_confidence"]
        - summary_df["accuracy_reliability"]
    )

    summary_df["confidence_reliability_gap_percent"] = (
        summary_df["confidence_reliability_gap"] * 100
    )

    # --------------------------------------------------------
    # Save class-wise summary
    # --------------------------------------------------------

    summary_path = (
        REPORTS_FOLDER
        / "xgboost_confidence_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nClass-wise confidence summary saved at:")
    print(summary_path)

    # --------------------------------------------------------
    # Print class-wise results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLASS-WISE CONFIDENCE & RELIABILITY")
    print("=" * 60)

    display_columns = [
        "attack_class",
        "support",
        "f1_score",
        "average_confidence",
        "accuracy_reliability",
        "confidence_reliability_gap"
    ]

    print(
        summary_df[display_columns]
        .round(4)
        .to_string(index=False)
    )

    # ========================================================
    # FIGURE 1 — AVERAGE CONFIDENCE BY CLASS
    # ========================================================

    print("\nGenerating confidence-by-class chart...")

    plot_df = summary_df.sort_values(
        "average_confidence",
        ascending=True
    )

    plt.figure(figsize=(13, 8))

    sns.barplot(
        data=plot_df,
        x="average_confidence",
        y="attack_class",
        hue="attack_class",
        legend=False
    )

    plt.title(
        "XGBoost Average Prediction Confidence by Attack Class",
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    plt.xlabel(
        "Average Prediction Confidence",
        fontsize=13,
        fontweight="bold"
    )

    plt.ylabel(
        "Attack Class",
        fontsize=13,
        fontweight="bold"
    )

    plt.xlim(0, 1.05)

    plt.tight_layout()

    confidence_class_path = (
        FIGURES_FOLDER
        / "xgboost_confidence_by_class.png"
    )

    plt.savefig(
        confidence_class_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confidence-by-class chart saved at:")
    print(confidence_class_path)

    # ========================================================
    # FIGURE 2 — CONFIDENCE DISTRIBUTION
    # ========================================================

    print("\nGenerating confidence distribution...")

    plt.figure(figsize=(12, 7))

    sns.histplot(
        data=results_df,
        x="confidence",
        bins=20,
        kde=True
    )

    plt.title(
        "XGBoost Prediction Confidence Distribution",
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    plt.xlabel(
        "Prediction Confidence",
        fontsize=13,
        fontweight="bold"
    )

    plt.ylabel(
        "Number of Predictions",
        fontsize=13,
        fontweight="bold"
    )

    plt.xlim(0, 1)

    plt.tight_layout()

    confidence_distribution_path = (
        FIGURES_FOLDER
        / "xgboost_confidence_distribution.png"
    )

    plt.savefig(
        confidence_distribution_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confidence distribution saved at:")
    print(confidence_distribution_path)

    # ========================================================
    # FIGURE 3 — CONFIDENCE VS EMPIRICAL RELIABILITY
    # ========================================================

    print("\nGenerating confidence vs reliability chart...")

    comparison_df = summary_df.melt(
        id_vars="attack_class",
        value_vars=[
            "average_confidence",
            "accuracy_reliability"
        ],
        var_name="measure",
        value_name="score"
    )

    comparison_df["measure"] = comparison_df[
        "measure"
    ].replace({
        "average_confidence": "Average Confidence",
        "accuracy_reliability": "Empirical Reliability"
    })

    plt.figure(figsize=(14, 8))

    sns.barplot(
        data=comparison_df,
        x="attack_class",
        y="score",
        hue="measure"
    )

    plt.title(
        "XGBoost Confidence vs Empirical Reliability",
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    plt.xlabel(
        "Attack Class",
        fontsize=13,
        fontweight="bold"
    )

    plt.ylabel(
        "Score",
        fontsize=13,
        fontweight="bold"
    )

    plt.ylim(0, 1.05)

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    comparison_path = (
        FIGURES_FOLDER
        / "xgboost_confidence_vs_reliability.png"
    )

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Confidence vs reliability chart saved at:")
    print(comparison_path)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("CONFIDENCE ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated reports:")
    print(f"1. {sample_report_path}")
    print(f"2. {summary_path}")

    print("\nGenerated figures:")
    print(f"1. {confidence_class_path}")
    print(f"2. {confidence_distribution_path}")
    print(f"3. {comparison_path}")

    print("\nNext research step:")
    print(
        "Analyze whether prediction confidence corresponds "
        "to empirical class-level reliability."
    )

    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()