from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# XGBOOST CONFIDENCE CALIBRATION ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "xgboost_confidence_analysis.csv"
)

REPORT_DIR = PROJECT_ROOT / "results" / "reports"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("XGBOOST CONFIDENCE CALIBRATION ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load existing confidence analysis
    # --------------------------------------------------------

    print("\nLoading confidence analysis...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Records loaded : {len(df):,}")

    # --------------------------------------------------------
    # 2. Create confidence bins
    # --------------------------------------------------------

    bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

    labels = [
        "0-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-95%",
        "95-100%"
    ]

    df["confidence_bin"] = pd.cut(
        df["confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # --------------------------------------------------------
    # 3. Calculate actual accuracy for each confidence bin
    # --------------------------------------------------------

    calibration = (
        df.groupby(
            "confidence_bin",
            observed=False
        )
        .agg(
            prediction_count=("correct_prediction", "count"),
            correct_predictions=("correct_prediction", "sum"),
            actual_accuracy=("correct_prediction", "mean"),
            average_confidence=("confidence", "mean")
        )
        .reset_index()
    )

    calibration["confidence_percent"] = (
        calibration["average_confidence"] * 100
    )

    calibration["actual_accuracy_percent"] = (
        calibration["actual_accuracy"] * 100
    )

    calibration["calibration_gap"] = (
        calibration["average_confidence"]
        - calibration["actual_accuracy"]
    )

    calibration["calibration_gap_percent"] = (
        calibration["calibration_gap"] * 100
    )

    # --------------------------------------------------------
    # 4. Display results
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CONFIDENCE CALIBRATION RESULTS")
    print("=" * 60)

    display_columns = [
        "confidence_bin",
        "prediction_count",
        "correct_predictions",
        "confidence_percent",
        "actual_accuracy_percent",
        "calibration_gap_percent"
    ]

    print(
        calibration[display_columns]
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 5. Save calibration report
    # --------------------------------------------------------

    report_path = (
        REPORT_DIR
        / "xgboost_confidence_calibration.csv"
    )

    calibration.to_csv(
        report_path,
        index=False
    )

    print(
        f"\nCalibration report saved at:\n{report_path}"
    )

    # --------------------------------------------------------
    # 6. Generate calibration chart
    # --------------------------------------------------------

    print("\nGenerating calibration chart...")

    plt.figure(figsize=(10, 6))

    plt.plot(
        calibration["confidence_percent"],
        calibration["actual_accuracy_percent"],
        marker="o",
        label="Actual Accuracy"
    )

    plt.plot(
        [0, 100],
        [0, 100],
        linestyle="--",
        label="Perfect Calibration"
    )

    plt.xlabel("Average Prediction Confidence (%)")
    plt.ylabel("Actual Accuracy (%)")
    plt.title("XGBoost Confidence Calibration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    chart_path = (
        FIGURE_DIR
        / "xgboost_confidence_calibration.png"
    )

    plt.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Calibration chart saved at:\n{chart_path}"
    )

    # --------------------------------------------------------
    # 7. Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CALIBRATION ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated report:")
    print(report_path)

    print("\nGenerated figure:")
    print(chart_path)


if __name__ == "__main__":
    main()