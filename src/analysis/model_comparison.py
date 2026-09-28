import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RF_REPORT = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "random_forest_classification_report.csv"
)

XGB_REPORT = (
    PROJECT_ROOT
    / "results"
    / "reports"
    / "xgboost_classification_report.csv"
)

RESULTS_FOLDER = PROJECT_ROOT / "results" / "reports"
FIGURES_FOLDER = PROJECT_ROOT / "results" / "figures"

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("RANDOM FOREST VS XGBOOST COMPARISON")
    print("=" * 60)

    # --------------------------------------------------------
    # Load evaluation reports
    # --------------------------------------------------------

    print("\nLoading Random Forest report...")
    rf_report = pd.read_csv(RF_REPORT, index_col=0)

    print("Loading XGBoost report...")
    xgb_report = pd.read_csv(XGB_REPORT, index_col=0)

    print("Reports loaded successfully!")

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    comparison = pd.DataFrame({
        "Random Forest": [
            rf_report.loc["accuracy", "precision"],
            rf_report.loc["macro avg", "precision"],
            rf_report.loc["macro avg", "recall"],
            rf_report.loc["macro avg", "f1-score"],
            rf_report.loc["weighted avg", "f1-score"]
        ],
        "XGBoost": [
            xgb_report.loc["accuracy", "precision"],
            xgb_report.loc["macro avg", "precision"],
            xgb_report.loc["macro avg", "recall"],
            xgb_report.loc["macro avg", "f1-score"],
            xgb_report.loc["weighted avg", "f1-score"]
        ]
    }, index=[
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1"
    ])

    print("\n" + "=" * 60)
    print("OVERALL MODEL COMPARISON")
    print("=" * 60)

    print("\n")
    print(comparison.round(4).to_string())

    # --------------------------------------------------------
    # Save overall comparison
    # --------------------------------------------------------

    comparison_path = RESULTS_FOLDER / "model_comparison.csv"

    comparison.to_csv(comparison_path)

    print("\nOverall comparison saved at:")
    print(comparison_path)

    # --------------------------------------------------------
    # Per-class F1 comparison
    # --------------------------------------------------------

    class_names = [
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

    f1_comparison = pd.DataFrame({
        "Random Forest": rf_report.loc[class_names, "f1-score"],
        "XGBoost": xgb_report.loc[class_names, "f1-score"]
    })

    print("\n" + "=" * 60)
    print("PER-CLASS F1-SCORE COMPARISON")
    print("=" * 60)

    print("\n")
    print(f1_comparison.round(2).to_string())

    # --------------------------------------------------------
    # Save per-class comparison
    # --------------------------------------------------------

    f1_path = RESULTS_FOLDER / "model_f1_comparison.csv"

    f1_comparison.to_csv(f1_path)

    print("\nPer-class F1 comparison saved at:")
    print(f1_path)

    # --------------------------------------------------------
    # Generate overall metrics chart
    # --------------------------------------------------------

    print("\nGenerating overall comparison chart...")

    plot_data = comparison.reset_index()

    plot_data = plot_data.melt(
        id_vars="index",
        var_name="Model",
        value_name="Score"
    )

    plt.figure(figsize=(12, 7))

    sns.barplot(
        data=plot_data,
        x="index",
        y="Score",
        hue="Model"
    )

    plt.title(
        "Random Forest vs XGBoost - Overall Performance",
        fontsize=18,
        fontweight="bold",
        pad=15
    )

    plt.xlabel(
        "Evaluation Metric",
        fontsize=13,
        fontweight="bold"
    )

    plt.ylabel(
        "Score",
        fontsize=13,
        fontweight="bold"
    )

    plt.ylim(0, 1.05)

    plt.xticks(rotation=20)

    plt.tight_layout()

    overall_chart_path = (
        FIGURES_FOLDER / "random_forest_vs_xgboost.png"
    )

    plt.savefig(
        overall_chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Overall comparison chart saved at:")
    print(overall_chart_path)

    # --------------------------------------------------------
    # Generate per-class F1 chart
    # --------------------------------------------------------

    print("\nGenerating per-class F1 comparison chart...")

    f1_plot_data = f1_comparison.reset_index()

    f1_plot_data = f1_plot_data.melt(
        id_vars="index",
        var_name="Model",
        value_name="F1-Score"
    )

    plt.figure(figsize=(15, 9))

    sns.barplot(
        data=f1_plot_data,
        x="F1-Score",
        y="index",
        hue="Model"
    )

    plt.title(
        "Random Forest vs XGBoost - F1-Score by Attack Type",
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

    class_chart_path = (
        FIGURES_FOLDER / "random_forest_vs_xgboost_f1.png"
    )

    plt.savefig(
        class_chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Per-class F1 comparison chart saved at:")
    print(class_chart_path)

    # --------------------------------------------------------
    # Determine better model using Macro F1
    # --------------------------------------------------------

    rf_macro_f1 = comparison.loc["Macro F1", "Random Forest"]
    xgb_macro_f1 = comparison.loc["Macro F1", "XGBoost"]

    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)

    if xgb_macro_f1 > rf_macro_f1:
        print("\nXGBoost has the higher Macro F1-score.")
        print("XGBoost is currently the better-performing model.")
    elif rf_macro_f1 > xgb_macro_f1:
        print("\nRandom Forest has the higher Macro F1-score.")
        print("Random Forest is currently the better-performing model.")
    else:
        print("\nBoth models have the same Macro F1-score.")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()