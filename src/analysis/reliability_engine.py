import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TEST_FEATURES_PATH = os.path.join(
    BASE_DIR, "dataset", "split", "X_test.csv"
)

TEST_LABELS_PATH = os.path.join(
    BASE_DIR, "dataset", "split", "y_test.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "xgboost_model.pkl"
)

CLASS_REPORT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "reports",
    "xgboost_classification_report.csv"
)

OUTPUT_REPORT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "reports",
    "reliability_analysis.csv"
)


# ============================================================
# RELIABILITY WEIGHTS
# ============================================================

CONFIDENCE_WEIGHT = 0.40
F1_WEIGHT = 0.40
RECALL_WEIGHT = 0.20


# ============================================================
# RELIABILITY CATEGORY
# ============================================================

def get_reliability_category(score):

    if score >= 0.80:
        return "High"

    elif score >= 0.60:
        return "Moderate"

    elif score >= 0.40:
        return "Low"

    else:
        return "Very Low"


# ============================================================
# MAIN RELIABILITY ENGINE
# ============================================================

def main():

    print("=" * 60)
    print("RELIABILITY ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print("\nLoading test data...")

    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = pd.read_csv(TEST_LABELS_PATH)

    # Handle one-column label CSV
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    print(f"Test Features Shape: {X_test.shape}")
    print(f"Test Labels Shape: {y_test.shape}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading XGBoost model...")

    model = joblib.load(MODEL_PATH)

    print("XGBoost model loaded successfully.")

    # --------------------------------------------------------
    # Load class-level performance
    # --------------------------------------------------------

    print("\nLoading class-level model performance...")

    report = pd.read_csv(CLASS_REPORT_PATH)

    print("Classification report loaded.")

    # Remove summary rows if present
    # Rename the class-name column
    report = report.rename(
        columns={"Unnamed: 0": "class"}
    )

    # Remove summary rows
    report = report[
        ~report["class"].isin(
            ["accuracy", "macro avg", "weighted avg"]
        )
    ].copy()

    # --------------------------------------------------------
    # Prediction probabilities
    # --------------------------------------------------------

    print("\nGenerating prediction probabilities...")

    probabilities = model.predict_proba(X_test)

    predicted_labels = np.argmax(probabilities, axis=1)

    confidence = np.max(probabilities, axis=1)

    print(f"Probability Matrix Shape: {probabilities.shape}")

    # --------------------------------------------------------
    # Create class performance lookup
    # --------------------------------------------------------

    class_performance = {}

    for _, row in report.iterrows():

        class_name = str(row["class"])

        class_performance[class_name] = {
            "f1": float(row["f1-score"]),
            "recall": float(row["recall"]),
            "support": int(row["support"])
        }

    # --------------------------------------------------------
    # Map numeric labels to class names
    # --------------------------------------------------------

    label_mapping = {}

    for class_name, values in class_performance.items():

        # Try to obtain numeric label from model classes
        for model_label, model_class in zip(
            model.classes_,
            model.classes_
        ):
            pass

    # XGBoost classes correspond to encoded labels
    # Use the known label encoder order from the project

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

    label_mapping = {
        index: name
        for index, name in enumerate(class_names)
    }

    # --------------------------------------------------------
    # Generate reliability results
    # --------------------------------------------------------

    results = []

    print("\nCalculating reliability scores...")

    for i in range(len(X_test)):

        predicted_label = int(predicted_labels[i])

        predicted_class = label_mapping.get(
            predicted_label,
            "Unknown"
        )

        prediction_confidence = float(confidence[i])

        performance = class_performance.get(
            predicted_class,
            {
                "f1": 0.0,
                "recall": 0.0,
                "support": 0
            }
        )

        class_f1 = performance["f1"]
        class_recall = performance["recall"]
        support = performance["support"]

        # ----------------------------------------------------
        # Reliability Score
        # ----------------------------------------------------

        reliability_score = (
            CONFIDENCE_WEIGHT * prediction_confidence
            +
            F1_WEIGHT * class_f1
            +
            RECALL_WEIGHT * class_recall
        )

        reliability_category = get_reliability_category(
            reliability_score
        )

        # ----------------------------------------------------
        # Limited evidence flag
        # ----------------------------------------------------

        if support < 10:
            evidence_status = "Limited Evidence"
        else:
            evidence_status = "Sufficient Evidence"

        results.append({
            "predicted_class": predicted_class,
            "confidence": prediction_confidence,
            "class_f1": class_f1,
            "class_recall": class_recall,
            "class_support": support,
            "reliability_score": reliability_score,
            "reliability_percent": reliability_score * 100,
            "reliability_category": reliability_category,
            "evidence_status": evidence_status
        })

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    reliability_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_REPORT_PATH),
        exist_ok=True
    )

    reliability_df.to_csv(
        OUTPUT_REPORT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nReliability Analysis Completed.")

    print(
        f"\nTotal Predictions: "
        f"{len(reliability_df):,}"
    )

    print(
        f"Average Reliability: "
        f"{reliability_df['reliability_percent'].mean():.2f}%"
    )

    print("\nReliability Category Distribution:")

    print(
        reliability_df[
            "reliability_category"
        ].value_counts()
    )

    print("\nEvidence Status Distribution:")

    print(
        reliability_df[
            "evidence_status"
        ].value_counts()
    )

    print("\nOutput saved to:")

    print(OUTPUT_REPORT_PATH)

    print("\n" + "=" * 60)
    print("RELIABILITY ENGINE FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()