import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "reports",
    "reliability_analysis.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "reports",
    "threat_scoring_analysis.csv"
)


# ============================================================
# ATTACK SEVERITY MAPPING
# Project-defined operational severity weights
# ============================================================

ATTACK_SEVERITY = {
    "BENIGN": 0,

    "PortScan": 60,

    "Web Attack - Brute Force": 70,

    "FTP-Patator": 75,
    "SSH-Patator": 75,
    "Web Attack - XSS": 75,

    "DoS Slowhttptest": 80,
    "DoS slowloris": 80,

    "DoS Hulk": 85,
    "DoS GoldenEye": 85,
    "Bot": 85,

    "DDoS": 90,
    "Heartbleed": 90,
    "Web Attack - SQL Injection": 90,

    "Infiltration": 95
}


# ============================================================
# THREAT SCORE WEIGHTS
# ============================================================

SEVERITY_WEIGHT = 0.50
RELIABILITY_WEIGHT = 0.30
CONFIDENCE_WEIGHT = 0.20


# ============================================================
# THREAT PRIORITY
# ============================================================

def get_threat_priority(score):

    if score >= 80:
        return "Critical"

    elif score >= 60:
        return "High"

    elif score >= 40:
        return "Medium"

    elif score >= 20:
        return "Low"

    else:
        return "Informational"


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("THREAT SCORING ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load reliability analysis
    # --------------------------------------------------------

    print("\nLoading reliability analysis...")

    if not os.path.exists(INPUT_PATH):

        print("\nERROR: Reliability analysis file not found.")
        print(INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)

    print(f"Input Shape: {df.shape}")

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "predicted_class",
        "confidence",
        "reliability_score"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print("\nERROR: Missing required columns:")

        for column in missing_columns:
            print(f"- {column}")

        return

    # --------------------------------------------------------
    # Map attack severity
    # --------------------------------------------------------

    print("\nAssigning attack severity...")

    df["severity_score"] = (
        df["predicted_class"]
        .map(ATTACK_SEVERITY)
        .fillna(0)
    )

    # --------------------------------------------------------
    # Convert confidence and reliability to percentages
    # --------------------------------------------------------

    df["confidence_percent"] = (
        df["confidence"] * 100
    )

    df["reliability_percent"] = (
        df["reliability_score"] * 100
    )

    # --------------------------------------------------------
    # Calculate threat score
    # --------------------------------------------------------

    print("\nCalculating threat scores...")

    df["threat_score"] = (
        SEVERITY_WEIGHT * df["severity_score"]
        +
        RELIABILITY_WEIGHT * df["reliability_percent"]
        +
        CONFIDENCE_WEIGHT * df["confidence_percent"]
    )

    # --------------------------------------------------------
    # BENIGN traffic should not generate a threat
    # --------------------------------------------------------

    benign_mask = (
        df["predicted_class"] == "BENIGN"
    )

    df.loc[
        benign_mask,
        "threat_score"
    ] = 0

    # --------------------------------------------------------
    # Assign priority
    # --------------------------------------------------------

    df["threat_priority"] = (
        df["threat_score"]
        .apply(get_threat_priority)
    )

    # BENIGN = No Threat
    df.loc[
        benign_mask,
        "threat_priority"
    ] = "No Threat"

    # --------------------------------------------------------
    # Round values
    # --------------------------------------------------------

    df["confidence_percent"] = (
        df["confidence_percent"].round(2)
    )

    df["reliability_percent"] = (
        df["reliability_percent"].round(2)
    )

    df["severity_score"] = (
        df["severity_score"].round(2)
    )

    df["threat_score"] = (
        df["threat_score"].round(2)
    )

    # --------------------------------------------------------
    # Save complete analysis
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nThreat Scoring Completed.")

    print(
        f"Total Predictions: {len(df):,}"
    )

    print(
        f"\nAverage Threat Score: "
        f"{df['threat_score'].mean():.2f}"
    )

    print("\nThreat Priority Distribution:")

    print(
        df["threat_priority"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Attack-wise summary
    # --------------------------------------------------------

    attack_summary = (
        df[
            df["predicted_class"] != "BENIGN"
        ]
        .groupby("predicted_class")
        .agg(
            prediction_count=("predicted_class", "size"),
            average_threat_score=("threat_score", "mean"),
            average_confidence=("confidence_percent", "mean"),
            average_reliability=("reliability_percent", "mean"),
            severity=("severity_score", "first")
        )
        .reset_index()
        .sort_values(
            "average_threat_score",
            ascending=False
        )
    )

    print("\nAttack-wise Threat Summary:")

    print(
        attack_summary.to_string(
            index=False
        )
    )

    print("\nOutput saved to:")

    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("THREAT SCORING ENGINE FINISHED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()