import os
import uuid
from datetime import datetime

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
    "threat_scoring_analysis.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "reports",
    "cti_alerts.csv"
)


# ============================================================
# ALERT GENERATOR
# ============================================================

def generate_alert(row, index):

    attack_type = row["predicted_class"]
    threat_score = float(row["threat_score"])
    confidence = float(row["confidence_percent"])
    reliability = float(row["reliability_percent"])
    priority = row["threat_priority"]
    evidence_status = row["evidence_status"]

    # --------------------------------------------------------
    # Ignore normal traffic
    # --------------------------------------------------------

    if attack_type == "BENIGN":
        return None

    # --------------------------------------------------------
    # Generate unique alert ID
    # --------------------------------------------------------

    alert_id = f"CTI-{index + 1:06d}-{uuid.uuid4().hex[:6].upper()}"

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------
    # Alert status
    # --------------------------------------------------------

    if priority == "Critical":
        status = "ACTIVE"

    elif priority == "High":
        status = "REVIEW"

    else:
        status = "MONITOR"

    # --------------------------------------------------------
    # Recommended action
    # --------------------------------------------------------

    if priority == "Critical":
        recommended_action = (
            "Immediate security investigation recommended"
        )

    elif priority == "High":
        recommended_action = (
            "Security review and further investigation recommended"
        )

    elif priority == "Medium":
        recommended_action = (
            "Monitor activity and investigate if repeated"
        )

    else:
        recommended_action = (
            "Continue monitoring"
        )

    # --------------------------------------------------------
    # Create alert
    # --------------------------------------------------------

    return {
        "alert_id": alert_id,
        "timestamp": timestamp,
        "attack_type": attack_type,
        "threat_score": threat_score,
        "priority": priority,
        "confidence_percent": confidence,
        "reliability_percent": reliability,
        "evidence_status": evidence_status,
        "status": status,
        "recommended_action": recommended_action
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CYBER THREAT INTELLIGENCE ALERT GENERATOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Load threat scoring results
    # --------------------------------------------------------

    print("\nLoading threat scoring analysis...")

    if not os.path.exists(INPUT_PATH):

        print("\nERROR: Input file not found:")
        print(INPUT_PATH)
        return

    df = pd.read_csv(INPUT_PATH)

    print(f"Input Shape: {df.shape}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = [
        "predicted_class",
        "threat_score",
        "threat_priority",
        "confidence_percent",
        "reliability_percent",
        "evidence_status"
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
    # Generate alerts
    # --------------------------------------------------------

    print("\nGenerating CTI alerts...")

    alerts = []

    for index, row in df.iterrows():

        alert = generate_alert(
            row,
            index
        )

        if alert is not None:
            alerts.append(alert)

    # --------------------------------------------------------
    # Create alert DataFrame
    # --------------------------------------------------------

    alerts_df = pd.DataFrame(alerts)

    # --------------------------------------------------------
    # Save alerts
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    alerts_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\nCTI Alert Generation Completed.")

    print(
        f"Total Predictions: {len(df):,}"
    )

    print(
        f"Total CTI Alerts: {len(alerts_df):,}"
    )

    if not alerts_df.empty:

        print("\nAlert Priority Distribution:")

        print(
            alerts_df["priority"]
            .value_counts()
        )

        print("\nAlert Status Distribution:")

        print(
            alerts_df["status"]
            .value_counts()
        )

        print("\nAttack Type Distribution:")

        print(
            alerts_df["attack_type"]
            .value_counts()
        )

        print("\nSample Alerts:")

        print(
            alerts_df.head(10).to_string(
                index=False
            )
        )

    print("\nOutput saved to:")

    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("CTI ALERT GENERATOR FINISHED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()