import os
import sqlite3
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
    "cti_alerts.csv"
)

DATABASE_DIR = os.path.join(
    BASE_DIR,
    "database"
)

DATABASE_PATH = os.path.join(
    DATABASE_DIR,
    "cyber_threat_intelligence.db"
)


# ============================================================
# DATABASE TABLE
# ============================================================

CREATE_TABLE_QUERY = """
CREATE TABLE IF NOT EXISTS alerts (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    alert_id TEXT UNIQUE NOT NULL,

    timestamp TEXT NOT NULL,

    attack_type TEXT NOT NULL,

    threat_score REAL NOT NULL,

    priority TEXT NOT NULL,

    confidence_percent REAL NOT NULL,

    reliability_percent REAL NOT NULL,

    evidence_status TEXT,

    status TEXT,

    recommended_action TEXT
);
"""


# ============================================================
# CREATE DATABASE
# ============================================================

def create_database():

    print("\nCreating database...")

    os.makedirs(
        DATABASE_DIR,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        CREATE_TABLE_QUERY
    )

    connection.commit()

    return connection


# ============================================================
# INSERT ALERTS
# ============================================================

def insert_alerts(connection, dataframe):

    print("\nInserting CTI alerts...")

    insert_query = """
    INSERT OR IGNORE INTO alerts (
        alert_id,
        timestamp,
        attack_type,
        threat_score,
        priority,
        confidence_percent,
        reliability_percent,
        evidence_status,
        status,
        recommended_action
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    records = []

    for _, row in dataframe.iterrows():

        records.append((
            row["alert_id"],
            row["timestamp"],
            row["attack_type"],
            float(row["threat_score"]),
            row["priority"],
            float(row["confidence_percent"]),
            float(row["reliability_percent"]),
            row["evidence_status"],
            row["status"],
            row["recommended_action"]
        ))

    connection.executemany(
        insert_query,
        records
    )

    connection.commit()

    print(
        f"Inserted/verified {len(records):,} alerts."
    )


# ============================================================
# VERIFY DATABASE
# ============================================================

def verify_database(connection):

    print("\nVerifying database...")

    cursor = connection.cursor()

    # Total alerts
    cursor.execute(
        "SELECT COUNT(*) FROM alerts;"
    )

    total_alerts = cursor.fetchone()[0]

    print(
        f"Total alerts in database: "
        f"{total_alerts:,}"
    )

    # Priority distribution
    cursor.execute("""
        SELECT priority, COUNT(*)
        FROM alerts
        GROUP BY priority
        ORDER BY COUNT(*) DESC;
    """)

    priority_results = cursor.fetchall()

    print("\nPriority Distribution:")

    for priority, count in priority_results:

        print(
            f"{priority}: {count:,}"
        )

    # Attack distribution
    cursor.execute("""
        SELECT attack_type, COUNT(*)
        FROM alerts
        GROUP BY attack_type
        ORDER BY COUNT(*) DESC;
    """)

    attack_results = cursor.fetchall()

    print("\nAttack Distribution:")

    for attack, count in attack_results:

        print(
            f"{attack}: {count:,}"
        )

    # Recent alerts
    cursor.execute("""
        SELECT
            alert_id,
            attack_type,
            threat_score,
            priority
        FROM alerts
        ORDER BY id DESC
        LIMIT 5;
    """)

    recent_alerts = cursor.fetchall()

    print("\nLatest 5 Alerts:")

    for alert in recent_alerts:

        print(
            f"ID: {alert[0]} | "
            f"Attack: {alert[1]} | "
            f"Score: {alert[2]} | "
            f"Priority: {alert[3]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CYBER THREAT INTELLIGENCE DATABASE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load CTI alerts
    # --------------------------------------------------------

    print("\nLoading CTI alerts...")

    if not os.path.exists(INPUT_PATH):

        print("\nERROR: CTI alert file not found:")
        print(INPUT_PATH)

        return

    dataframe = pd.read_csv(
        INPUT_PATH
    )

    print(
        f"CTI Alerts Loaded: "
        f"{len(dataframe):,}"
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = [
        "alert_id",
        "timestamp",
        "attack_type",
        "threat_score",
        "priority",
        "confidence_percent",
        "reliability_percent",
        "evidence_status",
        "status",
        "recommended_action"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        print("\nERROR: Missing columns:")

        for column in missing_columns:
            print(f"- {column}")

        return

    # --------------------------------------------------------
    # Create database
    # --------------------------------------------------------

    connection = create_database()

    # --------------------------------------------------------
    # Insert alerts
    # --------------------------------------------------------

    insert_alerts(
        connection,
        dataframe
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_database(
        connection
    )

    # --------------------------------------------------------
    # Close connection
    # --------------------------------------------------------

    connection.close()

    print("\nDatabase location:")

    print(DATABASE_PATH)

    print("\n" + "=" * 60)
    print("DATABASE SETUP COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()