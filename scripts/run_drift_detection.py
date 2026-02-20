"""Détection de drift par fenêtre temporelle.

Compare la distribution des features des prédictions récentes (Postgres)
contre les données de référence (training data).

Usage:
    DATABASE_URL=... python scripts/run_drift_detection.py
    DATABASE_URL=... python scripts/run_drift_detection.py --window-hours 48
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import psycopg2

# Permet l'import depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitoring.drift import compute_drift_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")
REFERENCE_DATA_PATH = Path("data/train_preprocessed.csv")
WINDOW_HOURS = int(os.environ.get("DRIFT_WINDOW_HOURS", "24"))
MIN_PREDICTIONS = int(os.environ.get("DRIFT_MIN_PREDICTIONS", "30"))
TOP_FEATURES = 30


def fetch_recent_predictions(conn, window_hours):
    """Récupère les features des prédictions récentes depuis Postgres."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    query = """
        SELECT features, created_at
        FROM predictions
        WHERE created_at >= %s AND features IS NOT NULL
        ORDER BY created_at
    """
    df = pd.read_sql(query, conn, params=(cutoff,))
    if len(df) == 0:
        return pd.DataFrame(), cutoff, datetime.now(timezone.utc)

    # Expand JSONB → colonnes
    features_df = pd.json_normalize(df["features"])
    window_start = df["created_at"].min()
    window_end = df["created_at"].max()
    return features_df, window_start, window_end


def load_reference_data(feature_columns, n_samples=5000):
    """Charge un échantillon du training data comme référence."""
    if not REFERENCE_DATA_PATH.exists():
        logger.error(f"{REFERENCE_DATA_PATH} introuvable.")
        sys.exit(1)

    ref = pd.read_csv(REFERENCE_DATA_PATH, nrows=n_samples)
    drop_cols = [c for c in ["SK_ID_CURR", "TARGET"] if c in ref.columns]
    ref = ref.drop(columns=drop_cols)

    # Aligner sur les colonnes des prédictions
    common = [c for c in feature_columns if c in ref.columns]
    return ref[common], common


def run_detection():
    """Exécute la détection de drift et stocke le rapport en Postgres."""
    if not DATABASE_URL:
        logger.error("DATABASE_URL non définie.")
        sys.exit(1)

    conn = psycopg2.connect(DATABASE_URL)

    # Récupérer les prédictions récentes
    prod_df, window_start, window_end = fetch_recent_predictions(conn, WINDOW_HOURS)

    if len(prod_df) < MIN_PREDICTIONS:
        logger.info(
            f"Seulement {len(prod_df)} prédictions dans la fenêtre "
            f"(minimum requis: {MIN_PREDICTIONS}). Skip."
        )
        conn.close()
        return

    logger.info(
        f"Analyse de {len(prod_df)} prédictions "
        f"({window_start} → {window_end})"
    )

    # Charger la référence et aligner les colonnes
    ref_df, common_cols = load_reference_data(prod_df.columns.tolist())
    prod_aligned = prod_df[common_cols]

    # KS test par feature (réutilise monitoring/drift.py)
    drift_report = compute_drift_report(ref_df, prod_aligned, top_n=TOP_FEATURES)

    n_analyzed = len(drift_report)
    n_drifted = int(drift_report["drift_detected"].sum())
    drift_pct = round(n_drifted / n_analyzed * 100, 1) if n_analyzed > 0 else 0.0

    # Déterminer le status
    if drift_pct > 30:
        status = "alert"
    elif drift_pct > 10:
        status = "warning"
    else:
        status = "ok"

    # Stocker dans drift_reports
    drift_details = drift_report.to_dict(orient="records")

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO drift_reports
            (window_start, window_end, n_predictions, n_features_analyzed,
             n_features_drifted, drift_percentage, drift_details, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                window_start,
                window_end,
                len(prod_df),
                n_analyzed,
                n_drifted,
                drift_pct,
                json.dumps(drift_details),
                status,
            ),
        )
    conn.commit()
    conn.close()

    logger.info(
        f"Drift report: {n_drifted}/{n_analyzed} features driftées "
        f"({drift_pct}%) — status: {status}"
    )


if __name__ == "__main__":
    run_detection()
