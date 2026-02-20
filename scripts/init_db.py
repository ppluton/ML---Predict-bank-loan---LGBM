"""Initialise le schema Postgres et seed les statistiques de référence du training data."""

import os
import sys
from pathlib import Path

import pandas as pd
import psycopg2

# Permet l'import depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATABASE_URL = os.environ.get("DATABASE_URL", "")
SQL_PATH = Path(__file__).parent / "init_db.sql"
TRAIN_DATA_PATH = Path("data/train_preprocessed.csv")


def init_schema(conn):
    """Exécute le DDL pour créer les tables."""
    sql = SQL_PATH.read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("Schema initialisé.")


def seed_training_reference(conn, sample_size=10000):
    """Calcule et insère les stats descriptives du training data.

    Ces stats servent de référence pour la détection de drift.
    """
    if not TRAIN_DATA_PATH.exists():
        print(f"WARN: {TRAIN_DATA_PATH} introuvable, skip seed.")
        return

    print(f"Chargement de {TRAIN_DATA_PATH} (sample={sample_size})...")
    df = pd.read_csv(TRAIN_DATA_PATH, nrows=sample_size)

    # Retirer les colonnes non-features
    drop_cols = [c for c in ["SK_ID_CURR", "TARGET"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    print(f"Calcul des stats pour {len(df.columns)} features...")
    with conn.cursor() as cur:
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            cur.execute(
                """
                INSERT INTO training_reference
                (feature_name, mean, std, median, q25, q75, n_samples)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (feature_name) DO UPDATE SET
                    mean=EXCLUDED.mean, std=EXCLUDED.std,
                    median=EXCLUDED.median, q25=EXCLUDED.q25, q75=EXCLUDED.q75,
                    n_samples=EXCLUDED.n_samples
                """,
                (
                    col,
                    float(series.mean()),
                    float(series.std()),
                    float(series.median()),
                    float(series.quantile(0.25)),
                    float(series.quantile(0.75)),
                    int(len(series)),
                ),
            )
    conn.commit()
    print(f"Stats de référence insérées pour {len(df.columns)} features.")


def main():
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL non définie.")
        sys.exit(1)

    conn = psycopg2.connect(DATABASE_URL)
    try:
        init_schema(conn)
        seed_training_reference(conn)
    finally:
        conn.close()

    print("Initialisation terminée.")


if __name__ == "__main__":
    main()
