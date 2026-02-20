"""Simule du trafic de production avec drift progressif.

Envoie les données du test set à l'API par batches :
- Batch 1-3 : données propres (pas de drift)
- Batch 4-6 : drift graduel (intensité croissante)
- Batch 7-8 : drift soudain (fort shift)

Les prédictions sont stockées dans Postgres avec timestamps,
permettant à run_drift_detection.py de détecter la progression du drift.

Usage:
    python scripts/simulate_production_traffic.py
    python scripts/simulate_production_traffic.py --batch-size 100 --n-batches 10 --delay 2
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

# Permet l'import depuis la racine du projet
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitoring.drift import simulate_drift

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000"
DATA_PATH = Path("data/test_preprocessed.csv")
FEATURE_NAMES_PATH = Path("artifacts/feature_names.json")


def send_batch(client, df, feature_names, batch_label, api_url):
    """Envoie un batch de prédictions à l'API."""
    successes = 0
    errors = 0
    for idx, row in df.iterrows():
        features = {col: float(row[col]) for col in feature_names if col in row.index}
        sk_id = int(row["SK_ID_CURR"]) if "SK_ID_CURR" in row.index else 100000 + idx

        try:
            resp = client.post(
                f"{api_url}/predict",
                json={"SK_ID_CURR": sk_id, "features": features},
                timeout=10.0,
            )
            if resp.status_code == 200:
                successes += 1
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"Requête échouée: {e}")

    logger.info(f"[{batch_label}] {successes} OK, {errors} erreurs sur {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Simule du trafic production avec drift progressif"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Prédictions par batch"
    )
    parser.add_argument(
        "--n-batches", type=int, default=8, help="Nombre de batches"
    )
    parser.add_argument(
        "--delay", type=float, default=3.0, help="Secondes entre batches"
    )
    parser.add_argument(
        "--api-url", type=str, default=API_URL, help="URL de l'API"
    )
    args = parser.parse_args()

    api_url = args.api_url

    # Charger le test set
    if not DATA_PATH.exists():
        logger.error(f"{DATA_PATH} introuvable.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    with open(FEATURE_NAMES_PATH) as f:
        model_features = json.load(f)

    feature_cols = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]

    client = httpx.Client()

    logger.info(
        f"Simulation: {args.n_batches} batches x {args.batch_size} prédictions"
    )

    for batch_idx in range(args.n_batches):
        # Échantillonner des lignes
        sample = df.sample(n=args.batch_size, random_state=batch_idx)

        # Appliquer un drift progressif
        if batch_idx < 3:
            drift_type, intensity = "none", 0.0
            label = f"Batch {batch_idx + 1}/{args.n_batches} [clean]"
        elif batch_idx < 6:
            drift_type = "gradual"
            intensity = round(0.1 * (batch_idx - 2), 1)  # 0.1, 0.2, 0.3
            label = f"Batch {batch_idx + 1}/{args.n_batches} [gradual i={intensity}]"
        else:
            drift_type = "sudden"
            intensity = round(0.3 + 0.2 * (batch_idx - 5), 1)  # 0.5, 0.7
            label = f"Batch {batch_idx + 1}/{args.n_batches} [sudden i={intensity}]"

        # Simuler le drift sur les features uniquement
        features_only = sample[feature_cols]
        drifted = simulate_drift(features_only, drift_type=drift_type, intensity=intensity)

        # Réassembler avec SK_ID_CURR
        drifted["SK_ID_CURR"] = sample["SK_ID_CURR"].values

        logger.info(f"Envoi: {label}")
        send_batch(client, drifted, model_features, label, api_url)

        if batch_idx < args.n_batches - 1:
            logger.info(f"Pause {args.delay}s...")
            time.sleep(args.delay)

    client.close()
    logger.info("Simulation terminée.")


if __name__ == "__main__":
    main()
