"""Connexion Postgres (Neon) pour le logging des prédictions et logs API."""

import json
import logging
import os
from contextlib import contextmanager

from psycopg2 import pool

logger = logging.getLogger("credit_scoring_api")

_pool = None


def get_pool():
    """Crée ou retourne le pool de connexions Postgres."""
    global _pool
    if _pool is None:
        database_url = os.environ.get("DATABASE_URL", "")
        if database_url:
            try:
                _pool = pool.SimpleConnectionPool(1, 5, database_url)
                logger.info('{"event": "db_pool_created"}')
            except Exception as e:
                logger.warning(f'{{"event": "db_pool_failed", "error": "{e}"}}')
    return _pool


@contextmanager
def get_connection():
    """Context manager pour obtenir une connexion du pool."""
    p = get_pool()
    if p is None:
        yield None
        return
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def log_prediction_to_db(
    sk_id_curr: int,
    probability: float,
    prediction: int,
    threshold: float,
    decision: str,
    inference_time_ms: float,
    features_dict: dict,
):
    """Insère une prédiction avec ses features dans Postgres.

    Fail-safe : ne lève pas d'exception si la DB est indisponible.
    """
    try:
        with get_connection() as conn:
            if conn is None:
                return
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                    (sk_id_curr, probability, prediction, threshold,
                     decision, inference_time_ms, features)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        sk_id_curr,
                        probability,
                        prediction,
                        threshold,
                        decision,
                        inference_time_ms,
                        json.dumps(features_dict),
                    ),
                )
    except Exception as e:
        logger.warning(f'{{"event": "db_prediction_log_failed", "error": "{e}"}}')


def log_api_event_to_db(payload: dict):
    """Insère un log API dans Postgres (appelé par Fluentd via /internal/logs).

    Fail-safe : ne lève pas d'exception si la DB est indisponible.
    """
    try:
        with get_connection() as conn:
            if conn is None:
                return False
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO api_logs (timestamp, level, event, message, extra)
                    VALUES (COALESCE(%s, NOW()), %s, %s, %s, %s)
                    """,
                    (
                        payload.get("timestamp"),
                        payload.get("level", "INFO"),
                        payload.get("event", "unknown"),
                        json.dumps(payload),
                        json.dumps(payload),
                    ),
                )
            return True
    except Exception as e:
        logger.warning(f'{{"event": "db_log_failed", "error": "{e}"}}')
        return False
