import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import Settings
from api.database import log_api_event_to_db, log_prediction_to_db
from api.predict import CreditScoringModel
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse

# --- Logging structuré JSON ---
logger = logging.getLogger("credit_scoring_api")
logger.setLevel(logging.INFO)

# Handler console avec format JSON
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage de l'application."""
    settings = Settings()
    app.state.model = CreditScoringModel(settings)
    app.state.settings = settings

    # Créer le dossier de logs si nécessaire
    settings.PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    _log_event("api_startup", model_type=type(app.state.model.model).__name__,
               n_features=len(app.state.model.feature_names),
               threshold=app.state.model.threshold)

    yield

    _log_event("api_shutdown")


app = FastAPI(
    title="Credit Scoring API",
    description="API de scoring crédit - Prédit le risque de défaut de paiement",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://oc6-dashboard.onrender.com",
        "http://localhost:8501",
        "http://localhost:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Vérifie que l'API et le modèle sont opérationnels."""
    model = app.state.model
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        n_features=len(model.feature_names),
        threshold=model.threshold,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Prédit le risque de défaut pour un client."""
    start = time.perf_counter()

    result = app.state.model.predict(request.features)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    # Log en JSONL (backup fichier local)
    if app.state.settings.LOG_PREDICTIONS:
        _log_prediction(
            request.SK_ID_CURR,
            result["probability"],
            result["prediction"],
            elapsed_ms,
        )

    # Log en Postgres (inputs + outputs + temps d'exécution)
    log_prediction_to_db(
        sk_id_curr=request.SK_ID_CURR,
        probability=result["probability"],
        prediction=result["prediction"],
        threshold=result["threshold"],
        decision=result["decision"],
        inference_time_ms=elapsed_ms,
        features_dict=request.features,
    )

    return PredictionResponse(
        SK_ID_CURR=request.SK_ID_CURR,
        inference_time_ms=elapsed_ms,
        **result,
    )


@app.get("/model-info")
def model_info():
    """Retourne les métadonnées du modèle."""
    model = app.state.model
    return {
        "model_type": type(model.model).__name__,
        "n_features": len(model.feature_names),
        "threshold": model.threshold,
        "metadata": model.metadata,
    }


@app.post("/internal/logs")
def ingest_log(payload: dict):
    """Reçoit les logs de Fluentd et les stocke dans Postgres."""
    success = log_api_event_to_db(payload)
    return {"status": "ok" if success else "no_db"}


def _log_event(event: str, **kwargs):
    """Émet un log structuré JSON sur stdout."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **kwargs,
    }
    logger.info(json.dumps(record, default=str))


def _log_prediction(
    client_id: int, probability: float, prediction: int, inference_time_ms: float
):
    """Enregistre une prédiction en JSONL (JSON Lines)."""
    log_path = app.state.settings.PREDICTIONS_LOG_PATH
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "SK_ID_CURR": client_id,
        "probability": round(probability, 6),
        "prediction": prediction,
        "inference_time_ms": inference_time_ms,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Log structuré sur stdout
    _log_event("prediction", SK_ID_CURR=client_id,
               probability=round(probability, 6),
               prediction=prediction,
               inference_time_ms=inference_time_ms)
