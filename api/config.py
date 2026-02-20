from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: Path = Path("artifacts/model.pkl")
    SCALER_PATH: Path = Path("artifacts/scaler.pkl")
    FEATURE_NAMES_PATH: Path = Path("artifacts/feature_names.json")
    METADATA_PATH: Path = Path("artifacts/model_metadata.json")
    ONNX_MODEL_PATH: Path = Path("artifacts/model.onnx")
    OPTIMAL_THRESHOLD: float = 0.494
    LOG_PREDICTIONS: bool = True
    PREDICTIONS_LOG_PATH: Path = Path("monitoring/predictions_log.jsonl")
    DATABASE_URL: str = ""
