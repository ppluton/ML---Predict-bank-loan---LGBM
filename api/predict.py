import json
import logging

import joblib
import numpy as np
import pandas as pd

from api.config import Settings

logger = logging.getLogger("credit_scoring_api")


class CreditScoringModel:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.model = joblib.load(self.settings.MODEL_PATH)
        with open(self.settings.FEATURE_NAMES_PATH) as f:
            self.feature_names = json.load(f)
        with open(self.settings.METADATA_PATH) as f:
            self.metadata = json.load(f)
        self.threshold = self.settings.OPTIMAL_THRESHOLD
        self._ort_session = None
        self._ort_input_name = None
        self._ort_output_name = None
        self._use_onnx = False

        onnx_path = self.settings.ONNX_MODEL_PATH
        if onnx_path.exists():
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self._ort_session = ort.InferenceSession(
                    str(onnx_path), opts, providers=["CPUExecutionProvider"]
                )
                self._ort_input_name = self._ort_session.get_inputs()[0].name
                self._ort_output_name = self._ort_session.get_outputs()[1].name
                self._use_onnx = True
                logger.info(json.dumps({"event": "onnx_loaded", "path": str(onnx_path)}))
            except Exception as e:
                logger.warning(json.dumps({"event": "onnx_load_failed", "error": str(e)}))

    def predict(self, client_data: dict) -> dict:
        proba = self._predict_onnx(client_data) if self._use_onnx else self._predict_lgbm(client_data)
        prediction = int(proba >= self.threshold)
        return {
            "probability": round(proba, 6),
            "prediction": prediction,
            "threshold": self.threshold,
            "decision": "REFUSED" if prediction == 1 else "APPROVED",
        }

    def _predict_lgbm(self, client_data: dict) -> float:
        df = pd.DataFrame([client_data])
        df = df.reindex(columns=self.feature_names, fill_value=0)
        return float(self.model.predict_proba(df)[:, 1][0])

    def _predict_onnx(self, client_data: dict) -> float:
        X = np.zeros((1, len(self.feature_names)), dtype=np.float32)
        for name, value in client_data.items():
            try:
                X[0, self.feature_names.index(name)] = float(value)
            except ValueError:
                pass
        outputs = self._ort_session.run([self._ort_output_name], {self._ort_input_name: X})
        return float(outputs[0][0][1])

    @property
    def inference_engine(self) -> str:
        return "onnx" if self._use_onnx else "lightgbm"
