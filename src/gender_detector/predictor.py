import joblib
import numpy as np
from dataclasses import dataclass
from src.gender_detector import config
from src.gender_detector.embedder import get_face_embedding


@dataclass
class Prediction:
    label: str
    confidence: float | None


class GenderPredictor:

    def __init__(self, model_path=config.MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model at {model_path}. Run `python -m src.gender_detector.train` first.")
        self.model = joblib.load(model_path)

    def predict(self, image_path: str) -> Prediction:
        embedding = get_face_embedding(image_path)
        return self.predict_from_embedding(embedding)

    def predict_from_embedding(self, embedding: np.ndarray) -> Prediction:
        features = embedding.reshape(1, -1)
        label = self.model.predict(features)[0]

        return Prediction(label=label, confidence=self._confidence(features))

    def _confidence(self, features: np.ndarray) -> float | None:
        if not hasattr(self.model, "predict_proba"):
            return None

        if getattr(self.model, "probability", True) is False:
            return None

        proba = self.model.predict_proba(features)[0]
        return float(proba.max())
