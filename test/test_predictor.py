import numpy as np
import pytest
from src.gender_detector import config
from src.gender_detector.train import load_vectors
from src.gender_detector.predictor import GenderPredictor, Prediction


@pytest.fixture(scope="module")
def predictor():
    if not config.MODEL_PATH.exists():
        pytest.skip("No trained model; run `python -m src.gender_detector.train` first.")
    return GenderPredictor()


def test_predict_from_embedding_returns_a_known_label(predictor):
    X, _ = load_vectors(config.TEST_VECTORS)
    prediction = predictor.predict_from_embedding(X[0])

    assert isinstance(prediction, Prediction)
    assert prediction.label in {"male", "female"}


def test_predictions_match_sklearn_directly(predictor):
    X, _ = load_vectors(config.TEST_VECTORS)
    sample = X[:5]

    for row in sample:
        ours = predictor.predict_from_embedding(np.array(row)).label
        sklearn_label = predictor.model.predict(row.reshape(1, -1))[0]
        assert ours == sklearn_label
