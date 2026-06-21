from src.gender_detector import config
from src.gender_detector.train import load_vectors, build_candidates, train_and_select


def test_vectors_load_with_expected_shape():
    X, y = load_vectors(config.TRAIN_VECTORS)
    assert X.shape[1] == 512
    assert len(X) == len(y)
    assert set(y) <= {"male", "female"}


def test_candidates_are_named_models():
    candidates = build_candidates()
    assert "svc" in candidates
    assert "logistic_regression" in candidates


def test_best_model_beats_a_coin_flip():
    model, name, score = train_and_select()
    assert score > 0.7
    assert name is not None
