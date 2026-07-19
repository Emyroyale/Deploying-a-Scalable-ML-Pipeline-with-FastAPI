import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from ml.model import compute_model_metrics, inference, train_model


def test_train_model_returns_random_forest():
    """
    Test that train_model returns a fitted RandomForestClassifier.
    """
    X_train = np.random.rand(20, 5)
    y_train = np.random.randint(0, 2, 20)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_expected_shape_and_type():
    """
    Test that inference returns one prediction per input row, as an array.
    """
    X_train = np.random.rand(20, 5)
    y_train = np.random.randint(0, 2, 20)
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_train.shape[0]


def test_compute_model_metrics_known_values():
    """
    Test that compute_model_metrics returns the correct precision,
    recall, and F1 for a known input.
    """
    y = np.array([1, 1, 0, 0, 1])
    preds = np.array([1, 0, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(0.8)
