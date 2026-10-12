"""One shared RMSE and strict observation alignment for every model."""

import numpy as np


def _vector(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def rmse(y_true, prediction):
    """Return scalar RMSE; reject shape mismatches, broadcasting and NaN/Inf.

    Inputs must be nonempty 1D real numeric vectors of exactly the same shape.
    Values are not clipped. Scaling avoids overflow when squaring large errors.
    """
    truth = _vector(y_true, "y_true")
    pred = _vector(prediction, "prediction")
    if truth.shape != pred.shape:
        raise ValueError("y_true and prediction must have identical shapes")
    with np.errstate(over="ignore", invalid="ignore"):
        error = truth - pred
    if not np.isfinite(error).all():
        raise ValueError("prediction errors overflowed float64")
    scale = float(np.max(np.abs(error)))
    if scale == 0.0:
        return 0.0
    return float(scale * np.sqrt(np.mean((error / scale) ** 2)))


def require_same_row_ids(expected_row_ids, prediction_row_ids):
    """Reject missing, repeated, extra or reordered observation IDs.

    Use also to compare the evaluation IDs of different models. No automatic
    sorting/alignment: a mistake must be visible to the experiment author.
    """
    expected = np.asarray(expected_row_ids)
    actual = np.asarray(prediction_row_ids)
    for name, values in [("expected_row_ids", expected), ("prediction_row_ids", actual)]:
        if values.ndim != 1 or values.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a 1D integer array")
        if (values < 0).any() or len(np.unique(values)) != len(values):
            raise ValueError(f"{name} must contain unique nonnegative IDs")
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise ValueError("observation row_id values/order do not match")


def rmse_by_row_id(y_true, prediction, expected_row_ids, prediction_row_ids):
    """Evaluate only after checking the complete observation order."""
    require_same_row_ids(expected_row_ids, prediction_row_ids)
    if np.asarray(y_true).shape != np.asarray(expected_row_ids).shape:
        raise ValueError("ratings and expected_row_ids have different shapes")
    if np.asarray(prediction).shape != np.asarray(prediction_row_ids).shape:
        raise ValueError("predictions and prediction_row_ids have different shapes")
    return rmse(y_true, prediction)
