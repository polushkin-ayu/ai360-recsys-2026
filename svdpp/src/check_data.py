"""Проверка RMSE, совпадения row_id и защиты от broadcasting."""
import math
import numpy as np
from src.metrics import rmse, rmse_by_row_id, require_same_row_ids

def require(condition, message):
    if not condition:
        raise ValueError(message)



def check_k6():
    manual = rmse([1, 3], [2, 1])
    require(math.isclose(manual, math.sqrt(2.5), rel_tol=0, abs_tol=1e-12), "manual RMSE failed")
    ids = np.array([0, 1], dtype=np.int64)
    aligned = rmse_by_row_id([1, 3], [2, 1], ids, ids)
    require(aligned == manual and type(manual) is float, "aligned RMSE/scalar type failed")
    invalid = {
        "different_lengths": lambda: rmse([1, 3], [2]),
        "broadcast_column": lambda: rmse([1, 3], [[2], [1]]),
        "broadcast_row": lambda: rmse([[1, 3]], [2, 1]),
        "both_2d": lambda: rmse([[1], [3]], [[2], [1]]),
        "scalar": lambda: rmse(1, 2),
        "empty": lambda: rmse([], []),
        "nan_truth": lambda: rmse([np.nan, 3], [2, 1]),
        "nan_prediction": lambda: rmse([1, 3], [2, np.nan]),
        "inf_truth": lambda: rmse([np.inf, 3], [2, 1]),
        "inf_prediction": lambda: rmse([1, 3], [2, -np.inf]),
        "complex": lambda: rmse([1 + 1j], [2]),
        "nonnumeric": lambda: rmse(["1"], ["2"]),
        "reordered_row_ids": lambda: rmse_by_row_id([1, 3], [2, 1], ids, ids[::-1]),
        "different_row_ids": lambda: require_same_row_ids(ids, [0, 2]),
        "duplicate_row_ids": lambda: require_same_row_ids([0, 0], [0, 0]),
        "missing_row_ids": lambda: require_same_row_ids(ids, [0]),
        "noninteger_row_ids": lambda: require_same_row_ids([0.0, 1.0], ids),
        "row_ids_length_mismatch": lambda: rmse_by_row_id([1, 3], [2, 1], [0], [0]),
    }
    rejected = []
    for name, function in invalid.items():
        try:
            function()
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError(f"K6 accepted invalid input: {name}")
    require(rmse([1, 3], [1, 3]) == 0, "zero error failed")
    require(math.isfinite(rmse([1e200], [0])), "large finite error overflowed")
    return {"status": "passed", "manual_rmse": manual, "expected": math.sqrt(2.5),
            "aligned_rmse": aligned, "invalid_inputs_rejected": rejected,
            "finite_large_values_checked": True, "zero_error_checked": True}



