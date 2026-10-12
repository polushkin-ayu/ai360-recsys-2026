from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.context_fm_experiment import (
    ContextFM,
    fit_one as fit_context,
    predict as predict_context,
    prepare_context_frames,
)
from src.data import MODEL_COLUMNS, load_prepared
from src.metrics import rmse_by_row_id
from src.models import (
    FactorizationMachine,
    TrainingConfig,
    count_parameters,
    fit_model,
    predict_frame,
)


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "movielens1m"

OUTPUT_DIR = (
    ROOT
    / "results"
    / "fm-parameter-lr-sweep"
)

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
HISTORY_DIR = OUTPUT_DIR / "histories"

PLAN_PATH = OUTPUT_DIR / "budget_plan.csv"
TUNING_PATH = OUTPUT_DIR / "tuning_results.csv"
SELECTED_PATH = OUTPUT_DIR / "best_by_validation.csv"
FINAL_PATH = OUTPUT_DIR / "final_results.csv"

PLOT_PATH = (
    OUTPUT_DIR
    / "fm_best_lr_test_rmse_vs_parameters.png"
)


# ============================================================
# Experiment
# ============================================================

MODELS = [
    "FM(u,i)",
    "FM(u,i,t)",
    "FM(u,i,t,f)",
]

BUDGETS = [
    100_000,
    200_000,
    300_000,
    400_000,
    500_000,
    1_000_000,
]

LEARNING_RATES = [
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
]

SEED = 42

EPOCHS = 100
BATCH_SIZE = 1024
PATIENCE = 10

REG_BIAS = 0.01
REG_FACTORS = 10.0

INIT_STD = 0.01

THREADS = 4


# ============================================================
# Helpers
# ============================================================

def safe_model_name(model_name: str) -> str:
    mapping = {
        "FM(u,i)": "fm_ui",
        "FM(u,i,t)": "fm_uit",
        "FM(u,i,t,f)": "fm_uitf",
    }

    return mapping[model_name]


def lr_tag(lr: float) -> str:
    return (
        f"{lr:.5g}"
        .replace(".", "p")
        .replace("-", "m")
    )


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def write_csv_atomic(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def upsert_csv(
    path: Path,
    row: dict,
    keys: list[str],
) -> None:
    old = read_csv_or_empty(path)

    new = pd.DataFrame([row])

    if old.empty:
        result = new
    else:
        result = pd.concat(
            [old, new],
            ignore_index=True,
        )

    result = result.drop_duplicates(
        subset=keys,
        keep="last",
    )

    write_csv_atomic(
        result,
        path,
    )


def checkpoint_path(
    model_name: str,
    budget: int,
    learning_rate: float,
) -> Path:
    return (
        CHECKPOINT_DIR
        / (
            f"{safe_model_name(model_name)}"
            f"_budget{budget}"
            f"_lr{lr_tag(learning_rate)}"
            f"_seed{SEED}.pt"
        )
    )


def history_path(
    model_name: str,
    budget: int,
    learning_rate: float,
) -> Path:
    return (
        HISTORY_DIR
        / (
            f"{safe_model_name(model_name)}"
            f"_budget{budget}"
            f"_lr{lr_tag(learning_rate)}"
            f"_seed{SEED}.csv"
        )
    )


# ============================================================
# Parameter counting
# ============================================================

def parameter_count_formula(
    model_name: str,
    k: int,
    *,
    n_users: int,
    n_items: int,
    n_days: int,
) -> int:

    if model_name == "FM(u,i)":
        # global bias
        # + user/item biases
        # + user/item latent factors

        base = (
            1
            + n_users
            + n_items
        )

        slope = (
            n_users
            + n_items
        )

        return base + slope * k

    if model_name == "FM(u,i,t)":
        # global bias
        # + user/item/day biases
        # + user/item/day latent factors

        n_features = (
            n_users
            + n_items
            + n_days
        )

        base = 1 + n_features
        slope = n_features

        return base + slope * k

    if model_name == "FM(u,i,t,f)":
        # same as above
        # + scalar frequency weight
        # + one k-dimensional frequency vector

        n_features = (
            n_users
            + n_items
            + n_days
        )

        base = (
            2
            + n_features
        )

        slope = (
            n_features
            + 1
        )

        return base + slope * k

    raise ValueError(
        f"Unknown model: {model_name}"
    )


def closest_k(
    model_name: str,
    budget: int,
    *,
    n_users: int,
    n_items: int,
    n_days: int,
) -> tuple[int, int]:

    # Estimate k, then check nearby
    # integer values exactly.

    if model_name == "FM(u,i)":
        base = 1 + n_users + n_items
        slope = n_users + n_items

    elif model_name == "FM(u,i,t)":
        n_features = (
            n_users
            + n_items
            + n_days
        )

        base = 1 + n_features
        slope = n_features

    elif model_name == "FM(u,i,t,f)":
        n_features = (
            n_users
            + n_items
            + n_days
        )

        base = 2 + n_features
        slope = n_features + 1

    else:
        raise ValueError(
            model_name
        )

    approximate = (
        budget - base
    ) / slope

    center = max(
        1,
        int(round(approximate)),
    )

    candidates = range(
        max(1, center - 3),
        center + 4,
    )

    best_k = None
    best_params = None
    best_difference = None

    for k in candidates:
        params = parameter_count_formula(
            model_name,
            k,
            n_users=n_users,
            n_items=n_items,
            n_days=n_days,
        )

        difference = abs(
            params - budget
        )

        if (
            best_difference is None
            or difference < best_difference
        ):
            best_difference = difference
            best_k = k
            best_params = params

    return (
        int(best_k),
        int(best_params),
    )


def build_budget_plan(
    data,
    n_days: int,
) -> pd.DataFrame:

    rows = []

    for model_name in MODELS:
        for budget in BUDGETS:

            k, actual_params = closest_k(
                model_name,
                budget,
                n_users=data.n_users,
                n_items=data.n_items,
                n_days=n_days,
            )

            rows.append(
                {
                    "model": model_name,
                    "budget": budget,
                    "k": k,
                    "actual_params": actual_params,
                    "difference": (
                        actual_params
                        - budget
                    ),
                }
            )

    plan = pd.DataFrame(rows)

    plan = plan.sort_values(
        [
            "model",
            "budget",
        ]
    ).reset_index(drop=True)

    write_csv_atomic(
        plan,
        PLAN_PATH,
    )

    return plan


# ============================================================
# Resume
# ============================================================

def tuning_already_recorded(
    model_name: str,
    budget: int,
    learning_rate: float,
    *,
    retry_failed: bool,
) -> bool:

    table = read_csv_or_empty(
        TUNING_PATH
    )

    if table.empty:
        return False

    mask = (
        (table["model"] == model_name)
        & (table["budget"] == budget)
        & np.isclose(
            table["learning_rate"],
            learning_rate,
        )
        & (table["seed"] == SEED)
    )

    matches = table[mask]

    if matches.empty:
        return False

    if retry_failed:
        return bool(
            (
                matches["status"]
                == "ok"
            ).any()
        )

    return True


# ============================================================
# Common RMSE
# ============================================================

def ui_tensors(frame):
    users = torch.as_tensor(
        frame[
            "user_idx"
        ].to_numpy(copy=True),
        dtype=torch.long,
    )

    items = torch.as_tensor(
        frame[
            "item_idx"
        ].to_numpy(copy=True),
        dtype=torch.long,
    )

    ratings = torch.as_tensor(
        frame[
            "rating"
        ].to_numpy(copy=True),
        dtype=torch.float32,
    )

    return (
        users,
        items,
        ratings,
    )


def evaluate_ui_rmse(
    model,
    frame,
) -> float:

    predictions, row_ids = (
        predict_frame(
            model,
            frame,
            batch_size=16384,
        )
    )

    return rmse_by_row_id(
        frame[
            "rating"
        ].to_numpy(),
        predictions,
        frame[
            "row_id"
        ].to_numpy(),
        row_ids,
    )


def evaluate_context_rmse(
    model,
    frame,
) -> float:

    predictions, row_ids = (
        predict_context(
            model,
            frame,
            batch_size=16384,
        )
    )

    return rmse_by_row_id(
        frame[
            "rating"
        ].to_numpy(),
        predictions,
        frame[
            "row_id"
        ].to_numpy(),
        row_ids,
    )


# ============================================================
# FM(u,i)
# ============================================================

def tune_fm_ui(
    data,
    plan_row,
    learning_rate: float,
) -> dict:

    model_name = "FM(u,i)"

    budget = int(
        plan_row["budget"]
    )

    k = int(
        plan_row["k"]
    )

    expected_params = int(
        plan_row["actual_params"]
    )

    train = (
        data.train[
            MODEL_COLUMNS
        ]
        .copy()
    )

    validation = (
        data.validation[
            MODEL_COLUMNS
        ]
        .copy()
    )

    (
        train_users,
        train_items,
        train_ratings,
    ) = ui_tensors(train)

    (
        val_users,
        val_items,
        val_ratings,
    ) = ui_tensors(validation)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = FactorizationMachine(
        n_users=data.n_users,
        n_items=data.n_items,
        n_factors=k,
        init_std=INIT_STD,
    )

    actual_params = (
        count_parameters(model)
    )

    if actual_params != expected_params:
        raise RuntimeError(
            f"{model_name}, budget={budget}: "
            f"planned {expected_params} parameters, "
            f"actual model has {actual_params}"
        )

    config = TrainingConfig(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        optimizer="adam",
        reg_bias=REG_BIAS,
        reg_factors=REG_FACTORS,
        normalize_regularization=True,
        patience=PATIENCE,
        min_delta=0.0,
        seed=SEED,
    )

    result = fit_model(
        model=model,
        train_user_idx=train_users,
        train_item_idx=train_items,
        train_rating=train_ratings,
        val_user_idx=val_users,
        val_item_idx=val_items,
        val_rating=val_ratings,
        config=config,
    )

    # fit_model has already restored
    # the best validation checkpoint.

    validation_rmse = (
        evaluate_ui_rmse(
            model,
            validation,
        )
    )

    ckpt = checkpoint_path(
        model_name,
        budget,
        learning_rate,
    )

    torch.save(
        {
            "model": model_name,
            "state_dict": model.state_dict(),
            "k": k,
            "budget": budget,
            "actual_params": actual_params,
            "learning_rate": learning_rate,
            "seed": SEED,
        },
        ckpt,
    )

    history = pd.DataFrame(
        result.history
    )

    history.to_csv(
        history_path(
            model_name,
            budget,
            learning_rate,
        ),
        index=False,
    )

    return {
        "validation_rmse": (
            validation_rmse
        ),
        "best_epoch": (
            result.best_epoch
        ),
        "fit_seconds": (
            result.fit_time_seconds
        ),
        "checkpoint": str(
            ckpt.relative_to(ROOT)
        ),
    }


# ============================================================
# FM(u,i,t) and FM(u,i,t,f)
# ============================================================

def tune_context_fm(
    *,
    model_name: str,
    use_frequency: bool,
    data,
    frames,
    n_days: int,
    plan_row,
    learning_rate: float,
) -> dict:

    budget = int(
        plan_row["budget"]
    )

    k = int(
        plan_row["k"]
    )

    expected_params = int(
        plan_row["actual_params"]
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = ContextFM(
        data.n_users,
        data.n_items,
        n_days,
        n_factors=k,
        use_frequency=use_frequency,
        init_std=INIT_STD,
    )

    actual_params = (
        count_parameters(model)
    )

    if actual_params != expected_params:
        raise RuntimeError(
            f"{model_name}, budget={budget}: "
            f"planned {expected_params} parameters, "
            f"actual model has {actual_params}"
        )

    result = fit_context(
        model,
        frames["train"],
        frames["validation"],
        seed=SEED,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=learning_rate,
        reg_bias=REG_BIAS,
        reg_factors=REG_FACTORS,
        patience=PATIENCE,
    )

    # fit_context restored best_state.

    validation_rmse = (
        evaluate_context_rmse(
            model,
            frames["validation"],
        )
    )

    ckpt = checkpoint_path(
        model_name,
        budget,
        learning_rate,
    )

    torch.save(
        {
            "model": model_name,
            "state_dict": model.state_dict(),
            "k": k,
            "n_days": n_days,
            "use_frequency": use_frequency,
            "budget": budget,
            "actual_params": actual_params,
            "learning_rate": learning_rate,
            "seed": SEED,
        },
        ckpt,
    )

    result["history"].to_csv(
        history_path(
            model_name,
            budget,
            learning_rate,
        ),
        index=False,
    )

    return {
        "validation_rmse": (
            validation_rmse
        ),
        "best_epoch": (
            result["best_epoch"]
        ),
        "fit_seconds": (
            result["fit_seconds"]
        ),
        "checkpoint": str(
            ckpt.relative_to(ROOT)
        ),
    }


# ============================================================
# Full tuning
# ============================================================

def run_tuning(
    plan: pd.DataFrame,
    data,
    frames,
    n_days: int,
    *,
    retry_failed: bool,
) -> None:

    total = (
        len(MODELS)
        * len(BUDGETS)
        * len(LEARNING_RATES)
    )

    current = 0

    for model_name in MODELS:

        model_plan = (
            plan[
                plan["model"]
                == model_name
            ]
            .sort_values("budget")
        )

        for _, plan_row in (
            model_plan.iterrows()
        ):

            budget = int(
                plan_row["budget"]
            )

            k = int(
                plan_row["k"]
            )

            actual_params = int(
                plan_row[
                    "actual_params"
                ]
            )

            for learning_rate in (
                LEARNING_RATES
            ):

                current += 1

                if tuning_already_recorded(
                    model_name,
                    budget,
                    learning_rate,
                    retry_failed=(
                        retry_failed
                    ),
                ):
                    print(
                        f"[{current}/{total}] "
                        f"SKIP "
                        f"{model_name} "
                        f"budget={budget:,} "
                        f"k={k} "
                        f"lr={learning_rate}"
                    )

                    continue

                print()
                print(
                    "=" * 80
                )

                print(
                    f"[{current}/{total}] "
                    f"{model_name}"
                )

                print(
                    f"budget       = {budget:,}"
                )

                print(
                    f"actual params= "
                    f"{actual_params:,}"
                )

                print(
                    f"k            = {k}"
                )

                print(
                    f"learning rate= "
                    f"{learning_rate}"
                )

                print(
                    "=" * 80
                )

                base_row = {
                    "model": model_name,
                    "budget": budget,
                    "actual_params": (
                        actual_params
                    ),
                    "k": k,
                    "learning_rate": (
                        learning_rate
                    ),
                    "seed": SEED,
                }

                try:

                    if (
                        model_name
                        == "FM(u,i)"
                    ):
                        result = tune_fm_ui(
                            data,
                            plan_row,
                            learning_rate,
                        )

                    elif (
                        model_name
                        == "FM(u,i,t)"
                    ):
                        result = (
                            tune_context_fm(
                                model_name=(
                                    model_name
                                ),
                                use_frequency=False,
                                data=data,
                                frames=frames,
                                n_days=n_days,
                                plan_row=(
                                    plan_row
                                ),
                                learning_rate=(
                                    learning_rate
                                ),
                            )
                        )

                    elif (
                        model_name
                        == "FM(u,i,t,f)"
                    ):
                        result = (
                            tune_context_fm(
                                model_name=(
                                    model_name
                                ),
                                use_frequency=True,
                                data=data,
                                frames=frames,
                                n_days=n_days,
                                plan_row=(
                                    plan_row
                                ),
                                learning_rate=(
                                    learning_rate
                                ),
                            )
                        )

                    else:
                        raise ValueError(
                            model_name
                        )

                    row = {
                        **base_row,
                        **result,

                        # Test is deliberately
                        # NOT evaluated here.
                        "test_rmse": np.nan,

                        "status": "ok",
                        "error": "",
                    }

                    print(
                        f"FINISHED | "
                        f"val RMSE = "
                        f"{row['validation_rmse']:.6f} "
                        f"| best epoch = "
                        f"{row['best_epoch']} "
                        f"| time = "
                        f"{row['fit_seconds']:.1f}s"
                    )

                except Exception as exc:

                    row = {
                        **base_row,
                        "validation_rmse": (
                            np.nan
                        ),
                        "test_rmse": np.nan,
                        "best_epoch": np.nan,
                        "fit_seconds": np.nan,
                        "checkpoint": "",
                        "status": "failed",
                        "error": repr(exc),
                    }

                    print(
                        "FAILED:",
                        repr(exc),
                    )

                upsert_csv(
                    TUNING_PATH,
                    row,
                    [
                        "model",
                        "budget",
                        "learning_rate",
                        "seed",
                    ],
                )


# ============================================================
# Select best LR using VALIDATION ONLY
# ============================================================

def select_best_learning_rates(
    plan: pd.DataFrame,
) -> pd.DataFrame:

    if not TUNING_PATH.exists():
        raise FileNotFoundError(
            f"{TUNING_PATH} not found. "
            f"Run --stage tune first."
        )

    tuning = pd.read_csv(
        TUNING_PATH
    )

    successful = tuning[
        tuning["status"] == "ok"
    ].copy()

    successful = successful.dropna(
        subset=[
            "validation_rmse"
        ]
    )

    selected_rows = []

    print()
    print(
        "=" * 80
    )
    print(
        "SELECTING BEST LR "
        "BY VALIDATION RMSE"
    )
    print(
        "=" * 80
    )

    for model_name in MODELS:

        for budget in BUDGETS:

            group = successful[
                (
                    successful["model"]
                    == model_name
                )
                & (
                    successful["budget"]
                    == budget
                )
                & (
                    successful["seed"]
                    == SEED
                )
            ].copy()

            if group.empty:
                raise RuntimeError(
                    f"No successful runs for "
                    f"{model_name}, "
                    f"budget={budget}"
                )

            best = (
                group.sort_values(
                    [
                        "validation_rmse",
                        "learning_rate",
                    ]
                )
                .iloc[0]
                .copy()
            )

            best[
                "best_learning_rate"
            ] = best[
                "learning_rate"
            ]

            selected_rows.append(
                best.to_dict()
            )

            print(
                f"{model_name:12s} "
                f"budget={budget:7,d} "
                f"k={int(best['k']):3d} "
                f"best_lr="
                f"{best['learning_rate']:<7g} "
                f"val="
                f"{best['validation_rmse']:.6f}"
            )

    selected = pd.DataFrame(
        selected_rows
    )

    selected = selected.sort_values(
        [
            "model",
            "budget",
        ]
    ).reset_index(
        drop=True
    )

    write_csv_atomic(
        selected,
        SELECTED_PATH,
    )

    print()
    print(
        f"Saved selection: "
        f"{SELECTED_PATH}"
    )

    return selected


# ============================================================
# Test evaluation
# ============================================================

def final_result_is_current(
    model_name: str,
    budget: int,
    learning_rate: float,
) -> bool:

    final = read_csv_or_empty(
        FINAL_PATH
    )

    if final.empty:
        return False

    mask = (
        (final["model"] == model_name)
        & (final["budget"] == budget)
        & np.isclose(
            final[
                "best_learning_rate"
            ],
            learning_rate,
        )
    )

    return bool(
        mask.any()
    )


def test_fm_ui(
    data,
    selected_row,
) -> float:

    k = int(
        selected_row["k"]
    )

    checkpoint = (
        ROOT
        / str(
            selected_row[
                "checkpoint"
            ]
        )
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model = FactorizationMachine(
        n_users=data.n_users,
        n_items=data.n_items,
        n_factors=k,
        init_std=INIT_STD,
    )

    model.load_state_dict(
        payload["state_dict"]
    )

    model.eval()

    return evaluate_ui_rmse(
        model,
        data.test[
            MODEL_COLUMNS
        ].copy(),
    )


def test_context_fm(
    *,
    model_name: str,
    use_frequency: bool,
    data,
    frames,
    n_days: int,
    selected_row,
) -> float:

    k = int(
        selected_row["k"]
    )

    checkpoint = (
        ROOT
        / str(
            selected_row[
                "checkpoint"
            ]
        )
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    model = ContextFM(
        data.n_users,
        data.n_items,
        n_days,
        n_factors=k,
        use_frequency=use_frequency,
        init_std=INIT_STD,
    )

    model.load_state_dict(
        payload["state_dict"]
    )

    model.eval()

    return evaluate_context_rmse(
        model,
        frames["test"],
    )


def evaluate_selected_on_test(
    selected: pd.DataFrame,
    data,
    frames,
    n_days: int,
) -> None:

    print()
    print(
        "=" * 80
    )
    print(
        "FINAL TEST EVALUATION"
    )
    print(
        "=" * 80
    )

    for _, row in (
        selected.sort_values(
            [
                "model",
                "budget",
            ]
        ).iterrows()
    ):

        model_name = str(
            row["model"]
        )

        budget = int(
            row["budget"]
        )

        k = int(
            row["k"]
        )

        learning_rate = float(
            row[
                "learning_rate"
            ]
        )

        if final_result_is_current(
            model_name,
            budget,
            learning_rate,
        ):
            print(
                f"SKIP TEST "
                f"{model_name} "
                f"budget={budget:,}"
            )

            continue

        print()
        print(
            f"TEST "
            f"{model_name} | "
            f"budget={budget:,} | "
            f"k={k} | "
            f"lr={learning_rate}"
        )

        if (
            model_name
            == "FM(u,i)"
        ):
            test_rmse = test_fm_ui(
                data,
                row,
            )

        elif (
            model_name
            == "FM(u,i,t)"
        ):
            test_rmse = (
                test_context_fm(
                    model_name=(
                        model_name
                    ),
                    use_frequency=False,
                    data=data,
                    frames=frames,
                    n_days=n_days,
                    selected_row=row,
                )
            )

        elif (
            model_name
            == "FM(u,i,t,f)"
        ):
            test_rmse = (
                test_context_fm(
                    model_name=(
                        model_name
                    ),
                    use_frequency=True,
                    data=data,
                    frames=frames,
                    n_days=n_days,
                    selected_row=row,
                )
            )

        else:
            raise ValueError(
                model_name
            )

        result = {
            "model": model_name,
            "budget": budget,
            "actual_params": int(
                row[
                    "actual_params"
                ]
            ),
            "k": k,
            "best_learning_rate": (
                learning_rate
            ),
            "validation_rmse": float(
                row[
                    "validation_rmse"
                ]
            ),
            "test_rmse": float(
                test_rmse
            ),
            "seed": SEED,
        }

        upsert_csv(
            FINAL_PATH,
            result,
            [
                "model",
                "budget",
            ],
        )

        print(
            f"TEST RMSE = "
            f"{test_rmse:.6f}"
        )

    print()
    print(
        f"Saved final results: "
        f"{FINAL_PATH}"
    )


# ============================================================
# Plot
# ============================================================

def make_plot() -> None:

    if not FINAL_PATH.exists():
        raise FileNotFoundError(
            f"{FINAL_PATH} not found. "
            f"Run --stage test first."
        )

    final = pd.read_csv(
        FINAL_PATH
    )

    expected_points = (
        len(MODELS)
        * len(BUDGETS)
    )

    # Keep only this experiment's
    # three models just in case.
    final = final[
        final["model"].isin(
            MODELS
        )
    ].copy()

    final = final.drop_duplicates(
        subset=[
            "model",
            "budget",
        ],
        keep="last",
    )

    missing = []

    for model_name in MODELS:
        for budget in BUDGETS:

            found = (
                (
                    final["model"]
                    == model_name
                )
                & (
                    final["budget"]
                    == budget
                )
            ).any()

            if not found:
                missing.append(
                    (
                        model_name,
                        budget,
                    )
                )

    if missing:
        raise RuntimeError(
            "Missing final points: "
            + ", ".join(
                f"{model}@{budget}"
                for model, budget
                in missing
            )
        )

    if len(final) != expected_points:
        print(
            f"Warning: expected "
            f"{expected_points} points, "
            f"found {len(final)}"
        )

    markers = {
        "FM(u,i)": "o",
        "FM(u,i,t)": "s",
        "FM(u,i,t,f)": "^",
    }

    fig, ax = plt.subplots(
        figsize=(11, 6.5),
        facecolor="white",
    )

    ax.set_facecolor(
        "white"
    )

    for model_name in MODELS:

        part = (
            final[
                final["model"]
                == model_name
            ]
            .sort_values(
                "budget"
            )
        )

        ax.plot(
            part["budget"],
            part["test_rmse"],
            marker=markers[
                model_name
            ],
            linewidth=2,
            markersize=7,
            label=model_name,
        )

    ax.set_xticks(
        BUDGETS,
        [
            "100k",
            "200k",
            "300k",
            "400k",
            "500k",
            "1M",
        ],
    )

    ax.set_xlabel(
        "Trainable parameter budget"
    )

    ax.set_ylabel(
        "Test RMSE"
    )

    ax.set_title(
        "MovieLens 1M: "
        "FM Variants — "
        "Best-LR Test RMSE vs Model Size"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=True
    )

    fig.tight_layout()

    PLOT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print()
    print(
        "=" * 80
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 80
    )

    columns = [
        "model",
        "budget",
        "actual_params",
        "k",
        "best_learning_rate",
        "validation_rmse",
        "test_rmse",
    ]

    print(
        final.sort_values(
            [
                "budget",
                "model",
            ]
        )[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved PNG: "
        f"{PLOT_PATH}"
    )


# ============================================================
# Summary of LR tuning
# ============================================================

def print_lr_summary() -> None:

    if not SELECTED_PATH.exists():
        return

    selected = pd.read_csv(
        SELECTED_PATH
    )

    print()
    print(
        "=" * 80
    )

    print(
        "BEST LR BY MODEL / "
        "PARAMETER BUDGET"
    )

    print(
        "=" * 80
    )

    print(
        selected[
            [
                "model",
                "budget",
                "actual_params",
                "k",
                "learning_rate",
                "validation_rmse",
            ]
        ]
        .sort_values(
            [
                "budget",
                "model",
            ]
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare FM(u,i), "
            "FM(u,i,t), and "
            "FM(u,i,t,f) under "
            "fixed parameter budgets "
            "with per-point LR tuning."
        )
    )

    parser.add_argument(
        "--stage",
        choices=[
            "plan",
            "tune",
            "select",
            "test",
            "plot",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "Retry tuning runs whose "
            "previous status was failed."
        ),
    )

    args = parser.parse_args()

    torch.set_num_threads(
        THREADS
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading MovieLens 1M..."
    )

    data = load_prepared(
        DATA_DIR
    )

    frames, n_days = (
        prepare_context_frames(
            data
        )
    )

    print(
        f"users = {data.n_users}"
    )

    print(
        f"items = {data.n_items}"
    )

    print(
        f"train days = {n_days}"
    )

    print(
        f"seed = {SEED}"
    )

    print(
        f"learning rates = "
        f"{LEARNING_RATES}"
    )

    # Always regenerate the plan.
    # It is cheap and guarantees
    # exact consistency with the data.

    plan = build_budget_plan(
        data,
        n_days,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "PARAMETER BUDGET PLAN"
    )

    print(
        "=" * 80
    )

    print(
        plan.to_string(
            index=False
        )
    )

    print()
    print(
        f"Total tuning runs: "
        f"{len(MODELS)} "
        f"x {len(BUDGETS)} "
        f"x {len(LEARNING_RATES)} "
        f"= "
        f"{len(MODELS) * len(BUDGETS) * len(LEARNING_RATES)}"
    )

    if args.stage == "plan":
        return

    if args.stage in {
        "tune",
        "all",
    }:
        run_tuning(
            plan,
            data,
            frames,
            n_days,
            retry_failed=(
                args.retry_failed
            ),
        )

    if args.stage in {
        "select",
        "all",
    }:
        select_best_learning_rates(
            plan
        )

    if args.stage in {
        "test",
        "all",
    }:

        if not SELECTED_PATH.exists():
            selected = (
                select_best_learning_rates(
                    plan
                )
            )

        else:
            selected = pd.read_csv(
                SELECTED_PATH
            )

        evaluate_selected_on_test(
            selected,
            data,
            frames,
            n_days,
        )

    if args.stage in {
        "plot",
        "all",
    }:
        make_plot()

    print_lr_summary()


if __name__ == "__main__":
    main()