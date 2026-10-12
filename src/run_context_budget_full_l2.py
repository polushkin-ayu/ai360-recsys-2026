from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.context_fm_experiment import (
    ContextFM,
    fit_one,
    predict,
    prepare_context_frames,
)
from src.data import load_prepared
from src.metrics import rmse_by_row_id
from src.models import count_parameters


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/processed/movielens1m"

PLAN_PATH = (
    ROOT
    / "results/parameter-budget-study/budget_plan.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results/context-budget-full-l2"
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "context_budget_results.csv"
)

HISTORY_DIR = OUTPUT_DIR / "histories"


# Только точки с текущего графика
BUDGETS = [
    100_000,
    200_000,
    300_000,
    500_000,
    1_000_000,
]

MODELS = [
    "FM(u,i,t)",
    "FM(u,i,t,f)",
]

SEED = 42

EPOCHS = 100
BATCH_SIZE = 1024
PATIENCE = 10

REG_BIAS = 0.01
REG_FACTORS = 10.0

DEFAULT_LR = 0.001


def load_old_best_lrs():
    """
    Пытаемся найти learning rates,
    которые уже были выбраны в старом
    Best-LR эксперименте.
    """

    candidates = [
        ROOT / "results/fm-parameter-lr-sweep/final_results.csv",
        ROOT / "results/fm-parameter-lr-sweep/best_by_validation.csv",
        ROOT / "results/parameter-lr-sweep/final_results.csv",
        ROOT / "results/parameter-lr-sweep/best_by_validation.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue

        df = pd.read_csv(path)

        if "best_learning_rate" in df.columns:
            lr_col = "best_learning_rate"
        elif "learning_rate" in df.columns:
            lr_col = "learning_rate"
        else:
            continue

        result = {}

        for _, row in df.iterrows():
            model = str(row["model"])

            if model not in MODELS:
                continue

            budget = int(row["budget"])

            if budget not in BUDGETS:
                continue

            result[
                (model, budget)
            ] = float(row[lr_col])

        if result:
            print(
                "Using old best LRs from:",
                path
            )

            return result

    print()
    print(
        "WARNING: old Best-LR CSV not found."
    )
    print(
        "Will use lr=0.001 for missing points."
    )

    return {}


def existing_results():
    if RESULTS_PATH.exists():
        return pd.read_csv(
            RESULTS_PATH
        )

    return pd.DataFrame()


def already_done(
    model_name,
    budget,
    learning_rate,
):
    df = existing_results()

    if df.empty:
        return False

    mask = (
        (df["model"] == model_name)
        & (df["budget"] == budget)
        & np.isclose(
            df["learning_rate"],
            learning_rate,
        )
    )

    return bool(mask.any())


def save_row(row):
    df = existing_results()

    if df.empty:
        df = pd.DataFrame([row])
    else:
        df = pd.concat(
            [
                df,
                pd.DataFrame([row]),
            ],
            ignore_index=True,
        )

    df = df.drop_duplicates(
        subset=[
            "model",
            "budget",
            "learning_rate",
            "seed",
        ],
        keep="last",
    )

    df = df.sort_values(
        [
            "model",
            "budget",
        ]
    ).reset_index(drop=True)

    df.to_csv(
        RESULTS_PATH,
        index=False,
    )


def evaluate(
    model,
    frame,
):
    pred, row_ids = predict(
        model,
        frame,
        batch_size=16384,
    )

    return rmse_by_row_id(
        frame["rating"].to_numpy(),
        pred,
        frame["row_id"].to_numpy(),
        row_ids,
    )


def main():
    torch.set_num_threads(4)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Not found: {PLAN_PATH}\n"
            "Run first:\n"
            "python -m src.parameter_budget_study"
        )

    plan = pd.read_csv(
        PLAN_PATH
    )

    data = load_prepared(
        DATA_DIR
    )

    frames, n_days = (
        prepare_context_frames(
            data
        )
    )

    best_lrs = load_old_best_lrs()

    print()
    print(
        "users:",
        data.n_users
    )

    print(
        "items:",
        data.n_items
    )

    print(
        "days:",
        n_days
    )

    print()

    for model_name in MODELS:

        use_frequency = (
            model_name
            == "FM(u,i,t,f)"
        )

        for budget in BUDGETS:

            match = plan[
                (plan["model"] == model_name)
                & (plan["budget"] == budget)
            ]

            if match.empty:
                raise RuntimeError(
                    f"No plan row for "
                    f"{model_name}, "
                    f"budget={budget}"
                )

            plan_row = match.iloc[0]

            k = int(
                plan_row["k"]
            )

            expected_params = int(
                plan_row[
                    "actual_params"
                ]
            )

            learning_rate = (
                best_lrs.get(
                    (
                        model_name,
                        budget,
                    ),
                    DEFAULT_LR,
                )
            )

            if already_done(
                model_name,
                budget,
                learning_rate,
            ):
                print(
                    "SKIP:",
                    model_name,
                    budget,
                    "lr=",
                    learning_rate,
                )

                continue

            print()
            print(
                "=" * 70
            )

            print(
                model_name
            )

            print(
                "budget:",
                budget
            )

            print(
                "k:",
                k
            )

            print(
                "lr:",
                learning_rate
            )

            print(
                "=" * 70
            )

            torch.manual_seed(SEED)
            np.random.seed(SEED)

            model = ContextFM(
                data.n_users,
                data.n_items,
                n_days,
                n_factors=k,
                use_frequency=(
                    use_frequency
                ),
                init_std=0.01,
            )

            actual_params = (
                count_parameters(
                    model
                )
            )

            if (
                actual_params
                != expected_params
            ):
                raise RuntimeError(
                    f"Parameter mismatch: "
                    f"plan={expected_params}, "
                    f"actual={actual_params}"
                )

            result = fit_one(
                model,
                frames["train"],
                frames["validation"],
                seed=SEED,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=(
                    learning_rate
                ),
                reg_bias=REG_BIAS,
                reg_factors=REG_FACTORS,
                patience=PATIENCE,
            )

            validation_rmse = evaluate(
                model,
                frames["validation"],
            )

            # Test смотрим только после
            # validation early stopping.
            test_rmse = evaluate(
                model,
                frames["test"],
            )

            history_name = (
                model_name
                .replace("(", "_")
                .replace(")", "")
                .replace(",", "")
            )

            result["history"].to_csv(
                HISTORY_DIR
                / (
                    f"{history_name}"
                    f"_budget{budget}"
                    f"_history.csv"
                ),
                index=False,
            )

            row = {
                "model":
                    model_name,

                "budget":
                    budget,

                "actual_params":
                    actual_params,

                "k":
                    k,

                "learning_rate":
                    learning_rate,

                "seed":
                    SEED,

                "best_epoch":
                    result[
                        "best_epoch"
                    ],

                "validation_rmse":
                    validation_rmse,

                "test_rmse":
                    test_rmse,

                "fit_seconds":
                    result[
                        "fit_seconds"
                    ],
            }

            save_row(row)

            print()
            print(
                f"DONE | "
                f"{model_name} | "
                f"{budget:,} | "
                f"k={k} | "
                f"lr={learning_rate} | "
                f"val={validation_rmse:.6f} | "
                f"test={test_rmse:.6f}"
            )

    print()
    print(
        "=" * 70
    )

    final = pd.read_csv(
        RESULTS_PATH
    )

    print(
        final[
            [
                "model",
                "budget",
                "k",
                "actual_params",
                "learning_rate",
                "validation_rmse",
                "test_rmse",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        RESULTS_PATH
    )


if __name__ == "__main__":
    main()