from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import MODEL_COLUMNS, load_prepared
from src.metrics import rmse_by_row_id

from src.models import (
    FactorizationMachine,
    PureMatrixFactorization,
    TrainingConfig,
    count_parameters,
    fit_model,
    predict_frame,
)

from src.context_fm_experiment import (
    ContextFM,
    prepare_context_frames,
    fit_one,
    predict as context_predict,
)


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "results/parameter-budget-study"
PLAN_PATH = OUTPUT_DIR / "budget_plan.csv"
RESULTS_PATH = OUTPUT_DIR / "results.csv"

SEED = 42


def save_row(row):
    if RESULTS_PATH.exists():
        df = pd.read_csv(RESULTS_PATH)
    else:
        df = pd.DataFrame()

    df = pd.concat(
        [df, pd.DataFrame([row])],
        ignore_index=True,
    )

    df = df.drop_duplicates(
        subset=["model", "budget", "seed"],
        keep="last",
    )

    df = df.sort_values(
        ["model", "budget", "seed"]
    ).reset_index(drop=True)

    df.to_csv(
        RESULTS_PATH,
        index=False,
    )


def already_done(model, budget, seed):
    if not RESULTS_PATH.exists():
        return False

    df = pd.read_csv(RESULTS_PATH)

    mask = (
        (df["model"] == model)
        & (df["budget"] == budget)
        & (df["seed"] == seed)
    )

    return bool(mask.any())


def tensors(frame):
    users = torch.as_tensor(
        frame["user_idx"].to_numpy(copy=True),
        dtype=torch.long,
    )

    items = torch.as_tensor(
        frame["item_idx"].to_numpy(copy=True),
        dtype=torch.long,
    )

    ratings = torch.as_tensor(
        frame["rating"].to_numpy(copy=True),
        dtype=torch.float32,
    )

    return users, items, ratings


def rmse_standard(
    model,
    frame,
    *,
    add_mean=0.0,
    clip=False,
):
    prediction, row_ids = predict_frame(
        model,
        frame,
        batch_size=16384,
    )

    prediction = (
        prediction + add_mean
    )

    if clip:
        prediction = np.clip(
            prediction,
            1.0,
            5.0,
        )

    return rmse_by_row_id(
        frame["rating"].to_numpy(),
        prediction,
        frame["row_id"].to_numpy(),
        row_ids,
    )


def train_mf(
    data,
    k,
    budget,
    expected_params,
):
    model_name = "MF"

    if already_done(
        model_name,
        budget,
        SEED,
    ):
        print(
            f"SKIP {model_name} "
            f"budget={budget}"
        )
        return

    print()
    print(
        f"=== {model_name} "
        f"budget={budget} k={k} ==="
    )

    train = data.train[MODEL_COLUMNS].copy()
    val = data.validation[MODEL_COLUMNS].copy()
    test = data.test[MODEL_COLUMNS].copy()

    train_mean = float(
        train["rating"].mean()
    )

    train_u, train_i, _ = tensors(train)
    val_u, val_i, _ = tensors(val)

    train_centered = torch.as_tensor(
        (
            train["rating"]
            - train_mean
        ).to_numpy(copy=True),
        dtype=torch.float32,
    )

    val_centered = torch.as_tensor(
        (
            val["rating"]
            - train_mean
        ).to_numpy(copy=True),
        dtype=torch.float32,
    )

    model = PureMatrixFactorization(
        n_users=data.n_users,
        n_items=data.n_items,
        n_factors=k,
        init_std=1.0 / np.sqrt(k),
    )

    config = TrainingConfig(
        epochs=100,
        batch_size=2048,
        learning_rate=0.001,
        optimizer="adam",
        reg_bias=0.0,
        reg_factors=0.00001,
        patience=15,
        seed=SEED,
    )

    result = fit_model(
        model=model,
        train_user_idx=train_u,
        train_item_idx=train_i,
        train_rating=train_centered,
        val_user_idx=val_u,
        val_item_idx=val_i,
        val_rating=val_centered,
        config=config,
    )

    val_rmse = rmse_standard(
        model,
        val,
        add_mean=train_mean,
        clip=True,
    )

    test_rmse = rmse_standard(
        model,
        test,
        add_mean=train_mean,
        clip=True,
    )

    actual_params = count_parameters(
        model
    )

    if actual_params != expected_params:
        raise RuntimeError(
            f"{model_name}: expected "
            f"{expected_params} params, "
            f"got {actual_params}"
        )

    history_path = (
        OUTPUT_DIR
        / f"MF_budget{budget}_history.csv"
    )

    pd.DataFrame(
        result.history
    ).to_csv(
        history_path,
        index=False,
    )

    row = {
        "model": model_name,
        "budget": budget,
        "k": k,
        "actual_params": actual_params,
        "seed": SEED,
        "best_epoch": result.best_epoch,
        "validation_rmse": val_rmse,
        "test_rmse": test_rmse,
        "fit_seconds": result.fit_time_seconds,
    }

    save_row(row)

    print(
        f"{model_name} "
        f"k={k} "
        f"params={actual_params} "
        f"val={val_rmse:.6f} "
        f"test={test_rmse:.6f}"
    )


def train_fm_ui(
    data,
    k,
    budget,
    expected_params,
):
    model_name = "FM(u,i)"

    if already_done(
        model_name,
        budget,
        SEED,
    ):
        print(
            f"SKIP {model_name} "
            f"budget={budget}"
        )
        return

    print()
    print(
        f"=== {model_name} "
        f"budget={budget} k={k} ==="
    )

    train = data.train[MODEL_COLUMNS].copy()
    val = data.validation[MODEL_COLUMNS].copy()
    test = data.test[MODEL_COLUMNS].copy()

    train_u, train_i, train_r = tensors(
        train
    )
    val_u, val_i, val_r = tensors(
        val
    )

    model = FactorizationMachine(
        n_users=data.n_users,
        n_items=data.n_items,
        n_factors=k,
        init_std=0.01,
    )

    config = TrainingConfig(
        epochs=100,
        batch_size=1024,
        learning_rate=0.001,
        optimizer="adam",
        reg_bias=0.01,
        reg_factors=10.0,
        normalize_regularization=True,
        patience=10,
        seed=SEED,
    )

    result = fit_model(
        model=model,
        train_user_idx=train_u,
        train_item_idx=train_i,
        train_rating=train_r,
        val_user_idx=val_u,
        val_item_idx=val_i,
        val_rating=val_r,
        config=config,
    )

    val_rmse = rmse_standard(
        model,
        val,
    )

    test_rmse = rmse_standard(
        model,
        test,
    )

    actual_params = count_parameters(
        model
    )

    if actual_params != expected_params:
        raise RuntimeError(
            f"{model_name}: expected "
            f"{expected_params} params, "
            f"got {actual_params}"
        )

    pd.DataFrame(
        result.history
    ).to_csv(
        OUTPUT_DIR
        / f"FM_ui_budget{budget}_history.csv",
        index=False,
    )

    row = {
        "model": model_name,
        "budget": budget,
        "k": k,
        "actual_params": actual_params,
        "seed": SEED,
        "best_epoch": result.best_epoch,
        "validation_rmse": val_rmse,
        "test_rmse": test_rmse,
        "fit_seconds": result.fit_time_seconds,
    }

    save_row(row)

    print(
        f"{model_name} "
        f"k={k} "
        f"params={actual_params} "
        f"val={val_rmse:.6f} "
        f"test={test_rmse:.6f}"
    )


def train_context(
    *,
    model_name,
    use_frequency,
    data,
    frames,
    n_days,
    k,
    budget,
    expected_params,
):
    if already_done(
        model_name,
        budget,
        SEED,
    ):
        print(
            f"SKIP {model_name} "
            f"budget={budget}"
        )
        return

    print()
    print(
        f"=== {model_name} "
        f"budget={budget} k={k} ==="
    )

    model = ContextFM(
        data.n_users,
        data.n_items,
        n_days,
        n_factors=k,
        use_frequency=use_frequency,
        init_std=0.01,
    )

    result = fit_one(
        model,
        frames["train"],
        frames["validation"],
        seed=SEED,
        epochs=100,
        batch_size=1024,
        learning_rate=0.001,
        reg_bias=0.01,
        reg_factors=10.0,
        patience=10,
    )

    val_pred, val_ids = context_predict(
        model,
        frames["validation"],
    )

    val_rmse = rmse_by_row_id(
        frames["validation"]["rating"].to_numpy(),
        val_pred,
        frames["validation"]["row_id"].to_numpy(),
        val_ids,
    )

    test_pred, test_ids = context_predict(
        model,
        frames["test"],
    )

    test_rmse = rmse_by_row_id(
        frames["test"]["rating"].to_numpy(),
        test_pred,
        frames["test"]["row_id"].to_numpy(),
        test_ids,
    )

    actual_params = count_parameters(
        model
    )

    if actual_params != expected_params:
        raise RuntimeError(
            f"{model_name}: expected "
            f"{expected_params} params, "
            f"got {actual_params}"
        )

    safe_name = (
        model_name
        .replace("(", "_")
        .replace(")", "")
        .replace(",", "")
    )

    result["history"].to_csv(
        OUTPUT_DIR
        / f"{safe_name}_budget{budget}_history.csv",
        index=False,
    )

    row = {
        "model": model_name,
        "budget": budget,
        "k": k,
        "actual_params": actual_params,
        "seed": SEED,
        "best_epoch": result["best_epoch"],
        "validation_rmse": val_rmse,
        "test_rmse": test_rmse,
        "fit_seconds": result["fit_seconds"],
    }

    save_row(row)

    print(
        f"{model_name} "
        f"k={k} "
        f"params={actual_params} "
        f"val={val_rmse:.6f} "
        f"test={test_rmse:.6f}"
    )


def import_svdpp_results(plan):
    svd_path = (
        ROOT
        / "svdpp"
        / "results"
        / "parameter-budget-study"
        / "seed42"
        / "metrics_by_run.csv"
    )

    if not svd_path.exists():
        print()
        print(
            "SVD++ results not found yet:"
        )
        print(svd_path)
        return

    svd = pd.read_csv(svd_path)

    for _, plan_row in (
        plan[plan["model"] == "SVD++"]
        .sort_values("budget")
        .iterrows()
    ):
        budget = int(plan_row["budget"])
        k = int(plan_row["k"])
        actual_params = int(
            plan_row["actual_params"]
        )

        match = svd[
            svd["k"].astype(int) == k
        ]

        if match.empty:
            print(
                f"SVD++ k={k} "
                "not completed"
            )
            continue

        row0 = match.iloc[0]

        if (
            "test_rmse" not in match.columns
            or pd.isna(row0["test_rmse"])
        ):
            print(
                f"SVD++ k={k}: "
                "test not evaluated yet"
            )
            continue

        row = {
            "model": "SVD++",
            "budget": budget,
            "k": k,
            "actual_params": actual_params,
            "seed": SEED,
            "best_epoch": int(
                row0["best_epoch"]
            ),
            "validation_rmse": float(
                row0["validation_rmse"]
            ),
            "test_rmse": float(
                row0["test_rmse"]
            ),
            "fit_seconds": float(
                row0["fit_seconds"]
            ),
        }

        save_row(row)

        print(
            f"Imported SVD++ "
            f"k={k} "
            f"test={row['test_rmse']:.6f}"
        )


def main():
    torch.set_num_threads(4)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not PLAN_PATH.exists():
        raise FileNotFoundError(
            f"Run first:\n"
            f"python -m src.parameter_budget_study\n"
            f"\nMissing: {PLAN_PATH}"
        )

    plan = pd.read_csv(
        PLAN_PATH
    )

    data = load_prepared(
        "data/processed/movielens1m"
    )

    frames, n_days = (
        prepare_context_frames(data)
    )

    print(
        "MovieLens 1M:",
        data.n_users,
        "users,",
        data.n_items,
        "items,",
        n_days,
        "days",
    )

    # ---------------------------------
    # MF
    # ---------------------------------

    for _, row in (
        plan[plan["model"] == "MF"]
        .sort_values("budget")
        .iterrows()
    ):
        train_mf(
            data,
            int(row["k"]),
            int(row["budget"]),
            int(row["actual_params"]),
        )

    # ---------------------------------
    # FM(u,i)
    # ---------------------------------

    for _, row in (
        plan[plan["model"] == "FM(u,i)"]
        .sort_values("budget")
        .iterrows()
    ):
        train_fm_ui(
            data,
            int(row["k"]),
            int(row["budget"]),
            int(row["actual_params"]),
        )

    # ---------------------------------
    # FM(u,i,t)
    # ---------------------------------

    for _, row in (
        plan[plan["model"] == "FM(u,i,t)"]
        .sort_values("budget")
        .iterrows()
    ):
        train_context(
            model_name="FM(u,i,t)",
            use_frequency=False,
            data=data,
            frames=frames,
            n_days=n_days,
            k=int(row["k"]),
            budget=int(row["budget"]),
            expected_params=int(
                row["actual_params"]
            ),
        )

    # ---------------------------------
    # FM(u,i,t,f)
    # ---------------------------------

    for _, row in (
        plan[
            plan["model"]
            == "FM(u,i,t,f)"
        ]
        .sort_values("budget")
        .iterrows()
    ):
        train_context(
            model_name="FM(u,i,t,f)",
            use_frequency=True,
            data=data,
            frames=frames,
            n_days=n_days,
            k=int(row["k"]),
            budget=int(row["budget"]),
            expected_params=int(
                row["actual_params"]
            ),
        )

    # ---------------------------------
    # SVD++ — импортируем после
    # отдельного optimized run
    # ---------------------------------

    import_svdpp_results(plan)

    print()
    print("=" * 80)

    if RESULTS_PATH.exists():
        results = pd.read_csv(
            RESULTS_PATH
        )

        print(
            results[
                [
                    "model",
                    "budget",
                    "k",
                    "actual_params",
                    "validation_rmse",
                    "test_rmse",
                ]
            ].to_string(index=False)
        )

        print()
        print(
            f"Saved: {RESULTS_PATH}"
        )


if __name__ == "__main__":
    main()