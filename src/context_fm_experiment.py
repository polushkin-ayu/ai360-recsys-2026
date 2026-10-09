from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.data import load_prepared
from src.metrics import rmse_by_row_id


class ContextFM(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        n_days,
        *,
        n_factors=100,
        use_frequency=False,
        init_std=0.01,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.n_days = n_days
        self.n_factors = n_factors
        self.use_frequency = use_frequency
        self.init_std = init_std

        # Sparse embeddings = намного легче обучение на MovieLens 1M.
        self.user_bias = nn.Embedding(n_users, 1, sparse=True)
        self.item_bias = nn.Embedding(n_items, 1, sparse=True)
        self.day_bias = nn.Embedding(n_days, 1, sparse=True)

        self.user_factors = nn.Embedding(
            n_users, n_factors, sparse=True
        )
        self.item_factors = nn.Embedding(
            n_items, n_factors, sparse=True
        )
        self.day_factors = nn.Embedding(
            n_days, n_factors, sparse=True
        )

        self.global_bias = nn.Parameter(
            torch.tensor(0.0)
        )

        if use_frequency:
            self.frequency_weight = nn.Parameter(
                torch.tensor(0.0)
            )
            self.frequency_factor = nn.Parameter(
                torch.empty(n_factors)
            )

        self.reset_parameters(0.0)

    def reset_parameters(self, global_mean):
        with torch.no_grad():
            self.global_bias.fill_(
                float(global_mean)
            )

            self.user_bias.weight.zero_()
            self.item_bias.weight.zero_()
            self.day_bias.weight.zero_()

            if self.use_frequency:
                self.frequency_weight.zero_()

        nn.init.normal_(
            self.user_factors.weight,
            mean=0.0,
            std=self.init_std,
        )

        nn.init.normal_(
            self.item_factors.weight,
            mean=0.0,
            std=self.init_std,
        )

        nn.init.normal_(
            self.day_factors.weight,
            mean=0.0,
            std=self.init_std,
        )

        if self.use_frequency:
            nn.init.normal_(
                self.frequency_factor,
                mean=0.0,
                std=self.init_std,
            )

    def forward_with_regularization(
        self,
        user_idx,
        item_idx,
        day_idx,
        frequency,
    ):
        # day_idx = -1 означает день,
        # которого вообще не было в train.
        safe_day = day_idx.clamp_min(0)
        day_mask = (day_idx >= 0).float()

        ub = self.user_bias(
            user_idx
        ).squeeze(-1)

        ib = self.item_bias(
            item_idx
        ).squeeze(-1)

        db = (
            self.day_bias(
                safe_day
            ).squeeze(-1)
            * day_mask
        )

        uf = self.user_factors(
            user_idx
        )

        itf = self.item_factors(
            item_idx
        )

        df = (
            self.day_factors(
                safe_day
            )
            * day_mask[:, None]
        )

        prediction = (
            self.global_bias
            + ub
            + ib
            + db
        )

        vectors = [
            uf,
            itf,
            df,
        ]

        # Регуляризуем только активные sparse-параметры.
        bias_reg = (
            ub.square().sum()
            + ib.square().sum()
            + db.square().sum()
        )

        factor_reg = (
            uf.square().sum()
            + itf.square().sum()
            + df.square().sum()
        )

        if self.use_frequency:
            frequency = frequency.to(
                prediction.dtype
            )

            prediction = (
                prediction
                + self.frequency_weight
                * frequency
            )

            frequency_vector = (
                self.frequency_factor[None, :]
                * frequency[:, None]
            )

            vectors.append(
                frequency_vector
            )

            bias_reg = (
                bias_reg
                + self.frequency_weight.square()
            )

            factor_reg = (
                factor_reg
                + self.frequency_factor
                .square()
                .sum()
            )

        interaction = torch.zeros_like(
            prediction
        )

        # Все pairwise interactions FM.
        for left in range(len(vectors)):
            for right in range(
                left + 1,
                len(vectors),
            ):
                interaction = (
                    interaction
                    + (
                        vectors[left]
                        * vectors[right]
                    ).sum(dim=1)
                )

        return (
            prediction + interaction,
            bias_reg,
            factor_reg,
        )

    def forward(
        self,
        user_idx,
        item_idx,
        day_idx,
        frequency,
    ):
        prediction, _, _ = (
            self.forward_with_regularization(
                user_idx,
                item_idx,
                day_idx,
                frequency,
            )
        )

        return prediction

    def sparse_parameters(self):
        return [
            self.user_bias.weight,
            self.item_bias.weight,
            self.day_bias.weight,
            self.user_factors.weight,
            self.item_factors.weight,
            self.day_factors.weight,
        ]

    def dense_parameters(self):
        params = [
            self.global_bias,
        ]

        if self.use_frequency:
            params += [
                self.frequency_weight,
                self.frequency_factor,
            ]

        return params


def prepare_context_frames(data):
    frames = {
        "train": data.train.copy().reset_index(
            drop=True
        ),
        "validation": (
            data.validation
            .copy()
            .reset_index(drop=True)
        ),
        "test": (
            data.test
            .copy()
            .reset_index(drop=True)
        ),
    }

    for name, frame in frames.items():
        if "timestamp" not in frame.columns:
            raise ValueError(
                f"{name}: timestamp column not found"
            )

        # Unix timestamp -> день.
        frame["day"] = (
            frame["timestamp"]
            .astype(np.int64)
            // 86400
        )

    train = frames["train"]

    # Только train определяет vocabulary времени.
    train_days = np.sort(
        train["day"].unique()
    )

    day_to_idx = {
        int(day): idx
        for idx, day in enumerate(
            train_days
        )
    }

    # f = число оценок пользователя в этот день.
    # Считаем ТОЛЬКО по train.
    train_user_day_count = (
        train
        .groupby(
            ["user_idx", "day"]
        )
        .size()
    )

    for frame in frames.values():
        frame["day_idx"] = (
            frame["day"]
            .map(day_to_idx)
            .fillna(-1)
            .astype(np.int64)
        )

        keys = pd.MultiIndex.from_frame(
            frame[
                ["user_idx", "day"]
            ]
        )

        counts = (
            train_user_day_count
            .reindex(
                keys,
                fill_value=0,
            )
            .to_numpy(
                dtype=np.float32
            )
        )

        # Rendle:
        # f = ln(number of ratings
        #        by user on that day)
        #
        # unseen user-day -> ln(1) = 0
        frame["frequency"] = np.log(
            np.maximum(
                counts,
                1.0,
            )
        ).astype(np.float32)

    return frames, len(train_days)


def make_tensors(frame):
    return (
        torch.as_tensor(
            frame[
                "user_idx"
            ].to_numpy(copy=True),
            dtype=torch.long,
        ),
        torch.as_tensor(
            frame[
                "item_idx"
            ].to_numpy(copy=True),
            dtype=torch.long,
        ),
        torch.as_tensor(
            frame[
                "day_idx"
            ].to_numpy(copy=True),
            dtype=torch.long,
        ),
        torch.as_tensor(
            frame[
                "frequency"
            ].to_numpy(copy=True),
            dtype=torch.float32,
        ),
        torch.as_tensor(
            frame[
                "rating"
            ].to_numpy(copy=True),
            dtype=torch.float32,
        ),
    )


def predict(
    model,
    frame,
    batch_size=16384,
):
    (
        users,
        items,
        days,
        frequency,
        _,
    ) = make_tensors(frame)

    predictions = []

    model.eval()

    with torch.no_grad():
        for start in range(
            0,
            len(frame),
            batch_size,
        ):
            stop = (
                start
                + batch_size
            )

            prediction = model(
                users[start:stop],
                items[start:stop],
                days[start:stop],
                frequency[start:stop],
            )

            predictions.append(
                prediction.cpu()
            )

    return (
        torch.cat(
            predictions
        ).numpy(),
        frame[
            "row_id"
        ].to_numpy(copy=True),
    )


def fit_one(
    model,
    train,
    validation,
    *,
    seed,
    epochs,
    batch_size,
    learning_rate,
    reg_bias,
    reg_factors,
    patience,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model.reset_parameters(
        float(
            train[
                "rating"
            ].mean()
        )
    )

    # SparseAdam обновляет только реально
    # использованные embeddings.
    sparse_optimizer = (
        torch.optim.SparseAdam(
            model.sparse_parameters(),
            lr=learning_rate,
        )
    )

    dense_optimizer = torch.optim.Adam(
        model.dense_parameters(),
        lr=learning_rate,
    )

    train_values = make_tensors(
        train
    )

    n_train = len(train)

    generator = (
        torch.Generator()
        .manual_seed(seed)
    )

    best_state = None
    best_epoch = 0
    best_val_rmse = float("inf")

    epochs_without_improvement = 0

    history = []

    started = time.perf_counter()

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        permutation = torch.randperm(
            n_train,
            generator=generator,
        )

        for start in range(
            0,
            n_train,
            batch_size,
        ):
            batch = permutation[
                start:
                start + batch_size
            ]

            users = train_values[0][batch]
            items = train_values[1][batch]
            days = train_values[2][batch]
            frequency = train_values[3][batch]
            ratings = train_values[4][batch]

            sparse_optimizer.zero_grad(
                set_to_none=True
            )

            dense_optimizer.zero_grad(
                set_to_none=True
            )

            (
                prediction,
                bias_reg,
                factor_reg,
            ) = (
                model.forward_with_regularization(
                    users,
                    items,
                    days,
                    frequency,
                )
            )

            loss = (
                F.mse_loss(
                    prediction,
                    ratings,
                )
                + (
                    reg_bias
                    / n_train
                )
                * bias_reg
                + (
                    reg_factors
                    / n_train
                )
                * factor_reg
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "non-finite training loss"
                )

            loss.backward()

            sparse_optimizer.step()
            dense_optimizer.step()

        val_pred, val_ids = predict(
            model,
            validation,
        )

        val_rmse = rmse_by_row_id(
            validation[
                "rating"
            ].to_numpy(),
            val_pred,
            validation[
                "row_id"
            ].to_numpy(),
            val_ids,
        )

        history.append({
            "epoch": epoch,
            "val_rmse": val_rmse,
        })

        print(
            f"epoch={epoch:03d} "
            f"val={val_rmse:.6f}",
            flush=True,
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

            if (
                epochs_without_improvement
                >= patience
            ):
                break

    if best_state is None:
        raise RuntimeError(
            "no best checkpoint"
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    return {
        "best_epoch": best_epoch,
        "validation_rmse": best_val_rmse,
        "fit_seconds": (
            time.perf_counter()
            - started
        ),
        "history": pd.DataFrame(
            history
        ),
    }


def run_model(
    *,
    model_name,
    use_frequency,
    data,
    frames,
    n_days,
    seeds,
    k,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    reg_bias,
    reg_factors,
    patience,
):
    rows = []

    for seed in seeds:
        print()
        print(
            f"=== {model_name} "
            f"seed={seed} ===",
            flush=True,
        )

        torch.manual_seed(seed)

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
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            reg_bias=reg_bias,
            reg_factors=reg_factors,
            patience=patience,
        )

        val_pred, val_ids = predict(
            model,
            frames["validation"],
        )

        val_rmse = rmse_by_row_id(
            frames["validation"][
                "rating"
            ].to_numpy(),
            val_pred,
            frames["validation"][
                "row_id"
            ].to_numpy(),
            val_ids,
        )

        # Test смотрим только после
        # validation-based early stopping.
        test_pred, test_ids = predict(
            model,
            frames["test"],
        )

        test_rmse = rmse_by_row_id(
            frames["test"][
                "rating"
            ].to_numpy(),
            test_pred,
            frames["test"][
                "row_id"
            ].to_numpy(),
            test_ids,
        )

        rows.append({
            "model": model_name,
            "seed": seed,
            "k": k,
            "best_epoch": result[
                "best_epoch"
            ],
            "validation_rmse": val_rmse,
            "test_rmse": test_rmse,
            "fit_seconds": result[
                "fit_seconds"
            ],
        })

        safe_name = (
            model_name
            .replace("(", "_")
            .replace(")", "")
            .replace(",", "")
        )

        result["history"].to_csv(
            output_dir
            / f"{safe_name}_seed{seed}_history.csv",
            index=False,
        )

        print(
            f"{model_name} "
            f"seed={seed} "
            f"best_epoch="
            f"{result['best_epoch']} "
            f"val={val_rmse:.6f} "
            f"test={test_rmse:.6f}",
            flush=True,
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "uit",
            "uitf",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--data-dir",
        default=(
            "data/processed/"
            "movielens1m"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "context-fm-paper"
        ),
    )

    parser.add_argument(
        "--k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[
            42,
            43,
            44,
        ],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--reg-bias",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--reg-factors",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    torch.set_num_threads(
        args.threads
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_prepared(
        args.data_dir
    )

    frames, n_days = (
        prepare_context_frames(
            data
        )
    )

    print(
        "train:",
        len(frames["train"]),
    )

    print(
        "validation:",
        len(frames["validation"]),
    )

    print(
        "test:",
        len(frames["test"]),
    )

    print(
        "users:",
        data.n_users,
    )

    print(
        "items:",
        data.n_items,
    )

    print(
        "train days:",
        n_days,
    )

    runs = []

    if args.model in {
        "uit",
        "all",
    }:
        runs.append(
            run_model(
                model_name="FM(u,i,t)",
                use_frequency=False,
                data=data,
                frames=frames,
                n_days=n_days,
                seeds=args.seeds,
                k=args.k,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                reg_bias=args.reg_bias,
                reg_factors=args.reg_factors,
                patience=args.patience,
            )
        )

    if args.model in {
        "uitf",
        "all",
    }:
        runs.append(
            run_model(
                model_name=(
                    "FM(u,i,t,f)"
                ),
                use_frequency=True,
                data=data,
                frames=frames,
                n_days=n_days,
                seeds=args.seeds,
                k=args.k,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                reg_bias=args.reg_bias,
                reg_factors=args.reg_factors,
                patience=args.patience,
            )
        )

    new_results = pd.concat(
        runs,
        ignore_index=True,
    )

    results_path = (
        output_dir
        / "context_fm_results.csv"
    )

    # Позволяет сначала запустить uit,
    # потом отдельно uitf.
    if results_path.exists():
        old_results = pd.read_csv(
            results_path
        )

        combined = pd.concat(
            [
                old_results,
                new_results,
            ],
            ignore_index=True,
        )

        combined = (
            combined
            .drop_duplicates(
                subset=[
                    "model",
                    "seed",
                    "k",
                ],
                keep="last",
            )
        )

    else:
        combined = new_results

    combined = (
        combined
        .sort_values(
            [
                "model",
                "k",
                "seed",
            ]
        )
        .reset_index(drop=True)
    )

    combined.to_csv(
        results_path,
        index=False,
    )

    print()
    print(
        f"Saved: {results_path}"
    )

    print(
        combined.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()