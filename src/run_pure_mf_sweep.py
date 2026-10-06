"""Strict observation alignment and stable PureMF sweep implementation (k from 1 to 100)."""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.data import MODEL_COLUMNS, load_prepared
from src.metrics import rmse_by_row_id
from src.models import (
    PureMatrixFactorization,
    TrainingConfig,
    fit_model,
    predict_frame,
    save_checkpoint,
)


def main() -> None:
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # 1. Загрузка подготовленных данных
    logger.info("Loading prepared dataset from data/processed/movielens1m")
    data = load_prepared("data/processed/movielens1m")

    train = data.train[MODEL_COLUMNS]
    val = data.validation[MODEL_COLUMNS]

    logger.info(f"Dataset loaded. Users: {data.n_users}, Items: {data.n_items}")
    logger.info(f"Train size: {len(train)}, Validation size: {len(val)}")

    # Считаем глобальное среднее рейтингов для стабильности PureMF
    train_mean = float(train["rating"].mean())
    logger.info(f"Global train rating mean: {train_mean:.4f}")

    # 2. Подготовка PyTorch-тензоров (центрируем таргет для обучения)
    device = torch.device("cpu")

    train_user = torch.as_tensor(train["user_idx"].to_numpy().copy(), dtype=torch.long, device=device)
    train_item = torch.as_tensor(train["item_idx"].to_numpy().copy(), dtype=torch.long, device=device)
    train_rating_centered = torch.as_tensor(
        (train["rating"] - train_mean).to_numpy().copy(),
        dtype=torch.float32,
        device=device
    )

    val_user = torch.as_tensor(val["user_idx"].to_numpy().copy(), dtype=torch.long, device=device)
    val_item = torch.as_tensor(val["item_idx"].to_numpy().copy(), dtype=torch.long, device=device)
    val_rating = torch.as_tensor(val["rating"].to_numpy().copy(), dtype=torch.float32, device=device)

    # 3. Настройка сетки гиперпараметров для k-sweep (от 1 до 100 включительно)
    k_grid = list(range(50, 201, 5))

    base_config = TrainingConfig(
        epochs=100,  # Можно уменьшить, patience сделает остальное
        batch_size=2048,  # Чуть больше батч для стабильности градиентов
        learning_rate=0.001,  # Снизили в 10 раз (было 0.01)
        optimizer="adam",
        reg_bias=0.0,
        reg_factors=0.0,
        patience=15,  # Ранняя остановка
        seed=42,
    )

    results_log = {}
    best_overall_rmse = float("inf")
    best_k = None
    best_model_state = None
    best_training_result = None

    logger.info(f"Starting sweep over k from {k_grid[0]} to {k_grid[-1]}...")

    # 4. Основной цикл обучения и оценки
    for k in k_grid:
        logger.info(f"--- Training PureMatrixFactorization with k={k} ---")

        torch.manual_seed(base_config.seed)
        np.random.seed(base_config.seed)

        init_std = 1.0 / np.sqrt(k)

        model = PureMatrixFactorization(
            n_users=data.n_users,
            n_items=data.n_items,
            n_factors=k,
            init_std=init_std,
        )

        training_result = fit_model(
            model=model,
            train_user_idx=train_user,
            train_item_idx=train_item,
            train_rating=train_rating_centered,
            val_user_idx=val_user,
            val_item_idx=val_item,
            val_rating=val_rating - train_mean,
            config=base_config,
        )

        predictions, row_ids = predict_frame(model, val)
        predictions_restored = np.clip(predictions + train_mean, 1.0, 5.0)

        val_rmse = rmse_by_row_id(
            val["rating"].to_numpy(),
            predictions_restored,
            val["row_id"].to_numpy(),
            row_ids,
        )

        logger.info(
            f"Completed k={k:3d} | Val RMSE: {val_rmse:.4f} | Best Epoch: {training_result.best_epoch} | Time: {training_result.fit_time_seconds:.1f}s"
        )

        results_log[k] = {
            "val_rmse": val_rmse,
            "best_epoch": training_result.best_epoch,
            "fit_time_seconds": training_result.fit_time_seconds,
        }

        if val_rmse < best_overall_rmse:
            best_overall_rmse = val_rmse
            best_k = k
            best_model_state = model
            best_training_result = training_result

    # 5. Сохранение результатов и артефактов
    results_dir = Path("results")
    checkpoints_dir = Path("checkpoints")
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    sweep_log_path = results_dir / "pure_mf_sweep.json"
    with open(sweep_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_k": best_k,
            "best_val_rmse": best_overall_rmse,
            "sweep": results_log
        }, f, indent=2)

    checkpoint_path = checkpoints_dir / "pure_mf_best.pt"
    save_checkpoint(checkpoint_path, best_model_state, best_training_result)

    logger.info("==================================================")
    logger.info(f"Sweep complete! Best k: {best_k} with Validation RMSE: {best_overall_rmse:.4f}")
    logger.info(f"Best model checkpoint saved to: {checkpoint_path}")
    logger.info(f"Sweep logs saved to: {sweep_log_path}")


if __name__ == "__main__":
    main()
