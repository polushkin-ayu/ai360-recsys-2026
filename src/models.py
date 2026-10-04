"""PyTorch bias baseline and second-order FM for user/item ratings.

This module only consumes integer indices prepared by the data team. It does
not create category mappings, split data or handle unknown IDs.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor, nn


def _validate_model_sizes(n_users: int, n_items: int) -> None:
    if not isinstance(n_users, int) or not isinstance(n_items, int):
        raise TypeError("n_users and n_items must be integers")
    if n_users <= 0 or n_items <= 0:
        raise ValueError("n_users and n_items must be positive")


def _validate_inputs(user_idx: Tensor, item_idx: Tensor) -> None:
    if user_idx.ndim != 1 or item_idx.ndim != 1:
        raise ValueError("user_idx and item_idx must be one-dimensional")
    if user_idx.shape != item_idx.shape:
        raise ValueError("user_idx and item_idx must have the same shape")
    if user_idx.dtype != torch.long or item_idx.dtype != torch.long:
        raise TypeError("user_idx and item_idx must have dtype torch.long")


class BiasModel(nn.Module):
    """Global intercept plus user and item biases.

    ``prediction = global_bias + user_bias[user] + item_bias[item]``
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        *,
        global_mean: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_model_sizes(n_users, n_items)
        self.n_users = n_users
        self.n_items = n_items
        self.global_bias = nn.Parameter(torch.tensor(float(global_mean)))
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self._reset_bias_parameters(global_mean)

    def _reset_bias_parameters(self, global_mean: float) -> None:
        with torch.no_grad():
            self.global_bias.fill_(float(global_mean))
            self.user_bias.weight.zero_()
            self.item_bias.weight.zero_()

    def reset_parameters(self, global_mean: float = 0.0) -> None:
        """Reset the intercept and all bias embeddings."""
        self._reset_bias_parameters(global_mean)

    def forward(self, user_idx: Tensor, item_idx: Tensor) -> Tensor:
        """Return one scalar prediction per user/item pair."""
        _validate_inputs(user_idx, item_idx)
        return (
            self.global_bias
            + self.user_bias(user_idx).squeeze(-1)
            + self.item_bias(item_idx).squeeze(-1)
        )

    def bias_l2(self) -> Tensor:
        """Squared L2 norm of regularized parameters (intercept excluded)."""
        return self.user_bias.weight.square().sum() + self.item_bias.weight.square().sum()

    def factor_l2(self) -> Tensor:
        """Zero for API compatibility with :class:`FactorizationMachine`."""
        return self.global_bias.new_zeros(())

    def get_config(self) -> dict[str, Any]:
        return {"n_users": self.n_users, "n_items": self.n_items}


class FactorizationMachine(BiasModel):
    """Second-order FM for two one-hot fields: user and item.

    With exactly these two active features the general FM interaction becomes
    ``dot(user_factors[user], item_factors[item])``. Thus this model is also
    matrix factorization with user/item biases.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        *,
        n_factors: int = 16,
        global_mean: float = 0.0,
        init_std: float = 0.01,
        fast_interaction: bool = True,
    ) -> None:
        if not isinstance(n_factors, int) or n_factors <= 0:
            raise ValueError("n_factors must be a positive integer")
        if init_std <= 0:
            raise ValueError("init_std must be positive")
        self.n_factors = n_factors
        self.init_std = float(init_std)
        super().__init__(n_users, n_items, global_mean=global_mean)
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.fast_interaction = fast_interaction
        self._reset_factor_parameters()

    def _reset_factor_parameters(self) -> None:
        nn.init.normal_(self.user_factors.weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.item_factors.weight, mean=0.0, std=self.init_std)

    def reset_parameters(self, global_mean: float = 0.0) -> None:
        """Reset biases and use a non-zero random factor initialization."""
        self._reset_bias_parameters(global_mean)
        self._reset_factor_parameters()

    def interaction_direct(self, user_idx: Tensor, item_idx: Tensor) -> Tensor:
        """Direct pair interaction, used as the reference implementation."""
        _validate_inputs(user_idx, item_idx)
        user_factors = self.user_factors(user_idx)
        item_factors = self.item_factors(item_idx)
        return (user_factors * item_factors).sum(dim=1)

    def interaction_fast(self, user_idx: Tensor, item_idx: Tensor) -> Tensor:
        """Equivalent FM sum-of-squares identity without enumerating pairs."""
        _validate_inputs(user_idx, item_idx)
        user_factors = self.user_factors(user_idx)
        item_factors = self.item_factors(item_idx)
        return 0.5 * (
            (user_factors + item_factors).square()
            - user_factors.square()
            - item_factors.square()
        ).sum(dim=1)

    def forward(self, user_idx: Tensor, item_idx: Tensor) -> Tensor:
        if self.fast_interaction:
            return super().forward(user_idx, item_idx) + self.interaction_fast(
                user_idx, item_idx
            )

        return super().forward(user_idx, item_idx) + self.interaction_direct(
                user_idx, item_idx
            )

    def factor_l2(self) -> Tensor:
        return (
            self.user_factors.weight.square().sum()
            + self.item_factors.weight.square().sum()
        )

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "n_factors": self.n_factors,
                "init_std": self.init_std,
                "fast_interaction": self.fast_interaction,
            }
        )
        return config


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters of the optional reference training loop."""

    epochs: int = 200
    batch_size: int = 1024
    learning_rate: float = 0.01
    optimizer: str = "adam"
    reg_bias: float = 0.01
    reg_factors: float = 0.05
    patience: Optional[int] = 20
    min_delta: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        if self.reg_bias < 0 or self.reg_factors < 0:
            raise ValueError("regularization coefficients must be non-negative")
        if self.patience is not None and self.patience <= 0:
            raise ValueError("patience must be positive or None")
        if self.min_delta < 0:
            raise ValueError("min_delta must be non-negative")


@dataclass
class TrainingResult:
    """Metrics and metadata returned by :func:`fit_model`."""

    history: list[dict[str, Optional[float]]]
    best_epoch: int
    fit_time_seconds: float


def regularized_mse_loss(
    model: BiasModel,
    prediction: Tensor,
    target: Tensor,
    *,
    n_train: int,
    reg_bias: float,
    reg_factors: float,
) -> Tensor:
    """MSE plus explicit L2 penalties; the global intercept is not penalized."""
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("prediction and target must have the same 1-D shape")
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    mse = nn.functional.mse_loss(prediction, target)
    return (
        mse
        + reg_bias / n_train * model.bias_l2()
        + reg_factors / n_train * model.factor_l2()
    )


def _validate_training_tensors(
    user_idx: Tensor, item_idx: Tensor, rating: Tensor, name: str
) -> None:
    _validate_inputs(user_idx, item_idx)
    if rating.ndim != 1 or rating.shape != user_idx.shape:
        raise ValueError(f"{name}_rating must match the index tensors")
    if not rating.is_floating_point():
        raise TypeError(f"{name}_rating must have a floating-point dtype")
    if rating.numel() == 0:
        raise ValueError(f"{name} tensors must not be empty")
    if not torch.isfinite(rating).all():
        raise ValueError(f"{name}_rating must contain only finite values")


def _rmse(prediction: Tensor, target: Tensor) -> float:
    return torch.sqrt(nn.functional.mse_loss(prediction, target)).item()


def fit_model(
    model: BiasModel,
    train_user_idx: Tensor,
    train_item_idx: Tensor,
    train_rating: Tensor,
    *,
    val_user_idx: Optional[Tensor] = None,
    val_item_idx: Optional[Tensor] = None,
    val_rating: Optional[Tensor] = None,
    config: Optional[TrainingConfig] = None,
) -> TrainingResult:
    """Reference PyTorch training loop with validation-based model selection.

    The input tensors remain owned by the caller. Batches are transferred to
    the device on which ``model`` already resides. Validation arguments must be
    either all present or all absent. Test data is intentionally not accepted.
    """
    config = config or TrainingConfig()
    _validate_training_tensors(
        train_user_idx, train_item_idx, train_rating, "train"
    )
    validation_values = (val_user_idx, val_item_idx, val_rating)
    has_validation = all(value is not None for value in validation_values)
    if any(value is not None for value in validation_values) and not has_validation:
        raise ValueError("all three validation tensors must be provided together")
    if has_validation:
        assert val_user_idx is not None and val_item_idx is not None
        assert val_rating is not None
        _validate_training_tensors(val_user_idx, val_item_idx, val_rating, "val")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    torch.manual_seed(config.seed)
    model.reset_parameters(global_mean=train_rating.mean().item())
    model.to(device=device, dtype=dtype)

    optimizer_class = torch.optim.Adam if config.optimizer == "adam" else torch.optim.SGD
    optimizer = optimizer_class(model.parameters(), lr=config.learning_rate)
    n_train = train_rating.numel()
    history: list[dict[str, Optional[float]]] = []
    best_epoch = 0
    best_val_rmse = float("inf")
    best_state: Optional[dict[str, Tensor]] = None
    epochs_without_improvement = 0
    start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(n_train, generator=generator)
        for start_idx in range(0, n_train, config.batch_size):
            batch = permutation[start_idx : start_idx + config.batch_size]
            users = train_user_idx[batch].to(device=device)
            items = train_item_idx[batch].to(device=device)
            ratings = train_rating[batch].to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(users, items)
            loss = regularized_mse_loss(
                model,
                prediction,
                ratings,
                n_train=n_train,
                reg_bias=config.reg_bias,
                reg_factors=config.reg_factors,
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_users_device = train_user_idx.to(device=device)
            train_items_device = train_item_idx.to(device=device)
            train_ratings_device = train_rating.to(device=device, dtype=dtype)
            train_prediction = model(train_users_device, train_items_device)
            train_loss = regularized_mse_loss(
                model,
                train_prediction,
                train_ratings_device,
                n_train=n_train,
                reg_bias=config.reg_bias,
                reg_factors=config.reg_factors,
            ).item()
            train_rmse = _rmse(train_prediction, train_ratings_device)
            validation_rmse: Optional[float] = None
            if has_validation:
                assert val_user_idx is not None and val_item_idx is not None
                assert val_rating is not None
                validation_prediction = model(
                    val_user_idx.to(device=device), val_item_idx.to(device=device)
                )
                validation_rmse = _rmse(
                    validation_prediction,
                    val_rating.to(device=device, dtype=dtype),
                )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_rmse": train_rmse,
                "val_rmse": validation_rmse,
            }
        )
        if has_validation:
            assert validation_rmse is not None
            if validation_rmse < best_val_rmse - config.min_delta:
                best_val_rmse = validation_rmse
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (
                config.patience is not None
                and epochs_without_improvement >= config.patience
            ):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif not has_validation:
        best_epoch = len(history)
    model.eval()
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        fit_time_seconds=time.perf_counter() - start,
    )


def count_parameters(model: nn.Module) -> int:
    """Number of trainable scalar parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def predict_frame(
    model: BiasModel,
    frame: Any,
    *,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a Team 1 prepared table and preserve its ``row_id`` order.

    The frame must contain one-dimensional integer ``row_id``, ``user_idx`` and
    ``item_idx`` columns. Prepared validation/test tables already exclude
    unknown IDs; passing the ``-1`` sentinel from an ``*_all.csv`` file is an
    error rather than an accidental lookup of the final embedding row.
    """
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")

    columns: dict[str, np.ndarray] = {}
    for name in ("row_id", "user_idx", "item_idx"):
        try:
            value = frame[name]
        except (KeyError, TypeError) as error:
            raise ValueError(f"frame must contain the {name!r} column") from error
        if hasattr(value, "to_numpy"):
            value = value.to_numpy()
        array = np.asarray(value)
        if array.ndim != 1:
            raise ValueError(f"frame column {name!r} must be one-dimensional")
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"frame column {name!r} must contain integers")
        columns[name] = array

    row_ids = columns["row_id"]
    user_values = columns["user_idx"]
    item_values = columns["item_idx"]
    if (
        not (len(row_ids) == len(user_values) == len(item_values))
        or len(row_ids) == 0
    ):
        raise ValueError("model input columns must have the same non-zero length")
    if (user_values < 0).any() or (item_values < 0).any():
        raise ValueError("unknown user/item indices must be filtered before prediction")
    if (user_values >= model.n_users).any() or (item_values >= model.n_items).any():
        raise ValueError("user/item index is outside the model vocabulary")

    device = next(model.parameters()).device
    users = torch.as_tensor(user_values, dtype=torch.long)
    items = torch.as_tensor(item_values, dtype=torch.long)
    predictions = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(row_ids), batch_size):
                stop = start + batch_size
                predictions.append(
                    model(users[start:stop].to(device), items[start:stop].to(device))
                    .detach()
                    .cpu()
                )
    finally:
        model.train(was_training)
    return torch.cat(predictions).numpy(), row_ids.copy()


def save_checkpoint(
    path: str | Path,
    model: BiasModel,
    training_result: Optional[TrainingResult] = None,
) -> None:
    """Save a standard state-dict checkpoint plus enough metadata to reload it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_class": type(model).__name__,
        "model_config": model.get_config(),
        "model_state_dict": model.state_dict(),
    }
    if training_result is not None:
        payload["training_result"] = asdict(training_result)
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[BiasModel, Optional[TrainingResult]]:
    """Load a checkpoint created by :func:`save_checkpoint`."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    model_classes = {
        "BiasModel": BiasModel,
        "FactorizationMachine": FactorizationMachine,
    }
    try:
        model_class = model_classes[payload["model_class"]]
    except KeyError as error:
        raise ValueError("checkpoint contains an unknown model class") from error
    model = model_class(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    result_data = payload.get("training_result")
    result = TrainingResult(**result_data) if result_data is not None else None
    return model, result
