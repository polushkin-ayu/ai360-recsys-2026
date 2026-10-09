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
from torch.nn import functional as F


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


class SparseFactorizationMachine(nn.Module):
    """General second-order FM for arbitrary real-valued feature vectors.

    Dense input is accepted by :meth:`forward` as a matrix ``[batch, n_features]``.
    Sparse input is accepted by :meth:`forward_sparse` in canonical CSR form:
    active feature IDs are unique and sorted within each row, values are flattened
    across rows, and ``row_offsets`` contains ``batch + 1`` boundaries including
    the final number of non-zero values.

    Both paths compute

    ``w0 + sum_i(w_i*x_i) + sum_{i<j}<v_i,v_j>*x_i*x_j``.
    """

    def __init__(
        self,
        n_features: int,
        *,
        n_factors: int = 16,
        global_mean: float = 0.0,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if not isinstance(n_features, int) or isinstance(n_features, bool):
            raise TypeError("n_features must be an integer")
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if not isinstance(n_factors, int) or isinstance(n_factors, bool):
            raise TypeError("n_factors must be an integer")
        if n_factors <= 0:
            raise ValueError("n_factors must be positive")
        if not np.isfinite(global_mean):
            raise ValueError("global_mean must be finite")
        if not np.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std must be positive and finite")

        self.n_features = n_features
        self.n_factors = n_factors
        self.init_std = float(init_std)
        self.global_bias = nn.Parameter(torch.tensor(float(global_mean)))
        self.linear_weights = nn.Embedding(n_features, 1)
        self.feature_factors = nn.Embedding(n_features, n_factors)
        self.reset_parameters(global_mean=global_mean)

    def reset_parameters(self, global_mean: float = 0.0) -> None:
        if not np.isfinite(global_mean):
            raise ValueError("global_mean must be finite")
        with torch.no_grad():
            self.global_bias.fill_(float(global_mean))
            self.linear_weights.weight.zero_()
        nn.init.normal_(self.feature_factors.weight, mean=0.0, std=self.init_std)

    def _validate_dense_input(self, features: Tensor) -> None:
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, n_features]")
        if features.shape[1] != self.n_features:
            raise ValueError(
                f"features must contain exactly {self.n_features} columns"
            )
        if not features.is_floating_point():
            raise TypeError("features must have a floating-point dtype")
        if not torch.isfinite(features).all():
            raise ValueError("features must contain only finite values")

    def forward(self, features: Tensor) -> Tensor:
        """Predict from a dense feature matrix of shape ``[batch, p]``."""
        self._validate_dense_input(features)
        dtype = self.global_bias.dtype
        if features.dtype != dtype:
            features = features.to(dtype=dtype)

        weights = self.linear_weights.weight.squeeze(-1)
        factors = self.feature_factors.weight
        linear = features @ weights
        summed_factors = features @ factors
        squared_factors = features.square() @ factors.square()
        interaction = 0.5 * (
            summed_factors.square() - squared_factors
        ).sum(dim=1)
        return self.global_bias + linear + interaction

    def _validate_sparse_input(
        self,
        feature_indices: Tensor,
        feature_values: Tensor,
        row_offsets: Tensor,
    ) -> None:
        if feature_indices.ndim != 1 or feature_values.ndim != 1:
            raise ValueError(
                "feature_indices and feature_values must be one-dimensional"
            )
        if feature_indices.shape != feature_values.shape:
            raise ValueError(
                "feature_indices and feature_values must have the same shape"
            )
        if row_offsets.ndim != 1 or row_offsets.numel() < 1:
            raise ValueError(
                "row_offsets must be a non-empty one-dimensional tensor"
            )
        if feature_indices.dtype != torch.long or row_offsets.dtype != torch.long:
            raise TypeError(
                "feature_indices and row_offsets must have dtype torch.long"
            )
        if not feature_values.is_floating_point():
            raise TypeError("feature_values must have a floating-point dtype")
        if not torch.isfinite(feature_values).all():
            raise ValueError("feature_values must contain only finite values")
        if feature_indices.numel() and (
            (feature_indices < 0).any()
            or (feature_indices >= self.n_features).any()
        ):
            raise ValueError("feature index is outside the model vocabulary")
        if row_offsets[0].item() != 0:
            raise ValueError("row_offsets must start with zero")
        if row_offsets[-1].item() != feature_indices.numel():
            raise ValueError("the final row offset must equal the number of values")
        if row_offsets.numel() > 1 and (row_offsets[1:] < row_offsets[:-1]).any():
            raise ValueError("row_offsets must be non-decreasing")
        if feature_indices.numel() > 1:
            counts = row_offsets[1:] - row_offsets[:-1]
            row_ids = torch.repeat_interleave(
                torch.arange(
                    counts.numel(), device=row_offsets.device, dtype=torch.long
                ),
                counts,
            )
            same_row = row_ids[1:] == row_ids[:-1]
            if (same_row & (feature_indices[1:] <= feature_indices[:-1])).any():
                raise ValueError(
                    "feature indices must be unique and increasing within each row"
                )

    def forward_sparse(
        self,
        feature_indices: Tensor,
        feature_values: Tensor,
        row_offsets: Tensor,
    ) -> Tensor:
        """Predict from a CSR batch without materializing a dense design matrix."""
        self._validate_sparse_input(feature_indices, feature_values, row_offsets)
        dtype = self.global_bias.dtype
        if feature_values.dtype != dtype:
            feature_values = feature_values.to(dtype=dtype)

        linear = F.embedding_bag(
            feature_indices,
            self.linear_weights.weight,
            row_offsets,
            mode="sum",
            per_sample_weights=feature_values,
            include_last_offset=True,
        ).squeeze(-1)
        summed_factors = F.embedding_bag(
            feature_indices,
            self.feature_factors.weight,
            row_offsets,
            mode="sum",
            per_sample_weights=feature_values,
            include_last_offset=True,
        )
        squared_factors = F.embedding_bag(
            feature_indices,
            self.feature_factors.weight.square(),
            row_offsets,
            mode="sum",
            per_sample_weights=feature_values.square(),
            include_last_offset=True,
        )
        interaction = 0.5 * (
            summed_factors.square() - squared_factors
        ).sum(dim=1)
        return self.global_bias + linear + interaction

    def bias_l2(self) -> Tensor:
        return self.linear_weights.weight.square().sum()

    def factor_l2(self) -> Tensor:
        return self.feature_factors.weight.square().sum()

    def get_config(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "n_factors": self.n_factors,
            "init_std": self.init_std,
        }
class PureMatrixFactorization(nn.Module):
    """
    Матричная факторизация без линейных сдвигов
    """

    def __init__(
            self,
            n_users: int,
            n_items: int,
            n_factors: int,
            global_mean: float = 0.0,
            init_std: float = 0.01,
            data_identity: dict | None = None,
    ) -> None:
        super().__init__()
        _validate_model_sizes(n_users, n_items)
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.init_std = float(init_std)
        self.data_identity = data_identity or {}

        # Глобальное среднее регистрируется как буфер без градиентов
        self.register_buffer("global_bias", torch.tensor(float(global_mean), dtype=torch.float32))

        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)

        self.reset_parameters()

    def reset_parameters(self, global_mean: float | None = None) -> None:
        if global_mean is not None:
            self.global_bias.fill_(float(global_mean))

        nn.init.normal_(self.user_factors.weight, mean=0.0, std=self.init_std)
        nn.init.normal_(self.item_factors.weight, mean=0.0, std=self.init_std)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        _validate_inputs(users, items)
        p_u = self.user_factors(users)
        q_i = self.item_factors(items)
        return self.global_bias + (p_u * q_i).sum(dim=1)

    def bias_l2(self) -> torch.Tensor:
        """Обучаемых смещений нет — возвращаем точный скалярный ноль."""
        return self.global_bias.new_zeros(())

    def factor_l2(self) -> torch.Tensor:
        """L2 норма факторных матриц."""
        return (
                self.user_factors.weight.square().sum()
                + self.item_factors.weight.square().sum()
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_factors": self.n_factors,
            "init_std": self.init_std,
            "data_identity": self.data_identity,
        }

    def verify_data_identity(self, identity: dict) -> None:
        if self.data_identity != identity:
            raise ValueError(
                f"Data identity mismatch: expected {self.data_identity}, got {identity}"
            )

@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters of the optional reference training loop."""

    epochs: int = 200
    batch_size: int = 1024
    learning_rate: float = 0.01
    optimizer: str = "adam"
    reg_bias: float = 0.01
    reg_factors: float = 0.05
    normalize_regularization: bool = False
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
    model: BiasModel | SparseFactorizationMachine,
    prediction: Tensor,
    target: Tensor,
    *,
    n_train: int,
    reg_bias: float,
    reg_factors: float,
    normalize_regularization: bool = False,
) -> Tensor:
    """MSE plus explicit L2 penalties; the global intercept is not penalized."""
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("prediction and target must have the same 1-D shape")
    if n_train <= 0:
        raise ValueError("n_train must be positive")

    mse = nn.functional.mse_loss(prediction, target)

    scale = n_train if normalize_regularization else 1

    return (
        mse
        + reg_bias / scale * model.bias_l2()
        + reg_factors / scale * model.factor_l2()
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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
                normalize_regularization=config.normalize_regularization,
            )
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
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
                normalize_regularization=config.normalize_regularization,
            ).item()
            train_rmse = _rmse(train_prediction, train_ratings_device)
            train_mse = nn.functional.mse_loss(train_prediction, train_ratings_device).item()
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
            if not np.isfinite(train_loss) or (validation_rmse is not None and not np.isfinite(validation_rmse)):
                raise ValueError("nonfinite evaluation metrics")

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_mse": train_mse,
                "train_loss_with_regularization": train_loss,
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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
    prediction = torch.cat(predictions).numpy()
    if prediction.shape != row_ids.shape or not np.isfinite(prediction).all():
        raise ValueError("predictions must be finite and match the input row count")
    return prediction, row_ids.copy()


def save_checkpoint(
    path: str | Path,
    model: BiasModel | SparseFactorizationMachine,
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
) -> tuple[BiasModel | SparseFactorizationMachine, Optional[TrainingResult]]:
    """Load a checkpoint created by :func:`save_checkpoint`."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    from src.svdpp import SVDPlusPlus
    from src.polynomial import (
        UserItemPolynomialRegression2,
        UserItemFactorizedPolynomialRegression,
    )

    model_classes = {
        "BiasModel": BiasModel,
        "FactorizationMachine": FactorizationMachine,
        "UserItemPolynomialRegression2": UserItemPolynomialRegression2,
        "UserItemFactorizedPolynomialRegression": (
            UserItemFactorizedPolynomialRegression
        ),
        "SVDPlusPlus": SVDPlusPlus,
        "SparseFactorizationMachine": SparseFactorizationMachine,
        "PureMatrixFactorization" : PureMatrixFactorization,
    }
    try:
        model_class = model_classes[payload["model_class"]]
    except KeyError as error:
        raise ValueError("checkpoint contains an unknown model class") from error
    model = model_class(**payload["model_config"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(map_location)
    model.eval()
    result_data = payload.get("training_result")
    result = TrainingResult(**result_data) if result_data is not None else None
    return model, result