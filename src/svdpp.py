"""Canonical SVD++ with immutable, deduplicated train-only histories."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models import BiasModel, _validate_inputs, _validate_model_sizes


def history_checksum(histories: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(histories, separators=(",", ":")).encode()).hexdigest()


def build_train_histories(train: Any, n_users: int, n_items: int) -> list[list[int]]:
    """Use only user/item membership; ratings and held-out frames are not inputs."""
    _validate_model_sizes(n_users, n_items)
    users, items = np.asarray(train["user_idx"]), np.asarray(train["item_idx"])
    if users.ndim != 1 or users.shape != items.shape:
        raise ValueError("train indices must be equally sized 1-D vectors")
    if users.dtype.kind not in "iu" or items.dtype.kind not in "iu":
        raise TypeError("train indices must be integers")
    if (users < 0).any() or (users >= n_users).any() or (items < 0).any() or (items >= n_items).any():
        raise ValueError("train user/item indices outside the mapping")
    histories = [set() for _ in range(n_users)]
    for user, item in zip(users, items, strict=True):
        histories[int(user)].add(int(item))
    return [sorted(items) for items in histories]


class SVDPlusPlus(BiasModel):
    """mu + bu + bi + dot(qi, pu + sum(yj)/sqrt(history length)).

    EmbeddingBag computes each user's exact sum once per forward. Its autograd
    graph is rebuilt on every call; no stale training cache or dense rating
    matrix is used. Item zero is an ordinary item, including in empty bags.
    """

    def __init__(self, n_users: int, n_items: int, *, histories: list[list[int]],
                 n_factors: int = 20, global_mean: float = 0.0,
                 init_std: float = 0.01, normalization: str = "sqrt",
                 data_identity: dict | None = None) -> None:
        if type(n_factors) is not int or n_factors <= 0:
            raise ValueError("n_factors must be a positive integer")
        if not np.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std must be finite and positive")
        if normalization != "sqrt":
            raise ValueError("only canonical sqrt normalization is supported")
        _validate_model_sizes(n_users, n_items)
        if len(histories) != n_users:
            raise ValueError("one history is required for every user")
        for bag in histories:
            if any(type(j) is not int or not 0 <= j < n_items for j in bag):
                raise ValueError("history item index outside the mapping")
            if bag != sorted(set(bag)):
                raise ValueError("histories must be sorted and unique")
        super().__init__(n_users, n_items, global_mean=global_mean)
        del self.global_bias
        self.register_buffer("global_bias", torch.tensor(float(global_mean)))
        self.n_factors, self.init_std = n_factors, float(init_std)
        self.normalization = normalization
        self.data_identity = dict(data_identity or {})
        self.histories = [list(bag) for bag in histories]
        self.history_checksum = history_checksum(self.histories)
        lengths = torch.tensor([len(bag) for bag in histories], dtype=torch.long)
        self.register_buffer("history_items", torch.tensor([j for bag in histories for j in bag], dtype=torch.long))
        self.register_buffer("history_offsets", torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)]))
        self.register_buffer("history_lengths", lengths)
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.implicit_factors = nn.Embedding(n_items, n_factors)
        self._reset_factors()

    def _reset_factors(self) -> None:
        for embedding in (self.user_factors, self.item_factors, self.implicit_factors):
            nn.init.normal_(embedding.weight, mean=0.0, std=self.init_std)

    def reset_parameters(self, global_mean: float = 0.0) -> None:
        self._reset_bias_parameters(global_mean)
        self._reset_factors()

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        _validate_inputs(user_idx, item_idx)
        if ((user_idx < 0) | (user_idx >= self.n_users)).any() or ((item_idx < 0) | (item_idx >= self.n_items)).any():
            raise ValueError("unknown or out-of-range user/item index")
        implicit = F.embedding_bag(self.history_items, self.implicit_factors.weight,
                                   self.history_offsets, mode="sum", include_last_offset=True)
        scale = self.history_lengths.clamp_min(1).to(implicit.dtype).rsqrt()
        implicit = implicit * scale[:, None]
        profile = self.user_factors(user_idx) + implicit[user_idx]
        return super().forward(user_idx, item_idx) + (profile * self.item_factors(item_idx)).sum(dim=1)

    def factor_l2(self) -> torch.Tensor:
        return sum(embedding.weight.square().sum() for embedding in
                   (self.user_factors, self.item_factors, self.implicit_factors))

    def get_config(self) -> dict:
        return dict(n_users=self.n_users, n_items=self.n_items,
                    n_factors=self.n_factors, init_std=self.init_std,
                    histories=self.histories, normalization=self.normalization,
                    data_identity=self.data_identity)

    def verify_data_identity(self, expected: dict) -> None:
        if self.data_identity != expected:
            raise ValueError("checkpoint data identity (split/mappings/history) does not match")
        if expected.get("history_checksum", self.history_checksum) != self.history_checksum:
            raise ValueError("checkpoint train history checksum does not match")
