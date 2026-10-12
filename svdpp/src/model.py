"""Каноническая SVD++: обучаемые b_u, b_i, p_u, q_i, y_j.

Прогноз: mu + b_u + b_i + q_i @ (p_u + sum(y_j) / sqrt(|N(u)|)).
N(u) содержит только фильмы из TRAIN; значения оценок не входят в сумму.
Формула из пакета от 6 октября; быстрый режим кеширует только индексы историй.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class Histories:
    """CSR histories; unique item membership from TRAIN only, without ratings."""
    def __init__(self, train, n_users):
        pairs = train[["user_idx", "item_idx"]].drop_duplicates().sort_values(["user_idx", "item_idx"])
        self.items = pairs.item_idx.to_numpy(dtype=np.int64)
        self.lengths = np.bincount(pairs.user_idx.to_numpy(), minlength=n_users)
        self.offsets = np.r_[0, self.lengths.cumsum()]
        self.checksum = hashlib.sha256(self.items.astype("<i8").tobytes() + self.offsets.astype("<i8").tobytes()).hexdigest()

    def enable_index_cache(self, max_entries=32):
        """Cache only immutable membership indices, never trainable Y/profile sums."""
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self._index_cache = {}
        self._cache_limit = max_entries

    def select(self, users, device):
        users = np.asarray(users, dtype=np.int64)
        cache = getattr(self, "_index_cache", None)
        key = (str(device), tuple(users)) if cache is not None else None
        if cache is not None and key in cache:
            return cache[key]
        lengths = self.lengths[users]
        offsets = np.r_[0, lengths.cumsum()]
        # Each selected history is concatenated exactly once, including empty bags.
        source = np.repeat(self.offsets[users], lengths)
        source += np.arange(offsets[-1]) - np.repeat(offsets[:-1], lengths)
        result = (
            torch.as_tensor(self.items[source], device=device, dtype=torch.long),
            torch.as_tensor(offsets, device=device, dtype=torch.long),
            torch.as_tensor(lengths, device=device, dtype=torch.float32),
        )
        if cache is not None:
            if len(cache) >= self._cache_limit:
                cache.pop(next(iter(cache)))
            cache[key] = result
        return result



class SVDPlusPlus(nn.Module):
    def __init__(self, n_users, n_items, k, mean, histories, init_base_std=0.1):
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.k, self.histories = k, histories
        self.register_buffer("mu", torch.tensor(float(mean), dtype=torch.float32))
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        self.p = nn.Embedding(n_users, k)
        self.q = nn.Embedding(n_items, k)
        self.y = nn.Embedding(n_items, k)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)
        # k * sigma^4 is constant: similar initial dot-product variance for all k.
        self.init_std = init_base_std / k ** 0.25
        for table in (self.p, self.q, self.y):
            nn.init.normal_(table.weight, std=self.init_std)

    def profiles(self, users):
        items, offsets, lengths = self.histories.select(users.detach().cpu().numpy(), users.device)
        implicit = F.embedding_bag(items, self.y.weight, offsets, mode="sum", include_last_offset=True)
        return self.p(users) + implicit * lengths.clamp_min(1).rsqrt()[:, None]

    def forward(self, users, items):
        unique, inverse = torch.unique(users, sorted=True, return_inverse=True)
        profiles = self.profiles(unique)
        return self.from_profiles(users, items, profiles[inverse])

    def from_profiles(self, users, items, profiles):
        return self.mu + self.bu(users).squeeze(-1) + self.bi(items).squeeze(-1) + (self.q(items) * profiles).sum(-1)

    def l2(self):
        return sum(table.weight.square().sum() for table in (self.bu, self.bi, self.p, self.q, self.y))

