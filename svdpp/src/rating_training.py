"""Rating mini-batches for SVD++ with exact, freshly computed history sums."""
from __future__ import annotations
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.model import SVDPlusPlus
from src.training import FixedOrderScorer, cpu_state, predict, save_torch
from src.data import write_json


class HistoryOperator:
    """Immutable TRAIN membership; sparse forward/transpose matrices cached once."""
    def __init__(self, histories, n_items):
        self.matrix = torch.sparse_csr_tensor(
            torch.tensor(histories.offsets), torch.tensor(histories.items),
            torch.ones(len(histories.items)), size=(len(histories.lengths), n_items),
            check_invariants=True)
        self.transpose = self.matrix.transpose(0, 1).to_sparse_csr()
        self.lengths = torch.tensor(histories.lengths)
        self.cache = {}

    def tensors(self, device, dtype):
        key = (str(device), dtype)
        if key not in self.cache:
            self.cache[key] = (self.matrix.to(device=device, dtype=dtype),
                               self.transpose.to(device=device, dtype=dtype),
                               self.lengths.to(device=device, dtype=dtype).clamp_min(1).rsqrt())
        return self.cache[key]


class _HistorySum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights, operator):
        matrix, transpose, normalization = operator.tensors(weights.device, weights.dtype)
        ctx.save_for_backward(transpose, normalization)
        return torch.sparse.mm(matrix, weights) * normalization[:, None]

    @staticmethod
    def backward(ctx, gradient):
        transpose, normalization = ctx.saved_tensors
        return torch.sparse.mm(transpose, (gradient * normalization[:, None]).contiguous()), None


class RatingSVDPlusPlus(SVDPlusPlus):
    """Same SVD++ parameters/formula; history sums use exact sparse products.

    H is cached, H @ Y is recomputed at every optimizer step. GPU uses the
    same sparse product implementation.
    """
    def __init__(self, n_users, n_items, k, mean, histories, init_std=.01, operator=None):
        super().__init__(n_users, n_items, k, mean, histories, init_std * k ** .25)
        self.history_operator = operator if operator is not None else HistoryOperator(histories, n_items)

    def profiles(self, users):
        implicit = _HistorySum.apply(self.y.weight, self.history_operator)
        return self.p(users) + implicit[users]


def regularization(model, n_train, reg_bias, reg_factors):
    bias = model.bu.weight.square().sum() + model.bi.weight.square().sum()
    factors = sum(table.weight.square().sum() for table in (model.p, model.q, model.y))
    return (reg_bias * bias + reg_factors * factors) / n_train


def rating_batches(n_ratings, batch_size, generator):
    """Every rating appears exactly once, including a possibly short last batch."""
    order = torch.randperm(n_ratings, generator=generator)
    return order.split(batch_size)


def tensors(frame, device):
    # pandas 3 can expose read-only arrays; torch.tensor owns a writable copy.
    return tuple(torch.tensor(frame[name].to_numpy(), dtype=dtype, device=device)
                 for name, dtype in (("user_idx", torch.long), ("item_idx", torch.long),
                                     ("rating", torch.float32)))


def fit_ratings(train, validation, histories, n_users, n_items, cfg, device, k, seed,
                output, signature, operator=None, scorers=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = RatingSVDPlusPlus(n_users, n_items, k, float(train.rating.mean()), histories,
                             cfg["init_std"], operator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_t, val_t = tensors(train, device), tensors(validation, device)
    train_score, val_score = scorers or (FixedOrderScorer(train), FixedOrderScorer(validation))
    n_updates = math.ceil(len(train) / cfg["batch_size"])
    last = output / "last.pt"
    history, best_state, best_val, best_epoch, elapsed, updates, first_epoch = [], None, math.inf, 0, 0., 0, 1
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Checkpoint belongs to different code/data/settings")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"].cpu())
        history, best_state, best_val, best_epoch, elapsed, updates = (
            state[name] for name in ("history", "best_state", "best_val", "best_epoch", "elapsed", "updates"))
        first_epoch = state["epoch"] + 1
        print(f"RESUME k={k} epoch={first_epoch} updates={updates}", flush=True)
    for epoch in range(first_epoch, cfg["max_epochs"] + 1):
        if history and history[-1]["epoch"] - best_epoch >= cfg["patience"]:
            break
        model.train()
        started = time.perf_counter()
        for step, rows in enumerate(rating_batches(len(train), cfg["batch_size"], generator), 1):
            rows = rows.to(device)
            users, items, ratings = (column[rows] for column in train_t)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(users, items) - ratings).square().mean()
            loss = loss + regularization(model, len(train), cfg["reg_bias"], cfg["reg_factors"])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            updates += 1
            if step % cfg.get("progress_every", 200) == 0:
                print(f"k={k} epoch={epoch} update={step}/{n_updates}", flush=True)
        train_rmse = train_score(predict(model, train_t, cfg["eval_batch_size"]))
        val_rmse = val_score(predict(model, val_t, cfg["eval_batch_size"]))
        penalty = regularization(model, len(train), cfg["reg_bias"], cfg["reg_factors"]).detach().item()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        elapsed += seconds
        if val_rmse < best_val:
            best_val, best_epoch, best_state = val_rmse, epoch, cpu_state(model)
        history.append(dict(epoch=epoch, train_rmse=train_rmse, validation_rmse=val_rmse,
                            train_mse=train_rmse ** 2, train_objective=train_rmse ** 2 + penalty,
                            epoch_seconds=seconds, updates_in_epoch=n_updates, optimizer_updates=updates))
        pd.DataFrame(history).to_csv(output / "history.csv", index=False, lineterminator="\n")
        write_json(output / "progress.json", dict(k=k, seed=seed, epoch=epoch, optimizer_updates=updates,
                    train_rmse=train_rmse, validation_rmse=val_rmse, best_validation_rmse=best_val,
                    best_epoch=best_epoch, elapsed_seconds=elapsed, completed=False))
        save_torch(last, dict(signature=signature, epoch=epoch, model=cpu_state(model),
                    optimizer=optimizer.state_dict(), generator=generator.get_state(), history=history,
                    best_state=best_state, best_val=best_val, best_epoch=best_epoch,
                    elapsed=elapsed, updates=updates))
        print(f"k={k} seed={seed} epoch={epoch} updates={updates} train={train_rmse:.6f} "
              f"val={val_rmse:.6f} best={best_val:.6f} sec={seconds:.2f}", flush=True)
    if best_state is None:
        raise RuntimeError("No completed epoch")
    model.load_state_dict(best_state)
    metrics = dict(k=k, seed=seed, train_rmse=train_score(predict(model, train_t, cfg["eval_batch_size"])),
                   validation_rmse=val_score(predict(model, val_t, cfg["eval_batch_size"])),
                   best_epoch=best_epoch, epochs_run=history[-1]["epoch"], updates_per_epoch=n_updates,
                   optimizer_updates=updates, fit_seconds=elapsed,
                   hit_epoch_cap=history[-1]["epoch"] == cfg["max_epochs"], time_limited=False,
                   learning_rate=cfg["learning_rate"], reg_bias=cfg["reg_bias"],
                   reg_factors=cfg["reg_factors"], batch_size=cfg["batch_size"], init_std=cfg["init_std"])
    save_torch(output / "model.pt", dict(state=cpu_state(model), metrics=metrics, signature=signature))
    write_json(output / "metrics.json", metrics)
    write_json(output / "progress.json", dict(**metrics, completed=True))
    return model, metrics, history
