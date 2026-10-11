"""Цикл обучения: группировка по пользователям, loss, Adam, early stopping.

fit_one() обучает одну модель и восстанавливает состояние лучшей эпохи.
Каждая оценка train участвует в обновлении один раз за эпоху.
"""
from __future__ import annotations
import math
import time
from pathlib import Path
import numpy as np
import torch
from src.model import SVDPlusPlus
from src.metrics import rmse, rmse_by_row_id, require_same_row_ids

def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)



class UserBatches:
    """Shuffle users and visit every training rating once per epoch."""
    def __init__(self, frame, n_users):
        self.order = np.argsort(frame.user_idx.to_numpy(), kind="stable")
        lengths = np.bincount(frame.user_idx.to_numpy(), minlength=n_users)
        self.offsets = np.r_[0, lengths.cumsum()]
        self.active_users = np.flatnonzero(lengths)

    def rows(self, users):
        lengths = self.offsets[users + 1] - self.offsets[users]
        offsets = np.r_[0, lengths.cumsum()]
        positions = np.repeat(self.offsets[users], lengths)
        positions += np.arange(offsets[-1]) - np.repeat(offsets[:-1], lengths)
        return self.order[positions]



def prepare_fixed_batches(batches, batch_users, device):
    """Prepare exact row coverage and inverse profile lookup once."""
    prepared = []
    for start in range(0, len(batches.active_users), batch_users):
        group = batches.active_users[start:start + batch_users]
        lengths = batches.offsets[group + 1] - batches.offsets[group]
        prepared.append((torch.as_tensor(batches.rows(group), device=device),
                         torch.as_tensor(group, device=device),
                         torch.as_tensor(np.repeat(np.arange(len(group)), lengths), device=device)))
    return prepared


def frame_tensors(frame, device):
    return tuple(torch.as_tensor(frame[name].to_numpy(), device=device, dtype=dtype)
                 for name, dtype in (("user_idx", torch.long), ("item_idx", torch.long), ("rating", torch.float32)))



@torch.no_grad()
def predict(model, tensors, batch_size):
    model.eval()
    users, items, _ = tensors
    profiles = model.profiles(torch.arange(model.p.num_embeddings, device=users.device))
    chunks = []
    for start in range(0, len(users), batch_size):
        u, i = users[start:start + batch_size], items[start:start + batch_size]
        chunks.append(model.from_profiles(u, i, profiles[u]).cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite predictions")
    return result



def score(frame, prediction):
    return float(rmse_by_row_id(frame.rating.to_numpy(), prediction, frame.row_id.to_numpy(), frame.row_id.to_numpy()))



class FixedOrderScorer:
    """Validate fixed observation IDs once; predictions follow predict() input order.

    Independent/external predictions must still use score() or supply their IDs.
    Shape, numeric type and finiteness are checked on every RMSE calculation.
    """
    def __init__(self, frame):
        self.row_ids = frame.row_id.to_numpy(copy=True)
        require_same_row_ids(self.row_ids, self.row_ids)
        self.targets = frame.rating.to_numpy(copy=True)
        rmse(self.targets, self.targets)
        self.row_ids.setflags(write=False)
        self.targets.setflags(write=False)

    def __call__(self, prediction, prediction_row_ids=None):
        if prediction_row_ids is not None:
            ids = np.asarray(prediction_row_ids)
            if ids.dtype.kind not in "iu" or not np.array_equal(ids, self.row_ids):
                raise ValueError("observation row_id values/order do not match")
        return float(rmse(self.targets, prediction))


def load_training_checkpoint(resume_path, signature, device,
                             initial_checkpoint=None, initial_signature=None):
    """An active continuation takes priority over the completed screening state."""
    if resume_path is not None and Path(resume_path).exists():
        path, expected = resume_path, signature
    elif initial_checkpoint is not None:
        path, expected = initial_checkpoint, initial_signature
    else:
        return None
    state = torch.load(path, map_location=device, weights_only=False)
    if expected is None or state["signature"] != expected:
        raise ValueError("active checkpoint belongs to another protocol")
    return state


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}



def fit_one(train, validation, histories, n_users, n_items, cfg, device, k, lr, reg, seed,
            resume_path=None, signature=None, max_epochs=None, min_epochs=1,
            initial_checkpoint=None, initial_signature=None, deadline=None, scorers=None):
    """MSE + lambda/N * global squared L2, Adam with weight_decay=0.

    Unequal rating counts per user batch require n_batches/N times batch SSE.
    Averaged over an epoch at fixed parameters this equals the full objective.
    """
    epochs = max_epochs if max_epochs is not None else cfg["max_epochs"]
    if type(min_epochs) is not int or min_epochs <= 0 or min_epochs > epochs:
        raise ValueError("min_epochs must be a positive integer <= the epoch budget")
    eval_every = cfg.get("eval_every", 1)
    if type(eval_every) is not int or eval_every <= 0:
        raise ValueError("eval_every must be a positive integer")
    train_score, val_score = scorers if scorers is not None else (FixedOrderScorer(train), FixedOrderScorer(validation))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SVDPlusPlus(n_users, n_items, k, float(train.rating.mean()), histories, cfg["init_base_std"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    generator = np.random.default_rng(seed)
    train_t = frame_tensors(train, device)
    val_t = frame_tensors(validation, device)
    batches = UserBatches(train, n_users)
    n_batches = math.ceil(len(batches.active_users) / cfg["batch_users"])
    prepared = (prepare_fixed_batches(batches, cfg["batch_users"], device)
                if cfg.get("fixed_user_batches", False) else [])
    history, best_state, best_rmse, best_epoch, stalled, elapsed = [], None, math.inf, 0, 0, 0.0
    first_epoch, last_epoch, time_limited = 1, 0, False
    state = load_training_checkpoint(resume_path, signature, device, initial_checkpoint, initial_signature)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.bit_generator.state = state["rng"]
        history, best_state, best_rmse, best_epoch, stalled, elapsed = (
            state[name] for name in ("history", "best_state", "best_rmse", "best_epoch", "stalled", "elapsed"))
        last_epoch = state.get("epoch", int(history[-1]["epoch"]))
        first_epoch = last_epoch + 1
        print(f"resume epoch {first_epoch} k={k} lr={lr:g} lambda={reg:g} seed={seed}", flush=True)
    for epoch in range(first_epoch, epochs + 1):
        # Always finish one epoch; a deadline is a bounded-budget result, not
        # evidence of convergence. Checks at epoch boundaries preserve resumability.
        if deadline is not None and time.perf_counter() >= deadline and best_state is not None:
            time_limited = True
            break
        if epoch > min_epochs and stalled >= cfg["patience"]:
            break
        synchronize(device)
        started = time.perf_counter()
        model.train()
        if prepared:
            epoch_batches = (prepared[index] for index in generator.permutation(len(prepared)))
        else:
            user_order = generator.permutation(batches.active_users)
            epoch_batches = ((torch.as_tensor(batches.rows(user_order[start:start + cfg["batch_users"]]),
                                             device=device), None, None)
                             for start in range(0, len(user_order), cfg["batch_users"]))
        for rows, group, inverse in epoch_batches:
            users, items, ratings = (values[rows] for values in train_t)
            optimizer.zero_grad(set_to_none=True)
            predictions = (model.from_profiles(users, items, model.profiles(group)[inverse])
                           if group is not None else model(users, items))
            loss = (predictions - ratings).square().sum() * (n_batches / len(train)) + reg / len(train) * model.l2()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss; inspect the learning rate")
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise FloatingPointError("nonfinite gradients")
            optimizer.step()
        synchronize(device)
        expired = deadline is not None and time.perf_counter() >= deadline
        evaluate = epoch == 1 or epoch % eval_every == 0 or epoch == epochs or expired
        if evaluate:
            train_rmse = train_score(predict(model, train_t, cfg["eval_batch_size"]))
            val_rmse = val_score(predict(model, val_t, cfg["eval_batch_size"]))
            penalty = reg / len(train) * model.l2().detach().item()
            if val_rmse < best_rmse:
                best_rmse, best_epoch = val_rmse, epoch
                best_state = cpu_state(model)
            stalled = epoch - best_epoch
        synchronize(device)
        seconds = time.perf_counter() - started
        elapsed += seconds
        last_epoch = epoch
        if evaluate:
            history.append(dict(epoch=epoch, train_rmse=train_rmse, validation_rmse=val_rmse,
                                train_mse=train_rmse ** 2, train_objective=train_rmse ** 2 + penalty,
                                epoch_seconds=seconds))
        should_stop = epoch >= min_epochs and stalled >= cfg["patience"]
        expired = deadline is not None and time.perf_counter() >= deadline
        time_limited = expired and epoch < epochs
        if evaluate and (epoch == 1 or epoch % cfg["log_every"] == 0 or should_stop or epoch == epochs or expired):
            print(f"k={k} lr={lr:g} lambda={reg:g} seed={seed} epoch={epoch} "
                  f"train={train_rmse:.6f} val={val_rmse:.6f} best={best_rmse:.6f} sec={seconds:.2f}", flush=True)
        if resume_path is not None and (epoch == 1 or epoch % cfg["checkpoint_every"] == 0 or epoch == epochs or should_stop or expired):
            save_torch(resume_path, dict(signature=signature, epoch=epoch, model=cpu_state(model),
                       optimizer=optimizer.state_dict(), rng=generator.bit_generator.state,
                       history=history, best_state=best_state, best_rmse=best_rmse,
                       best_epoch=best_epoch, stalled=stalled, elapsed=elapsed))
        if should_stop or expired:
            break
    if best_state is None:
        raise RuntimeError("no successful training epoch")
    model.load_state_dict(best_state)
    best_train = train_score(predict(model, train_t, cfg["eval_batch_size"]))
    best_val = val_score(predict(model, val_t, cfg["eval_batch_size"]))
    if abs(best_val - best_rmse) > 2e-6 or abs(model.mu.item() - float(train.rating.mean())) > 1e-6:
        raise RuntimeError("best-state restore or fixed train mean check failed")
    return model, dict(k=k, learning_rate=lr, regularization=reg, seed=seed,
                      train_rmse=best_train, validation_rmse=best_val, best_epoch=best_epoch,
                      epochs_run=last_epoch, hit_epoch_cap=last_epoch == epochs, time_limited=time_limited,
                      fit_seconds=elapsed, n_parameters=sum(p.numel() for p in model.parameters()),
                      init_std=model.init_std), history

