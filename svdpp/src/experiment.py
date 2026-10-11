"""Полный эксперимент SVD++ / ML1M: подбор по validation и закрытый test.

python -m src.experiment --help
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from src.data import ROOT, sha256
from src.metrics import rmse_by_row_id


from src.plotting import (plot_loss_history, plot_final_loss_curves,
                          plot_rmse_vs_k, plot_tuning)
from src.model import Histories, SVDPlusPlus
from src.training import (save_torch, UserBatches, frame_tensors, predict, score,
                          synchronize, cpu_state, fit_one)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()



def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))



def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)



def git_info():
    try:
        return {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}



def environment(device):
    return {
        "python": platform.python_version(), "os": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_version": torch.version.cuda,
    }



def validate_frame(frame, n_users, n_items, name):
    if frame.empty or not frame.row_id.is_unique or not np.isfinite(frame.rating).all():
        raise ValueError(f"invalid {name} rows")
    for column, size in (("user_idx", n_users), ("item_idx", n_items)):
        if frame[column].dtype.kind not in "iu" or not frame[column].between(0, size - 1).all():
            raise ValueError(f"{name}: invalid/unknown indices")



def validate_search_values(learning_rates, regularizations):
    if any(not math.isfinite(v) or v <= 0 for v in learning_rates):
        raise ValueError("finite positive learning rates required")
    if any(not math.isfinite(v) or v < 0 for v in regularizations):
        raise ValueError("finite nonnegative regularizations required")


def tuning_strategy(cfg):
    """Absent strategy preserves the exhaustive protocol in older configs."""
    strategy = dict(cfg.get("tuning_strategy", {"mode": "grid"}))
    if strategy.get("mode") not in ("grid", "two_stage"):
        raise ValueError("tuning_strategy.mode must be grid or two_stage")
    if strategy["mode"] == "two_stage":
        for field in ("screening_epochs", "candidates_per_learning_rate"):
            if type(strategy.get(field)) is not int or strategy[field] <= 0:
                raise ValueError(f"tuning_strategy.{field} must be a positive integer")
        if strategy["screening_epochs"] > cfg["max_epochs"]:
            raise ValueError("screening_epochs must not exceed max_epochs")
    return strategy


def shortlist_candidates(rows, count_per_lr):
    """Keep the best lambdas separately for each LR, including slow starters."""
    chosen = []
    for lr in sorted({row["learning_rate"] for row in rows}):
        group = sorted((row for row in rows if row["learning_rate"] == lr),
                       key=lambda row: (row["validation_rmse"], row["regularization"]))
        chosen.extend(group[:count_per_lr])
    return chosen


def context(config_path, device_override=None):
    cfg = read_json(ROOT / config_path)
    for field in ("max_epochs", "patience", "batch_users", "eval_batch_size", "checkpoint_every", "log_every", "threads"):
        if type(cfg[field]) is not int or cfg[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("k_grid", "tuning_seeds", "final_seeds", "learning_rates", "regularizations"):
        if not cfg[field] or len(set(cfg[field])) != len(cfg[field]):
            raise ValueError(f"{field} must be nonempty and unique")
    if set(cfg["tuning_seeds"]) & set(cfg["final_seeds"]):
        raise ValueError("final seeds must be different from tuning seeds")
    if len(cfg["tuning_seeds"]) < 2 or len(cfg["final_seeds"]) < 2:
        raise ValueError("at least two seeds are required for sample standard deviation")
    if any(k <= 0 or type(k) is not int for k in cfg["k_grid"]):
        raise ValueError("positive integer k required")
    validate_search_values(cfg["learning_rates"], cfg["regularizations"])
    strategy = tuning_strategy(cfg)
    torch.set_num_threads(cfg["threads"])
    requested = device_override or cfg["device"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(device.type == "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    directory = ROOT / cfg["data_dir"]
    meta = read_json(directory / "metadata.json")
    if meta["dataset"] != "MovieLens 1M":
        raise ValueError("MovieLens 1M required")
    reference = read_json(ROOT / "results/movielens1m-data/reproducibility.json")
    if reference["split_id"] != cfg["expected_split_id"]:
        raise ValueError("unexpected shared split reference")
    if digest(meta["artifact_sha256"]) != meta["split"]["id"]:
        raise ValueError("split identifier does not match metadata artifact hashes")
    for filename, expected in meta["artifact_sha256"].items():
        if sha256(directory / filename) != expected:
            raise ValueError(f"prepared data checksum mismatch: {filename}")
    # Windows writes this auxiliary JSON with CRLF; Linux writes it with LF.
    # Every CSV must match the shared reference byte-for-byte in either case.
    for filename, expected in reference["artifact_sha256"].items():
        if filename != "selected_users.json" and meta["artifact_sha256"].get(filename) != expected:
            raise ValueError(f"data differ from the shared reference: {filename}")
    train, val = (pd.read_csv(directory / filename) for filename in ("train.csv", "validation.csv"))
    for name, frame in (("train", train), ("validation", val)):
        if len(frame) != cfg["expected_sizes"][name]:
            raise ValueError(f"unexpected {name} size")
        validate_frame(frame, meta["n_users"], meta["n_items"], name)
    if set(train.row_id) & set(val.row_id):
        raise ValueError("train/validation overlap")
    if read_json(directory / "selected_users.json") != sorted(int(value) for value in train.user_id.unique()):
        raise ValueError("selected-users JSON disagrees with the shared user dictionary")
    histories = Histories(train, meta["n_users"])
    sources = ["src/experiment.py", "src/model.py", "src/training.py", "src/data.py", "src/check_data.py", "src/metrics.py", "src/prepare_movielens1m.py", "src/plotting.py",
               "results/movielens1m-data/reproducibility.json"]
    protocol = {
        "config": cfg, "data_sha256": meta["artifact_sha256"], "split_id": meta["split"]["id"],
        "histories_checksum": histories.checksum,
        "reference_split_id": reference["split_id"], "all_shared_csvs_byte_identical": True,
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
        "environment": environment(device), "mean": float(train.rating.mean()),
        "objective": "MSE + lambda/N * (||bu||^2+||bi||^2+||P||^2+||Q||^2+||Y||^2)",
        "optimizer": "Adam, weight_decay=0, no learning-rate scheduler, float32, unclipped predictions",
        "selection": "per-k minimum mean validation RMSE over tuning seeds; ties by lambda then lr",
        "tuning_strategy": strategy,
    }
    return dict(cfg=cfg, device=device, directory=directory, meta=meta, train=train, validation=val,
                histories=histories, protocol=protocol, output=ROOT / cfg["output_dir"],
                checkpoints=ROOT / cfg["checkpoint_dir"])



def ensure_protocol(ctx, training=False):
    path = ctx["output"] / "protocol.json"
    fingerprint = digest(ctx["protocol"])
    if path.exists():
        if read_json(path)["fingerprint"] != fingerprint:
            raise ValueError("code/config/data/environment changed; use a NEW output/checkpoint directory")
    else:
        ctx["output"].mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ctx["output"] / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in ctx["protocol"].get("source_sha256", {}):
                archive.write(ROOT / name, name)
            archive.writestr("experiment_config.json", json.dumps(ctx["protocol"].get("config", {}), indent=2))
        write_json(path, dict(fingerprint=fingerprint, protocol=ctx["protocol"], git=git_info()))
    if training and (ctx["output"] / "frozen.json").exists():
        raise ValueError("protocol frozen; training and tuning are closed for this study")
    return fingerprint



def trial_id(stage, k, lr, reg, seed):
    return f"{stage}_k{k}_s{seed}_" + digest([lr, reg])[:16]



def trial(ctx, stage, k, lr, reg, seed):
    fingerprint = ensure_protocol(ctx, training=True)
    identifier = trial_id(stage, k, lr, reg, seed)
    record_path = ctx["output"] / "records" / (identifier + ".json")
    signature = digest([fingerprint, stage, k, lr, reg, seed])
    if record_path.exists():
        record = read_json(record_path)
        if record["signature"] != signature:
            raise ValueError("trial protocol mismatch")
        for name, expected in record["files"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"completed artifact changed: {name}")
        print(f"reuse {identifier}", flush=True)
        return record
    resume_path = ctx["checkpoints"] / "active" / (identifier + ".pt")
    strategy = tuning_strategy(ctx["cfg"])
    options, screening = {}, None
    if strategy["mode"] == "two_stage" and stage in ("screen", "tune"):
        options["min_epochs"] = strategy["screening_epochs"]
        if stage == "screen":
            options["max_epochs"] = strategy["screening_epochs"]
        elif seed == ctx["cfg"]["tuning_seeds"][0]:
            screen_id = trial_id("screen", k, lr, reg, seed)
            screening = read_json(ctx["output"] / "records" / (screen_id + ".json"))
            expected = digest([fingerprint, "screen", k, lr, reg, seed])
            if screening["signature"] != expected:
                raise ValueError("screening checkpoint belongs to another protocol")
            for name, checksum in screening["files"].items():
                if sha256(ROOT / name) != checksum:
                    raise ValueError(f"screening artifact changed: {name}")
            options.update(initial_checkpoint=ROOT / screening["checkpoint"], initial_signature=expected)
    model, metrics, history = fit_one(ctx["train"], ctx["validation"], ctx["histories"],
                  ctx["meta"]["n_users"], ctx["meta"]["n_items"], ctx["cfg"], ctx["device"], k, lr, reg, seed,
                  resume_path=resume_path, signature=signature, **options)
    hist_path = ctx["output"] / "histories" / (identifier + ".csv")
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(hist_path, index=False, lineterminator="\n")
    files = {str(hist_path.relative_to(ROOT)): sha256(hist_path)}
    checkpoint = None
    if stage == "screen":
        # fit_one restored the best weights in memory; retain LAST weights, Adam,
        # RNG, history and stopping counters from the resumable checkpoint.
        checkpoint_path = ctx["checkpoints"] / "screen" / (identifier + ".pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        shutil.copyfile(resume_path, temporary)
        temporary.replace(checkpoint_path)
        checkpoint = str(checkpoint_path.relative_to(ROOT))
        files[checkpoint] = sha256(checkpoint_path)
    if screening is not None:
        metrics["continued_from_screening"] = True
        metrics["additional_fit_seconds"] = metrics["fit_seconds"] - screening["metrics"]["fit_seconds"]
    if stage == "final":
        checkpoint_path = ctx["checkpoints"] / "final" / (identifier + ".pt")
        save_torch(checkpoint_path, dict(signature=signature, fingerprint=fingerprint, state=cpu_state(model), metrics=metrics))
        checkpoint = str(checkpoint_path.relative_to(ROOT))
        files[checkpoint] = sha256(checkpoint_path)
        restored = SVDPlusPlus(ctx["meta"]["n_users"], ctx["meta"]["n_items"], k,
                    ctx["train"].rating.mean(), ctx["histories"], ctx["cfg"]["init_base_std"]).to(ctx["device"])
        restored.load_state_dict(torch.load(checkpoint_path, map_location=ctx["device"], weights_only=True)["state"])
        reload_rmse = score(ctx["validation"], predict(restored, frame_tensors(ctx["validation"], ctx["device"]), ctx["cfg"]["eval_batch_size"]))
        if abs(reload_rmse - metrics["validation_rmse"]) > 2e-6:
            raise RuntimeError("checkpoint reload changed the RMSE")
    record = dict(id=identifier, stage=stage, signature=signature, metrics=metrics, checkpoint=checkpoint,
                  files=files, git=git_info())
    write_json(record_path, record)
    if stage == "final":
        plot_loss_history(hist_path, ctx["output"] / "final_curves" / (identifier + "_loss.png"),
                          metrics["best_epoch"])
    resume_path.unlink(missing_ok=True)
    return record



def require_complete_selection(ctx, k):
    path = ctx["output"] / "selected" / f"k{k}.json"
    selected = read_json(path)
    space = read_json(ctx["output"] / "search_space.json")[str(k)]
    if selected["space_digest"] != digest(space):
        raise ValueError(f"k={k}: expanded search unfinished; run tune again")
    strategy = tuning_strategy(ctx["cfg"])
    if selected.get("strategy", {"mode": "grid"}) != strategy:
        raise ValueError(f"k={k}: tuning strategy changed; run tune again")
    if strategy["mode"] == "two_stage":
        shortlist = read_json(ctx["output"] / "shortlists" / f"k{k}.json")
        if (shortlist["space_digest"] != digest(space) or shortlist["strategy"] != strategy
                or selected["shortlist_digest"] != digest(shortlist)):
            raise ValueError(f"k={k}: screening/confirmation unfinished; run tune again")
    return selected



def tune(ctx, ks, extra_lrs, extra_regs):
    ensure_protocol(ctx, training=True)
    path = ctx["output"] / "search_space.json"
    spaces = read_json(path) if path.exists() else {}
    cfg = ctx["cfg"]
    strategy = tuning_strategy(cfg)
    for k in ks:
        old = spaces.get(str(k), {"learning_rates": [], "regularizations": []})
        lrs = sorted(set(cfg["learning_rates"] + old["learning_rates"] + extra_lrs))
        regs = sorted(set(cfg["regularizations"] + old["regularizations"] + extra_regs))
        validate_search_values(lrs, regs)
        space = {"learning_rates": lrs, "regularizations": regs}
        spaces[str(k)] = space
        write_json(path, spaces)
        shortlist = None
        pairs = list(itertools.product(lrs, regs))
        if strategy["mode"] == "two_stage":
            screening_rows = []
            seed = cfg["tuning_seeds"][0]
            for lr, reg in pairs:
                metrics = trial(ctx, "screen", k, lr, reg, seed)["metrics"]
                screening_rows.append(dict(k=k, learning_rate=lr, regularization=reg, seed=seed,
                                           validation_rmse=metrics["validation_rmse"],
                                           best_epoch=metrics["best_epoch"], epochs_run=metrics["epochs_run"]))
                pd.DataFrame(screening_rows).to_csv(ctx["output"] / f"screening_k{k}.csv", index=False)
            candidates = shortlist_candidates(screening_rows, strategy["candidates_per_learning_rate"])
            shortlist = dict(k=k, space_digest=digest(space), strategy=strategy, screening_seed=seed,
                             candidates=candidates, test_used=False)
            write_json(ctx["output"] / "shortlists" / f"k{k}.json", shortlist)
            pairs = [(row["learning_rate"], row["regularization"]) for row in candidates]
            plot_tuning(ctx["output"], [k])
        rows = []
        for lr, reg in pairs:
            runs = [trial(ctx, "tune", k, lr, reg, seed)["metrics"] for seed in cfg["tuning_seeds"]]
            values = np.array([r["validation_rmse"] for r in runs])
            rows.append(dict(k=k, learning_rate=lr, regularization=reg, validation_rmse_mean=float(values.mean()),
                             validation_rmse_std=float(values.std(ddof=1)),
                             n_seeds=len(runs), hit_epoch_cap=any(r["hit_epoch_cap"] for r in runs)))
            pd.DataFrame(rows).to_csv(ctx["output"] / f"tuning_k{k}.csv", index=False)
        rows.sort(key=lambda row: (row["validation_rmse_mean"], row["regularization"], row["learning_rate"]))
        best = rows[0]
        diagnostics = []
        if best["learning_rate"] in (lrs[0], lrs[-1]):
            diagnostics.append("learning_rate_on_search_boundary")
        if best["regularization"] in (regs[0], regs[-1]):
            diagnostics.append("regularization_on_search_boundary")
        if best["hit_epoch_cap"]:
            diagnostics.append("selected_candidate_hit_epoch_cap")
        if shortlist is not None:
            chosen_pairs = set(pairs)
            if any(row["best_epoch"] > strategy["screening_epochs"] - 10
                   for row in screening_rows
                   if (row["learning_rate"], row["regularization"]) not in chosen_pairs):
                diagnostics.append("discarded_screening_candidate_best_near_budget; consider_longer_screening_or_more_candidates")
        selection = dict(**best, space_digest=digest(space), diagnostics=diagnostics,
                         test_used=False, selection_seeds=cfg["tuning_seeds"], strategy=strategy)
        if shortlist is not None:
            selection["shortlist_digest"] = digest(shortlist)
        write_json(ctx["output"] / "selected" / f"k{k}.json", selection)
        plot_tuning(ctx["output"], [k])
        print(f"SELECTED k={k}: lr={best['learning_rate']:g}, lambda={best['regularization']:g}, "
              f"validation={best['validation_rmse_mean']:.6f}, diagnostics={diagnostics}", flush=True)



def final_fit(ctx, ks):
    ensure_protocol(ctx, training=True)
    for k in ks:
        selected = require_complete_selection(ctx, k)
        for seed in ctx["cfg"]["final_seeds"]:
            trial(ctx, "final", k, selected["learning_rate"], selected["regularization"], seed)
        plot_final_loss_curves(ctx["output"])



def diagnose(ctx):
    ensure_protocol(ctx)
    rows = []
    for k in ctx["cfg"]["k_grid"]:
        selected = require_complete_selection(ctx, k)
        caps, missing = [], []
        for seed in ctx["cfg"]["final_seeds"]:
            identifier = trial_id("final", k, selected["learning_rate"], selected["regularization"], seed)
            path = ctx["output"] / "records" / (identifier + ".json")
            if not path.exists():
                missing.append(seed)
            elif read_json(path)["metrics"]["hit_epoch_cap"]:
                caps.append(seed)
        rows.append(dict(k=k, learning_rate=selected["learning_rate"], regularization=selected["regularization"],
                         validation_rmse_mean=selected["validation_rmse_mean"],
                         selection_diagnostics=selected["diagnostics"],
                         final_seeds_at_epoch_cap=caps, missing_final_seeds=missing))
    write_json(ctx["output"] / "diagnostics.json", rows)
    print(json.dumps(rows, indent=2), flush=True)



def freeze(ctx):
    fingerprint = ensure_protocol(ctx)
    path = ctx["output"] / "frozen.json"
    if path.exists():
        print("Already frozen; test may be continued.", flush=True)
        return
    selections, runs = {}, []
    for k in ctx["cfg"]["k_grid"]:
        selected = require_complete_selection(ctx, k)
        selections[str(k)] = selected
        for seed in ctx["cfg"]["final_seeds"]:
            identifier = trial_id("final", k, selected["learning_rate"], selected["regularization"], seed)
            record = read_json(ctx["output"] / "records" / (identifier + ".json"))
            if record["signature"] != digest([fingerprint, "final", k, selected["learning_rate"], selected["regularization"], seed]):
                raise ValueError("final trial signature mismatch")
            for name, expected in record["files"].items():
                if sha256(ROOT / name) != expected:
                    raise ValueError(f"final artifact changed: {name}")
            runs.append(record)
    means = {k: float(np.mean([r["metrics"]["validation_rmse"] for r in runs if r["metrics"]["k"] == k])) for k in ctx["cfg"]["k_grid"]}
    selected_k = min(means, key=lambda k: (means[k], k))
    frozen = dict(fingerprint=fingerprint, selections=selections, selected_k_by_validation=selected_k,
                  runs=runs, test_started=False,
                  policy="All predeclared k and final seeds; choose k by validation before test. No refit on train+validation.")
    write_json(path, frozen)
    write_tables(ctx, frozen)
    diagnostics = {str(k): {
        "selection": selections[str(k)]["diagnostics"],
        "final_seeds_at_epoch_cap": [r["metrics"]["seed"] for r in runs
                                     if r["metrics"]["k"] == k and r["metrics"]["hit_epoch_cap"]],
    } for k in ctx["cfg"]["k_grid"]}
    write_json(ctx["output"] / "diagnostics.json", diagnostics)
    if any(value["selection"] or value["final_seeds_at_epoch_cap"] for value in diagnostics.values()):
        print("Inspect diagnostics.json: search-boundary optima or epoch caps require review before test.", flush=True)
    print(f"FROZEN; selected k by validation={selected_k}; test not evaluated.", flush=True)



def write_tables(ctx, frozen):
    rows = []
    for run in frozen["runs"]:
        row = {**run["metrics"], "id": run["id"], "test_rmse": run.get("test_rmse"), "split_id": ctx["meta"]["split"]["id"]}
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(ctx["output"] / "metrics_by_run.csv", index=False)
    summary = []
    for k, group in frame.groupby("k", sort=True):
        row = dict(k=int(k), n_seeds=len(group), learning_rate=float(group.learning_rate.iloc[0]),
                   regularization=float(group.regularization.iloc[0]))
        for metric in ("train_rmse", "validation_rmse", "test_rmse", "fit_seconds"):
            values = group[metric].dropna().astype(float)
            row[metric + "_mean"] = float(values.mean()) if len(values) else None
            row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        row["test_n_seeds"] = int(group.test_rmse.notna().sum())
        row["any_epoch_cap"] = bool(group.hit_epoch_cap.any())
        summary.append(row)
    result = pd.DataFrame(summary)
    result.to_csv(ctx["output"] / "metrics_summary.csv", index=False)
    return result



def evaluate_test(ctx):
    fingerprint = ensure_protocol(ctx)
    path = ctx["output"] / "frozen.json"
    frozen = read_json(path)
    if frozen["fingerprint"] != fingerprint:
        raise ValueError("frozen fingerprint mismatch")
    # Mark access before parsing test targets, including on interrupted runs.
    frozen["test_started"] = True
    write_json(path, frozen)
    test = pd.read_csv(ctx["directory"] / "test.csv")
    if len(test) != ctx["cfg"]["expected_sizes"]["test"]:
        raise ValueError("unexpected test size")
    validate_frame(test, ctx["meta"]["n_users"], ctx["meta"]["n_items"], "test")
    if set(test.row_id) & (set(ctx["train"].row_id) | set(ctx["validation"].row_id)):
        raise ValueError("test overlaps train/validation")
    test_t = frame_tensors(test, ctx["device"])
    pred_dir = ctx["output"] / "test_predictions"
    pred_dir.mkdir(exist_ok=True)
    for run in frozen["runs"]:
        for name, expected in run["files"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"frozen artifact changed: {name}")
        if "test_rmse" in run:
            if sha256(ROOT / run["test_prediction_path"]) != run["test_prediction_sha256"]:
                raise ValueError("test predictions changed")
            continue
        metrics = run["metrics"]
        saved = torch.load(ROOT / run["checkpoint"], map_location=ctx["device"], weights_only=True)
        if saved["fingerprint"] != fingerprint or saved["signature"] != run["signature"]:
            raise ValueError("checkpoint identity mismatch")
        model = SVDPlusPlus(ctx["meta"]["n_users"], ctx["meta"]["n_items"], metrics["k"],
                  ctx["train"].rating.mean(), ctx["histories"], ctx["cfg"]["init_base_std"]).to(ctx["device"])
        model.load_state_dict(saved["state"])
        prediction = predict(model, test_t, ctx["cfg"]["eval_batch_size"])
        prediction_path = pred_dir / (run["id"] + ".csv")
        pd.DataFrame({"row_id": test.row_id, "rating": test.rating, "prediction": prediction}).to_csv(prediction_path, index=False)
        run.update(test_rmse=score(test, prediction), test_prediction_path=str(prediction_path.relative_to(ROOT)),
                   test_prediction_sha256=sha256(prediction_path))
        write_json(path, frozen)
        print(f"TEST {run['id']}: {run['test_rmse']:.6f}", flush=True)
    frozen["test_complete"] = True
    write_json(path, frozen)
    write_tables(ctx, frozen)



def plots(ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frozen = read_json(ctx["output"] / "frozen.json")
    table = write_tables(ctx, frozen)
    plot_rmse_vs_k(table, ctx["output"], frozen.get("test_complete", False))
    plot_final_loss_curves(ctx["output"], frozen["runs"])
    for run in frozen["runs"]:
        plot_loss_history(ctx["output"] / "histories" / (run["id"] + ".csv"),
                          ctx["output"] / "final_curves" / (run["id"] + "_loss.png"),
                          run["metrics"]["best_epoch"])
    plot_tuning(ctx["output"])
    n_rows = math.ceil(len(ctx["cfg"]["k_grid"]) / 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, n_rows * 3), squeeze=False)
    for ax, k in zip(axes.flat, ctx["cfg"]["k_grid"]):
        run = next(r for r in frozen["runs"] if r["metrics"]["k"] == k)
        history = pd.read_csv(ctx["output"] / "histories" / (run["id"] + ".csv"))
        ax.plot(history.epoch, history.train_rmse, label="Train")
        ax.plot(history.epoch, history.validation_rmse, label="Validation")
        ax.axvline(run["metrics"]["best_epoch"], color="gray", linestyle="--")
        ax.set(title=f"k={k}, seed={run['metrics']['seed']}", xlabel="Epoch", ylabel="RMSE")
        ax.grid(alpha=0.2)
        ax.legend()
    for ax in list(axes.flat)[len(ctx["cfg"]["k_grid"]):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(ctx["output"] / "learning_curves.png", dpi=160)
    plt.close(fig)



def audit(ctx):
    lengths = ctx["histories"].lengths
    report = dict(split_id=ctx["meta"]["split"]["id"], train=len(ctx["train"]), validation=len(ctx["validation"]),
                  reference_split_id=ctx["protocol"]["reference_split_id"], all_shared_csvs_byte_identical=True,
                  n_users=ctx["meta"]["n_users"], n_items=ctx["meta"]["n_items"],
                  test_targets_parsed=False, fixed_train_mean=ctx["protocol"]["mean"],
                  train_history_memberships=int(lengths.sum()), history_checksum=ctx["histories"].checksum,
                  history_min=int(lengths.min()), history_max=int(lengths.max()), environment=environment(ctx["device"]))
    print(json.dumps(report, indent=2), flush=True)



def benchmark(ctx, ks):
    rows = []
    for k in ks:
        _, metrics, history = fit_one(ctx["train"], ctx["validation"], ctx["histories"],
                     ctx["meta"]["n_users"], ctx["meta"]["n_items"], ctx["cfg"], ctx["device"],
                     k, 0.003, 1.0, 999, max_epochs=3)
        rows.append(dict(k=k, measured_seconds=metrics["fit_seconds"],
                         epoch_seconds=[r["epoch_seconds"] for r in history],
                         validation_rmse=metrics["validation_rmse"], quality_result=False))
    write_json(ctx["output"] / "benchmark.json", dict(device=environment(ctx["device"]), runs=rows, test_used=False))



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "benchmark", "tune", "diagnose", "fit", "freeze", "test", "plot", "plot-tuning"))
    parser.add_argument("--config", default="configs/svdpp-ml1m.json")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--k", type=int, nargs="+")
    parser.add_argument("--extra-lrs", type=float, nargs="+", default=[])
    parser.add_argument("--extra-regs", type=float, nargs="+", default=[])
    args = parser.parse_args()
    ctx = context(args.config, args.device)
    ks = args.k or ([2, 200] if args.stage == "benchmark" else ctx["cfg"]["k_grid"])
    if any(k not in ctx["cfg"]["k_grid"] for k in ks):
        raise ValueError("--k must be part of the predefined k grid")
    if args.stage == "audit":
        audit(ctx)
    elif args.stage == "benchmark":
        benchmark(ctx, ks)
    elif args.stage == "tune":
        tune(ctx, ks, args.extra_lrs, args.extra_regs)
    elif args.stage == "fit":
        final_fit(ctx, ks)
    elif args.stage == "diagnose":
        diagnose(ctx)
    elif args.stage == "freeze":
        freeze(ctx)
    elif args.stage == "test":
        evaluate_test(ctx)
    elif args.stage == "plot-tuning":
        ensure_protocol(ctx)
        paths = plot_tuning(ctx["output"], args.k)
        print(f"Saved {len(paths)} tuning plots to {ctx['output'] / 'tuning_plots'}", flush=True)
    else:
        ensure_protocol(ctx)
        plots(ctx)


if __name__ == "__main__":
    main()
