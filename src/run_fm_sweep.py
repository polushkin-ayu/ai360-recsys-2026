"""Train user/item FM on a fixed split; publish metrics, keep weights local.

Run from the repository root: python -m src.run_fm_sweep --config configs/fm-k-sweep.json
Test ratings are never used for fitting, selection, or reported scores.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import ROOT, load_prepared, sha256
from src.metrics import rmse_by_row_id
from src.models import (
    FactorizationMachine, TrainingConfig, count_parameters,
    fit_model, predict_frame, save_checkpoint, load_checkpoint,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def row_digest(frame):
    # Canonical integer representation: independent of OS CSV line endings.
    return hashlib.sha256(frame.row_id.to_numpy(dtype="<i8").tobytes()).hexdigest()


def tensors(frame):
    return (
        torch.tensor(frame.user_idx.to_numpy(copy=True), dtype=torch.long),
        torch.tensor(frame.item_idx.to_numpy(copy=True), dtype=torch.long),
        torch.tensor(frame.rating.to_numpy(copy=True), dtype=torch.float32),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fm-k-sweep.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_dir"]
    local = ROOT / config["local_dir"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new experiment_id/output_dir; existing results must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    data = load_prepared(config["data_dir"])
    train_tensors, validation_tensors = tensors(data.train), tensors(data.validation)
    provenance = {
        "experiment_id": config["experiment_id"],
        "command": f"python -m src.run_fm_sweep --config {args.config}",
        "commit": git("rev-parse", "HEAD"),
        "working_tree_status_before_run": git("status", "--porcelain", "--untracked-files=normal"),
        "config_sha256": sha256(config_path),
        "source_sha256": {name: sha256(ROOT / name) for name in [
            "src/models.py", "src/data.py", "src/metrics.py", "src/run_fm_sweep.py"]},
        "config": config,
        "environment": {
            "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "torch": torch.__version__, "os": platform.platform(),
            "cpu": platform.processor(), "device": "cpu", "threads": torch.get_num_threads(),
            "deterministic_algorithms": True,
        },
        "data": {
            "split_id": data.metadata["split"]["id"],
            "source": data.metadata["source"], "subset": data.metadata["config"]["subset"],
            "split": data.metadata["config"]["split"],
            "n_users": data.n_users, "n_items": data.n_items,
            "n_train": len(data.train), "n_validation": len(data.validation),
            "n_test": len(data.test), "coverage": data.metadata["coverage"],
            "train_row_ids_sha256": row_digest(data.train),
            "validation_row_ids_sha256": row_digest(data.validation),
            "test_row_ids_sha256": row_digest(data.test),
        },
        "evaluation": {
            "metric": "RMSE via src.metrics.rmse_by_row_id", "clipping": False,
            "test_evaluated": False, "selection": "minimum mean validation RMSE across seeds",
            "timing": "fit_model internal wall clock, training and epoch evaluation; excludes initialization, IO and plotting",
            "scope": "user/item FM; fixed regularization, not a completed hyperparameter search or baseline comparison",
            "team_approval": "k grid and seeds proposed for this pilot; confirmation pending",
        },
    }
    write_json(output / "manifest.json", provenance)
    rows = []
    for k in config["k_values"]:
        for seed in config["seeds"]:
            run_id = f"fm_k{k}_seed{seed}"
            training = TrainingConfig(**config["training"], seed=seed)
            model = FactorizationMachine(data.n_users, data.n_items, n_factors=k, init_std=config["init_std"])
            result = fit_model(model, *train_tensors,
                val_user_idx=validation_tensors[0], val_item_idx=validation_tensors[1],
                val_rating=validation_tensors[2], config=training)
            history = pd.DataFrame(result.history)
            if not np.isfinite(history[["train_loss", "train_rmse", "val_rmse"]].to_numpy()).all():
                raise ValueError(f"Nonfinite history in {run_id}")
            history.epoch = history.epoch.astype(int)
            history.to_csv(output / f"{run_id}_history.csv", index=False)
            scores = {}
            for name, frame in [("train", data.train), ("validation", data.validation)]:
                prediction, ids = predict_frame(model, frame)
                scores[name] = rmse_by_row_id(frame.rating.to_numpy(), prediction, frame.row_id.to_numpy(), ids)
                pd.DataFrame({"row_id": ids, "prediction": prediction}).to_csv(local / f"{run_id}_{name}_predictions.csv", index=False)
            np.testing.assert_allclose(scores["validation"], history.val_rmse.min(), rtol=1e-6, atol=1e-7)
            checkpoint = local / f"{run_id}.pt"
            save_checkpoint(checkpoint, model, result)
            restored, restored_result = load_checkpoint(checkpoint)
            restored_prediction, restored_ids = predict_frame(restored, data.validation)
            original_prediction, original_ids = predict_frame(model, data.validation)
            np.testing.assert_array_equal(restored_ids, original_ids)
            np.testing.assert_array_equal(restored_prediction, original_prediction)
            if restored_result.best_epoch != result.best_epoch:
                raise ValueError("Checkpoint metadata differs")
            row = {
                "run_id": run_id, "model": "FM(user,item)", "k": k, "seed": seed,
                "split_id": provenance["data"]["split_id"],
                "validation_row_ids_sha256": provenance["data"]["validation_row_ids_sha256"],
                "train_rmse": scores["train"], "val_rmse": scores["validation"], "test_rmse": None,
                "best_epoch": result.best_epoch, "epochs_run": len(history),
                "fit_seconds": result.fit_time_seconds, "n_parameters": count_parameters(model),
                "reg_bias": training.reg_bias, "reg_factors": training.reg_factors,
                "learning_rate": training.learning_rate, "batch_size": training.batch_size,
                "optimizer": training.optimizer, "clipping": False, "device": "cpu",
                "code_commit": provenance["commit"], "history_file": f"{run_id}_history.csv",
                "checkpoint_local": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256(checkpoint),
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
            print(f"{run_id}: train={scores['train']:.6f}, validation={scores['validation']:.6f}, best_epoch={result.best_epoch}, seconds={result.fit_time_seconds:.2f}", flush=True)
    summary = pd.DataFrame(rows)
    aggregate = summary.groupby(["model", "k"], as_index=False).agg(
        val_rmse_mean=("val_rmse", "mean"), val_rmse_std=("val_rmse", "std"),
        train_rmse_mean=("train_rmse", "mean"), fit_seconds_mean=("fit_seconds", "mean"),
        n_seeds=("seed", "count"), n_parameters=("n_parameters", "first"))
    aggregate.to_csv(output / "aggregate.csv", index=False)
    selected = aggregate.sort_values(
        ["val_rmse_mean", "k"]
    ).iloc[0]

    write_json(output / "selection.json", {
        "selected_k": int(selected.k),
        "val_rmse_mean": float(selected.val_rmse_mean),

        "val_rmse_std": (
            None
            if pd.isna(selected.val_rmse_std)
            else float(selected.val_rmse_std)
        ),

        "test_evaluated": False,
        "criterion": "lowest mean validation RMSE across available seeds; tie broken by smaller k",
        "limitations": "Selection only within this grid at fixed regularization; subject to team review",
    })
    print(aggregate.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
