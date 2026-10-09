"""Auditable pilot -> frozen sweep -> separate final test evaluation for PureMF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import zipfile

import matplotlib
import numpy as np
import pandas as pd
import torch

from src.data import ROOT, load_prepared, sha256
from src.metrics import require_same_row_ids, rmse_by_row_id
from src.models import (
    PureMatrixFactorization,
    TrainingConfig,
    count_parameters,
    fit_model,
    load_checkpoint,
    predict_frame,
    save_checkpoint,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    files = sorted(ROOT.glob("src/*.py")) + sorted(ROOT.glob("tests/*.py"))
    files += [ROOT / "configs/mf-study.json"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in files if p.exists()}


def git_provenance():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "code_commit": git("rev-parse", "HEAD"),
        "dirty_worktree": bool(git("status", "--porcelain")),
    }


def setup(config):
    if config.get("models") != ["mf"]:
        raise ValueError("this experiment only supports PureMatrixFactorization ('mf')")
    if config["clipping"] is not False or config["mu"] != "fixed_train_mean":
        raise ValueError("this study requires no clipping and fixed train mean")
    if config["dtype"] != "float32":
        raise ValueError("the measured study uses float32")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])


def environment(config):
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return dict(
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        pandas=pd.__version__,
        matplotlib=matplotlib.__version__,
        os=platform.platform(),
        cpu=cpu,
        device=config["device"],
        dtype=config["dtype"],
        threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )


def data_identity(data):
    return dict(
        split_id=data.metadata["split"]["id"],
        n_users=data.n_users,
        n_items=data.n_items,
        artifact_sha256=data.metadata["artifact_sha256"],
    )


def context(config_path):
    path = Path(config_path).resolve()
    config = read_json(path)
    setup(config)
    data = load_prepared(config["data_dir"], verify=True)
    identity = data_identity(data)
    fingerprint = dict(
        config_hash=sha256(path),
        config_json_source=path.read_text(encoding="utf-8"),
        source_sha256=source_hashes(),
        data_identity=identity,
        environment=environment(config),
    )
    return config, data, fingerprint


def tensors(frame):
    return (
        torch.as_tensor(frame.user_idx.to_numpy(), dtype=torch.long),
        torch.as_tensor(frame.item_idx.to_numpy(), dtype=torch.long),
        torch.as_tensor(frame.rating.to_numpy(), dtype=torch.float32),
    )


def create_model(name, k, data, config, identity):
    if name != "mf":
        raise ValueError("this experiment only trains PureMatrixFactorization ('mf')")
    mean = tensors(data.train)[2].mean().item()
    model = PureMatrixFactorization(
        n_users=data.n_users,
        n_items=data.n_items,
        n_factors=k,
        global_mean=mean,
        init_std=config["init_std"],
        data_identity=identity,
    )
    return model.to(config["device"])


def save_predictions(path, frame, prediction, row_ids):
    require_same_row_ids(frame.row_id.to_numpy(), row_ids)
    pd.DataFrame(dict(row_id=row_ids, rating=frame.rating.to_numpy(), prediction=prediction)).to_csv(
        path, index=False
    )


def verify_run_files(run, include_test=False):
    files = dict(run["files"])
    if include_test:
        files.update(run.get("test_files", {}))
    for path, expected in files.items():
        if not (ROOT / path).exists() or sha256(ROOT / path) != expected:
            raise ValueError(f"completed run artifact missing or changed: {path}")


def load_study_checkpoint(path, identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("data_identity") != identity or payload.get("fixed_mean") is not True:
        raise ValueError("checkpoint split/mappings/fixed-mean mismatch")
    model, result = load_checkpoint(path)
    if not isinstance(model, PureMatrixFactorization):
        raise ValueError("the study checkpoint must contain PureMatrixFactorization")
    model.verify_data_identity(identity)
    return model, result


def execute_run(run_id, name, k, seed, candidate, config, data, fingerprint, *, pilot=False):
    output = ROOT / config["output_dir"]
    signature = digest(
        dict(fingerprint=fingerprint, model=name, k=k, seed=seed, candidate=candidate, pilot=pilot)
    )
    record = output / ("pilot_runs" if pilot else "runs") / f"{run_id}.json"
    if record.exists():
        run = read_json(record)
        if run["signature"] != signature:
            raise ValueError(f"refusing to mix protocols for completed run {run_id}")
        verify_run_files(run)
        print(f"resume {run_id}", flush=True)
        return run

    training = {**config["training"], **candidate, "seed": seed}
    model = create_model(name, k, data, config, fingerprint["data_identity"])
    fixed_mean = model.global_bias.item()
    train_u, train_i, train_r = tensors(data.train)
    val_u, val_i, val_r = tensors(data.validation)
    provenance = git_provenance()

    result = fit_model(
        model,
        train_u,
        train_i,
        train_r,
        val_user_idx=val_u,
        val_item_idx=val_i,
        val_rating=val_r,
        config=TrainingConfig(**training),
    )

    if model.global_bias.item() != fixed_mean or model.global_bias.requires_grad:
        raise RuntimeError("global mean changed during training")

    train_pred, train_ids = predict_frame(model, data.train)
    val_pred, val_ids = predict_frame(model, data.validation)
    train_rmse = rmse_by_row_id(data.train.rating, train_pred, data.train.row_id, train_ids)
    validation_rmse = rmse_by_row_id(
        data.validation.rating, val_pred, data.validation.row_id, val_ids
    )

    best = result.history[result.best_epoch - 1]
    if abs(validation_rmse - best["val_rmse"]) > 2e-6:
        raise RuntimeError("best validation checkpoint was not restored")

    subfolder = "pilot" if pilot else "main"
    checkpoint = ROOT / config["checkpoint_dir"] / subfolder / f"{run_id}.pt"
    save_checkpoint(checkpoint, model, result)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload.update(data_identity=fingerprint["data_identity"], fixed_mean=True, signature=signature)
    torch.save(payload, checkpoint)

    reloaded, _ = load_study_checkpoint(checkpoint, fingerprint["data_identity"])
    restored_pred, restored_ids = predict_frame(reloaded, data.validation)
    require_same_row_ids(val_ids, restored_ids)
    reload_max_error = float(np.max(np.abs(restored_pred - val_pred)))
    if reload_max_error > 2e-6:
        raise RuntimeError("checkpoint predictions changed after reload")

    history_path = output / "histories" / f"{run_id}.csv"
    prediction_path = output / "predictions" / f"{run_id}_validation.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result.history).to_csv(history_path, index=False)
    save_predictions(prediction_path, data.validation, val_pred, val_ids)

    n_train = len(data.train)
    penalty = (
        training["reg_bias"] * model.bias_l2().item()
        + training["reg_factors"] * model.factor_l2().item()
    ) / n_train

    metrics = dict(
        run_id=run_id,
        model=name,
        k=k,
        initialization_seed=seed,
        split_id=fingerprint["data_identity"]["split_id"],
        fixed_global_mean=fixed_mean,
        config_hash=fingerprint["config_hash"],
        **provenance,
        best_epoch=result.best_epoch,
        epochs_run=len(result.history),
        train_mse=train_rmse**2,
        train_loss_with_regularization=train_rmse**2 + penalty,
        train_rmse=train_rmse,
        validation_rmse=validation_rmse,
        test_rmse=None,
        parameter_count=count_parameters(model),
        fit_seconds=result.fit_time_seconds,
        device=config["device"],
    )

    run = dict(
        signature=signature,
        metrics=metrics,
        training_config=training,
        checkpoint=str(checkpoint.relative_to(ROOT)),
        history_csv=str(history_path.relative_to(ROOT)),
        validation_predictions=str(prediction_path.relative_to(ROOT)),
        reload_max_error=reload_max_error,
        files={
            str(p.relative_to(ROOT)): sha256(p)
            for p in (checkpoint, history_path, prediction_path)
        },
    )
    atomic_json(record, run)
    print(
        f"{run_id}: val={validation_rmse:.6f}, best_epoch={result.best_epoch}, epochs={len(result.history)}, seconds={result.fit_time_seconds:.2f}",
        flush=True,
    )
    return run


def summarize(runs):
    table = pd.DataFrame([r["metrics"] for r in runs])
    rows = []
    for (name, k), group in table.groupby(["model", "k"], dropna=False):
        row = dict(model=name, k=None if pd.isna(k) else int(k), n_runs=len(group))
        for metric in ("train_rmse", "validation_rmse", "test_rmse", "fit_seconds"):
            values = group[metric].dropna().astype(float)
            row[metric + "_mean"] = float(values.mean()) if len(values) else None
            row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        rows.append(row)
    return table, pd.DataFrame(rows)


def write_tables(output, runs):
    table, summary = summarize(runs)
    table.to_csv(output / "metrics_by_run.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    return summary


def expected_runs(config):
    for k in config["k_grid"]:
        for seed in config["initialization_seeds"]:
            yield f"mf_k{k}_s{seed}", "mf", k, seed


def pilot(config, data, fingerprint):
    output = ROOT / config["output_dir"]
    if (output / "manifest.json").exists() and read_json(output / "manifest.json").get(
        "test_access_started_at"
    ):
        raise ValueError("test has already been viewed; pilot/model selection is closed")
    runs, chosen = [], {}
    for name in ("mf",):
        candidates = config["pilot_candidates"]
        attempts = []
        for i, candidate in enumerate(candidates):
            run_id = f"pilot_{name}_c{i}_s{config['pilot_seed']}"
            run = execute_run(
                run_id,
                name,
                config["pilot_k"],
                config["pilot_seed"],
                candidate,
                config,
                data,
                fingerprint,
                pilot=True,
            )
            runs.append(run)
            attempts.append((run["metrics"]["validation_rmse"], i, candidate))
        chosen[name] = min(attempts, key=lambda x: (x[0], x[1]))[2]

    report = dict(
        created_at=now(),
        fingerprint=fingerprint,
        selection="minimum pilot validation RMSE, ties by candidate order",
        chosen=chosen,
        runs=runs,
        test_targets_used=False,
    )
    path = output / "pilot_report.json"
    if path.exists() and read_json(path)["fingerprint"] != fingerprint:
        raise ValueError("pilot report belongs to a different protocol")
    atomic_json(path, report)

    snapshot = output / "code_snapshot.zip"
    with zipfile.ZipFile(snapshot, "w", zipfile.ZIP_DEFLATED) as archive:
        for p_str in fingerprint["source_sha256"]:
            archive.write(ROOT / p_str, p_str)
        archive.writestr("configs/mf-study.json", fingerprint["config_json_source"])
    print("Pilot chosen settings: " + json.dumps(chosen), flush=True)


def sweep(config, data, fingerprint):
    output = ROOT / config["output_dir"]
    report = read_json(output / "pilot_report.json")
    if report["fingerprint"] != fingerprint:
        raise ValueError("source/config/data/environment differs from completed pilot")
    frozen_path = output / "frozen_protocol.json"
    frozen = dict(
        fingerprint=fingerprint,
        config=config,
        chosen=report["chosen"],
        code_provenance=git_provenance(),
        expected_runs=[x[0] for x in expected_runs(config)],
        test_policy="evaluate entire predefined grid after validation selection; no test-based selection or refit",
    )
    if frozen_path.exists():
        stored = read_json(frozen_path)
        for key in ("fingerprint", "config", "chosen", "expected_runs"):
            if stored[key] != frozen[key]:
                raise ValueError("frozen protocol mismatch")
    else:
        frozen["frozen_at"] = now()
        atomic_json(frozen_path, frozen)

    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path).get("test_access_started_at"):
        print(
            "Sweep already frozen and test access started; no new selection or training",
            flush=True,
        )
        return

    runs = []
    for run_id, name, k, seed in expected_runs(config):
        run = execute_run(run_id, name, k, seed, report["chosen"][name], config, data, fingerprint)
        runs.append(run)
        write_tables(output, runs)
    summary = write_tables(output, runs)

    selected_k = {}
    representatives = {}
    for name in ("mf",):
        group = summary[summary.model == name].sort_values(["validation_rmse_mean", "k"])
        selected_k[name] = int(group.iloc[0].k)

    for name in ("mf",):
        candidates = [
            r
            for r in runs
            if r["metrics"]["model"] == name and r["metrics"]["k"] == selected_k[name]
        ]
        median = float(np.median([r["metrics"]["validation_rmse"] for r in candidates]))
        representatives[name] = min(
            candidates,
            key=lambda r: (
                abs(r["metrics"]["validation_rmse"] - median),
                r["metrics"]["initialization_seed"],
            ),
        )["metrics"]["run_id"]

    manifest = dict(
        created_at=now(),
        config=config,
        fingerprint=fingerprint,
        frozen_protocol_sha256=sha256(frozen_path),
        pilot_report_sha256=sha256(output / "pilot_report.json"),
        selected_k_by_mean_validation=selected_k,
        training_curve_runs=representatives,
        runs=runs,
        missing_runs=[],
        test_evaluated_at=None,
    )
    atomic_json(manifest_path, manifest)
    print("Validation selected k: " + json.dumps(selected_k), flush=True)


def final_evaluate(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    config = manifest["config"]
    setup(config)
    output = ROOT / config["output_dir"]

    frozen = read_json(output / "frozen_protocol.json")
    if sha256(output / "frozen_protocol.json") != manifest["frozen_protocol_sha256"]:
        raise ValueError("frozen protocol has changed")
    if sha256(output / "pilot_report.json") != manifest["pilot_report_sha256"]:
        raise ValueError("pilot report has changed after selection")

    expected = [x[0] for x in expected_runs(config)]
    if [r["metrics"]["run_id"] for r in manifest["runs"]] != expected or manifest["missing_runs"]:
        raise ValueError("full predefined sweep must finish before test")

    data = load_prepared(config["data_dir"], verify=True)
    identity = data_identity(data)

    if not manifest.get("test_access_started_at"):
        manifest["test_access_started_at"] = now()
        atomic_json(manifest_path, manifest)

    common_ids = data.test.row_id.to_numpy()
    for run in manifest["runs"]:
        if run["metrics"]["test_rmse"] is not None:
            continue
        model, _ = load_study_checkpoint(ROOT / run["checkpoint"], identity)
        model.to(config["device"])
        prediction, ids = predict_frame(model, data.test)
        require_same_row_ids(common_ids, ids)
        value = rmse_by_row_id(data.test.rating, prediction, common_ids, ids)
        path = output / "predictions" / f"{run['metrics']['run_id']}_test.csv"
        save_predictions(path, data.test, prediction, ids)
        run["metrics"]["test_rmse"] = value
        run["test_files"] = {str(path.relative_to(ROOT)): sha256(path)}
        atomic_json(output / "runs" / f"{run['metrics']['run_id']}.json", run)
        atomic_json(manifest_path, manifest)
        print(f"test {run['metrics']['run_id']}: {value:.6f}", flush=True)

    manifest["test_evaluated_at"] = manifest.get("test_evaluated_at") or now()
    atomic_json(manifest_path, manifest)
    write_tables(output, manifest["runs"])


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage", required=True, choices=("pilot", "sweep", "final-evaluate"))
    parser.add_argument("--config", default="configs/mf-study.json")
    parser.add_argument("--manifest", default="results/mf/manifest.json")
    args = parser.parse_args()

    if args.stage == "final-evaluate":
        final_evaluate(args.manifest)
    else:
        config, data, fingerprint = context(args.config)
        (pilot if args.stage == "pilot" else sweep)(config, data, fingerprint)


if __name__ == "__main__":
    main()