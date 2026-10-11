"""Resume time-limited fast-study trials without a wall-clock limit.

The original study stays immutable. Candidates, epoch caps and patience stay
fixed; only trials marked time_limited resume their model/Adam/RNG checkpoints.
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import pandas as pd
import torch

from src.data import ROOT, sha256
from src.experiment import (context, digest, ensure_protocol, evaluate_test, plots,
                            read_json, trial_id, write_json, write_tables)
from src.training import FixedOrderScorer, cpu_state, fit_one, save_torch


def verify_files(record):
    for name, expected in record["files"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Saved artifact changed: {name}")


def load_source(source):
    saved = read_json(source / "protocol.json")
    protocol = saved["protocol"]
    if digest(protocol) != saved["fingerprint"]:
        raise ValueError("Source protocol fingerprint mismatch")
    for name, expected in protocol["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Training code changed since the source run: {name}")
    records = {}
    for path in sorted((source / "records").glob("*.json")):
        record = read_json(path)
        if record["stage"] not in ("screen", "tune"):
            continue
        m = record["metrics"]
        expected = digest([saved["fingerprint"], record["stage"], m["k"],
                           m["learning_rate"], m["regularization"], m["seed"]])
        if record["signature"] != expected:
            raise ValueError(f"Source trial identity mismatch: {record['id']}")
        verify_files(record)
        records[record["id"]] = record
    cfg, seed = protocol["config"], protocol["fast_seed"]
    for k in cfg["k_grid"]:
        for lr, reg in itertools.product(cfg["learning_rates"], cfg["regularizations"]):
            if trial_id("screen", k, lr, reg, seed) not in records:
                raise ValueError(f"Incomplete source screening for k={k}")
        confirmed = [r for r in records.values() if r["stage"] == "tune" and r["metrics"]["k"] == k]
        count = cfg["tuning_strategy"]["candidates_per_learning_rate"]
        if any(sum(r["metrics"]["learning_rate"] == lr for r in confirmed) != count
               for lr in cfg["learning_rates"]):
            raise ValueError(f"Incomplete source confirmation for k={k}")
    return saved, records


def complete_trial(ctx, original, fingerprint, source_fingerprint, scorers):
    m, stage, identifier = original["metrics"], original["stage"], original["id"]
    signature = digest([fingerprint, stage, m["k"], m["learning_rate"], m["regularization"], m["seed"]])
    record_path = ctx["output"] / "records" / (identifier + ".json")
    if record_path.exists():
        record = read_json(record_path)
        if record["signature"] != signature or record["metrics"].get("time_limited", False):
            raise ValueError("Invalid completed continuation record")
        verify_files(record)
        return record
    ensure_protocol(ctx, training=True)
    verify_files(original)
    history_path = ctx["output"] / "histories" / (identifier + ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if m.get("time_limited", False):
        last = ctx["checkpoints"] / "active" / (identifier + ".pt")
        if not last.exists():
            state = torch.load(ROOT / original["last_checkpoint"], map_location="cpu", weights_only=False)
            if state["signature"] != original["signature"] or state["epoch"] != m["epochs_run"]:
                raise ValueError("Source last checkpoint mismatch")
            state["signature"] = signature
            save_torch(last, state)
        cap = (ctx["cfg"]["tuning_strategy"]["screening_epochs"]
               if stage == "screen" else ctx["cfg"]["max_epochs"])
        print(f"CONTINUE {identifier}: epoch {m['epochs_run']} -> at most {cap}; no time limit", flush=True)
        model, metrics, history = fit_one(
            ctx["train"], ctx["validation"], ctx["histories"],
            ctx["meta"]["n_users"], ctx["meta"]["n_items"], ctx["cfg"], ctx["device"],
            m["k"], m["learning_rate"], m["regularization"], m["seed"],
            resume_path=last, signature=signature, deadline=None, max_epochs=cap,
            min_epochs=ctx["cfg"]["tuning_strategy"]["screening_epochs"], scorers=scorers)
        if metrics["time_limited"] or metrics["epochs_run"] < m["epochs_run"]:
            raise RuntimeError("Continuation did not complete normally")
        checkpoint = ctx["checkpoints"] / "best" / (identifier + ".pt")
        save_torch(checkpoint, dict(fingerprint=fingerprint, signature=signature,
                                  state=cpu_state(model), metrics=metrics))
        pd.DataFrame(history).to_csv(history_path, index=False)
        files = {str(p.relative_to(ROOT)): sha256(p) for p in (last, checkpoint, history_path)}
        checkpoint_signature, checkpoint_fingerprint = signature, fingerprint
    else:
        # Reuse weights and optimizer files read-only; copy the small plot history.
        source_history = next(ROOT / name for name in original["files"] if name.endswith(".csv"))
        history_path.write_bytes(source_history.read_bytes())
        checkpoint = ROOT / original["checkpoint"]
        last = ROOT / original["last_checkpoint"]
        files = {name: value for name, value in original["files"].items() if not name.endswith(".csv")}
        files[str(history_path.relative_to(ROOT))] = sha256(history_path)
        metrics = dict(m)
        checkpoint_signature, checkpoint_fingerprint = original["signature"], source_fingerprint
    record = dict(id=identifier, stage=stage, signature=signature, metrics=metrics,
                  checkpoint=str(checkpoint.relative_to(ROOT)), last_checkpoint=str(last.relative_to(ROOT)),
                  checkpoint_signature=checkpoint_signature, checkpoint_fingerprint=checkpoint_fingerprint,
                  files=files, source_signature=original["signature"],
                  source_time_limited=m.get("time_limited", False), source_epochs=m["epochs_run"],
                  additional_epochs=metrics["epochs_run"] - m["epochs_run"])
    write_json(record_path, record)
    return record


def run(ctx, source, saved_source, originals):
    started = time.perf_counter()
    frozen_path = ctx["output"] / "frozen.json"
    fingerprint = ensure_protocol(ctx, training=not frozen_path.exists())
    cfg, seed = ctx["cfg"], saved_source["protocol"]["fast_seed"]
    limited = sum(r["metrics"].get("time_limited", False) for r in originals.values())
    write_json(ctx["output"] / "completion_status.json", dict(complete=False, source=str(source),
               source_time_limited_runs=limited, wall_time_limit=None))
    if not frozen_path.exists():
        ctx["histories"].enable_index_cache()
        scorers = FixedOrderScorer(ctx["train"]), FixedOrderScorer(ctx["validation"])
        selections, finals, spaces = {}, [], {}
        for k in cfg["k_grid"]:
            screened = {}
            for lr, reg in itertools.product(cfg["learning_rates"], cfg["regularizations"]):
                original = originals[trial_id("screen", k, lr, reg, seed)]
                screened[(lr, reg)] = complete_trial(ctx, original, fingerprint,
                                                    saved_source["fingerprint"], scorers)
            pd.DataFrame([r["metrics"] for r in screened.values()]).to_csv(
                ctx["output"] / f"screening_k{k}.csv", index=False)
            # Keep exactly the original candidates: this request completes trials,
            # rather than starting a new search after observing the source test.
            original_tunes = sorted((r for r in originals.values()
                                     if r["stage"] == "tune" and r["metrics"]["k"] == k),
                                    key=lambda r: (r["metrics"]["learning_rate"], r["metrics"]["regularization"]))
            candidates = [screened[(r["metrics"]["learning_rate"], r["metrics"]["regularization"])]["metrics"]
                          for r in original_tunes]
            write_json(ctx["output"] / "shortlists" / f"k{k}.json",
                       dict(candidates=candidates, policy="Candidates fixed in source study", test_used=False))
            confirmed = [complete_trial(ctx, r, fingerprint, saved_source["fingerprint"], scorers)
                         for r in original_tunes]
            pd.DataFrame([dict(k=k, learning_rate=r["metrics"]["learning_rate"],
                regularization=r["metrics"]["regularization"], validation_rmse_mean=r["metrics"]["validation_rmse"],
                validation_rmse_std=None, n_seeds=1) for r in confirmed]).to_csv(
                    ctx["output"] / f"tuning_k{k}.csv", index=False)
            best = min(confirmed, key=lambda r: (r["metrics"]["validation_rmse"],
                       r["metrics"]["regularization"], r["metrics"]["learning_rate"]))
            m = best["metrics"]
            selections[str(k)] = m
            final_id = trial_id("final", k, m["learning_rate"], m["regularization"], seed)
            signature = digest([fingerprint, "final", k, m["learning_rate"], m["regularization"], seed])
            saved = torch.load(ROOT / best["checkpoint"], map_location="cpu", weights_only=True)
            if saved["signature"] != best["checkpoint_signature"] or saved["fingerprint"] != best["checkpoint_fingerprint"]:
                raise ValueError("Selected model identity mismatch")
            checkpoint = ctx["checkpoints"] / "final" / (final_id + ".pt")
            save_torch(checkpoint, dict(fingerprint=fingerprint, signature=signature, state=saved["state"], metrics=m))
            history_path = ctx["output"] / "histories" / (final_id + ".csv")
            history_path.write_bytes((ctx["output"] / "histories" / (best["id"] + ".csv")).read_bytes())
            final = dict(id=final_id, stage="final", signature=signature, metrics=m,
                         checkpoint=str(checkpoint.relative_to(ROOT)),
                         files={str(p.relative_to(ROOT)): sha256(p) for p in (checkpoint, history_path)})
            write_json(ctx["output"] / "records" / (final_id + ".json"), final)
            finals.append(final)
            spaces[str(k)] = dict(learning_rates=cfg["learning_rates"], regularizations=cfg["regularizations"])
            write_json(ctx["output"] / "search_space.json", spaces)
        selected_k = min(cfg["k_grid"], key=lambda k: (selections[str(k)]["validation_rmse"], k))
        write_json(frozen_path, dict(fingerprint=fingerprint, selections=selections, runs=finals,
                   selected_k_by_validation=selected_k, test_started=False,
                   source_test_already_evaluated=read_json(source / "frozen.json").get("test_complete", False),
                   policy="Original candidates completed without time limits; choose by validation; no final refit"))
        write_tables(ctx, read_json(frozen_path))
    evaluate_test(ctx)
    plots(ctx)
    records = [read_json(p) for p in (ctx["output"] / "records").glob("*.json")]
    trials = [r for r in records if r["stage"] in ("screen", "tune")]
    remaining = sum(r["metrics"].get("time_limited", False) for r in trials)
    if remaining or len(trials) != len(originals):
        raise RuntimeError("Incomplete continuation")
    frozen = read_json(frozen_path)
    status = dict(complete=True, source=str(source), seed=seed, wall_time_limit=None,
                  source_time_limited_runs=limited, completed_time_limited_runs=limited,
                  remaining_time_limited_runs=remaining,
                  additional_epochs=sum(r["additional_epochs"] for r in trials),
                  invocation_seconds=time.perf_counter()-started,
                  selected_models_time_limited=sum(r["metrics"]["time_limited"] for r in frozen["runs"]),
                  selected_models_at_epoch_cap=sum(r["metrics"]["hit_epoch_cap"] for r in frozen["runs"]),
                  all_k_evaluated=True, converged=False,
                  quality_note="Original epoch caps/patience and one seed retained; no convergence claim")
    write_json(ctx["output"] / "completion_status.json", status)
    print(f"COMPLETED: {limited} time-limited trials; {status['additional_epochs']} additional epochs; "
          f"{status['invocation_seconds']/60:.1f} minutes; results: {ctx['output']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/svdpp-ml1m-fast/seed11")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output")
    parser.add_argument("--checkpoints")
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    saved, originals = load_source(source)
    seed = saved["protocol"]["fast_seed"]
    output = (ROOT / (args.output or f"results/svdpp-ml1m-fast-completed/seed{seed}")).resolve()
    checkpoints = (ROOT / (args.checkpoints or f"artifacts/svdpp-ml1m-fast-completed/seed{seed}")).resolve()
    original_checkpoints = (ROOT / saved["protocol"]["config"]["checkpoint_dir"]).resolve()
    if output.is_relative_to(source) or checkpoints.is_relative_to(original_checkpoints):
        parser.error("Continuation must use separate output/checkpoint directories")
    cfg = dict(saved["protocol"]["config"], output_dir=str(output.relative_to(ROOT)),
               checkpoint_dir=str(checkpoints.relative_to(ROOT)))
    config_path = output / "completion_config.json"
    write_json(config_path, cfg)
    ctx = context(config_path, args.device)
    original = saved["protocol"]
    for key in ("data_sha256", "split_id", "histories_checksum", "mean", "objective", "optimizer"):
        if ctx["protocol"][key] != original[key]:
            raise ValueError(f"Source and continuation differ: {key}")
    ctx["protocol"].update(source_fingerprint=saved["fingerprint"],
        source_protocol_sha256=sha256(source / "protocol.json"),
        source_record_manifest=digest({key: digest(value) for key, value in originals.items()}),
        fast_seed=seed, wall_budget_seconds=None,
        selection="Original candidates fixed; complete time-limited trials; validation-only selection",
        independent_final_repeats=False)
    for name in ("fast.py", "src/fast_experiment.py", "complete_fast.py"):
        ctx["protocol"]["source_sha256"][name] = sha256(ROOT / name)
    print(f"Continue {sum(r['metrics'].get('time_limited', False) for r in originals.values())} trials; "
          f"reuse {sum(not r['metrics'].get('time_limited', False) for r in originals.values())}; no time limit", flush=True)
    run(ctx, source, saved, originals)


if __name__ == "__main__":
    main()
