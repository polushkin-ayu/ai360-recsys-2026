"""Short single-seed study over the complete k/lr/lambda grid.

This is a separate, explicitly budget-limited protocol, not a convergence claim.
All train ratings are retained; test is opened only after every k is frozen.
"""
from __future__ import annotations
import argparse
import itertools
import math
from pathlib import Path
import time

import pandas as pd
import torch

from src.data import ROOT, sha256
from src.experiment import (context, digest, ensure_protocol, evaluate_test, plots,
                            read_json, shortlist_candidates, trial_id, write_json, write_tables)
from src.training import FixedOrderScorer, cpu_state, fit_one, save_torch


class HourBudget:
    """Redistribute unused training time; reserve time for test and figures."""
    def __init__(self, seconds, work, clock=time.perf_counter):
        self.clock = clock
        self.finish = clock() + seconds
        self.training_finish = self.finish - min(180., seconds * .1)
        self.remaining_work = float(work)

    def allocate(self, work):
        now = self.clock()
        if work <= 0:
            return now
        if now >= self.finish - 30.:
            raise TimeoutError("Study time budget exhausted; unfinished study is not frozen/tested")
        share = max(0., self.training_finish - now) * work / max(work, self.remaining_work)
        self.remaining_work = max(0., self.remaining_work - work)
        return now + share


def plan(cfg):
    pairs = len(cfg["learning_rates"]) * len(cfg["regularizations"])
    retained = len(cfg["learning_rates"]) * min(len(cfg["regularizations"]),
                         cfg["tuning_strategy"]["candidates_per_learning_rate"])
    screen = cfg["tuning_strategy"]["screening_epochs"]
    return dict(k_count=len(cfg["k_grid"]), screening_runs=len(cfg["k_grid"]) * pairs,
                confirmation_runs=len(cfg["k_grid"]) * retained,
                maximum_training_epochs=len(cfg["k_grid"]) *
                    (pairs * screen + retained * (cfg["max_epochs"] - screen)),
                screening_epochs=screen, confirmation_epochs=cfg["max_epochs"],
                batch_users=cfg["batch_users"], eval_every=cfg.get("eval_every", 1),
                training_ratings="all", seeds_per_model=1, independent_final_repeats=False)


def _trial(ctx, stage, k, lr, reg, seed, deadline, scorers, screening=None):
    fingerprint = ensure_protocol(ctx, training=True)
    identifier = trial_id(stage, k, lr, reg, seed)
    signature = digest([fingerprint, stage, k, lr, reg, seed])
    record_path = ctx["output"] / "records" / (identifier + ".json")
    if record_path.exists():
        record = read_json(record_path)
        if record["signature"] != signature:
            raise ValueError("fast trial protocol changed")
        for name, expected in record["files"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"completed fast artifact changed: {name}")
        return record
    screen_epochs = ctx["cfg"]["tuning_strategy"]["screening_epochs"]
    options = dict(deadline=deadline, scorers=scorers, min_epochs=screen_epochs)
    if stage == "screen":
        options["max_epochs"] = screen_epochs
    if screening is not None:
        if screening["signature"] != digest([fingerprint, "screen", k, lr, reg, seed]):
            raise ValueError("screening checkpoint signature mismatch")
        for name, expected in screening["files"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"screening artifact changed: {name}")
        options.update(initial_checkpoint=ROOT / screening["last_checkpoint"],
                       initial_signature=screening["signature"])
    last = ctx["checkpoints"] / "active" / (identifier + ".pt")
    model, metrics, history = fit_one(ctx["train"], ctx["validation"], ctx["histories"],
                    ctx["meta"]["n_users"], ctx["meta"]["n_items"], ctx["cfg"], ctx["device"],
                    k, lr, reg, seed, resume_path=last, signature=signature, **options)
    history_path = ctx["output"] / "histories" / (identifier + ".csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)
    best = ctx["checkpoints"] / "best" / (identifier + ".pt")
    save_torch(best, dict(fingerprint=fingerprint, signature=signature,
                         state=cpu_state(model), metrics=metrics))
    files = {str(path.relative_to(ROOT)): sha256(path) for path in (history_path, last, best)}
    record = dict(id=identifier, stage=stage, signature=signature, metrics=metrics,
                  checkpoint=str(best.relative_to(ROOT)), last_checkpoint=str(last.relative_to(ROOT)), files=files)
    write_json(record_path, record)
    return record


def run(ctx, seed, seconds, started=None):
    started = time.perf_counter() if started is None else started
    cfg = ctx["cfg"]
    if seed not in cfg["tuning_seeds"]:
        raise ValueError("seed must belong to the predefined tuning_seeds")
    ctx["histories"].enable_index_cache()
    ctx["protocol"] = dict(ctx["protocol"], fast_plan=plan(cfg), fast_seed=seed,
        wall_budget_seconds=seconds,
        selection="Single-seed bounded screening per LR, then validation selection; selected weights reused as final")
    for name in ("fast.py", "src/fast_experiment.py"):
        ctx["protocol"]["source_sha256"][name] = sha256(ROOT / name)
    frozen_path = ctx["output"] / "frozen.json"
    fingerprint = ensure_protocol(ctx, training=not frozen_path.exists())
    try:
        if not frozen_path.exists():
            scorers = FixedOrderScorer(ctx["train"]), FixedOrderScorer(ctx["validation"])
            pairs = list(itertools.product(cfg["learning_rates"], cfg["regularizations"]))
            screen_epochs = cfg["tuning_strategy"]["screening_epochs"]
            retained = len(cfg["learning_rates"]) * min(len(cfg["regularizations"]),
                               cfg["tuning_strategy"]["candidates_per_learning_rate"])
            work = sum((len(pairs) * screen_epochs + retained * (cfg["max_epochs"] - screen_epochs))
                       * (1 + k / 8) for k in cfg["k_grid"])
            remaining_seconds = max(31., seconds - (time.perf_counter() - started))
            budget = HourBudget(remaining_seconds, work)
            spaces, selections, finals = {}, {}, []
            for k in cfg["k_grid"]:
                spaces[str(k)] = dict(learning_rates=cfg["learning_rates"], regularizations=cfg["regularizations"])
                write_json(ctx["output"] / "search_space.json", spaces)
                screened, rows = {}, []
                for lr, reg in pairs:
                    deadline = budget.allocate(screen_epochs * (1 + k / 8))
                    record = _trial(ctx, "screen", k, lr, reg, seed, deadline, scorers)
                    screened[(lr, reg)] = record
                    rows.append(record["metrics"])
                pd.DataFrame(rows).to_csv(ctx["output"] / f"screening_k{k}.csv", index=False)
                candidates = shortlist_candidates(rows, cfg["tuning_strategy"]["candidates_per_learning_rate"])
                write_json(ctx["output"] / "shortlists" / f"k{k}.json",
                           dict(candidates=candidates, test_used=False))
                confirmed, summary = [], []
                for candidate in candidates:
                    lr, reg = candidate["learning_rate"], candidate["regularization"]
                    deadline = budget.allocate((cfg["max_epochs"] - screen_epochs) * (1 + k / 8))
                    record = _trial(ctx, "tune", k, lr, reg, seed, deadline, scorers, screened[(lr, reg)])
                    confirmed.append(record)
                    summary.append(dict(k=k, learning_rate=lr, regularization=reg,
                                        validation_rmse_mean=record["metrics"]["validation_rmse"],
                                        validation_rmse_std=None, n_seeds=1))
                pd.DataFrame(summary).to_csv(ctx["output"] / f"tuning_k{k}.csv", index=False)
                best = min(confirmed, key=lambda r: (r["metrics"]["validation_rmse"],
                          r["metrics"]["regularization"], r["metrics"]["learning_rate"]))
                selections[str(k)] = best["metrics"]
                # Reuse the selected model: no independent final refit in this mode.
                m = best["metrics"]
                final_id = trial_id("final", k, m["learning_rate"], m["regularization"], seed)
                signature = digest([fingerprint, "final", k, m["learning_rate"], m["regularization"], seed])
                saved = torch.load(ROOT / best["checkpoint"], map_location="cpu", weights_only=True)
                checkpoint = ctx["checkpoints"] / "final" / (final_id + ".pt")
                save_torch(checkpoint, dict(signature=signature, fingerprint=fingerprint,
                                           state=saved["state"], metrics=m))
                history_path = ctx["output"] / "histories" / (final_id + ".csv")
                history_path.write_bytes((ctx["output"] / "histories" / (best["id"] + ".csv")).read_bytes())
                files = {str(path.relative_to(ROOT)): sha256(path) for path in (checkpoint, history_path)}
                final = dict(id=final_id, stage="final", signature=signature, metrics=m,
                             checkpoint=str(checkpoint.relative_to(ROOT)), files=files)
                write_json(ctx["output"] / "records" / (final_id + ".json"), final)
                finals.append(final)
            selected_k = min(cfg["k_grid"], key=lambda k: (selections[str(k)]["validation_rmse"], k))
            frozen = dict(fingerprint=fingerprint, selections=selections, runs=finals,
                          selected_k_by_validation=selected_k, test_started=False,
                          policy="All k completed under a short single-seed budget; no independent final repeats or convergence guarantee")
            write_json(frozen_path, frozen)
            write_tables(ctx, frozen)
        # Decisions are frozen before any test targets are parsed.
        evaluate_test(ctx)
        plots(ctx)
        elapsed = time.perf_counter() - started
        frozen = read_json(frozen_path)
        write_json(ctx["output"] / "fast_status.json", dict(complete=True, all_k_evaluated=True, seed=seed,
                   invocation_seconds=elapsed, within_requested_budget=elapsed <= seconds,
                   within_one_hour=elapsed < 3600, plan=plan(cfg), converged=False,
                   selected_models_time_limited=sum(r["metrics"].get("time_limited", False) for r in frozen["runs"]),
                   quality_note="Limited epochs/time; convergence not established"))
        print(f"Single-seed study completed in {elapsed / 60:.1f} minutes; results: {ctx['output']}", flush=True)
    except TimeoutError:
        write_json(ctx["output"] / "fast_status.json", dict(complete=False, seed=seed,
                   invocation_seconds=time.perf_counter() - started,
                   test_used=read_json(frozen_path).get("test_started", False) if frozen_path.exists() else False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/svdpp-ml1m-fast.json")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--minutes", type=float, default=55.)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--plan", action="store_true", help="Show work budget without loading data or training")
    args = parser.parse_args()
    if not math.isfinite(args.minutes) or not 5 <= args.minutes < 60:
        parser.error("--minutes must be at least 5 and less than 60")
    if args.plan:
        import json
        print(json.dumps(plan(read_json(ROOT / args.config)), indent=2))
        return
    started = time.perf_counter()
    ctx = context(args.config, args.device)
    # Each of the three requested seeds gets an isolated, reusable study.
    ctx["output"] = ctx["output"] / f"seed{args.seed}"
    ctx["checkpoints"] = ctx["checkpoints"] / f"seed{args.seed}"
    ctx["cfg"] = dict(ctx["cfg"], output_dir=str(ctx["output"].relative_to(ROOT)),
                      checkpoint_dir=str(ctx["checkpoints"].relative_to(ROOT)))
    ctx["protocol"] = dict(ctx["protocol"], config=ctx["cfg"])
    try:
        run(ctx, args.seed, args.minutes * 60, started=started)
    except TimeoutError as error:
        parser.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
