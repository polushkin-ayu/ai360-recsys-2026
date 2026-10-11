"""Графики сохранённых метрик: обучение, подбор и итоговая оценка."""
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def _save(figure, output_path, dpi=180):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if figure._suptitle is not None:
        height = figure.get_figheight()
        figure._suptitle.set_y(1 - .08 / height)
        figure.tight_layout(rect=(0, 0, 1, 1 - .55 / height))
    else:
        figure.tight_layout()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_history(history_path, output_path, best_epoch=None):
    history = pd.read_csv(history_path)
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(history.epoch, history.train_rmse, label="Train")
    axes.plot(history.epoch, history.validation_rmse, label="Validation")
    if best_epoch is not None:
        axes.axvline(best_epoch, color="gray", linestyle="--", label="Best validation epoch")
    axes.set(xlabel="Epoch", ylabel="RMSE", title="SVD++ / MovieLens 1M")
    axes.legend()
    axes.grid(alpha=0.25)
    return _save(figure, output_path)


def plot_loss_history(history_path, output_path, best_epoch=None):
    """Full train objective evaluated after each epoch, alongside unpenalized MSE."""
    history = pd.read_csv(history_path)
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(history.epoch, history.train_objective, label="Train loss: MSE + lambda/N * L2")
    axes.plot(history.epoch, history.train_mse, linestyle="--", label="Train MSE")
    if best_epoch is not None:
        axes.axvline(best_epoch, color="gray", linestyle=":", label="Best validation epoch")
    axes.set(xlabel="Epoch", ylabel="Train loss", title="SVD++ / train objective after each epoch")
    axes.legend()
    axes.grid(alpha=0.25)
    return _save(figure, output_path)


def _error_bars(values):
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return values if np.isfinite(values).any() else None


def plot_rmse_vs_k(table, output_dir, test_complete=False):
    """Final-seed mean +/- sample SD; test is shown only after evaluation."""
    output_dir = Path(output_dir)
    table = table.sort_values("k")
    figure, axes = plt.subplots(figsize=(10, 5))
    axes.errorbar(table.k, table.validation_rmse_mean, yerr=_error_bars(table.validation_rmse_std),
                  marker="o", capsize=3, label="Validation")
    if test_complete:
        axes.errorbar(table.k, table.test_rmse_mean, yerr=_error_bars(table.test_rmse_std),
                      marker="s", capsize=3, label="Test")
    axes.set(xlabel="Number of latent factors k", ylabel="RMSE (unclipped)",
             title="SVD++ / final runs after per-k hyperparameter selection")
    axes.grid(alpha=0.25)
    axes.legend(title="Mean over seeds; SD shown when available")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    paths = []
    for extension in ("png", "pdf"):
        path = output_dir / ("rmse_vs_k." + extension)
        figure.savefig(path, dpi=180)
        paths.append(path)
    axes.set_xscale("log")
    axes.set_xticks(table.k, labels=[str(int(k)) for k in table.k], rotation=45)
    axes.set_xlabel("Number of latent factors k (log scale)")
    paths.append(_save(figure, output_dir / "rmse_vs_k_log.png"))
    return paths


def _records(output_dir, stage):
    return [record for path in sorted((Path(output_dir) / "records").glob("*.json"))
            if (record := json.loads(path.read_text(encoding="utf-8")))["stage"] == stage]


def plot_final_loss_curves(output_dir, runs=None):
    """One panel per k, one loss curve per final seed; no averaging/padding epochs."""
    output_dir = Path(output_dir)
    runs = _records(output_dir, "final") if runs is None else runs
    if not runs:
        return None
    ks = sorted({run["metrics"]["k"] for run in runs})
    n_columns = min(3, len(ks))
    n_rows = math.ceil(len(ks) / n_columns)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4.8, n_rows * 3.4), squeeze=False)
    for axis, k in zip(axes.flat, ks):
        for run in sorted((r for r in runs if r["metrics"]["k"] == k), key=lambda r: r["metrics"]["seed"]):
            history = pd.read_csv(output_dir / "histories" / (run["id"] + ".csv"))
            line, = axis.plot(history.epoch, history.train_objective, label=f"seed={run['metrics']['seed']}")
            best = history.loc[history.epoch == run["metrics"]["best_epoch"]]
            axis.scatter(best.epoch, best.train_objective, color=line.get_color(), s=22, zorder=3)
        axis.set(title=f"k={k}", xlabel="Epoch", ylabel="Train loss (MSE + L2)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in list(axes.flat)[len(ks):]:
        axis.set_visible(False)
    figure.suptitle("Final training / train loss per seed; dots = best validation epoch")
    return _save(figure, output_dir / "final_loss_curves.png", dpi=160)


def _plot_tuning_curves(output_dir, best, runs, path):
    selected = [run for run in runs
                if run["metrics"]["k"] == best["k"]
                and run["metrics"]["learning_rate"] == best["learning_rate"]
                and run["metrics"]["regularization"] == best["regularization"]]
    if not selected:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, run in enumerate(sorted(selected, key=lambda r: r["metrics"]["seed"])):
        history = pd.read_csv(output_dir / "histories" / (run["id"] + ".csv"))
        color = f"C{index % 10}"
        seed = run["metrics"]["seed"]
        axes[0].plot(history.epoch, history.validation_rmse, color=color, label=f"Validation, seed={seed}")
        axes[0].plot(history.epoch, history.train_rmse, color=color, linestyle="--", label=f"Train, seed={seed}")
        axes[1].plot(history.epoch, history.train_objective, color=color, label=f"Loss, seed={seed}")
        axes[1].plot(history.epoch, history.train_mse, color=color, linestyle="--", label=f"MSE, seed={seed}")
    axes[0].set(ylabel="RMSE")
    axes[1].set(ylabel="Train loss / MSE")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"Best completed tuning candidate / k={int(best['k'])}, "
                   f"lr={best['learning_rate']:g}, lambda={best['regularization']:g}")
    return _save(figure, path)


def _plot_screening(output_dir, spaces, ks):
    """Single-seed screening scores are kept separate from confirmed mean/SD."""
    paths = []
    for csv_path in sorted(output_dir.glob("screening_k*.csv")):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        k = int(frame.k.iloc[0])
        if ks is not None and k not in ks:
            continue
        space = spaces.get(str(k), {})
        lrs = sorted(space.get("learning_rates", frame.learning_rate.unique()))
        regs = sorted(space.get("regularizations", frame.regularization.unique()))
        matrix = frame.pivot(index="learning_rate", columns="regularization",
                             values="validation_rmse").reindex(index=lrs, columns=regs)
        figure, axes = plt.subplots(figsize=(max(7, len(regs) * 1.45), max(4, len(lrs) * 1.1 + 2)))
        cmap = plt.get_cmap("viridis_r").with_extremes(bad="#eeeeee")
        colors = axes.imshow(np.ma.masked_invalid(matrix.to_numpy()), aspect="auto", cmap=cmap)
        figure.colorbar(colors, ax=axes, label="Screening validation RMSE (one seed)")
        axes.set_xticks(range(len(regs)), labels=[f"{reg:g}" for reg in regs])
        axes.set_yticks(range(len(lrs)), labels=[f"{lr:g}" for lr in lrs])
        axes.set(xlabel="lambda (categorical, includes zero)", ylabel="Learning rate",
                 title=f"Screening k={k} / seed={int(frame.seed.iloc[0])}, "
                       f"{int(frame.epochs_run.max())} epochs; outlined = retained")
        low, high = colors.get_clim()
        for row in range(len(lrs)):
            for column in range(len(regs)):
                value = matrix.iloc[row, column]
                if pd.notna(value):
                    axes.text(column, row, f"{value:.4f}", ha="center", va="center",
                              color="white" if value > (low + high) / 2 else "black")
        shortlist_path = output_dir / "shortlists" / f"k{k}.json"
        if shortlist_path.exists():
            shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
            for candidate in shortlist["candidates"]:
                axes.add_patch(Rectangle((regs.index(candidate["regularization"]) - .5,
                                          lrs.index(candidate["learning_rate"]) - .5), 1, 1,
                                         fill=False, edgecolor="#e83e8c", linewidth=2.5))
        paths.append(_save(figure, output_dir / "tuning_plots" / f"screening_heatmap_k{k}.png"))
    return paths


def plot_tuning(output_dir, ks=None):
    """Plot completed candidates only; usable before freeze and without reading test."""
    output_dir = Path(output_dir)
    graph_dir = output_dir / "tuning_plots"
    graph_dir.mkdir(parents=True, exist_ok=True)
    spaces_path = output_dir / "search_space.json"
    spaces = json.loads(spaces_path.read_text(encoding="utf-8")) if spaces_path.exists() else {}
    runs = _records(output_dir, "tune")
    best_rows, paths = [], _plot_screening(output_dir, spaces, ks)
    for csv_path in sorted(output_dir.glob("tuning_k*.csv")):
        frame = pd.read_csv(csv_path)
        if frame.empty:
            continue
        k = int(frame.k.iloc[0])
        best = frame.sort_values(["validation_rmse_mean", "regularization", "learning_rate"]).iloc[0]
        best_rows.append(best)
        if ks is not None and k not in ks:
            continue
        space = spaces.get(str(k), {})
        lrs = sorted(space.get("learning_rates", frame.learning_rate.unique()))
        regs = sorted(space.get("regularizations", frame.regularization.unique()))
        matrix = frame.pivot(index="learning_rate", columns="regularization",
                             values="validation_rmse_mean").reindex(index=lrs, columns=regs)
        spread = frame.pivot(index="learning_rate", columns="regularization",
                             values="validation_rmse_std").reindex(index=lrs, columns=regs)
        figure, axes = plt.subplots(figsize=(max(7, len(regs) * 1.45), max(4, len(lrs) * 1.1 + 2)))
        cmap = plt.get_cmap("viridis_r").with_extremes(bad="#eeeeee")
        colors = axes.imshow(np.ma.masked_invalid(matrix.to_numpy()), aspect="auto", cmap=cmap)
        figure.colorbar(colors, ax=axes, label="Mean validation RMSE")
        axes.set_xticks(range(len(regs)), labels=[f"{reg:g}" for reg in regs])
        axes.set_yticks(range(len(lrs)), labels=[f"{lr:g}" for lr in lrs])
        axes.set(xlabel="lambda (categorical, includes zero)", ylabel="Learning rate",
                 title=f"Tuning k={k} / completed candidates; SD when multiple seeds")
        low, high = colors.get_clim()
        for row in range(len(lrs)):
            for column in range(len(regs)):
                mean, std = matrix.iloc[row, column], spread.iloc[row, column]
                if pd.notna(mean):
                    color = "white" if mean > (low + high) / 2 else "black"
                    axes.text(column, row, f"{mean:.4f}" + (f"\n+/- {std:.4f}" if pd.notna(std) else ""), ha="center", va="center",
                              fontsize=9, color=color)
        axes.add_patch(Rectangle((regs.index(best["regularization"]) - .5,
                                  lrs.index(best["learning_rate"]) - .5), 1, 1,
                                 fill=False, edgecolor="#e83e8c", linewidth=2.5))
        paths.append(_save(figure, graph_dir / f"heatmap_k{k}.png"))
        curve = _plot_tuning_curves(output_dir, best, runs, graph_dir / f"learning_curves_k{k}.png")
        if curve is not None:
            paths.append(curve)
    if best_rows:
        table = pd.DataFrame(best_rows).sort_values("k")
        figure, axes = plt.subplots(figsize=(10, 5))
        axes.errorbar(table.k, table.validation_rmse_mean, yerr=_error_bars(table.validation_rmse_std),
                      marker="o", capsize=3, label="Best completed candidate per k")
        axes.set(xlabel="Number of latent factors k", ylabel="Validation RMSE",
                 title="Hyperparameter tuning / validation mean; SD when multiple seeds")
        axes.grid(alpha=0.25)
        axes.legend()
        paths.append(_save(figure, graph_dir / "best_validation_vs_k.png"))
    return paths
