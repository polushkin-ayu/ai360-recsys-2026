"""Plot saved FM results without loading data, weights, or retraining models."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/fm-k-sweep-20261006")
    args = parser.parse_args()
    root = Path(args.results)
    aggregate = pd.read_csv(root / "aggregate.csv").sort_values("k")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(aggregate.k, aggregate.val_rmse_mean, yerr=aggregate.val_rmse_std,
                marker="o", capsize=4, label="FM(user,item): mean +/- sample SD, 3 seeds")
    ax.set(xlabel="k (latent dimensions)", ylabel="Validation RMSE", title="MovieLens subset: FM at fixed regularization")
    ax.set_xticks(aggregate.k)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "validation_rmse_by_k.png", dpi=180)
    plt.close(fig)
    summary = pd.read_csv(root / "summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for _, row in summary[summary.seed == summary.seed.min()].iterrows():
        history = pd.read_csv(root / row.history_file)
        axes[0].plot(history.epoch, history.train_loss, label=f"k={row.k}")
        axes[1].plot(history.epoch, history.val_rmse, label=f"k={row.k}")
    for ax, ylabel in zip(axes, ["Train loss (MSE + L2)", "Validation RMSE"]):
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle(f"Training curves, seed={int(summary.seed.min())}; each run stops independently")
    fig.tight_layout()
    fig.savefig(root / "training_curves_seed42.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
