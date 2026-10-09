"""Generate publication-ready plots and sync test artifacts for PureMF study."""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    output_dir = Path("../results/mf")
    summary_path = output_dir / "metrics_summary.csv"
    runs_path = output_dir / "metrics_by_run.csv"
    manifest_path = output_dir / "manifest.json"

    if not summary_path.exists() or not runs_path.exists():
        raise FileNotFoundError("Run sweep and final-evaluate first to generate metrics CSVs.")

    df_summary = pd.read_csv(summary_path).sort_values("k")
    df_runs = pd.read_csv(runs_path)

    # 1. Синхронизация папки tests/ (копии тестовых предсказаний)
    tests_dir = output_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    if predictions_dir.exists():
        for p_file in predictions_dir.glob("*_test.csv"):
            dest_file = tests_dir / p_file.name
            if not dest_file.exists():
                dest_file.write_text(p_file.read_text(encoding="utf-8"), encoding="utf-8")

    # Настройка графического стиля
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11

    k_vals = df_summary["k"].values

    # 2. Построение rmse vs k_validation (.pdf / .png)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    val_mean = df_summary["validation_rmse_mean"].values
    val_std = df_summary["validation_rmse_std"].values

    ax.plot(k_vals, val_mean, marker="o", linestyle="-", linewidth=2, color="#1f77b4", label="Validation RMSE (Mean)")
    if not np.isnan(val_std).all():
        ax.fill_between(k_vals, val_mean - val_std, val_mean + val_std, color="#1f77b4", alpha=0.2, label="±1 std")

    best_row = df_summary.loc[df_summary["validation_rmse_mean"].idxmin()]
    best_k = int(best_row["k"])
    best_val = best_row["validation_rmse_mean"]
    ax.scatter([best_k], [best_val], color="red", s=120, zorder=5, label=f"Best k={best_k} ({best_val:.4f})")

    ax.set_title("Pure Matrix Factorization: Validation RMSE vs Latent Factors (k)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Latent Factors (k)", fontsize=11)
    ax.set_ylabel("Validation RMSE", fontsize=11)
    ax.set_xticks(k_vals)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()

    plt.savefig(output_dir / "rmse vs k_validation.pdf")
    plt.savefig(output_dir / "rmse vs k_validation.png", dpi=300)
    plt.close()

    # 3. Построение rmse vs k_test (.pdf / .png)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    test_mean = df_summary["test_rmse_mean"].values
    test_std = df_summary["test_rmse_std"].values

    if not np.isnan(test_mean).all():
        ax.plot(k_vals, test_mean, marker="s", linestyle="--", linewidth=2, color="#2ca02c", label="Test RMSE (Mean)")
        if not np.isnan(test_std).all():
            ax.fill_between(k_vals, test_mean - test_std, test_mean + test_std, color="#2ca02c", alpha=0.2, label="±1 std")

        best_test_row = df_summary.loc[df_summary["test_rmse_mean"].idxmin()]
        best_test_k = int(best_test_row["k"])
        best_test_val = best_test_row["test_rmse_mean"]
        ax.scatter([best_test_k], [best_test_val], color="purple", s=120, zorder=5, label=f"Best Test k={best_test_k} ({best_test_val:.4f})")

    ax.set_title("Pure Matrix Factorization: Test RMSE vs Latent Factors (k)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Latent Factors (k)", fontsize=11)
    ax.set_ylabel("Test RMSE", fontsize=11)
    ax.set_xticks(k_vals)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    plt.tight_layout()

    plt.savefig(output_dir / "rmse vs k_test.pdf")
    plt.savefig(output_dir / "rmse vs k_test.png", dpi=300)
    plt.close()

    # 4. Построение training_curves (.pdf / .png)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    histories_dir = output_dir / "histories"

    run_ids_to_plot = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        curve_runs = manifest.get("training_curve_runs", {})
        run_ids_to_plot = list(curve_runs.values())

    if not run_ids_to_plot and histories_dir.exists():
        rep_k = [10, 50, 100, 200, 300]
        for r in df_runs.itertuples():
            if r.k in rep_k and getattr(r, "initialization_seed", None) == 11:
                run_ids_to_plot.append(r.run_id)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(run_ids_to_plot), 1)))

    for idx, run_id in enumerate(run_ids_to_plot):
        history_csv = histories_dir / f"{run_id}.csv"
        if history_csv.exists():
            hist_df = pd.read_csv(history_csv)
            k_val = df_runs.loc[df_runs["run_id"] == run_id, "k"].values
            k_str = f"k={k_val[0]}" if len(k_val) > 0 else run_id
            ax.plot(hist_df["epoch"], hist_df["val_rmse"], label=f"{k_str}", linewidth=1.8, color=colors[idx])

    ax.set_title("Pure Matrix Factorization: Validation RMSE Training Curves", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation RMSE", fontsize=11)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9, loc="upper right")
    plt.tight_layout()

    plt.savefig(output_dir / "training_curves.pdf")
    plt.savefig(output_dir / "training_curves.png", dpi=300)
    plt.close()

    print("Success! All PDF/PNG plots and tests/ structure generated in results/mf/")


if __name__ == "__main__":
    main()