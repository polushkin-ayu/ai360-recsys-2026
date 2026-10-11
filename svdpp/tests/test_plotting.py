"""Проверки графиков на искусственных метриках; обучение не запускается."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import plotting
from src.experiment import tune


class PlottingTests(unittest.TestCase):
    def history(self, path, epochs=(1, 2, 3)):
        frame = pd.DataFrame({
            "epoch": epochs,
            "train_rmse": np.linspace(1.1, .8, len(epochs)),
            "validation_rmse": np.linspace(1.2, .95, len(epochs)),
            "train_mse": np.linspace(1.21, .64, len(epochs)),
            "train_objective": np.linspace(1.4, .7, len(epochs)),
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        return frame

    def test_loss_uses_saved_objective_and_real_epoch_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            history = self.history(root / "history.csv", (1, 3, 7))
            save = plotting._save
            def inspect(figure, path, **kwargs):
                lines = figure.axes[0].lines
                np.testing.assert_array_equal(lines[0].get_xdata(), history.epoch)
                np.testing.assert_allclose(lines[0].get_ydata(), history.train_objective)
                np.testing.assert_allclose(lines[1].get_ydata(), history.train_mse)
                self.assertEqual(list(lines[2].get_xdata()), [3, 3])
                return save(figure, path, **kwargs)
            with patch("src.plotting._save", side_effect=inspect):
                plotting.plot_loss_history(root / "history.csv", root / "loss.png", 3)
            self.assertGreater((root / "loss.png").stat().st_size, 1000)

    def test_rmse_vs_k_has_validation_and_test_only_after_evaluation(self):
        table = pd.DataFrame({
            "k": [3, 1], "validation_rmse_mean": [.94, .99],
            "validation_rmse_std": [.01, .02], "test_rmse_mean": [.95, 1.0],
            "test_rmse_std": [.02, .03], "train_rmse_mean": [.1, .2],
        })
        save = plotting._save
        with tempfile.TemporaryDirectory() as folder:
            for complete in (False, True):
                def inspect(figure, path, **kwargs):
                    series = {container.get_label(): container.lines[0] for container in figure.axes[0].containers}
                    self.assertEqual(set(series), {"Validation", "Test"} if complete else {"Validation"})
                    np.testing.assert_array_equal(series["Validation"].get_xdata(), [1, 3])
                    np.testing.assert_allclose(np.asarray(series["Validation"].get_ydata(), dtype=float), [.99, .94])
                    if complete:
                        np.testing.assert_allclose(np.asarray(series["Test"].get_ydata(), dtype=float), [1., .95])
                    return save(figure, path, **kwargs)
                with patch("src.plotting._save", side_effect=inspect):
                    paths = plotting.plot_rmse_vs_k(table, Path(folder) / str(complete), complete)
                self.assertTrue(all(path.is_file() for path in paths))

    def test_partial_tuning_zero_lambda_and_final_seeds_with_different_lengths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "records").mkdir()
            rows = [
                dict(k=1, learning_rate=.003, regularization=0., validation_rmse_mean=.98, validation_rmse_std=.01),
                dict(k=1, learning_rate=.03, regularization=0., validation_rmse_mean=1.1, validation_rmse_std=.02),
                dict(k=1, learning_rate=.003, regularization=1., validation_rmse_mean=1.02, validation_rmse_std=.02),
            ]
            pd.DataFrame(rows).to_csv(root / "tuning_k1.csv", index=False)
            (root / "search_space.json").write_text(json.dumps({
                "1": {"learning_rates": [.003, .03], "regularizations": [0., .1, 1., 10.]}
            }), encoding="utf-8")
            for stage, seeds in (("tune", [11, 22]), ("final", [101, 102])):
                for index, seed in enumerate(seeds):
                    identifier = f"{stage}_{seed}"
                    self.history(root / "histories" / (identifier + ".csv"), (1, 2) if index == 0 else (1, 2, 3))
                    record = {"id": identifier, "stage": stage, "metrics": {
                        "k": 1, "seed": seed, "learning_rate": .003, "regularization": 0., "best_epoch": 2,
                    }}
                    (root / "records" / (identifier + ".json")).write_text(json.dumps(record), encoding="utf-8")
            save = plotting._save
            def inspect(figure, path, **kwargs):
                if Path(path).name == "heatmap_k1.png":
                    axis = figure.axes[0]
                    self.assertEqual([label.get_text() for label in axis.get_xticklabels()], ["0", "0.1", "1", "10"])
                    self.assertEqual(int(np.ma.getmaskarray(axis.images[0].get_array()).sum()), 5)
                if Path(path).name == "final_loss_curves.png":
                    self.assertEqual([len(line.get_xdata()) for line in figure.axes[0].lines], [2, 3])
                return save(figure, path, **kwargs)
            with patch("src.plotting._save", side_effect=inspect):
                paths = plotting.plot_tuning(root)
                final_path = plotting.plot_final_loss_curves(root)
            self.assertEqual({path.name for path in paths}, {
                "heatmap_k1.png", "learning_curves_k1.png", "best_validation_vs_k.png"})
            self.assertTrue(final_path.is_file())
            self.assertFalse((root / "frozen.json").exists())

    def test_tune_automatically_plots_completed_candidates_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ctx = {"output": root, "cfg": {
                "learning_rates": [.003, .03], "regularizations": [0., .1],
                "tuning_seeds": [11, 22],
            }}
            def metrics(ctx, stage, k, lr, reg, seed):
                return {"metrics": {"validation_rmse": 1. + lr + reg + seed / 10000, "hit_epoch_cap": False}}
            with patch("src.experiment.ensure_protocol"), patch("src.experiment.trial", side_effect=metrics), \
                    patch("src.experiment.fit_one", side_effect=AssertionError("Training must not run")):
                tune(ctx, [1], [], [])
            self.assertTrue((root / "tuning_plots/heatmap_k1.png").is_file())
            self.assertTrue((root / "tuning_plots/best_validation_vs_k.png").is_file())

    def test_screening_graph_separates_one_seed_and_marks_each_retained_lr(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [dict(k=2, learning_rate=lr, regularization=reg, seed=11,
                         validation_rmse=(.95 if lr == .003 else .85) + reg / 100,
                         best_epoch=80, epochs_run=100)
                    for lr in (.003, .03) for reg in (0., .1, 1., 10.)]
            pd.DataFrame(rows).to_csv(root / "screening_k2.csv", index=False)
            (root / "shortlists").mkdir()
            (root / "shortlists/k2.json").write_text(json.dumps({"candidates": [
                {"learning_rate": .003, "regularization": 0.},
                {"learning_rate": .03, "regularization": .1},
            ]}), encoding="utf-8")
            save = plotting._save
            def inspect(figure, path, **kwargs):
                axis = figure.axes[0]
                self.assertEqual(axis.images[0].get_array().shape, (2, 4))
                self.assertEqual(int(np.ma.getmaskarray(axis.images[0].get_array()).sum()), 0)
                self.assertEqual(len(axis.patches), 2)
                self.assertIn("seed=11, 100 epochs", axis.get_title())
                self.assertNotIn("SD", axis.get_title())
                return save(figure, path, **kwargs)
            with patch("src.plotting._save", side_effect=inspect):
                paths = plotting.plot_tuning(root)
            self.assertEqual([path.name for path in paths], ["screening_heatmap_k2.png"])
            self.assertTrue(paths[0].is_file())
            self.assertFalse((root / "frozen.json").exists())

    def test_single_seed_figures_do_not_invent_standard_deviation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            table = pd.DataFrame(dict(k=[1, 2], validation_rmse_mean=[.98, .95],
                                     validation_rmse_std=[None, None], test_rmse_mean=[.99, .96],
                                     test_rmse_std=[None, None]))
            paths = plotting.plot_rmse_vs_k(table, root, test_complete=True)
            self.assertTrue(all(path.is_file() for path in paths))
            pd.DataFrame(dict(k=[1, 1], learning_rate=[.003, .03], regularization=[0., 0.],
                              validation_rmse_mean=[.98, .95], validation_rmse_std=[None, None],
                              n_seeds=[1, 1])).to_csv(root / "tuning_k1.csv", index=False)
            paths = plotting.plot_tuning(root)
            self.assertEqual({path.name for path in paths}, {"heatmap_k1.png", "best_validation_vs_k.png"})
            self.assertTrue(all(path.is_file() for path in paths))

    def tearDown(self):
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
