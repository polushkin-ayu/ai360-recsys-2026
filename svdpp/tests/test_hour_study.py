"""Hour-mode checks with mocked fitting; model math checks do not optimize weights."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src import fast_experiment as fast
from src.experiment import read_json
from src.model import Histories, SVDPlusPlus
from src.training import UserBatches, prepare_fixed_batches, save_torch


class HourStudyTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(row_id=[0, 1, 2, 3, 4], user_idx=[0, 0, 1, 2, 2],
                                 item_idx=[0, 1, 1, 0, 2], rating=[4., 2., 3., 5., 1.]))

    def test_plan_and_redistribution_leave_time_for_reports(self):
        cfg = read_json(Path(__file__).resolve().parents[1] / "configs/svdpp-ml1m-fast.json")
        result = fast.plan(cfg)
        self.assertEqual(result["k_count"], 18)
        self.assertEqual(result["screening_runs"], 144)
        self.assertEqual(result["confirmation_runs"], 36)
        self.assertEqual(result["maximum_training_epochs"], 1944)
        now = [0.]
        budget = fast.HourBudget(3300., 10., clock=lambda: now[0])
        first = budget.allocate(2.)
        self.assertEqual(first, 624.)
        now[0] = 100.  # Earlier work finished well before its allocation.
        second = budget.allocate(8.)
        self.assertEqual(second, 3120.)
        self.assertGreater(budget.finish - second, 0.)
        now[0] = 3290.
        with self.assertRaises(TimeoutError):
            budget.allocate(1.)

    def test_cached_indices_keep_profiles_live_and_prepared_batches_cover_every_row(self):
        frame = self.frame()
        histories = Histories(frame, 4)
        histories.enable_index_cache(2)
        first = histories.select(np.array([0, 1]), "cpu")
        second = histories.select(np.array([0, 1]), "cpu")
        self.assertIs(first, second)
        model = SVDPlusPlus(4, 3, 2, 3., histories)
        with torch.no_grad():
            before = model.profiles(torch.tensor([0, 1])).clone()
            model.y.weight.add_(1.)
            after = model.profiles(torch.tensor([0, 1]))
        self.assertFalse(torch.equal(before, after))
        batches = prepare_fixed_batches(UserBatches(frame, 4), 2, "cpu")
        seen = []
        with torch.no_grad():
            for rows, group, inverse in batches:
                seen.extend(rows.tolist())
                users = torch.tensor(frame.user_idx.to_numpy(copy=True))[rows]
                items = torch.tensor(frame.item_idx.to_numpy(copy=True))[rows]
                direct = model(users, items)
                prepared = model.from_profiles(users, items, model.profiles(group)[inverse])
                torch.testing.assert_close(direct, prepared)
        self.assertEqual(sorted(seen), list(range(len(frame))))
        histories.select(np.array([2]), "cpu")
        histories.select(np.array([3]), "cpu")
        self.assertLessEqual(len(histories._index_cache), 2)

    def test_entire_hour_workflow_freezes_all_k_before_test_without_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["fast.py", "src/fast_experiment.py"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture", encoding="utf-8")
            cfg = dict(k_grid=[1, 2], learning_rates=[.003, .03], regularizations=[0., 1.],
                       tuning_seeds=[11, 22, 33], final_seeds=[101, 102], max_epochs=6,
                       batch_users=2, eval_every=2,
                       tuning_strategy=dict(screening_epochs=2, candidates_per_learning_rate=1))
            ctx = dict(cfg=cfg, protocol={"source_sha256": {}}, output=root / "results",
                       checkpoints=root / "weights", histories=Histories(self.frame(), 4),
                       train=self.frame(), validation=self.frame(), device=torch.device("cpu"),
                       meta={"n_users": 4, "n_items": 3, "split": {"id": "fixture"}})
            calls = []
            class Model:
                def state_dict(self):
                    return {"weight": torch.tensor([1.])}
            def fake_fit(*args, **kwargs):
                calls.append((args[7], args[8], args[9], args[10], kwargs))
                if kwargs.get("initial_checkpoint"):
                    self.assertTrue(Path(kwargs["initial_checkpoint"]).exists())
                hist = [dict(epoch=1, train_rmse=.9, validation_rmse=1., train_mse=.81,
                             train_objective=.82)]
                metrics = dict(k=args[7], learning_rate=args[8], regularization=args[9], seed=args[10],
                               validation_rmse=1. + args[8] + args[9], train_rmse=.9,
                               best_epoch=1, epochs_run=1, hit_epoch_cap=False, fit_seconds=.01,
                               time_limited=True)
                save_torch(kwargs["resume_path"], dict(signature=kwargs["signature"], epoch=1))
                return Model(), metrics, hist
            def test_after_freeze(ctx):
                frozen = read_json(ctx["output"] / "frozen.json")
                self.assertEqual({r["metrics"]["k"] for r in frozen["runs"]}, {1, 2})
                self.assertEqual({r["metrics"]["seed"] for r in frozen["runs"]}, {11})
                self.assertFalse(frozen["test_started"])
                self.assertTrue(all((root / r["checkpoint"]).exists() for r in frozen["runs"]))
            with patch("src.fast_experiment.ROOT", root), patch("src.experiment.ROOT", root), \
                    patch("src.fast_experiment.fit_one", side_effect=fake_fit), \
                    patch("src.fast_experiment.evaluate_test", side_effect=test_after_freeze), \
                    patch("src.fast_experiment.plots"):
                fast.run(ctx, 11, 3300.)
            self.assertEqual(len(calls), 12)  # Eight screen + four confirmations, no final refits.
            self.assertTrue(all(call[3] == 11 for call in calls))
            self.assertEqual(sum("initial_checkpoint" in call[4] for call in calls), 4)
            self.assertTrue(read_json(ctx["output"] / "fast_status.json")["complete"])
            self.assertFalse(read_json(ctx["output"] / "fast_status.json")["converged"])

    def test_expired_study_is_not_frozen_or_tested(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cfg = dict(k_grid=[1], learning_rates=[.003], regularizations=[0.], tuning_seeds=[11],
                       max_epochs=6, batch_users=2, tuning_strategy=dict(screening_epochs=2,
                       candidates_per_learning_rate=1))
            ctx = dict(cfg=cfg, protocol={"source_sha256": {}}, output=root, histories=Histories(self.frame(), 4),
                       train=self.frame(), validation=self.frame())
            with patch("src.fast_experiment.sha256", return_value="fixture"), \
                    patch("src.fast_experiment.ensure_protocol", return_value="fixture"), \
                    patch("src.fast_experiment.HourBudget.allocate", side_effect=TimeoutError), \
                    patch("src.fast_experiment.fit_one", side_effect=AssertionError("Training must not run")), \
                    patch("src.fast_experiment.evaluate_test") as evaluate:
                with self.assertRaises(TimeoutError):
                    fast.run(ctx, 11, 3300.)
                evaluate.assert_not_called()
            self.assertFalse((root / "frozen.json").exists())
            self.assertFalse(read_json(root / "fast_status.json")["complete"])


if __name__ == "__main__":
    unittest.main()
