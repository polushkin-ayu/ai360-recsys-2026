"""Search, scoring and continuation checks; no model training is performed."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.experiment import (require_complete_selection,
                            shortlist_candidates, trial, tune, tuning_strategy)
from src.training import FixedOrderScorer, fit_one, load_training_checkpoint, save_torch, score


class FastTuningTests(unittest.TestCase):
    def config(self):
        return dict(k_grid=[2], learning_rates=[.003, .03], regularizations=[0., .1, 1., 10.],
                    tuning_seeds=[11, 22, 33], final_seeds=[101, 102], max_epochs=12,
                    tuning_strategy=dict(mode="two_stage", screening_epochs=5,
                                         candidates_per_learning_rate=1))

    def context(self, root):
        return dict(cfg=self.config(), output=root / "results", checkpoints=root / "weights",
                    protocol={"fixture": "no-training"}, device=torch.device("cpu"),
                    train=None, validation=None, histories=None, meta={"n_users": 2, "n_items": 2})

    def test_fixed_order_scorer_checks_ids_once_and_retains_prediction_checks(self):
        frame = pd.DataFrame({"row_id": [9, 2, 7], "rating": [4., 2., 3.]})
        from src.training import require_same_row_ids
        with patch("src.training.require_same_row_ids", wraps=require_same_row_ids) as check:
            scorer = FixedOrderScorer(frame)
            self.assertEqual(check.call_count, 1)
            expected = score(frame, np.array([3., 2., 4.]))
            with patch("numpy.unique", side_effect=AssertionError("IDs must not be revalidated")):
                for _ in range(4):
                    self.assertEqual(scorer(np.array([3., 2., 4.])), expected)
                    self.assertEqual(scorer(np.array([3., 2., 4.]), np.array([9, 2, 7])), expected)
            self.assertEqual(check.call_count, 1)
        for prediction in (np.array([1., 2.]), np.array([1., np.nan, 3.]),
                           np.array([1., np.inf, 3.]), np.ones((3, 1))):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                scorer(prediction)
        for ids in ([2, 9, 7], [9, 2, 2], [9, 2, 7, 8], [9., 2., 7.]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                scorer(np.ones(3), ids)
        # Snapshots cannot silently follow a subsequently changed source frame.
        frame.loc[0, "rating"] = 100.
        frame.loc[0, "row_id"] = 100
        self.assertEqual(scorer(np.array([3., 2., 4.])), expected)
        for ids in ([1, 1, 2], [-1, 1, 2], [0., 1., 2.]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                FixedOrderScorer(pd.DataFrame({"row_id": ids, "rating": [1., 2., 3.]}))

    def test_strategy_validation_and_slow_lr_shortlist(self):
        self.assertEqual(tuning_strategy({}), {"mode": "grid"})
        for update in ({"screening_epochs": 0}, {"screening_epochs": 13},
                       {"screening_epochs": True}, {"candidates_per_learning_rate": 0},
                       {"mode": "invalid"}):
            cfg = self.config()
            cfg["tuning_strategy"].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                tuning_strategy(cfg)
        rows = [dict(learning_rate=lr, regularization=reg, validation_rmse=value)
                for lr, reg, value in [(.03, 0., .8), (.03, .1, .81),
                                       (.003, 0., .98), (.003, .1, .95), (.003, 1., .95)]]
        chosen = shortlist_candidates(rows, 1)
        self.assertEqual([(r["learning_rate"], r["regularization"]) for r in chosen],
                         [(.003, .1), (.03, 0.)])
        self.assertEqual(len(shortlist_candidates(rows, 2)), 4)

    def test_two_stage_search_covers_grid_confirms_both_lrs_and_invalidates_expansion(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx = self.context(Path(folder))
            ctx["output"].mkdir()
            calls = []
            def fake_trial(ctx, stage, k, lr, reg, seed):
                calls.append((stage, lr, reg, seed))
                # Slow LR is worse than every fast candidate at screening time.
                value = (1. if lr == .003 else .8) + abs(reg - .1) / 100 + seed / 10000
                return {"metrics": dict(validation_rmse=value, best_epoch=5, epochs_run=5,
                                        hit_epoch_cap=False)}
            with patch("src.experiment.ensure_protocol"), patch("src.experiment.trial", side_effect=fake_trial), \
                    patch("src.experiment.plot_tuning"), \
                    patch("src.experiment.fit_one", side_effect=AssertionError("Training must not run")):
                tune(ctx, [2], [], [])
                screen = [call for call in calls if call[0] == "screen"]
                confirm = [call for call in calls if call[0] == "tune"]
                self.assertEqual(len(screen), 8)
                self.assertEqual({call[3] for call in screen}, {11})
                self.assertEqual(len(confirm), 6)
                self.assertEqual({call[1] for call in confirm}, {.003, .03})
                self.assertEqual({call[2] for call in confirm}, {.1})
                self.assertEqual({call[3] for call in confirm}, {11, 22, 33})
                selection = require_complete_selection(ctx, 2)
                self.assertEqual(selection["n_seeds"], 3)
                self.assertFalse(selection["test_used"])
                self.assertEqual(len(pd.read_csv(ctx["output"] / "screening_k2.csv")), 8)
                self.assertEqual(len(pd.read_csv(ctx["output"] / "tuning_k2.csv")), 2)
                # Interrupted grid expansion must not expose an old selection.
                with patch("src.experiment.trial", side_effect=RuntimeError("interrupted")):
                    with self.assertRaises(RuntimeError):
                        tune(ctx, [2], [], [2.])
                with self.assertRaises(ValueError):
                    require_complete_selection(ctx, 2)
                calls.clear()
                tune(ctx, [2], [], [2.])
                self.assertEqual(sum(call[0] == "screen" for call in calls), 10)
                require_complete_selection(ctx, 2)

    def test_trial_continues_last_screening_state_preserves_resume_and_reuses_completed_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ctx = self.context(root)
            calls = []
            class Model:
                def state_dict(self):
                    return {"weight": torch.tensor([1.])}  # best state, not last state
            def fake_fit(*args, **kwargs):
                calls.append(kwargs)
                state = load_training_checkpoint(kwargs["resume_path"], kwargs["signature"], ctx["device"],
                                                 kwargs.get("initial_checkpoint"), kwargs.get("initial_signature"))
                if state is not None:
                    self.assertEqual(state["model"]["weight"].item(), 5.)
                    self.assertEqual(state["best_state"]["weight"].item(), 1.)
                    self.assertEqual(state["optimizer"]["step"], 5)
                    self.assertEqual(state["rng"], {"fixture": 11})
                    self.assertEqual(len(state["history"]), 5)
                history = [dict(epoch=i, train_rmse=1., validation_rmse=.9, train_mse=1.,
                                train_objective=1.1) for i in range(1, 6)]
                save_torch(kwargs["resume_path"], dict(signature=kwargs["signature"],
                           model={"weight": torch.tensor([5.])}, optimizer={"step": 5},
                           rng={"fixture": 11}, history=history, best_state=Model().state_dict(),
                           best_rmse=.9, best_epoch=2, stalled=3, elapsed=10.))
                metrics = dict(k=args[7], learning_rate=args[8], regularization=args[9], seed=args[10],
                               validation_rmse=.9, best_epoch=2, epochs_run=5, hit_epoch_cap=True,
                               fit_seconds=10. if state is None else 12.)
                return Model(), metrics, history
            with patch("src.experiment.ROOT", root), patch("src.experiment.fit_one", side_effect=fake_fit):
                screening = trial(ctx, "screen", 2, .003, .1, 11)
                self.assertEqual(calls[0]["min_epochs"], 5)
                self.assertEqual(calls[0]["max_epochs"], 5)
                self.assertTrue((root / screening["checkpoint"]).exists())
                confirmed = trial(ctx, "tune", 2, .003, .1, 11)
                self.assertEqual(calls[1]["min_epochs"], 5)
                self.assertNotIn("max_epochs", calls[1])
                self.assertEqual(confirmed["metrics"]["additional_fit_seconds"], 2.)
                trial(ctx, "tune", 2, .003, .1, 22)
                self.assertNotIn("initial_checkpoint", calls[2])
                with patch("src.experiment.fit_one", side_effect=AssertionError("Completed fit repeated")):
                    self.assertEqual(trial(ctx, "screen", 2, .003, .1, 11)["id"], screening["id"])
                    self.assertEqual(trial(ctx, "tune", 2, .003, .1, 11)["id"], confirmed["id"])
                # Corrupted screening state must never be adopted by another run.
                checkpoint = root / screening["checkpoint"]
                checkpoint.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    trial(ctx, "screen", 2, .003, .1, 11)

    def test_epoch_floor_and_continuation_with_model_and_optimizer_replaced(self):
        # Exercise the actual epoch/resume loop with a parameter-free stand-in.
        # No SVD++ model or working optimizer is constructed or trained.
        train = pd.DataFrame(dict(row_id=[0, 1], user_idx=[0, 1], item_idx=[0, 1], rating=[3., 3.]))
        val = train.assign(row_id=[2, 3])
        epochs = []
        class Model:
            def __init__(self, *args):
                self.marker = 0
                self.init_std = .1
                self.mu = torch.tensor(3.)
            def to(self, device):
                return self
            def train(self):
                self.marker += 1
                epochs.append(self.marker)
            def parameters(self):
                return []
            def __call__(self, users, items):
                return torch.zeros(len(users), requires_grad=True)
            def l2(self):
                return torch.tensor(0.)
            def state_dict(self):
                return {"marker": torch.tensor(self.marker)}
            def load_state_dict(self, state):
                self.marker = int(state["marker"])
        class Optimizer:
            def __init__(self, *args, **kwargs):
                self.steps = 0
            def zero_grad(self, **kwargs):
                pass
            def step(self):
                self.steps += 1  # No parameter update.
            def state_dict(self):
                return {"steps": self.steps}
            def load_state_dict(self, state):
                self.steps = state["steps"]
        current = None
        def model_factory(*args):
            nonlocal current
            current = Model()
            return current
        def scorer_factory(frame):
            return lambda predictions: 1. if frame is train else (.9 if current.marker == 1 else 1.)
        cfg = dict(max_epochs=8, patience=2, batch_users=2, eval_batch_size=2,
                   init_base_std=.1, checkpoint_every=1, log_every=100)
        args = (train, val, None, 2, 2, cfg, torch.device("cpu"), 2, .003, 1., 11)
        with tempfile.TemporaryDirectory() as folder, \
                patch("src.training.SVDPlusPlus", side_effect=model_factory), \
                patch("src.training.torch.optim.Adam", Optimizer), \
                patch("src.training.FixedOrderScorer", side_effect=scorer_factory), \
                patch("src.training.predict", return_value=np.zeros(2)):
            screen_path = Path(folder) / "screen.pt"
            _, screened, _ = fit_one(*args, max_epochs=5, min_epochs=5,
                                     resume_path=screen_path, signature="screen")
            self.assertEqual(epochs, [1, 2, 3, 4, 5])
            self.assertEqual(screened["best_epoch"], 1)
            saved = torch.load(screen_path, weights_only=False)
            self.assertEqual(saved["model"]["marker"].item(), 5)
            self.assertEqual(saved["best_state"]["marker"].item(), 1)
            epochs.clear()
            _, confirmed, _ = fit_one(*args, min_epochs=5, signature="full",
                                      initial_checkpoint=screen_path, initial_signature="screen")
            self.assertEqual(epochs, [])  # Already stalled, but only after the floor.
            self.assertEqual(confirmed["epochs_run"], 5)
            epochs.clear()
            _, other_seed, _ = fit_one(*args, min_epochs=5)
            self.assertEqual(epochs, [1, 2, 3, 4, 5])
            self.assertEqual(other_seed["epochs_run"], 5)
            # A still-improving first seed must continue at epoch 6, with its
            # original history and optimizer counter, instead of starting over.
            epochs.clear()
            def improving_scorer(frame):
                return lambda prediction: 1. if frame is train else 1. - current.marker / 100
            screen_path.unlink()
            with patch("src.training.FixedOrderScorer", side_effect=improving_scorer):
                fit_one(*args, max_epochs=5, min_epochs=5,
                        resume_path=screen_path, signature="screen")
                epochs.clear()
                active = Path(folder) / "confirmation.pt"
                _, confirmed, history = fit_one(*args, min_epochs=5, signature="full",
                                                resume_path=active, initial_checkpoint=screen_path,
                                                initial_signature="screen")
                self.assertEqual(epochs, [6, 7, 8])
                self.assertEqual([row["epoch"] for row in history], list(range(1, 9)))
                self.assertEqual(confirmed["best_epoch"], 8)
                saved = torch.load(active, weights_only=False)
                self.assertEqual(saved["optimizer"]["steps"], 8)
                # Sparse metric histories must resume from the saved epoch,
                # not from their shorter row count.
                sparse_cfg = dict(cfg, eval_every=3)
                sparse_args = (*args[:5], sparse_cfg, *args[6:])
                sparse = Path(folder) / "sparse.pt"
                _, metrics, history = fit_one(*sparse_args, max_epochs=5, min_epochs=5,
                                              resume_path=sparse, signature="sparse")
                self.assertEqual([row["epoch"] for row in history], [1, 3, 5])
                self.assertEqual(metrics["epochs_run"], 5)
                epochs.clear()
                _, metrics, history = fit_one(*sparse_args, min_epochs=5, resume_path=sparse,
                                              signature="sparse")
                self.assertEqual(epochs, [6, 7, 8])
                self.assertEqual(metrics["epochs_run"], 8)
                self.assertEqual([row["epoch"] for row in history], [1, 3, 5, 6, 8])
                # A consumed allocation still yields one complete evaluated
                # epoch, with an explicit time_limited flag, never convergence.
                _, limited, history = fit_one(*sparse_args, min_epochs=5, deadline=-1.)
                self.assertEqual(limited["epochs_run"], 1)
                self.assertEqual([row["epoch"] for row in history], [1])
                self.assertTrue(limited["time_limited"])

    def test_active_checkpoint_has_priority_and_signatures_are_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            screen, active = root / "screen.pt", root / "active.pt"
            save_torch(screen, {"signature": "screen", "epoch": 5})
            save_torch(active, {"signature": "full", "epoch": 8})
            state = load_training_checkpoint(active, "full", "cpu", screen, "screen")
            self.assertEqual(state["epoch"], 8)
            with self.assertRaises(ValueError):
                load_training_checkpoint(active, "wrong", "cpu", screen, "screen")
            active.unlink()
            self.assertEqual(load_training_checkpoint(active, "full", "cpu", screen, "screen")["epoch"], 5)
            with self.assertRaises(ValueError):
                load_training_checkpoint(active, "full", "cpu", screen, "wrong")


if __name__ == "__main__":
    unittest.main()
