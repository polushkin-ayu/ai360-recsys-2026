import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.experiment import (Histories, SVDPlusPlus, UserBatches, cpu_state,
                            diagnose, digest, ensure_protocol, evaluate_test, final_fit,
                            fit_one, freeze, plots, predict, frame_tensors,
                            read_json, tune, validate_search_values, write_json)


class SVDppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def frame(self):
        return pd.DataFrame(dict(row_id=[0, 1, 2, 3, 4], user_idx=[0, 0, 1, 2, 2],
                                 item_idx=[0, 1, 1, 0, 2], rating=[4., 2., 3., 5., 1.]))

    def config(self):
        return dict(k_grid=[2, 3], learning_rates=[0.003, 0.01], regularizations=[0.1, 1.],
                    tuning_seeds=[11, 22], final_seeds=[101, 102], max_epochs=3, patience=2,
                    batch_users=2, eval_batch_size=2, init_base_std=0.1, checkpoint_every=1,
                    log_every=10, expected_sizes={"test": 2})

    def test_search_values_allow_zero_regularization(self):
        validate_search_values([0.003, 0.03], [0., 0.1, 1., 10.])
        for lr, reg in ((0., 0.1), (-0.003, 0.1), (0.003, -0.1),
                        (float("inf"), 0.), (0.003, float("inf")),
                        (float("nan"), 0.), (0.003, float("nan"))):
            with self.subTest(lr=lr, reg=reg), self.assertRaises(ValueError):
                validate_search_values([lr], [reg])

    def test_forward_matches_direct_formula_and_empty_bag(self):
        train = self.frame()
        histories = Histories(train, 4)
        model = SVDPlusPlus(4, 3, 5, 3., histories)
        with torch.no_grad():
            model.bu.weight.copy_(torch.arange(4).float()[:, None] / 10)
            model.bi.weight.copy_(torch.arange(3).float()[:, None] / 10)
        users = torch.tensor([0, 1, 2, 3, 0])
        items = torch.tensor([0, 2, 1, 0, 1])
        expected = []
        for user, item in zip(users, items):
            bag = sorted(set(train.loc[train.user_idx == int(user), "item_idx"]))
            implicit = model.y.weight[bag].sum(0) / np.sqrt(len(bag)) if bag else torch.zeros(5)
            expected.append(model.mu + model.bu.weight[user, 0] + model.bi.weight[item, 0]
                            + model.q.weight[item] @ (model.p.weight[user] + implicit))
        torch.testing.assert_close(model(users, items), torch.stack(expected))
        self.assertNotEqual(model.q.weight.data_ptr(), model.y.weight.data_ptr())
        self.assertFalse(model.mu.requires_grad)

    def test_histories_are_unique_and_do_not_depend_on_ratings(self):
        train = self.frame()
        changed = pd.concat([train.assign(rating=-100), train.iloc[:1]], ignore_index=True)
        first, second = Histories(train, 4), Histories(changed, 4)
        self.assertEqual(first.checksum, second.checksum)
        self.assertEqual(list(first.lengths), [2, 1, 2, 0])

    def test_implicit_gradient_matches_finite_difference(self):
        model = SVDPlusPlus(3, 3, 2, 3., Histories(self.frame(), 3)).double()
        u, i = torch.tensor([0, 2]), torch.tensor([2, 1])
        model(u, i).sum().backward()
        derivative = model.y.weight.grad[0, 0].item()
        self.assertNotEqual(derivative, 0)
        delta = 1e-6
        with torch.no_grad():
            model.y.weight[0, 0] += delta
            high = model(u, i).sum().item()
            model.y.weight[0, 0] -= 2 * delta
            low = model(u, i).sum().item()
        self.assertAlmostEqual(derivative, (high - low) / (2 * delta), places=7)

    def test_grouped_batch_objective_has_correct_full_gradient(self):
        train = self.frame()
        model = SVDPlusPlus(3, 3, 2, 3., Histories(train, 3))
        full = copy.deepcopy(model)
        t = frame_tensors(train, torch.device("cpu"))
        full_loss = (full(t[0], t[1]) - t[2]).square().mean() + 0.7 / len(train) * full.l2()
        full_loss.backward()
        batches = UserBatches(train, 3)
        user_groups = [np.array([0, 1]), np.array([2])]
        seen = []
        for group in user_groups:
            rows = batches.rows(group)
            seen.extend(rows.tolist())
            prediction = model(t[0][rows], t[1][rows])
            loss = (prediction - t[2][rows]).square().sum() * (2 / len(train)) + 0.7 / len(train) * model.l2()
            (loss / 2).backward()
        self.assertEqual(sorted(seen), list(range(len(train))))
        for first, second in zip(model.parameters(), full.parameters()):
            torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-5)

    def test_prediction_cache_matches_forward(self):
        train = self.frame()
        model = SVDPlusPlus(3, 3, 2, 3., Histories(train, 3))
        tensors = frame_tensors(train, torch.device("cpu"))
        with torch.no_grad():
            direct = model(tensors[0], tensors[1]).numpy()
        np.testing.assert_allclose(predict(model, tensors, 2), direct, atol=1e-6)

    def test_resume_matches_uninterrupted_training(self):
        train = self.frame()
        val = pd.DataFrame(dict(row_id=[5, 6], user_idx=[0, 1], item_idx=[2, 0], rating=[2., 4.]))
        cfg = self.config()
        cfg.update(max_epochs=6, patience=10)
        histories = Histories(train, 3)
        args = (train, val, histories, 3, 3, cfg, torch.device("cpu"), 2, .003, 1., 11)
        full, _, _ = fit_one(*args)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "resume.pt"
            fit_one(*args, resume_path=path, signature="test", max_epochs=2)
            resumed, metrics, _ = fit_one(*args, resume_path=path, signature="test")
            self.assertEqual(metrics["epochs_run"], 6)
            for name, value in cpu_state(full).items():
                torch.testing.assert_close(value, cpu_state(resumed)[name], atol=0, rtol=0)

    def test_pipeline_selection_freeze_test_and_lock(self):
        train = self.frame()
        val = pd.DataFrame(dict(row_id=[5, 6], user_idx=[0, 1], item_idx=[2, 0], rating=[2., 4.]))
        test = pd.DataFrame(dict(row_id=[7, 8], user_idx=[1, 2], item_idx=[2, 1], rating=[3., 2.]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            test.to_csv(root / "test.csv", index=False)
            cfg = self.config()
            cfg["k_grid"] = [1, 3]
            cfg["regularizations"] = [0., 1.]
            ctx = dict(cfg=cfg, device=torch.device("cpu"), directory=root, train=train, validation=val,
                       meta={"n_users": 3, "n_items": 3, "split": {"id": "synthetic-test"}},
                       histories=Histories(train, 3), protocol={"test_fixture": True},
                       output=root / "results", checkpoints=root / "weights")
            with patch("src.experiment.ROOT", root):
                tune(ctx, [1, 3], [], [])
                # Completed runs can be safely reused.
                tune(ctx, [1, 3], [], [])
                diagnose(ctx)
                final_fit(ctx, [1, 3])
                diagnose(ctx)
                freeze(ctx)
                self.assertFalse(read_json(root / "results/frozen.json")["test_started"])
                with self.assertRaises(ValueError):
                    tune(ctx, [1], [], [])
                evaluate_test(ctx)
                evaluate_test(ctx)
                plots(ctx)
                frozen = read_json(root / "results/frozen.json")
                self.assertTrue(frozen["test_complete"])
                self.assertEqual(len(frozen["runs"]), 4)
                summary = pd.read_csv(root / "results/metrics_summary.csv")
                self.assertEqual(summary.test_n_seeds.tolist(), [2, 2])
                self.assertTrue((root / "results/rmse_vs_k.png").is_file())

    def test_changed_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx = {"output": Path(folder), "protocol": {"a": 1}}
            ensure_protocol(ctx)
            ctx["protocol"]["a"] = 2
            with self.assertRaises(ValueError):
                ensure_protocol(ctx)


if __name__ == "__main__":
    unittest.main()
