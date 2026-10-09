import copy
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models import (TrainingConfig, count_parameters,
                        fit_model, load_checkpoint, predict_frame, regularized_mse_loss, save_checkpoint)
from src.run_svdpp import load_study_checkpoint
from src.svdpp import SVDPlusPlus, build_train_histories, history_checksum

DIAGNOSTICS = {}


def direct(model, users, items):
    """Independent NumPy formula, without any model prediction methods."""
    p = model.user_factors.weight.detach().cpu().numpy()
    q = model.item_factors.weight.detach().cpu().numpy()
    y = model.implicit_factors.weight.detach().cpu().numpy()
    bu = model.user_bias.weight.detach().cpu().numpy()[:, 0]
    bi = model.item_bias.weight.detach().cpu().numpy()[:, 0]
    result = []
    for u, i in zip(users, items, strict=True):
        bag = model.histories[int(u)]
        h = y[bag].sum(axis=0) / np.sqrt(len(bag)) if bag else np.zeros(model.n_factors)
        result.append(model.global_bias.item() + bu[u] + bi[i] + np.dot(q[i], p[u] + h))
    return np.asarray(result)


class SVDPlusPlusTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(11)
        self.histories = [[0], [0, 1, 2, 3], []]
        self.model = SVDPlusPlus(3, 4, histories=self.histories, n_factors=2, global_mean=3).double()
        with torch.no_grad():
            self.model.user_factors.weight.copy_(torch.tensor([[.2, -.3], [1., .5], [-.2, .7]]))
            self.model.item_factors.weight.copy_(torch.tensor([[.3, -.1], [1., -.5], [.4, .8], [-.6, .2]]))
            self.model.implicit_factors.weight.copy_(torch.tensor([[.7, -.4], [.1, .9], [-.5, .3], [.6, -.2]]))
            self.model.user_bias.weight[:, 0].copy_(torch.tensor([.1, -.2, .3]))
            self.model.item_bias.weight[:, 0].copy_(torch.tensor([-.3, .2, -.1, .5]))
        self.users = torch.tensor([0, 1, 1, 2], dtype=torch.long)
        self.items = torch.tensor([0, 2, 3, 1], dtype=torch.long)
        self.target = torch.tensor([2., 4., 3., 5.], dtype=torch.float64)

    def test_independent_forward(self):
        actual = self.model(self.users, self.items).detach().numpy()
        expected = direct(self.model, self.users.numpy(), self.items.numpy())
        error = float(np.max(np.abs(actual - expected)))
        DIAGNOSTICS["forward_max_abs_error"] = error
        DIAGNOSTICS["forward_tolerance"] = 1e-12
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_sqrt_lengths_zero_item_and_empty_bags(self):
        model = SVDPlusPlus(3, 4, histories=self.histories, n_factors=1).double()
        with torch.no_grad():
            model.user_factors.weight.zero_()
            model.item_factors.weight.fill_(1.)
            model.implicit_factors.weight.copy_(torch.tensor([[2.], [1.], [3.], [4.]]))
        actual = model(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 0]))
        torch.testing.assert_close(actual, torch.tensor([2., 5., 0.], dtype=torch.float64), rtol=0, atol=0)

    def test_zero_implicit_factors_direct_formula(self):
        with torch.no_grad():
            self.model.implicit_factors.weight.zero_()
        expected = direct(self.model, self.users.numpy(), self.items.numpy())
        np.testing.assert_allclose(self.model(self.users, self.items).detach().numpy(), expected, rtol=0, atol=1e-12)

    def test_zero_factors_direct_linear_terms(self):
        with torch.no_grad():
            for name in ("user_factors", "item_factors", "implicit_factors"):
                getattr(self.model, name).weight.zero_()
        expected = direct(self.model, self.users.numpy(), self.items.numpy())
        np.testing.assert_allclose(self.model(self.users, self.items).detach().numpy(), expected, rtol=0, atol=0)

    def test_train_histories_membership_and_deduplication(self):
        train = pd.DataFrame(dict(user_idx=[1, 0, 1, 1], item_idx=[3, 0, 1, 3], rating=[1, 2, 3, 4]))
        held_out = pd.DataFrame(dict(user_idx=[0, 2], item_idx=[2, 3]))
        expected = [[0], [1, 3], []]
        self.assertEqual(build_train_histories(train, 3, 4), expected)
        held_out.item_idx = [1, 2]
        train.rating = [-100, 0, 100, 7]
        self.assertEqual(build_train_histories(train, 3, 4), expected)
        self.assertNotIn(2, expected[0])
        # A validation target is never inserted by predict.
        snapshot = history_checksum(self.model.histories)
        self.model(torch.tensor([0]), torch.tensor([2]))
        self.assertEqual(history_checksum(self.model.histories), snapshot)

    def test_finite_differences_full_regularization(self):
        def independent_loss():
            prediction = direct(self.model, self.users.numpy(), self.items.numpy())
            mse = np.mean((prediction - self.target.numpy()) ** 2)
            bias_norm = sum(float(getattr(self.model, name).weight.detach().square().sum()) for name in ("user_bias", "item_bias"))
            factor_norm = sum(float(getattr(self.model, name).weight.detach().square().sum()) for name in ("user_factors", "item_factors", "implicit_factors"))
            return mse + .07 * bias_norm + .11 * factor_norm
        regularized_mse_loss(self.model, self.model(self.users, self.items), self.target,
                             n_train=19, reg_bias=.07, reg_factors=.11).backward()
        coordinates = [("user_factors", (1, 0)), ("item_factors", (3, 1)),
                       ("implicit_factors", (0, 0)), ("implicit_factors", (2, 1)),
                       ("user_bias", (1, 0)), ("item_bias", (0, 0))]
        errors = []
        for name, index in coordinates:
            parameter = getattr(self.model, name).weight
            analytic = parameter.grad[index].item()
            old, eps = parameter[index].item(), 1e-6
            with torch.no_grad():
                parameter[index] = old + eps
                plus = independent_loss()
                parameter[index] = old - eps
                minus = independent_loss()
                parameter[index] = old
            errors.append(abs(analytic - (plus - minus) / (2 * eps)))
        DIAGNOSTICS["finite_difference_max_abs_error"] = max(errors)
        DIAGNOSTICS["finite_difference_epsilon"] = 1e-6
        DIAGNOSTICS["finite_difference_tolerance"] = 2e-8
        self.assertLess(max(errors), 2e-8)
        self.assertIsNone(self.model.global_bias.grad)
        self.assertNotIn("global_bias", dict(self.model.named_parameters()))

    def test_repeated_user_gradient_accumulates(self):
        batched = copy.deepcopy(self.model)
        individual = copy.deepcopy(self.model)
        batched(self.users, self.items).sum().backward()
        for u, i in zip(self.users, self.items, strict=True):
            individual(u[None], i[None]).sum().backward()
        errors = []
        for a, b in zip(batched.parameters(), individual.parameters(), strict=True):
            errors.append((a.grad - b.grad).abs().max().item())
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=1e-12)
        DIAGNOSTICS["repeated_user_gradient_max_abs_error"] = max(errors)

    def test_one_sgd_step_and_no_stale_cache(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.03)
        before_prediction = self.model(self.users, self.items).detach().clone()
        regularized_mse_loss(self.model, self.model(self.users, self.items), self.target,
                             n_train=19, reg_bias=.07, reg_factors=.11).backward()
        before = {name: (p.detach().clone(), p.grad.detach().clone()) for name, p in self.model.named_parameters()}
        optimizer.step()
        for name, p in self.model.named_parameters():
            torch.testing.assert_close(
                p,
                before[name][0] - .03 * before[name][1],
                rtol=0,
                atol=1e-15,
            )
        after = self.model(self.users, self.items).detach().numpy()
        self.assertFalse(np.array_equal(after, before_prediction.numpy()))
        np.testing.assert_allclose(after, direct(self.model, self.users.numpy(), self.items.numpy()), rtol=0, atol=1e-12)
        self.assertEqual(self.model.global_bias.item(), 3.)

    def test_representable_toy_learning(self):
        # Threshold and budget are set before fitting; this is a technical fixture.
        threshold, epochs = 1e-4, 600
        users = torch.tensor([0, 0, 0, 1, 1, 1])
        items = torch.tensor([0, 1, 2, 0, 1, 2])
        target = torch.tensor([2.1, 3.2, 3.7, 4.0, 2.4, 1.5])
        model = SVDPlusPlus(2, 3, histories=[[0, 1, 2], [0, 1, 2]], n_factors=3,
                            global_mean=target.mean().item())
        initial = (model(users, items) - target).square().mean().item()
        result = fit_model(model, users, items, target, config=TrainingConfig(
            epochs=epochs, batch_size=6, learning_rate=.03, reg_bias=0, reg_factors=0, patience=None, seed=17))
        final = (model(users, items) - target).square().mean().item()
        DIAGNOSTICS["toy_learning"] = dict(initial_mse=initial, final_mse=final, threshold=threshold, epochs=epochs, seed=17)
        self.assertLess(final, threshold)
        self.assertLess(final, initial)
        self.assertEqual(len(result.history), epochs)

    def test_prediction_order_checkpoint_identity_and_shapes(self):
        frame = pd.DataFrame(dict(row_id=[19, 2, 13, 7], user_idx=self.users.numpy(), item_idx=self.items.numpy()))
        expected, ids = predict_frame(self.model, frame, batch_size=2)
        np.testing.assert_array_equal(ids, frame.row_id)
        self.assertEqual(expected.shape, (4,))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.model.float())
            restored, _ = load_checkpoint(path)
            actual, _ = predict_frame(restored, frame)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
            self.assertEqual(restored.histories, self.histories)
            self.assertFalse(restored.global_bias.requires_grad)
            with self.assertRaisesRegex(ValueError, "identity"):
                restored.verify_data_identity({"split_id": "wrong"})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_study_checkpoint(path, {"split_id": "wrong"})

    def test_bad_indices_histories_and_inputs(self):
        for users, items in (([-1], [0]), ([3], [0]), ([0], [-1]), ([0], [4])):
            with self.assertRaises(ValueError):
                self.model(torch.tensor(users), torch.tensor(items))
        with self.assertRaises(TypeError):
            self.model(torch.tensor([0.]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            self.model(torch.tensor([[0]]), torch.tensor([[0]]))
        for histories in ([[0, 0], [], []], [[-1], [], []], [[1, 0], [], []], [[], []]):
            with self.assertRaises(ValueError):
                SVDPlusPlus(3, 4, histories=histories)
        with self.assertRaises(ValueError):
            build_train_histories(dict(user_idx=np.array([0]), item_idx=np.array([-1])), 3, 4)

    def test_seed_reset_parameter_count_and_fixed_mean(self):
        models = [SVDPlusPlus(3, 4, histories=self.histories, n_factors=2) for _ in range(2)]
        config = TrainingConfig(epochs=5, batch_size=2, seed=9)
        results = [fit_model(model, self.users, self.items, self.target.float(), config=config) for model in models]
        for key in models[0].state_dict():
            torch.testing.assert_close(models[0].state_dict()[key], models[1].state_dict()[key], rtol=0, atol=0)
        self.assertEqual(results[0].history, results[1].history)
        self.assertEqual(count_parameters(models[0]), 3 + 4 + 2 * (3 + 2 * 4))
        self.assertEqual(count_parameters(SVDPlusPlus(140, 1329, histories=[[] for _ in range(140)], n_factors=20)), 1469 + 2798 * 20)
        self.assertEqual(models[0].global_bias.item(), self.target.float().mean().item())
        self.assertFalse(models[0].global_bias.requires_grad)
        self.assertGreater(models[0].factor_l2().item(), 0)
        torch.manual_seed(18)
        models[0].reset_parameters(global_mean=3.4)
        first_y = models[0].implicit_factors.weight.detach().clone()
        torch.manual_seed(18)
        models[0].reset_parameters(global_mean=3.4)
        torch.testing.assert_close(first_y, models[0].implicit_factors.weight, rtol=0, atol=0)
        self.assertGreater(first_y.abs().sum().item(), 0)
        DIAGNOSTICS["same_seed_state_dict_max_error"] = 0.0

    def test_best_validation_restore_and_metrics(self):
        model = SVDPlusPlus(3, 4, histories=self.histories, n_factors=2)
        validation_target = torch.full((4,), 1.)
        result = fit_model(model, self.users, self.items, self.target.float(),
                           val_user_idx=self.users, val_item_idx=self.items, val_rating=validation_target,
                           config=TrainingConfig(epochs=70, patience=5, seed=12))
        best = min(result.history, key=lambda row: row["val_rmse"])
        self.assertEqual(result.best_epoch, int(best["epoch"]))
        restored_rmse = (model(self.users, self.items) - validation_target).square().mean().sqrt().item()
        self.assertAlmostEqual(restored_rmse, best["val_rmse"], places=7)
        self.assertLess(len(result.history), 70)
        for row in result.history:
            self.assertAlmostEqual(row["train_rmse"] ** 2, row["train_mse"], places=6)
            self.assertGreaterEqual(row["train_loss_with_regularization"], row["train_mse"])
        self.assertFalse(any("test" in name for name in inspect.signature(fit_model).parameters))
        self.assertFalse(any("test" in name for name in inspect.signature(build_train_histories).parameters))
        DIAGNOSTICS["best_restore_abs_error"] = abs(restored_rmse - best["val_rmse"])

    def test_buffers_follow_device_and_dtype(self):
        model = self.model.to(dtype=torch.float64, device="cpu")
        self.assertEqual(model.history_items.device, model.implicit_factors.weight.device)
        self.assertEqual(model.history_offsets.dtype, torch.long)
        self.assertEqual(model(self.users, self.items).dtype, torch.float64)
        if torch.cuda.is_available():
            cuda_model = copy.deepcopy(model).cuda()
            torch.testing.assert_close(cuda_model(self.users.cuda(), self.items.cuda()).cpu(), model(self.users, self.items), rtol=1e-8, atol=1e-8)
        DIAGNOSTICS["cuda_test_performed"] = torch.cuda.is_available()


if __name__ == "__main__":
    unittest.main()
