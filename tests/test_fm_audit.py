"""FM/data contract audit; no training loop, optimizer or loss is exercised.

The dense oracle is test-only. Production supports exactly two unit-valued
one-hot fields. One unsupported-domain test is explicitly skipped, and two
expected failures document unfixed numerical/input-validation findings.
"""

import unittest

import numpy as np
import pandas as pd
import torch

from src.data import ROOT, RAW_COLUMNS, build_mappings, encode, load_prepared
from src.models import FactorizationMachine, predict_frame


def naive_interaction(x, factors):
    result = x.new_zeros(x.shape[0])
    for i in range(factors.shape[0]):
        for j in range(i + 1, factors.shape[0]):
            result = result + torch.dot(factors[i], factors[j]) * x[:, i] * x[:, j]
    return result


def dense_parameters(model):
    linear = torch.cat((model.user_bias.weight[:, 0], model.item_bias.weight[:, 0]))
    factors = torch.cat((model.user_factors.weight, model.item_factors.weight))
    return linear, factors


def dense_inputs(model, users, items):
    x = model.global_bias.new_zeros((len(users), model.n_users + model.n_items))
    rows = torch.arange(len(users), device=users.device)
    x[rows, users] = 1
    x[rows, model.n_users + items] = 1
    return x


class FMMathematicsAuditTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(2026)

    def manual_model(self, fast=True):
        model = FactorizationMachine(1, 2, n_factors=2, fast_interaction=fast).double()
        with torch.no_grad():
            model.global_bias.fill_(0.5)
            model.user_bias.weight.copy_(torch.tensor([[0.1]], dtype=torch.float64))
            model.item_bias.weight.copy_(torch.tensor([[0.2], [-0.3]], dtype=torch.float64))
            model.user_factors.weight.copy_(torch.tensor([[1., 2.]], dtype=torch.float64))
            model.item_factors.weight.copy_(torch.tensor([[3., 4.], [-1., 2.]], dtype=torch.float64))
        return model

    def test_two_features_manual_interaction(self):
        for fast in (True, False):
            model = self.manual_model(fast)
            users, items = torch.tensor([0]), torch.tensor([0])
            # x=(1,1,0): dot([1,2],[3,4]) = 11, total = .5+.1+.2+11.
            torch.testing.assert_close(model.interaction_fast(users, items), torch.tensor([11.], dtype=torch.float64))
            torch.testing.assert_close(model(users, items), torch.tensor([11.8], dtype=torch.float64))

    def test_weighted_manual_example_via_parameter_scaling(self):
        model = self.manual_model()
        linear, factors = dense_parameters(model)
        x = torch.tensor([[1., 2., 0.]], dtype=torch.float64)
        expected = model.global_bias + x @ linear + naive_interaction(x, factors)
        torch.testing.assert_close(expected, torch.tensor([23.], dtype=torch.float64))
        # Only for this fixed x: absorb value 2 into the selected w and V row.
        # This does NOT add support for feature_values to production forward.
        with torch.no_grad():
            model.item_bias.weight[0].mul_(2)
            model.item_factors.weight[0].mul_(2)
        torch.testing.assert_close(model(torch.tensor([0]), torch.tensor([0])), expected)

    @unittest.skip("forward requires both user and item with value 1; a single active feature is outside its domain")
    def test_single_nonzero_feature_has_zero_interaction(self):
        pass

    def test_zero_item_factors_have_zero_interaction(self):
        # Tests diagonal cancellation in production, not support for missing fields.
        model = self.manual_model()
        with torch.no_grad():
            model.item_factors.weight.zero_()
        users, items = torch.tensor([0, 0]), torch.tensor([0, 1])
        for interaction in (model.interaction_fast, model.interaction_direct):
            torch.testing.assert_close(interaction(users, items), torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)

    def test_naive_equals_optimized(self):
        for seed in range(5):
            for dtype in (torch.float32, torch.float64):
                torch.manual_seed(seed)
                model = FactorizationMachine(3, 4, n_factors=5, init_std=0.3).to(dtype=dtype)
                with torch.no_grad():
                    model.global_bias.fill_(0.7)
                    model.user_bias.weight.normal_()
                    model.item_bias.weight.normal_()
                users = torch.randint(3, (11,))
                items = torch.randint(4, (11,))
                x = dense_inputs(model, users, items)
                linear, factors = dense_parameters(model)
                oracle = naive_interaction(x, factors)
                for fast in (True, False):
                    with self.subTest(seed=seed, dtype=dtype, fast=fast):
                        model.fast_interaction = fast
                        actual = model.interaction_fast(users, items) if fast else model.interaction_direct(users, items)
                        torch.testing.assert_close(actual, oracle, rtol=1e-5, atol=1e-7)
                        torch.testing.assert_close(model(users, items), model.global_bias + x @ linear + oracle, rtol=1e-5, atol=1e-7)

    def test_V_zero_equals_linear_model(self):
        for fast in (True, False):
            model = self.manual_model(fast)
            with torch.no_grad():
                model.user_factors.weight.zero_()
                model.item_factors.weight.zero_()
            torch.testing.assert_close(model(torch.tensor([0, 0]), torch.tensor([0, 1])), torch.tensor([0.8, 0.3], dtype=torch.float64))

    def test_zero_biases_leave_only_interaction(self):
        for fast in (True, False):
            model = self.manual_model(fast)
            with torch.no_grad():
                model.global_bias.zero_()
                model.user_bias.weight.zero_()
                model.item_bias.weight.zero_()
            torch.testing.assert_close(model(torch.tensor([0, 0]), torch.tensor([0, 1])), torch.tensor([11., 3.], dtype=torch.float64))

    def test_batch_shape(self):
        for fast in (True, False):
            model = self.manual_model(fast)
            for size in (0, 1, 7):
                users = torch.zeros(size, dtype=torch.long)
                items = torch.arange(size) % 2
                actual = model(users, items)
                self.assertEqual(actual.shape, (size,))
                if size:
                    separate = torch.cat([model(u.reshape(1), i.reshape(1)) for u, i in zip(users, items)])
                    torch.testing.assert_close(actual, separate)

    def test_model_parameters_registered(self):
        model = self.manual_model()
        parameters = dict(model.named_parameters())
        expected = {"global_bias", "user_bias.weight", "item_bias.weight", "user_factors.weight", "item_factors.weight"}
        self.assertEqual(set(parameters), expected)
        self.assertEqual(set(model.state_dict()), expected)
        self.assertEqual(sum(p.numel() for p in parameters.values()), 1 + 3 + 3 * 2)
        self.assertTrue(all(p.requires_grad for p in parameters.values()))
        self.assertTrue(all(p.dtype == torch.float64 for p in parameters.values()))
        model(torch.tensor([0]), torch.tensor([0])).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters.values()))

    def test_forward_gradients_match_naive_oracle(self):
        for fast in (True, False):
            model = self.manual_model(fast)
            users, items = torch.tensor([0, 0, 0]), torch.tensor([0, 1, 0])
            x = dense_inputs(model, users, items)
            linear, factors = dense_parameters(model)
            oracle = model.global_bias + x @ linear + naive_interaction(x, factors)
            parameters = tuple(model.parameters())
            actual_grad = torch.autograd.grad(model(users, items).sum(), parameters)
            oracle_grad = torch.autograd.grad(oracle.sum(), parameters)
            for actual, expected in zip(actual_grad, oracle_grad):
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_initialization_and_zero_factor_stationary_point(self):
        model = FactorizationMachine(2, 3, n_factors=4, global_mean=3.)
        self.assertEqual(model.global_bias.item(), 3.)
        self.assertEqual(torch.count_nonzero(model.user_bias.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(model.item_bias.weight).item(), 0)
        self.assertGreater(torch.count_nonzero(model.user_factors.weight).item(), 0)
        self.assertGreater(torch.count_nonzero(model.item_factors.weight).item(), 0)
        with torch.no_grad():
            model.user_factors.weight.zero_()
            model.item_factors.weight.zero_()
        model(torch.tensor([0]), torch.tensor([0])).sum().backward()
        self.assertEqual(torch.count_nonzero(model.user_factors.weight.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(model.item_factors.weight.grad).item(), 0)

    def test_invalid_input_shapes_dtypes_and_indices_rejected(self):
        model = self.manual_model()
        for users, items, error in [
            (torch.tensor(0), torch.tensor(0), ValueError),
            (torch.tensor([[0]]), torch.tensor([[0]]), ValueError),
            (torch.tensor([0, 0]), torch.tensor([0]), ValueError),
            (torch.tensor([0.]), torch.tensor([0]), TypeError),
            (torch.tensor([-1]), torch.tensor([0]), IndexError),
            (torch.tensor([1]), torch.tensor([0]), IndexError),
            (torch.tensor([0]), torch.tensor([2]), IndexError),
        ]:
            with self.subTest(users=users, items=items), self.assertRaises(error):
                model(users, items)

    @unittest.expectedFailure
    def test_fast_interaction_retains_accuracy_for_unbalanced_factors(self):
        """Known MINOR finding: subtraction cancellation in default float32 path."""
        model = FactorizationMachine(1, 1, n_factors=1)
        with torch.no_grad():
            model.user_factors.weight.fill_(10000.)
            model.item_factors.weight.fill_(0.0001)
        indices = torch.tensor([0])
        torch.testing.assert_close(model.interaction_fast(indices, indices), torch.tensor([1.]), rtol=1e-5, atol=1e-7)

    @unittest.expectedFailure
    def test_nonfinite_init_std_rejected(self):
        """Known MINOR finding: positive infinity silently creates invalid factors."""
        with self.assertRaises(ValueError):
            FactorizationMachine(1, 1, init_std=float("inf"))


class FeatureIndexAuditTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            [0, 5, 5, 4, 100], [1, 17, 52, 3, 101], [2, 5, 99, 5, 102],
        ], columns=RAW_COLUMNS)

    def test_feature_indices_valid(self):
        users, items = build_mappings(self.frame)
        frame = encode(self.frame, users, items)
        self.assertEqual(set(users.values()), set(range(len(users))))
        self.assertEqual(set(items.values()), set(range(len(items))))
        self.assertEqual(frame.user_idx.dtype, np.dtype("int64"))
        self.assertEqual(frame.item_idx.dtype, np.dtype("int64"))
        self.assertTrue(frame.user_idx.between(0, len(users) - 1).all())
        self.assertTrue(frame.item_idx.between(0, len(items) - 1).all())
        self.assertFalse(set(users.values()) & {len(users) + i for i in items.values()})
        self.assertEqual(users[5], items[5])  # Same local index is intentional.
        model = FactorizationMachine(len(users), len(items))
        self.assertIsNot(model.user_factors.weight, model.item_factors.weight)
        self.assertNotEqual(model.user_factors.weight.data_ptr(), model.item_factors.weight.data_ptr())
        self.assertEqual(model(torch.from_numpy(frame.user_idx.to_numpy(copy=True)), torch.from_numpy(frame.item_idx.to_numpy(copy=True))).shape, (3,))

    def test_mapping_invariant_to_row_order_and_target(self):
        changed = self.frame.sample(frac=1, random_state=31).copy()
        changed["rating"] = 1
        self.assertEqual(build_mappings(self.frame), build_mappings(changed))

    def test_unknown_ids_are_not_embedding_rows(self):
        users, items = build_mappings(self.frame)
        unseen = pd.DataFrame([[3, 1000, 5, 3, 103], [4, 5, 1000, 2, 104]], columns=RAW_COLUMNS)
        encoded = encode(unseen, users, items)
        self.assertEqual(encoded.user_idx.tolist(), [-1, 0])
        self.assertEqual(encoded.item_idx.tolist(), [0, -1])
        with self.assertRaisesRegex(ValueError, "unknown"):
            predict_frame(FactorizationMachine(len(users), len(items)), encoded)


@unittest.skipUnless((ROOT / "data/processed/team1/metadata.json").exists(), "local prepared data is unavailable")
class PreparedPipelineAuditTests(unittest.TestCase):
    def test_saved_feature_indices_and_predictions(self):
        data = load_prepared()
        self.assertEqual(set(data.user_mapping.values()), set(range(data.n_users)))
        self.assertEqual(set(data.item_mapping.values()), set(range(data.n_items)))
        model = FactorizationMachine(data.n_users, data.n_items, n_factors=3)
        for name in ("train", "validation", "test"):
            with self.subTest(partition=name):
                frame = getattr(data, name)
                for kind, mapping in (("user", data.user_mapping), ("item", data.item_mapping)):
                    np.testing.assert_array_equal(frame[f"{kind}_idx"], frame[f"{kind}_id"].map(mapping))
                    self.assertTrue(frame[f"{kind}_idx"].between(0, len(mapping) - 1).all())
                prediction, row_ids = predict_frame(model, frame, batch_size=257)
                np.testing.assert_array_equal(row_ids, frame.row_id)
                self.assertEqual(prediction.shape, (len(frame),))
                self.assertTrue(np.isfinite(prediction).all())
                users = torch.tensor(frame.user_idx.to_numpy(copy=True), dtype=torch.long)
                items = torch.tensor(frame.item_idx.to_numpy(copy=True), dtype=torch.long)
                expected = (model.global_bias + model.user_bias(users).squeeze(-1)
                            + model.item_bias(items).squeeze(-1)
                            + (model.user_factors(users) * model.item_factors(items)).sum(dim=1))
                np.testing.assert_allclose(prediction, expected.detach().numpy(), rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
