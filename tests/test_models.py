import tempfile
import unittest
from pathlib import Path

import torch

from src.models import (
    BiasModel,
    FactorizationMachine,
    TrainingConfig,
    count_parameters,
    fit_model,
    load_checkpoint,
    regularized_mse_loss,
    save_checkpoint,
)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.users = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        self.items = torch.tensor([0, 1, 0, 2, 1, 2], dtype=torch.long)
        self.ratings = torch.tensor([3.0, 2.0, 4.0, 5.0, 1.0, 2.5])

    def test_bias_forward(self) -> None:
        model = BiasModel(3, 3, global_mean=3.0)
        with torch.no_grad():
            model.user_bias.weight[:, 0] = torch.tensor([0.1, 0.2, 0.3])
            model.item_bias.weight[:, 0] = torch.tensor([-0.5, 0.0, 0.5])
        actual = model(torch.tensor([0, 2]), torch.tensor([0, 2]))
        torch.testing.assert_close(actual, torch.tensor([2.6, 3.8]))

    def test_fast_and_direct_fm_are_equal(self) -> None:
        model = FactorizationMachine(3, 3, n_factors=4)
        torch.testing.assert_close(
            model.interaction_direct(self.users, self.items),
            model.interaction_fast(self.users, self.items),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_fm_is_matrix_factorization_with_bias(self) -> None:
        model = FactorizationMachine(3, 3, n_factors=2, global_mean=3.0)
        expected = (
            model.global_bias
            + model.user_bias(self.users).squeeze(-1)
            + model.item_bias(self.items).squeeze(-1)
            + (
                model.user_factors(self.users) * model.item_factors(self.items)
            ).sum(dim=1)
        )
        torch.testing.assert_close(model(self.users, self.items), expected)

    def test_zero_factors_reduce_fm_to_bias(self) -> None:
        bias = BiasModel(3, 3, global_mean=3.0)
        fm = FactorizationMachine(3, 3, n_factors=2, global_mean=3.0)
        with torch.no_grad():
            fm.user_bias.weight.copy_(bias.user_bias.weight)
            fm.item_bias.weight.copy_(bias.item_bias.weight)
            fm.user_factors.weight.zero_()
            fm.item_factors.weight.zero_()
        torch.testing.assert_close(
            fm(self.users, self.items), bias(self.users, self.items)
        )

    def test_autograd_matches_finite_difference_with_regularization(self) -> None:
        model = FactorizationMachine(3, 3, n_factors=2).double()
        target = self.ratings.double()
        loss = regularized_mse_loss(
            model,
            model(self.users, self.items),
            target,
            n_train=len(target),
            reg_bias=0.07,
            reg_factors=0.11,
        )
        loss.backward()
        analytical = model.user_factors.weight.grad[1, 0].item()
        epsilon = 1e-6
        parameter = model.user_factors.weight
        with torch.no_grad():
            original = parameter[1, 0].item()
            parameter[1, 0] = original + epsilon
            plus = regularized_mse_loss(
                model,
                model(self.users, self.items),
                target,
                n_train=len(target),
                reg_bias=0.07,
                reg_factors=0.11,
            ).item()
            parameter[1, 0] = original - epsilon
            minus = regularized_mse_loss(
                model,
                model(self.users, self.items),
                target,
                n_train=len(target),
                reg_bias=0.07,
                reg_factors=0.11,
            ).item()
            parameter[1, 0] = original
        numerical = (plus - minus) / (2.0 * epsilon)
        self.assertAlmostEqual(analytical, numerical, places=7)

    def test_one_standard_sgd_step(self) -> None:
        model = BiasModel(3, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = regularized_mse_loss(
            model,
            model(self.users, self.items),
            self.ratings,
            n_train=len(self.ratings),
            reg_bias=0.01,
            reg_factors=0.0,
        )
        optimizer.zero_grad()
        loss.backward()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()}
        optimizer.step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name] - 0.1 * gradients[name])

    def test_fit_and_standard_checkpoint(self) -> None:
        model = FactorizationMachine(3, 3, n_factors=3)
        result = fit_model(
            model,
            self.users,
            self.items,
            self.ratings,
            config=TrainingConfig(epochs=80, batch_size=6, patience=None, seed=5),
        )
        self.assertLess(result.history[-1]["train_rmse"], result.history[0]["train_rmse"])
        self.assertEqual(count_parameters(model), 1 + 3 + 3 + (3 + 3) * 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fm.pt"
            save_checkpoint(path, model, result)
            restored, restored_result = load_checkpoint(path)
            torch.testing.assert_close(
                model(self.users, self.items), restored(self.users, self.items)
            )
            self.assertEqual(restored_result.best_epoch, result.best_epoch)

    def test_training_seed_is_reproducible(self) -> None:
        config = TrainingConfig(epochs=5, batch_size=3, patience=None, seed=17)
        first = FactorizationMachine(3, 3, n_factors=2)
        second = FactorizationMachine(3, 3, n_factors=2)
        fit_model(first, self.users, self.items, self.ratings, config=config)
        fit_model(second, self.users, self.items, self.ratings, config=config)
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        ):
            torch.testing.assert_close(first_parameter, second_parameter)


if __name__ == "__main__":
    unittest.main()
