import tempfile
import unittest
from pathlib import Path

import torch

from src.models import (
    BiasModel,
    FactorizationMachine,
    TrainingConfig,
    fit_model,
    load_checkpoint,
    save_checkpoint,
)

from src.polynomial import (
    PolynomialRegression2,
    FactorizedPolynomialRegression,
    UserItemPolynomialRegression2,
    UserItemFactorizedPolynomialRegression,
)


class PolynomialRegressionTests(unittest.TestCase):

    def test_polynomial_manual(self):
        model = PolynomialRegression2(
            n_features=3
        ).double()

        with torch.no_grad():
            model.bias.fill_(1.0)

            model.linear_weights.copy_(
                torch.tensor(
                    [2.0, 3.0, 4.0],
                    dtype=torch.float64,
                )
            )

            # Feature pairs:
            # (0, 1)
            # (0, 2)
            # (1, 2)
            model.pair_weights.copy_(
                torch.tensor(
                    [5.0, 6.0, 7.0],
                    dtype=torch.float64,
                )
            )

        x = torch.tensor(
            [[1.0, 2.0, 3.0]],
            dtype=torch.float64,
        )

        prediction = model(x)

        # 1
        # + 2*1 + 3*2 + 4*3
        # + 5*1*2
        # + 6*1*3
        # + 7*2*3
        #
        # = 91

        expected = torch.tensor(
            [91.0],
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            prediction,
            expected,
        )

    def test_factorized_coefficient(self):
        model = FactorizedPolynomialRegression(
            n_features=3,
            n_factors=2,
        ).double()

        with torch.no_grad():
            model.factors.copy_(
                torch.tensor(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                    ],
                    dtype=torch.float64,
                )
            )

        coefficient = model.pair_coefficient(
            0,
            1,
        )

        # <v0, v1>
        # = 1*3 + 2*4
        # = 11

        expected = torch.tensor(
            11.0,
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            coefficient,
            expected,
        )

    def test_direct_and_fast_are_equal(self):
        torch.manual_seed(42)

        model = FactorizedPolynomialRegression(
            n_features=8,
            n_factors=4,
        ).double()

        x = torch.randn(
            5,
            8,
            dtype=torch.float64,
        )

        direct = model.interaction_direct(x)
        fast = model.interaction_fast(x)

        torch.testing.assert_close(
            direct,
            fast,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_user_item_models_follow_project_architecture(self):
        polynomial = UserItemPolynomialRegression2(
            n_users=3,
            n_items=4,
        )

        factorized = UserItemFactorizedPolynomialRegression(
            n_users=3,
            n_items=4,
            n_factors=2,
        )

        self.assertIsInstance(
            polynomial,
            BiasModel,
        )

        self.assertIsInstance(
            factorized,
            BiasModel,
        )

        self.assertIsInstance(
            factorized,
            FactorizationMachine,
        )

    def test_user_item_polynomial_manual(self):
        model = UserItemPolynomialRegression2(
            n_users=2,
            n_items=3,
            global_mean=3.0,
        )

        with torch.no_grad():
            model.user_bias.weight[:, 0].copy_(
                torch.tensor(
                    [0.2, -0.1],
                    dtype=torch.float32,
                )
            )

            model.item_bias.weight[:, 0].copy_(
                torch.tensor(
                    [0.5, 0.0, -0.4],
                    dtype=torch.float32,
                )
            )

            # Pair:
            # user = 0
            # item = 2
            #
            # Flattened index:
            # 0 * n_items + 2 = 2
            model.pair_weights.weight[
                2,
                0,
            ] = 0.7

        users = torch.tensor(
            [0],
            dtype=torch.long,
        )

        items = torch.tensor(
            [2],
            dtype=torch.long,
        )

        prediction = model(
            users,
            items,
        )

        # 3.0 global mean
        # + 0.2 user bias
        # - 0.4 item bias
        # + 0.7 pair interaction
        #
        # = 3.5

        expected = torch.tensor(
            [3.5],
            dtype=torch.float32,
        )

        torch.testing.assert_close(
            prediction,
            expected,
        )

    def test_factorized_user_item_matches_canonical_fm(self):
        torch.manual_seed(42)

        canonical = FactorizationMachine(
            n_users=3,
            n_items=4,
            n_factors=2,
        )

        factorized = UserItemFactorizedPolynomialRegression(
            n_users=3,
            n_items=4,
            n_factors=2,
        )

        factorized.load_state_dict(
            canonical.state_dict()
        )

        users = torch.tensor(
            [0, 1, 2],
            dtype=torch.long,
        )

        items = torch.tensor(
            [1, 2, 3],
            dtype=torch.long,
        )

        with torch.no_grad():
            canonical_prediction = canonical(
                users,
                items,
            )

            factorized_prediction = factorized(
                users,
                items,
            )

        torch.testing.assert_close(
            canonical_prediction,
            factorized_prediction,
        )

    def test_user_item_models_work_with_fit_model(self):
        users = torch.tensor(
            [0, 0, 1, 1, 2, 2],
            dtype=torch.long,
        )

        items = torch.tensor(
            [0, 1, 0, 1, 2, 3],
            dtype=torch.long,
        )

        ratings = torch.tensor(
            [5.0, 4.0, 3.0, 2.0, 5.0, 1.0],
            dtype=torch.float32,
        )

        config = TrainingConfig(
            epochs=2,
            batch_size=3,
            patience=None,
            seed=42,
        )

        polynomial = UserItemPolynomialRegression2(
            n_users=3,
            n_items=4,
        )

        factorized = UserItemFactorizedPolynomialRegression(
            n_users=3,
            n_items=4,
            n_factors=2,
        )

        polynomial_result = fit_model(
            polynomial,
            users,
            items,
            ratings,
            config=config,
        )

        factorized_result = fit_model(
            factorized,
            users,
            items,
            ratings,
            config=config,
        )

        self.assertEqual(
            len(polynomial_result.history),
            2,
        )

        self.assertEqual(
            len(factorized_result.history),
            2,
        )

    def test_polynomial_checkpoint_roundtrip(self):
        model = UserItemPolynomialRegression2(
            n_users=3,
            n_items=4,
            global_mean=3.5,
        )

        users = torch.tensor(
            [0, 1, 2],
            dtype=torch.long,
        )

        items = torch.tensor(
            [1, 2, 3],
            dtype=torch.long,
        )

        with torch.no_grad():
            before = model(
                users,
                items,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "polynomial.pt"
            )

            save_checkpoint(
                path,
                model,
            )

            loaded_model, _ = load_checkpoint(
                path
            )

            with torch.no_grad():
                after = loaded_model(
                    users,
                    items,
                )

        self.assertIsInstance(
            loaded_model,
            UserItemPolynomialRegression2,
        )

        torch.testing.assert_close(
            before,
            after,
        )

    def test_factorized_polynomial_checkpoint_roundtrip(self):
        torch.manual_seed(42)

        model = UserItemFactorizedPolynomialRegression(
            n_users=3,
            n_items=4,
            n_factors=2,
        )

        users = torch.tensor(
            [0, 1, 2],
            dtype=torch.long,
        )

        items = torch.tensor(
            [1, 2, 3],
            dtype=torch.long,
        )

        with torch.no_grad():
            before = model(
                users,
                items,
            )

        with tempfile.TemporaryDirectory() as directory:
            path = (
                Path(directory)
                / "factorized.pt"
            )

            save_checkpoint(
                path,
                model,
            )

            loaded_model, _ = load_checkpoint(
                path
            )

            with torch.no_grad():
                after = loaded_model(
                    users,
                    items,
                )

        self.assertIsInstance(
            loaded_model,
            UserItemFactorizedPolynomialRegression,
        )

        torch.testing.assert_close(
            before,
            after,
        )


if __name__ == "__main__":
    unittest.main()