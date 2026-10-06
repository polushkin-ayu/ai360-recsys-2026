import unittest

import torch

from src.polynomial import (
    PolynomialRegression2,
    FactorizedPolynomialRegression,
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

            # Пары:
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


if __name__ == "__main__":
    unittest.main()
