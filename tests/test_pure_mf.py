import unittest
import torch
import numpy as np
from src.models import PureMatrixFactorization, TrainingConfig, fit_model, predict_frame

class PureMFTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.n_users = 10
        self.n_items = 15
        self.model = PureMatrixFactorization(self.n_users, self.n_items, n_factors=5)
        self.users = torch.tensor([0, 1, 2], dtype=torch.long)
        self.items = torch.tensor([1, 2, 3], dtype=torch.long)

    def test_forward_shape(self):
        preds = self.model(self.users, self.items)
        self.assertEqual(preds.shape, (3,))

    def test_factor_l2(self):
        l2 = self.model.factor_l2()
        self.assertGreater(l2.item(), 0)

    def test_toy_fit(self):
        ratings = torch.tensor([3.0, 4.0, 2.0])
        config = TrainingConfig(epochs=5, batch_size=3, learning_rate=0.01, seed=42)
        result = fit_model(
            model=self.model,
            train_user_idx=self.users,
            train_item_idx=self.items,
            train_rating=ratings,
            config=config
        )
        self.assertEqual(len(result.history), 5)

if __name__ == "__main__":
    unittest.main()