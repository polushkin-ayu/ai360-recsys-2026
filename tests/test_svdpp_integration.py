import tempfile
import unittest
from pathlib import Path
import pandas as pd
import torch
from src.svdpp import Histories, SVDPlusPlus
from src.svdpp_legacy import SVDPlusPlus as LegacySVDPlusPlus
from src.models import save_checkpoint, load_checkpoint


class IntegrationTests(unittest.TestCase):
    def test_public_model_uses_current_implementation(self):
        frame = pd.DataFrame(dict(user_idx=[0, 1], item_idx=[0, 1]))
        histories = Histories(frame, 2)
        model = SVDPlusPlus(2, 2, 3, 3.5, histories)
        self.assertEqual(SVDPlusPlus.__module__, "svdpp.src.model")
        with torch.no_grad():
            value = model(torch.tensor([0, 1]), torch.tensor([1, 0]))
        self.assertEqual(value.shape, (2,))
        self.assertTrue(torch.isfinite(value).all())

    def test_shared_loader_still_reads_historical_svdpp_checkpoint(self):
        model = LegacySVDPlusPlus(2, 2, histories=[[0], [1]], n_factors=2, global_mean=3.5)
        users, items = torch.tensor([0, 1]), torch.tensor([1, 0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            save_checkpoint(path, model)
            restored, result = load_checkpoint(path)
        self.assertIsNone(result)
        with torch.no_grad():
            torch.testing.assert_close(model(users, items), restored(users, items))


if __name__ == "__main__":
    unittest.main()
