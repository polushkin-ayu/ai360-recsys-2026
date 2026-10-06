import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from src.data import RAW_COLUMNS, load_prepared
from src.prepare_movielens1m import prepare_frame, check_frame, read_ratings, fetch_ratings


class MovieLens1MTests(unittest.TestCase):
    def fixture(self):
        # Distinct pairs, users and items missing in some splits exercise unknown handling.
        frame = pd.DataFrame({"row_id": np.arange(20), "user_id": np.arange(20) // 4 + 1,
                              "item_id": np.arange(20) % 7 + 1, "rating": np.arange(20) % 5 + 1,
                              "timestamp": np.arange(20) + 100})[RAW_COLUMNS]
        config = {"source": {"url": "fixture", "ratings_sha256": "0"*64},
                  "expected": {"ratings": 20, "users": 5, "items": 7},
                  "subset": {"rule": "all_ratings"},
                  "split": {"seed": 42, "fractions": [0.7, 0.15, 0.15]}}
        return frame, config

    def test_membership_reproduction_and_train_only_encoding(self):
        full, config = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary)/"first", Path(temporary)/"second"
            a = prepare_frame(full, config, first)
            b = prepare_frame(full, config, second)
            self.assertEqual(a["artifact_sha256"], b["artifact_sha256"])
            self.assertEqual(a["split"]["id"], b["split"]["id"])
            data = load_prepared(first)
            order = np.random.Generator(np.random.PCG64(42)).permutation(20)
            self.assertEqual(set(data.train.row_id), set(order[:14]))
            self.assertEqual(set(pd.read_csv(first/"validation_all.csv").row_id), set(order[14:17]))
            self.assertEqual(set(pd.read_csv(first/"test_all.csv").row_id), set(order[17:]))
            self.assertEqual(set(data.user_mapping), set(data.train.user_id))
            for name in ["validation", "test"]:
                all_rows = pd.read_csv(first/f"{name}_all.csv")
                kept = getattr(data, name)
                self.assertEqual(set(kept.row_id), set(all_rows.loc[all_rows.known_user & all_rows.known_item].row_id))
            self.assertEqual(check_frame(full, config, first)["status"], "passed")
            self.assertNotIn(b"\r\n", (first/"train.csv").read_bytes())

    def test_corrupted_table_rejected(self):
        full, config = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            prepare_frame(full, config, output)
            frame = pd.read_csv(output/"train.csv")
            frame.loc[0, "rating"] = 5 if frame.loc[0, "rating"] != 5 else 1
            frame.to_csv(output/"train.csv", index=False)
            with self.assertRaisesRegex(ValueError, "checksum"):
                check_frame(full, config, output)

    def test_config_and_duplicate_rows_rejected(self):
        full, config = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            config["split"]["fractions"] = [0.8, 0.2, 0.2]
            with self.assertRaises(ValueError):
                prepare_frame(full, config, Path(temporary))
            source = Path(temporary)/"ratings.dat"
            source.write_text("1::2::5::100\n1::2::4::101\n")
            with self.assertRaisesRegex(ValueError, "repeated"):
                read_ratings(source)

    def test_zip_checksum_and_double_colon_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root/"raw"
            raw.mkdir()
            ratings = b"1::2::5::100\n2::3::4::200\n"
            archive = raw/"movielens1m.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                for name in ["ratings.dat", "README", "movies.dat", "users.dat"]:
                    zipped.writestr(name, ratings if name == "ratings.dat" else b"fixture")
            config = {"raw_dir": "raw", "source": {"ratings_sha256": hashlib.sha256(ratings).hexdigest(),
                "archives": [{"sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                              "bytes": archive.stat().st_size, "prefix": "", "url": "fixture"}]}}
            path, _ = fetch_ratings(config, root)
            frame = read_ratings(path)
            self.assertEqual(frame.row_id.tolist(), [0, 1])
            self.assertEqual(frame.rating.tolist(), [5, 4])
            config["source"]["ratings_sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "ratings.dat checksum"):
                fetch_ratings(config, root)


if __name__ == "__main__":
    unittest.main()
