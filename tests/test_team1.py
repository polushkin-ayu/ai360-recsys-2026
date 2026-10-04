"""Small protocol/metric edge cases; actual MovieLens is checked by check_data."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.check_data import check_k5, check_k6
from src.data import (
    ROOT, RAW_COLUMNS, build_mappings, coverage, encode, load_config, load_prepared,
    select_subset, sha256, split_subset, verify_archive,
)
from src.metrics import require_same_row_ids, rmse, rmse_by_row_id


class MetricsTests(unittest.TestCase):
    def test_all_k6_checks(self):
        report = check_k6()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["invalid_inputs_rejected"]), 18)

    def test_ids_not_silently_reordered(self):
        with self.assertRaises(ValueError):
            rmse_by_row_id([1, 3], [1, 3], [10, 11], [11, 10])

    def test_same_model_ids(self):
        require_same_row_ids([10, 11], [10, 11])
        with self.assertRaises(ValueError):
            require_same_row_ids([10, 11], [10, 12])

    def test_large_integer_input_avoids_integer_square_overflow(self):
        self.assertEqual(rmse(np.array([10**12], dtype=np.int64), [0]), 10**12)

    def test_finite_subtraction_overflow_detected(self):
        with self.assertRaises(ValueError):
            rmse([1e308], [-1e308])


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            [row, row // 4 + 1, row % 5 + 1, row % 5 + 1, 1000 + row]
            for row in range(40)
        ], columns=RAW_COLUMNS)

    def test_subset_keeps_all_user_ratings_and_ignores_target(self):
        subset, users = select_subset(self.frame, 42, 15)
        changed = self.frame.copy()
        changed["rating"] = 6 - changed.rating
        other, other_users = select_subset(changed, 42, 15)
        self.assertEqual(users, other_users)
        self.assertEqual(subset.row_id.tolist(), other.row_id.tolist())
        self.assertEqual(len(subset), 16)
        expected = self.frame[self.frame.user_id.isin(users)].reset_index(drop=True)
        pd.testing.assert_frame_equal(subset, expected)

    def test_seeded_subset_is_repeatable(self):
        first, users = select_subset(self.frame, 42, 15)
        second, other_users = select_subset(self.frame, 42, 15)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(users, other_users)

    def test_invalid_subset_sizes(self):
        for size in [0, -1, 41, 2.5]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                select_subset(self.frame, 42, size)

    def test_row_split_complete_disjoint_and_repeatable(self):
        parts, rule = split_subset(self.frame, 42, [0.7, 0.15, 0.15])
        repeated, _ = split_subset(self.frame, 42, [0.7, 0.15, 0.15])
        self.assertEqual(rule, "row_random")
        self.assertEqual([len(x) for x in parts.values()], [28, 6, 6])
        self._check_parts(parts, self.frame)
        for name in parts:
            pd.testing.assert_frame_equal(parts[name], repeated[name])

    def _check_parts(self, parts, original):
        combined = pd.concat(parts.values()).sort_values("row_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(combined, original)
        names = list(parts)
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                self.assertFalse(set(parts[name].row_id) & set(parts[other].row_id))
                self.assertFalse(set(zip(parts[name].user_id, parts[name].item_id)) &
                                 set(zip(parts[other].user_id, parts[other].item_id)))

    def test_duplicate_pairs_split_as_whole_groups_no_rows_removed(self):
        extra = self.frame.iloc[[0, 0, 7, 18]].copy()
        extra["row_id"] = range(40, 44)
        duplicate = pd.concat([self.frame, extra], ignore_index=True)
        parts, rule = split_subset(duplicate, 42, [0.7, 0.15, 0.15])
        self.assertEqual(rule, "pair_group_random")
        self._check_parts(parts, duplicate)
        again, _ = split_subset(duplicate, 42, [0.7, 0.15, 0.15])
        for name in parts:
            pd.testing.assert_frame_equal(parts[name], again[name])

    def test_invalid_split_fractions(self):
        for fractions in [[0.7, 0.2, 0.2], [1, 0, 0], [0.7, 0.3], [np.nan, 0.15, 0.15]]:
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                split_subset(self.frame, 42, fractions)

    def test_too_small_split_is_rejected(self):
        with self.assertRaises(ValueError):
            split_subset(self.frame.iloc[:2], 42, [0.7, 0.15, 0.15])

    def test_train_only_mappings_and_all_unknown_cases(self):
        train = pd.DataFrame([[0, 10, 100, 5, 10], [1, 20, 200, 2, 11]], columns=RAW_COLUMNS)
        users, items = build_mappings(train)
        self.assertEqual(users, {10: 0, 20: 1})
        self.assertEqual(items, {100: 0, 200: 1})
        evaluation = pd.DataFrame([
            [2, 10, 100, 4, 12], [3, 99, 100, 3, 13],
            [4, 10, 999, 2, 14], [5, 99, 999, 1, 15],
        ], columns=RAW_COLUMNS)
        encoded = encode(evaluation, users, items)
        self.assertEqual(encoded.user_idx.tolist(), [0, -1, 0, -1])
        self.assertEqual(encoded.item_idx.tolist(), [0, 0, -1, -1])
        self.assertEqual(coverage(encoded), {
            "before": 4, "unknown_user_rows": 2, "unknown_item_rows": 2,
            "unknown_users": 1, "unknown_items": 1, "unknown_both_rows": 1,
            "excluded_rows": 3, "after": 1, "coverage": 0.25,
        })
        self.assertEqual(encoded.row_id.tolist(), [2, 3, 4, 5])
        self.assertNotIn(99, users)
        self.assertNotIn(999, items)

    def test_invalid_config_rejected(self):
        default = json.loads(Path("configs/team1-data.json").read_text())
        for section, key, value in [("subset", "seed", -1), ("subset", "target_ratings", 0),
                                    ("split", "seed", True), ("source", "archive_sha256", "bad")]:
            config = copy.deepcopy(default)
            config[section][key] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config))
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    load_config(path)

    def test_archive_size_and_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.zip"
            path.write_bytes(b"bad archive")
            with self.assertRaises(ValueError):
                verify_archive(path, {"archive_bytes": 1, "archive_sha256": sha256(path)})
            with self.assertRaises(ValueError):
                verify_archive(path, {"archive_bytes": path.stat().st_size, "archive_sha256": "0" * 64})


@unittest.skipUnless((ROOT / "data/processed/team1/metadata.json").exists(),
                     "run python -m src.prepare_data first for actual-artifact tests")
class SavedArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "prepared"
        shutil.copytree(ROOT / "data/processed/team1", self.path)
        self.config = load_config(ROOT / "configs/team1-data.json")

    def refresh_manifest(self):
        """Even with recomputed checksums, semantic leakage must be caught."""
        path = self.path / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["artifact_sha256"] = {
            name: sha256(self.path / name) for name in metadata["artifact_sha256"]
        }
        metadata["split"]["id"] = hashlib.sha256(
            json.dumps(metadata["artifact_sha256"], sort_keys=True).encode()
        ).hexdigest()
        path.write_text(json.dumps(metadata))

    def test_real_saved_pipeline_k5(self):
        self.assertEqual(check_k5(self.path, self.config)["status"], "passed")

    def test_modified_table_rejected_by_loader(self):
        path = self.path / "validation.csv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_prepared(self.path)

    def test_train_mapping_leak_rejected_even_with_valid_checksums(self):
        path = self.path / "item_mapping.csv"
        mapping = pd.read_csv(path)
        mapping.loc[len(mapping)] = [999999, len(mapping)]
        mapping.to_csv(path, index=False)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "train-only"):
            check_k5(self.path, self.config)

    def test_reordered_evaluation_ids_rejected_even_with_valid_checksums(self):
        path = self.path / "validation_row_ids.csv"
        ids = pd.read_csv(path)
        ids.iloc[::-1].to_csv(path, index=False)
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "order do not match"):
            check_k5(self.path, self.config)

    def test_wrong_coverage_rejected(self):
        path = self.path / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["coverage"]["test"]["coverage"] = 1.0
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "coverage metadata"):
            check_k5(self.path, self.config)


if __name__ == "__main__":
    unittest.main()
