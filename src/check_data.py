"""Executable K5/K6 evidence; run independently against saved artifacts."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .data import (
    ROOT, RAW_COLUMNS, SPLITS, coverage, describe, environment, load_config,
    load_prepared, read_ratings, select_subset, sha256, verify_archive, write_json,
)
from .metrics import require_same_row_ids, rmse, rmse_by_row_id


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_k5(directory, config, root=ROOT):
    """Check integrity, raw row lineage, complete subset, split and mappings."""
    directory = Path(directory)
    bundle = load_prepared(directory)
    metadata = bundle.metadata
    require(metadata["config"] == config, "metadata and supplied config differ")
    raw = Path(root) / config["raw_dir"]
    verify_archive(raw / "ml-100k.zip", config["source"])
    ratings_path = raw / "ml-100k/u.data"
    require(sha256(ratings_path) == config["source"]["ratings_sha256"], "raw u.data checksum mismatch")
    full = read_ratings(ratings_path)
    require(describe(full) == metadata["full_data"], "source summary mismatch")
    for key in ["ratings", "users", "items"]:
        require(metadata["full_data"][key] == config["expected"][key], f"unexpected source {key}")
    require(metadata["source"] == {
        "url": config["source"]["url"],
        "archive_sha256": config["source"]["archive_sha256"],
        "ratings_sha256": config["source"]["ratings_sha256"],
    }, "metadata source provenance mismatch")
    subset = pd.read_csv(directory / "subset.csv")
    expected_subset, expected_users = select_subset(full, **config["subset"])
    pd.testing.assert_frame_equal(subset, expected_subset)
    require(json.loads((directory / "selected_users.json").read_text()) == expected_users,
            "selected users differ from the seeded first-prefix rule")
    require(describe(subset) == metadata["subset"], "subset summary mismatch")
    require(len(expected_users) == metadata["selected_users"], "selected user count mismatch")
    parts = {"train": bundle.train,
             "validation": pd.read_csv(directory / "validation_all.csv"),
             "test": pd.read_csv(directory / "test_all.csv")}
    row_sets, pair_sets = {}, {}
    for name, frame in parts.items():
        require(frame.row_id.is_unique, f"duplicate row_id in {name}")
        require(frame.row_id.is_monotonic_increasing, f"unsorted row_id in {name}")
        row_sets[name] = set(frame.row_id)
        pair_sets[name] = set(zip(frame.user_id, frame.item_id))
        pd.testing.assert_frame_equal(
            frame[RAW_COLUMNS].reset_index(drop=True),
            subset.loc[subset.row_id.isin(row_sets[name]), RAW_COLUMNS].reset_index(drop=True),
        )
    intersections = {}
    pair_intersections = {}
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            key = f"{first}/{second}"
            intersections[key] = len(row_sets[first] & row_sets[second])
            pair_intersections[key] = len(pair_sets[first] & pair_sets[second])
            require(intersections[key] == 0, f"row overlap in {key}")
            require(pair_intersections[key] == 0, f"user/item pair overlap in {key}")
    require(set.union(*row_sets.values()) == set(subset.row_id), "split union differs from subset")
    train = parts["train"]
    for name, mapping in [("user", bundle.user_mapping), ("item", bundle.item_mapping)]:
        values = sorted(train[f"{name}_id"].unique())
        expected = {int(value): index for index, value in enumerate(values)}
        table = pd.read_csv(directory / f"{name}_mapping.csv")
        require(table[f"{name}_id"].is_unique and table[f"{name}_idx"].is_unique,
                f"duplicate entries in {name} mapping")
        require(mapping == expected, f"{name} mapping is not exactly the sorted train-only mapping")
        require(metadata[f"n_{name}s"] == len(expected), f"wrong n_{name}s")
    for name, frame in parts.items():
        for kind, mapping in [("user", bundle.user_mapping), ("item", bundle.item_mapping)]:
            expected = frame[f"{kind}_id"].map(mapping).fillna(-1).astype("int64")
            require(np.array_equal(expected, frame[f"{kind}_idx"]), f"incorrect {kind} indices in {name}")
            require(frame[f"known_{kind}"].dtype.kind == "b", f"invalid known_{kind} flags")
            require(np.array_equal(expected >= 0, frame[f"known_{kind}"]), f"incorrect known_{kind} in {name}")
    require((train.known_user & train.known_item).all(), "train contains unknown IDs")
    evaluation = {}
    for name in ["validation", "test"]:
        frame = parts[name]
        known = getattr(bundle, name)
        expected = frame.loc[frame.known_user & frame.known_item].reset_index(drop=True)
        pd.testing.assert_frame_equal(known, expected)
        ids = pd.read_csv(directory / f"{name}_row_ids.csv").row_id.to_numpy()
        require_same_row_ids(known.row_id.to_numpy(), ids)
        stats = coverage(frame)
        require(stats == metadata["coverage"][name], f"incorrect {name} coverage metadata")
        evaluation[name] = {**stats, "row_ids_file": f"{name}_row_ids.csv",
                            "row_ids_sha256": sha256(directory / f"{name}_row_ids.csv")}
    sizes = {name: len(part) for name, part in parts.items()}
    require(sizes == metadata["split"]["sizes_before_filter"], "split size metadata mismatch")
    expected_rule = "pair_group_random" if subset.duplicated(["user_id", "item_id"]).any() else "row_random"
    require(metadata["split"]["rule"] == expected_rule, "split rule metadata mismatch")
    split_id = hashlib.sha256(json.dumps(metadata["artifact_sha256"], sort_keys=True).encode()).hexdigest()
    require(split_id == metadata["split"]["id"], "split ID mismatch")
    return {
        "status": "passed", "row_intersections": intersections,
        "pair_intersections": pair_intersections, "union_equals_subset": True,
        "raw_row_lineage_verified": True, "all_ratings_of_selected_users": True,
        "seeded_subset_verified": True, "mappings_exactly_train_only": True,
        "known_flags_and_indices_verified": True, "artifact_checksums_verified": True,
        "split_id": split_id, "sizes_before_filter": sizes, "evaluation": evaluation,
        "model_id_comparison": "pending model implementations; require_same_row_ids is provided",
    }


def check_k6():
    manual = rmse([1, 3], [2, 1])
    require(math.isclose(manual, math.sqrt(2.5), rel_tol=0, abs_tol=1e-12), "manual RMSE failed")
    ids = np.array([0, 1], dtype=np.int64)
    aligned = rmse_by_row_id([1, 3], [2, 1], ids, ids)
    require(aligned == manual and type(manual) is float, "aligned RMSE/scalar type failed")
    invalid = {
        "different_lengths": lambda: rmse([1, 3], [2]),
        "broadcast_column": lambda: rmse([1, 3], [[2], [1]]),
        "broadcast_row": lambda: rmse([[1, 3]], [2, 1]),
        "both_2d": lambda: rmse([[1], [3]], [[2], [1]]),
        "scalar": lambda: rmse(1, 2),
        "empty": lambda: rmse([], []),
        "nan_truth": lambda: rmse([np.nan, 3], [2, 1]),
        "nan_prediction": lambda: rmse([1, 3], [2, np.nan]),
        "inf_truth": lambda: rmse([np.inf, 3], [2, 1]),
        "inf_prediction": lambda: rmse([1, 3], [2, -np.inf]),
        "complex": lambda: rmse([1 + 1j], [2]),
        "nonnumeric": lambda: rmse(["1"], ["2"]),
        "reordered_row_ids": lambda: rmse_by_row_id([1, 3], [2, 1], ids, ids[::-1]),
        "different_row_ids": lambda: require_same_row_ids(ids, [0, 2]),
        "duplicate_row_ids": lambda: require_same_row_ids([0, 0], [0, 0]),
        "missing_row_ids": lambda: require_same_row_ids(ids, [0]),
        "noninteger_row_ids": lambda: require_same_row_ids([0.0, 1.0], ids),
        "row_ids_length_mismatch": lambda: rmse_by_row_id([1, 3], [2, 1], [0], [0]),
    }
    rejected = []
    for name, function in invalid.items():
        try:
            function()
        except ValueError:
            rejected.append(name)
        else:
            raise ValueError(f"K6 accepted invalid input: {name}")
    require(rmse([1, 3], [1, 3]) == 0, "zero error failed")
    require(math.isfinite(rmse([1e200], [0])), "large finite error overflowed")
    return {"status": "passed", "manual_rmse": manual, "expected": math.sqrt(2.5),
            "aligned_rmse": aligned, "invalid_inputs_rejected": rejected,
            "finite_large_values_checked": True, "zero_error_checked": True}


def provenance(root=ROOT):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit, status = None, "Git metadata unavailable"
    paths = sorted(Path(root).glob("src/*.py")) + sorted(Path(root).glob("tests/*.py"))
    paths += [Path(root) / "configs/team1-data.json", Path(root) / "requirements-data.txt"]
    return {"base_commit": commit, "uncommitted_changes": bool(status),
            "source_sha256": {str(path.relative_to(root)): sha256(path) for path in paths if path.is_file()},
            "note": "No commit/push performed. Reproduce with the supplied patch, not the base commit alone."}


def make_report(directory, config, command, config_path, root=ROOT):
    metadata = json.loads((Path(directory) / "metadata.json").read_text())
    require(metadata["config_sha256"] == sha256(config_path), "config file checksum mismatch")
    return {"run_id": "team1-data-2026-10-04", "command": command,
            "config_sha256": sha256(config_path), "environment": environment(),
            "provenance": provenance(root), "source": metadata["source"],
            "full_data": metadata["full_data"], "subset": metadata["subset"],
            "selected_users": metadata["selected_users"],
            "n_users": metadata["n_users"], "n_items": metadata["n_items"],
            "config": config, "K5": check_k5(directory, config, root), "K6": check_k6()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/team1-data.json")
    parser.add_argument("--report", default="results/team1-data-checks.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = load_config(config_path)
    command = f"python -m src.check_data --config {args.config} --report {args.report}"
    report = make_report(ROOT / config["output_dir"], config, command, config_path)
    write_json(ROOT / args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
