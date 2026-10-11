"""Prepare/check the shared MovieLens 1M split without changing 100K data.

python -m src.prepare_movielens1m --config configs/movielens1m-data.json
All CSVs use LF line endings for identical cross-platform artifact checksums.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src.data import (
    ROOT, RAW_COLUMNS, SPLITS, build_mappings, coverage, describe, encode,
    environment, load_prepared, sha256, split_subset, validate_fractions, write_json,
)
from src.check_data import check_k6
from src.metrics import require_same_row_ids


def require(condition, message):
    if not condition:
        raise ValueError(message)


def row_hash(frame):
    return hashlib.sha256(frame.row_id.to_numpy(dtype="<i8").tobytes()).hexdigest()


def fetch_ratings(config, root=ROOT):
    raw = Path(root) / config["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "movielens1m.zip"
    sources = config["source"]["archives"]
    if not archive.exists():
        failures = []
        for source in sources:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=raw, suffix=".download", delete=False) as stream:
                    temporary = Path(stream.name)
                    with urllib.request.urlopen(source["url"], timeout=60) as response:
                        stream.write(response.read(source["bytes"] + 1))
                require(temporary.stat().st_size == source["bytes"] and sha256(temporary) == source["sha256"], "archive checksum/size mismatch")
                temporary.replace(archive)
                break
            except Exception as error:
                failures.append(f"{source['url']}: {type(error).__name__}: {error}")
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        require(archive.exists(), "Download failed. Cache a pinned archive at " + str(archive) + ". " + "; ".join(failures))
    digest = sha256(archive)
    matching = [s for s in sources if s["sha256"] == digest and s["bytes"] == archive.stat().st_size]
    require(len(matching) == 1, "cached archive is not one of the pinned Kaggle/GroupLens archives")
    source = matching[0]
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None, "archive CRC check failed")
        for member in ["ratings.dat", "README", "movies.dat", "users.dat"]:
            (raw / member).write_bytes(zipped.read(source["prefix"] + member))
    ratings = raw / "ratings.dat"
    require(sha256(ratings) == config["source"]["ratings_sha256"], "ratings.dat checksum mismatch")
    return ratings, source


def read_ratings(path):
    frame = pd.read_csv(path, sep="::", engine="python", header=None, names=RAW_COLUMNS[1:], dtype="int64")
    require(not frame.empty and not frame.isna().any().any(), "empty/missing source values")
    require(frame.rating.between(1, 5).all(), "ratings outside 1..5")
    require((frame[["user_id", "item_id"]] > 0).all().all(), "invalid source IDs")
    require(not frame.duplicated(["user_id", "item_id"]).any(), "unexpected repeated user/item pair")
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return frame


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def prepare_frame(full, config, output):
    validate_fractions(config["split"]["fractions"])
    require(config["subset"]["rule"] == "all_ratings", "1M preparation must retain all source rows")
    summary = describe(full)
    for name, expected in config["expected"].items():
        require(summary[name] == expected, f"unexpected source {name}")
    parts, rule = split_subset(full, **config["split"])
    users, items = build_mappings(parts["train"])
    output.mkdir(parents=True, exist_ok=True)
    save_csv(full, output / "subset.csv")
    write_json(output / "selected_users.json", sorted(int(v) for v in full.user_id.unique()))
    for name, mapping in [("user", users), ("item", items)]:
        save_csv(pd.DataFrame(list(mapping.items()), columns=[f"{name}_id", f"{name}_idx"]), output / f"{name}_mapping.csv")
    coverages = {}
    for name, frame in parts.items():
        encoded = encode(frame, users, items)
        if name == "train":
            save_csv(encoded, output / "train.csv")
        else:
            save_csv(encoded, output / f"{name}_all.csv")
            known = encoded.loc[encoded.known_user & encoded.known_item].reset_index(drop=True)
            save_csv(known, output / f"{name}.csv")
            save_csv(known[["row_id"]], output / f"{name}_row_ids.csv")
            coverages[name] = coverage(encoded)
    names = ["subset.csv", "selected_users.json", "user_mapping.csv", "item_mapping.csv", "train.csv"]
    names += [f"{name}{suffix}.csv" for name in ["validation", "test"] for suffix in ["", "_all", "_row_ids"]]
    hashes = {name: sha256(output / name) for name in names}
    # Stable split identifier is independent of downloaded ZIP container and OS.
    split_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    metadata = {
        "schema_version": 1, "dataset": "MovieLens 1M", "config": config,
        "source": {"url": config["source"]["url"], "ratings_sha256": config["source"]["ratings_sha256"]},
        "row_id_rule": "zero-based original ratings.dat line number before splitting",
        "full_data": summary, "subset": summary, "selected_users": summary["users"],
        "n_users": len(users), "n_items": len(items), "artifact_sha256": hashes,
        "split": {"id": split_id, "rule": rule, "seed": config["split"]["seed"],
                  "requested_fractions": config["split"]["fractions"],
                  "sizes_before_filter": {name: len(frame) for name, frame in parts.items()},
                  "order": "ascending original row_id", "rounding": "floor train/validation; remainder test"},
        "coverage": coverages,
    }
    write_json(output / "metadata.json", metadata)
    return metadata


def check_frame(full, config, output):
    bundle = load_prepared(output)
    metadata = bundle.metadata
    require(metadata["config"] == config, "metadata/config mismatch")
    pd.testing.assert_frame_equal(pd.read_csv(output / "subset.csv"), full)
    require(describe(full) == metadata["full_data"], "source summary mismatch")
    expected, rule = split_subset(full, **config["split"])
    parts = {"train": bundle.train, "validation": pd.read_csv(output / "validation_all.csv"), "test": pd.read_csv(output / "test_all.csv")}
    row_sets, pair_sets = {}, {}
    for name, frame in parts.items():
        pd.testing.assert_frame_equal(frame[RAW_COLUMNS], expected[name])
        require(frame.row_id.is_unique and frame.row_id.is_monotonic_increasing, "row_id order/uniqueness")
        row_sets[name] = set(frame.row_id)
        pair_sets[name] = set(zip(frame.user_id, frame.item_id))
        for kind, mapping in [("user", bundle.user_mapping), ("item", bundle.item_mapping)]:
            correct = frame[f"{kind}_id"].map(mapping).fillna(-1).astype("int64").to_numpy()
            require(np.array_equal(correct, frame[f"{kind}_idx"]), "incorrect encoded indices")
            require(np.array_equal(correct >= 0, frame[f"known_{kind}"]), "incorrect known flags")
    intersections, pair_intersections = {}, {}
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i+1:]:
            key = f"{first}/{second}"
            intersections[key] = len(row_sets[first] & row_sets[second])
            pair_intersections[key] = len(pair_sets[first] & pair_sets[second])
            require(intersections[key] == pair_intersections[key] == 0, "split overlaps")
    require(set.union(*row_sets.values()) == set(full.row_id), "split does not cover source")
    for kind, mapping in [("user", bundle.user_mapping), ("item", bundle.item_mapping)]:
        correct = {int(value): index for index, value in enumerate(sorted(bundle.train[f"{kind}_id"].unique()))}
        require(mapping == correct, "mapping not train-only")
        require(metadata[f"n_{kind}s"] == len(correct), "vocabulary size mismatch")
    evaluation = {}
    for name in ["validation", "test"]:
        filtered = parts[name].loc[parts[name].known_user & parts[name].known_item].reset_index(drop=True)
        actual = getattr(bundle, name)
        pd.testing.assert_frame_equal(filtered, actual)
        ids = pd.read_csv(output / f"{name}_row_ids.csv").row_id.to_numpy()
        require_same_row_ids(actual.row_id.to_numpy(), ids)
        stats = coverage(parts[name])
        require(stats == metadata["coverage"][name], "coverage mismatch")
        evaluation[name] = {**stats, "canonical_row_ids_sha256": row_hash(actual)}
    hashes = metadata["artifact_sha256"]
    require(hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() == metadata["split"]["id"], "split ID mismatch")
    sizes = {name: len(frame) for name, frame in parts.items()}
    require(sizes == metadata["split"]["sizes_before_filter"] and rule == metadata["split"]["rule"], "split metadata mismatch")
    return {"status": "passed", "source_lineage_verified": True, "seeded_membership_verified": True,
            "train_only_mappings_verified": True, "all_source_rows_retained": True,
            "row_intersections": intersections, "pair_intersections": pair_intersections,
            "sizes_before_filter": sizes, "evaluation": evaluation, "split_id": metadata["split"]["id"],
            "train_canonical_row_ids_sha256": row_hash(bundle.train)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/movielens1m-data.json")
    parser.add_argument("--report", default="data/movielens1m-checks.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ratings, archive_source = fetch_ratings(config)
    full = read_ratings(ratings)
    summary = describe(full)
    require(all(summary[k] == v for k, v in config["expected"].items()), "source dimensions differ")
    output = ROOT / config["output_dir"]
    if not args.check_only:
        prepare_frame(full, config, output)
    checks = check_frame(full, config, output)
    report = {"run_id": "movielens1m-shared-split-20261006", "dataset": config["dataset"],
              "config": config, "config_sha256": sha256(config_path), "downloaded_archive": archive_source,
              "kaggle_and_official_ratings_byte_identical": True,
              "source_summary": summary, "environment": environment(),
              "K5": checks, "K6": check_k6(), "test_scores_computed": False}
    write_json(ROOT / args.report, report)
    print(json.dumps({"source": summary["ratings"], "split_id": checks["split_id"],
                      "sizes_before_filter": checks["sizes_before_filter"],
                      "evaluation": checks["evaluation"], "K5": "passed", "K6": "passed"}, indent=2))


if __name__ == "__main__":
    main()
