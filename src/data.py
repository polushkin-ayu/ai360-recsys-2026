"""Deterministic MovieLens preparation and the shared model input interface."""

from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_COLUMNS = ["row_id", "user_id", "item_id", "rating", "timestamp"]
MODEL_COLUMNS = ["row_id", "user_idx", "item_idx", "rating"]
SPLITS = ("train", "validation", "test")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for section in ["subset", "split"]:
        seed = config[section]["seed"]
        if type(seed) is not int or seed < 0:
            raise ValueError("seeds must be nonnegative integers")
    target = config["subset"]["target_ratings"]
    if type(target) is not int or target <= 0:
        raise ValueError("target_ratings must be a positive integer")
    validate_fractions(config["split"]["fractions"])
    if not config["source"]["url"].startswith("https://"):
        raise ValueError("the dataset source must use HTTPS")
    for key in ["archive_sha256", "ratings_sha256"]:
        value = config["source"][key]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"invalid SHA-256: {key}")
    return config


def validate_fractions(fractions):
    values = np.asarray(fractions, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("split fractions must contain three positive finite values")
    if not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("split fractions must sum to one")
    return values


def fetch_source(config, root=ROOT):
    """Check every cached archive; only publish a fully verified download."""
    source = config["source"]
    raw = Path(root) / config["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "ml-100k.zip"
    if not archive.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=raw, suffix=".download", delete=False) as temp:
                temporary = Path(temp.name)
                with urllib.request.urlopen(source["url"], timeout=30) as response:
                    # The configured length also bounds unexpected downloads.
                    content = response.read(source["archive_bytes"] + 1)
                temp.write(content)
            verify_archive(temporary, source)
            temporary.replace(archive)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    verify_archive(archive, source)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("MovieLens ZIP CRC check failed")
        # Extract only these explicit member names, never arbitrary ZIP paths.
        for name in ["u.data", "README", "u.info"]:
            destination = raw / "ml-100k" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(zipped.read(f"ml-100k/{name}"))
    ratings = raw / "ml-100k/u.data"
    if sha256(ratings) != source["ratings_sha256"]:
        raise ValueError("MovieLens u.data SHA-256 does not match configuration")
    write_json(raw / "CHECKSUMS.json", {
        "source": source["url"], "ml-100k.zip": sha256(archive),
        "ml-100k/u.data": sha256(ratings),
    })
    return ratings


def verify_archive(path, source):
    if Path(path).stat().st_size != source["archive_bytes"]:
        raise ValueError("unexpected MovieLens ZIP size")
    if sha256(path) != source["archive_sha256"]:
        raise ValueError("MovieLens ZIP SHA-256 does not match configuration")


def read_ratings(path):
    """row_id is the zero-based original line number in the pinned u.data."""
    frame = pd.read_csv(path, sep="\t", header=None, names=RAW_COLUMNS[1:])
    if frame.isna().any().any():
        raise ValueError("source contains missing values")
    if any(frame[column].dtype.kind not in "iu" for column in frame.columns):
        raise ValueError("source columns must be integers")
    if not frame.rating.between(1, 5).all() or (frame[["user_id", "item_id"]] <= 0).any().any():
        raise ValueError("invalid rating range or user/item ID")
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return frame


def describe(frame):
    def counts(column):
        values = frame.groupby(column).size()
        return {key: float(value) for key, value in values.describe().items()}

    return {
        "ratings": len(frame), "users": int(frame.user_id.nunique()),
        "items": int(frame.item_id.nunique()),
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items()},
        "rating_min": int(frame.rating.min()), "rating_max": int(frame.rating.max()),
        "duplicate_pair_rows": int(frame.duplicated(["user_id", "item_id"], keep=False).sum()),
        "duplicate_pair_excess": int(frame.duplicated(["user_id", "item_id"]).sum()),
        "rating_distribution": {str(key): int(value) for key, value in frame.rating.value_counts().sort_index().items()},
        "ratings_per_user": counts("user_id"), "ratings_per_item": counts("item_id"),
    }


def select_subset(frame, seed, target_ratings):
    """Shuffle sorted users; stop at the first prefix reaching the target."""
    if type(target_ratings) is not int or target_ratings <= 0 or target_ratings > len(frame):
        raise ValueError("target_ratings must be within the source size")
    counts = frame.groupby("user_id", sort=True).size()
    shuffled = np.random.Generator(np.random.PCG64(seed)).permutation(counts.index.to_numpy())
    count = int(np.searchsorted(np.cumsum(counts.loc[shuffled].to_numpy()), target_ratings)) + 1
    users = shuffled[:count]
    subset = frame.loc[frame.user_id.isin(users)].sort_values("row_id").reset_index(drop=True)
    return subset, users.tolist()


def split_subset(subset, seed, fractions):
    """Random row split, or random pair-group split when pairs repeat."""
    values = validate_fractions(fractions)
    repeated = subset.duplicated(["user_id", "item_id"]).any()
    if repeated:
        units = subset[["user_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    else:
        units = subset[["row_id"]].reset_index(drop=True)
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(units))
    first = int(len(units) * values[0])
    second = first + int(len(units) * values[1])
    labels = np.empty(len(units), dtype=object)
    labels[order[:first]] = "train"
    labels[order[first:second]] = "validation"
    labels[order[second:]] = "test"
    assigned = subset.merge(units.assign(split=labels), on=list(units.columns), validate="many_to_one")
    parts = {
        name: assigned.loc[assigned.split == name, RAW_COLUMNS].sort_values("row_id").reset_index(drop=True)
        for name in SPLITS
    }
    if any(part.empty for part in parts.values()):
        raise ValueError("split is too small: every partition must have rows")
    return parts, "pair_group_random" if repeated else "row_random"


def build_mappings(train):
    return tuple({int(value): index for index, value in enumerate(sorted(train[column].unique()))}
                 for column in ["user_id", "item_id"])


def encode(frame, user_mapping, item_mapping):
    encoded = frame.copy()
    encoded["user_idx"] = encoded.user_id.map(user_mapping).fillna(-1).astype("int64")
    encoded["item_idx"] = encoded.item_id.map(item_mapping).fillna(-1).astype("int64")
    encoded["known_user"] = encoded.user_idx >= 0
    encoded["known_item"] = encoded.item_idx >= 0
    return encoded


def coverage(frame):
    unknown_users = ~frame.known_user
    unknown_items = ~frame.known_item
    excluded = unknown_users | unknown_items
    return {
        "before": len(frame), "unknown_user_rows": int(unknown_users.sum()),
        "unknown_item_rows": int(unknown_items.sum()),
        "unknown_users": int(frame.loc[unknown_users, "user_id"].nunique()),
        "unknown_items": int(frame.loc[unknown_items, "item_id"].nunique()),
        "unknown_both_rows": int((unknown_users & unknown_items).sum()),
        "excluded_rows": int(excluded.sum()), "after": int((~excluded).sum()),
        "coverage": float((~excluded).mean()),
    }


def environment():
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "os": platform.platform(), "device": "CPU"}


def prepare(config, config_path, root=ROOT):
    raw_path = fetch_source(config, root)
    full = read_ratings(raw_path)
    summary = describe(full)
    for key in ["ratings", "users", "items"]:
        if summary[key] != config["expected"][key]:
            raise ValueError(f"unexpected MovieLens {key}: {summary[key]}")
    subset, users = select_subset(full, **config["subset"])
    parts, rule = split_subset(subset, **config["split"])
    user_mapping, item_mapping = build_mappings(parts["train"])
    output = Path(root) / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output / "subset.csv", index=False)
    write_json(output / "selected_users.json", users)
    for name, mapping in [("user", user_mapping), ("item", item_mapping)]:
        pd.DataFrame(list(mapping.items()), columns=[f"{name}_id", f"{name}_idx"]).to_csv(output / f"{name}_mapping.csv", index=False)
    coverages = {}
    for name, part in parts.items():
        encoded = encode(part, user_mapping, item_mapping)
        if name == "train":
            encoded.to_csv(output / "train.csv", index=False)
        else:
            encoded.to_csv(output / f"{name}_all.csv", index=False)
            known = encoded.loc[encoded.known_user & encoded.known_item].reset_index(drop=True)
            known.to_csv(output / f"{name}.csv", index=False)
            known[["row_id"]].to_csv(output / f"{name}_row_ids.csv", index=False)
            coverages[name] = coverage(encoded)
    names = ["subset.csv", "selected_users.json", "user_mapping.csv", "item_mapping.csv", "train.csv"]
    names += [f"{name}{suffix}.csv" for name in ["validation", "test"] for suffix in ["", "_all", "_row_ids"]]
    hashes = {name: sha256(output / name) for name in names}
    split_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    metadata = {
        "schema_version": 1, "config": config, "config_sha256": sha256(config_path),
        "source": {"url": config["source"]["url"], "archive_sha256": config["source"]["archive_sha256"], "ratings_sha256": sha256(raw_path)},
        "row_id_rule": "zero-based line number in the pinned original ml-100k/u.data",
        "full_data": summary, "subset": describe(subset), "selected_users": len(users),
        "n_users": len(user_mapping), "n_items": len(item_mapping),
        "split": {"id": split_id, "rule": rule, "seed": config["split"]["seed"],
                  "requested_fractions": config["split"]["fractions"],
                  "sizes_before_filter": {name: len(part) for name, part in parts.items()},
                  "order": "ascending original row_id; predictions must preserve this order",
                  "rounding": "floor train and validation unit counts; remainder to test",
                  "duplicate_policy": "keep all observations; split whole user/item groups when duplicates exist"},
        "coverage": coverages, "artifact_sha256": hashes, "environment": environment(),
    }
    write_json(output / "metadata.json", metadata)
    return output, metadata


@dataclass
class PreparedData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    user_mapping: dict
    item_mapping: dict
    metadata: dict

    @property
    def n_users(self):
        return self.metadata["n_users"]

    @property
    def n_items(self):
        return self.metadata["n_items"]


def load_prepared(directory="data/processed/team1", verify=True):
    """Load known-ID partitions, train dictionaries and complete metadata.

    Relative paths are resolved against the repository root, not the shell cwd.
    Verify hashes by default so models cannot unknowingly use changed files.
    Validation/test *_all.csv retain excluded observations with index -1.
    """
    path = ROOT / directory
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if verify:
        for name, expected in metadata["artifact_sha256"].items():
            if sha256(path / name) != expected:
                raise ValueError(f"prepared artifact checksum mismatch: {name}")
    mappings = []
    for name in ["user", "item"]:
        table = pd.read_csv(path / f"{name}_mapping.csv")
        mappings.append(dict(zip(table[f"{name}_id"].astype(int), table[f"{name}_idx"].astype(int))))
    frames = [pd.read_csv(path / f"{name}.csv") for name in SPLITS]
    return PreparedData(*frames, *mappings, metadata)
