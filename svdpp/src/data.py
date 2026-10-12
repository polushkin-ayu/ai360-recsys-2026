"""Общий split MovieLens 1M и словари ID, построенные только по train."""
from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
import sys
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
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")



def validate_fractions(fractions):
    values = np.asarray(fractions, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("split fractions must contain three positive finite values")
    if not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("split fractions must sum to one")
    return values



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



def load_prepared(directory="data/processed/movielens1m", verify=True):
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



