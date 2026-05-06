from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sample")
    parser.add_argument("--users", type=int, default=80)
    parser.add_argument("--source-items", type=int, default=120)
    parser.add_argument("--target-items", type=int, default=100)
    parser.add_argument("--semantic-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    user_vecs = {f"u{u}": np_rng.normal(size=args.semantic_dim).astype(float).tolist() for u in range(args.users)}
    item_vecs = {}
    for i in range(args.source_items):
        item_vecs[f"source:s{i}"] = np_rng.normal(size=args.semantic_dim).astype(float).tolist()
    for i in range(args.target_items):
        item_vecs[f"target:t{i}"] = np_rng.normal(size=args.semantic_dim).astype(float).tolist()

    rows = []
    timestamp = 1
    for u in range(args.users):
        source_pos = rng.sample(range(args.source_items), k=12)
        target_pos = rng.sample(range(args.target_items), k=10)
        for pos, item in enumerate(source_pos):
            split = "train" if pos < 10 else "valid"
            rows.append(["source", f"u{u}", f"s{item}", 1, timestamp, split])
            timestamp += 1
        for pos, item in enumerate(target_pos):
            split = "train" if pos < 8 else ("valid" if pos == 8 else "test")
            rows.append(["target", f"u{u}", f"t{item}", 1, timestamp, split])
            timestamp += 1

    with (output / "interactions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "user_id", "item_id", "label", "timestamp", "split"])
        writer.writerows(rows)
    with (output / "user_semantics.json").open("w", encoding="utf-8") as f:
        json.dump(user_vecs, f)
    with (output / "item_semantics.json").open("w", encoding="utf-8") as f:
        json.dump(item_vecs, f)
    print(f"wrote sample data to {output}")


if __name__ == "__main__":
    main()

