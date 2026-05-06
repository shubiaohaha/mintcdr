from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from mintcdr.semantic import OfflineSemanticEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess two Amazon-style CSV files into MINTCDR format.")
    parser.add_argument("--source", required=True, help="CSV with user_id,item_id,rating,timestamp[,text]")
    parser.add_argument("--target", required=True, help="CSV with user_id,item_id,rating,timestamp[,text]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--semantic-dim", type=int, default=64)
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    encoder = OfflineSemanticEncoder(args.semantic_dim)

    source_rows = _read_rows(Path(args.source), args.positive_threshold)
    target_rows = _read_rows(Path(args.target), args.positive_threshold)
    source_split = _split_by_user(source_rows, train_ratio=0.8, valid_ratio=0.2)
    target_split = _split_by_user(target_rows, train_ratio=0.8, valid_ratio=0.1)

    with (output / "interactions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "user_id", "item_id", "label", "timestamp", "split"])
        for row, split in source_split:
            writer.writerow(["source", row["user_id"], row["item_id"], 1, row["timestamp"], split])
        for row, split in target_split:
            writer.writerow(["target", row["user_id"], row["item_id"], 1, row["timestamp"], split])

    users = sorted({r["user_id"] for r in source_rows + target_rows})
    user_sem = {u: encoder.encode(f"user {u}").tolist() for u in users}
    item_sem = {}
    for domain, rows in [("source", source_rows), ("target", target_rows)]:
        item_text = {}
        for row in rows:
            item_text.setdefault(row["item_id"], row.get("text") or row["item_id"])
        for item, text in item_text.items():
            item_sem[f"{domain}:{item}"] = encoder.encode(text).tolist()

    with (output / "user_semantics.json").open("w", encoding="utf-8") as f:
        json.dump(user_sem, f)
    with (output / "item_semantics.json").open("w", encoding="utf-8") as f:
        json.dump(item_sem, f)
    print(f"wrote processed data to {output}")


def _read_rows(path: Path, threshold: float) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row.get("rating", 1.0)) >= threshold:
                rows.append(row)
    return rows


def _split_by_user(rows: list[dict[str, str]], train_ratio: float, valid_ratio: float):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["user_id"]].append(row)
    output = []
    for user_rows in grouped.values():
        user_rows = sorted(user_rows, key=lambda x: int(x.get("timestamp", 0)))
        n = len(user_rows)
        train_end = max(1, int(n * train_ratio))
        valid_end = train_end + max(1, int(n * valid_ratio)) if n >= 3 else train_end
        for idx, row in enumerate(user_rows):
            if idx < train_end:
                split = "train"
            elif idx < valid_end:
                split = "valid"
            else:
                split = "test"
            output.append((row, split))
    return output


if __name__ == "__main__":
    main()

