from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


SOURCE = 0
TARGET = 1


@dataclass(frozen=True)
class Interaction:
    domain: int
    user: int
    item: int
    label: float
    timestamp: int
    split: str
    raw_user: str
    raw_item: str


class CDRDataModule:
    def __init__(self, root: str | Path, max_seq_len: int = 50, neg_per_pos: int = 1, seed: int = 2026):
        self.root = Path(root)
        self.max_seq_len = max_seq_len
        self.neg_per_pos = neg_per_pos
        self.seed = seed

        self.user2id: dict[str, int] = {}
        self.source_item2id: dict[str, int] = {}
        self.target_item2id: dict[str, int] = {}
        self.interactions: list[Interaction] = []
        self.source_item_sem: np.ndarray | None = None
        self.target_item_sem: np.ndarray | None = None
        self.user_sem: np.ndarray | None = None
        self._load()

    @property
    def num_users(self) -> int:
        return len(self.user2id)

    @property
    def num_source_items(self) -> int:
        return len(self.source_item2id)

    @property
    def num_target_items(self) -> int:
        return len(self.target_item2id)

    @property
    def semantic_dim(self) -> int:
        if self.user_sem is not None:
            return int(self.user_sem.shape[1])
        raise RuntimeError("semantic features are not loaded")

    def _user_id(self, raw: str) -> int:
        if raw not in self.user2id:
            self.user2id[raw] = len(self.user2id)
        return self.user2id[raw]

    def _item_id(self, raw: str, domain: int) -> int:
        mapping = self.source_item2id if domain == SOURCE else self.target_item2id
        if raw not in mapping:
            mapping[raw] = len(mapping)
        return mapping[raw]

    def _load(self) -> None:
        path = self.root / "interactions.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing interactions file: {path}")

        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                domain = SOURCE if row["domain"].lower() == "source" else TARGET
                user = self._user_id(row["user_id"])
                item = self._item_id(row["item_id"], domain)
                rows.append(
                    Interaction(
                        domain=domain,
                        user=user,
                        item=item,
                        label=float(row["label"]),
                        timestamp=int(row.get("timestamp", 0)),
                        split=row.get("split", "train"),
                        raw_user=row["user_id"],
                        raw_item=row["item_id"],
                    )
                )
        self.interactions = rows
        self.user_sem = self._load_user_semantics()
        self.source_item_sem, self.target_item_sem = self._load_item_semantics()

    def _load_user_semantics(self) -> np.ndarray:
        path = self.root / "user_semantics.json"
        data = _read_json(path)
        dim = len(next(iter(data.values())))
        out = np.zeros((self.num_users, dim), dtype=np.float32)
        for raw, idx in self.user2id.items():
            out[idx] = np.asarray(data.get(raw, np.zeros(dim)), dtype=np.float32)
        return out

    def _load_item_semantics(self) -> tuple[np.ndarray, np.ndarray]:
        path = self.root / "item_semantics.json"
        data = _read_json(path)
        dim = len(next(iter(data.values())))
        source = np.zeros((self.num_source_items, dim), dtype=np.float32)
        target = np.zeros((self.num_target_items, dim), dtype=np.float32)
        for raw, idx in self.source_item2id.items():
            source[idx] = np.asarray(data.get(f"source:{raw}", np.zeros(dim)), dtype=np.float32)
        for raw, idx in self.target_item2id.items():
            target[idx] = np.asarray(data.get(f"target:{raw}", np.zeros(dim)), dtype=np.float32)
        return source, target

    def make_sequences(self, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
        source_hist: dict[int, list[tuple[int, int]]] = defaultdict(list)
        target_hist: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for x in self.interactions:
            if x.split != split or x.label <= 0:
                continue
            hist = source_hist if x.domain == SOURCE else target_hist
            hist[x.user].append((x.timestamp, x.item))

        src = np.zeros((self.num_users, self.max_seq_len), dtype=np.int64)
        tgt = np.zeros((self.num_users, self.max_seq_len), dtype=np.int64)
        for user, values in source_hist.items():
            src[user] = _pad_sequence([i + 1 for _, i in sorted(values)], self.max_seq_len)
        for user, values in target_hist.items():
            tgt[user] = _pad_sequence([i + 1 for _, i in sorted(values)], self.max_seq_len)
        return src, tgt

    def dataset(self, split: str) -> "InteractionDataset":
        src_seq, tgt_seq = self.make_sequences("train")
        rows = [x for x in self.interactions if x.split == split]
        if split == "train" and self.neg_per_pos > 0:
            rows = self._with_negative_samples(rows)
        return InteractionDataset(rows, src_seq, tgt_seq, self.user_sem, self.source_item_sem, self.target_item_sem)

    def _with_negative_samples(self, rows: list[Interaction]) -> list[Interaction]:
        rng = random.Random(self.seed)
        positives = [x for x in rows if x.label > 0]
        seen = {(x.domain, x.user, x.item) for x in self.interactions if x.label > 0}
        out = list(rows)
        for x in positives:
            item_count = self.num_source_items if x.domain == SOURCE else self.num_target_items
            for _ in range(self.neg_per_pos):
                for _try in range(50):
                    item = rng.randrange(item_count)
                    if (x.domain, x.user, item) not in seen:
                        out.append(
                            Interaction(x.domain, x.user, item, 0.0, x.timestamp, x.split, x.raw_user, f"neg-{item}")
                        )
                        break
        rng.shuffle(out)
        return out

    def target_ground_truth(self, split: str = "test") -> dict[int, set[int]]:
        truth: dict[int, set[int]] = defaultdict(set)
        for x in self.interactions:
            if x.domain == TARGET and x.split == split and x.label > 0:
                truth[x.user].add(x.item)
        return truth

    def target_train_seen(self) -> dict[int, set[int]]:
        seen: dict[int, set[int]] = defaultdict(set)
        for x in self.interactions:
            if x.domain == TARGET and x.split == "train" and x.label > 0:
                seen[x.user].add(x.item)
        return seen


class InteractionDataset(Dataset):
    def __init__(
        self,
        rows: list[Interaction],
        source_seq: np.ndarray,
        target_seq: np.ndarray,
        user_sem: np.ndarray,
        source_item_sem: np.ndarray,
        target_item_sem: np.ndarray,
    ):
        self.rows = rows
        self.source_seq = torch.as_tensor(source_seq, dtype=torch.long)
        self.target_seq = torch.as_tensor(target_seq, dtype=torch.long)
        self.user_sem = torch.as_tensor(user_sem, dtype=torch.float32)
        self.source_item_sem = torch.as_tensor(source_item_sem, dtype=torch.float32)
        self.target_item_sem = torch.as_tensor(target_item_sem, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        x = self.rows[idx]
        item_sem = self.source_item_sem[x.item] if x.domain == SOURCE else self.target_item_sem[x.item]
        return {
            "domain": torch.tensor(x.domain, dtype=torch.long),
            "user": torch.tensor(x.user, dtype=torch.long),
            "item": torch.tensor(x.item, dtype=torch.long),
            "label": torch.tensor(x.label, dtype=torch.float32),
            "source_seq": self.source_seq[x.user],
            "target_seq": self.target_seq[x.user],
            "user_sem": self.user_sem[x.user],
            "item_sem": item_sem,
        }


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pad_sequence(values: Iterable[int], max_len: int) -> np.ndarray:
    arr = np.zeros(max_len, dtype=np.int64)
    values = list(values)[-max_len:]
    if values:
        arr[-len(values) :] = values
    return arr

