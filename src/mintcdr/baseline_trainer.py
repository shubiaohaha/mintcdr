from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mintcdr.baselines import CrossDomainMF, TargetMF
from mintcdr.config import get_device
from mintcdr.data import CDRDataModule, TARGET
from mintcdr.losses import bce_loss
from mintcdr.metrics import aggregate_ranking_metrics
from mintcdr.trainer import set_seed


class BaselineTrainer:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        set_seed(int(cfg.get("seed", 2026)))
        self.device = get_device(cfg.get("device", "auto"))
        self.data = CDRDataModule(
            cfg["data"]["root"],
            max_seq_len=int(cfg["data"].get("max_seq_len", 50)),
            neg_per_pos=int(cfg["data"].get("neg_per_pos", 1)),
            seed=int(cfg.get("seed", 2026)),
        )
        model_name = cfg.get("baseline", {}).get("name", "tgt").lower()
        dim = int(cfg["model"].get("embedding_dim", 64))
        if model_name in {"tgt", "target_mf"}:
            self.model = TargetMF(self.data.num_users, self.data.num_target_items, dim)
            self.target_only = True
        elif model_name in {"cdr_mf", "dual_mf"}:
            self.model = CrossDomainMF(self.data.num_users, self.data.num_source_items, self.data.num_target_items, dim)
            self.target_only = False
        else:
            raise ValueError(f"unknown baseline: {model_name}")
        self.model.to(self.device)
        self.run_dir = Path(cfg.get("run_dir", f"runs/{model_name}"))
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def fit(self) -> dict[str, float]:
        train_cfg = self.cfg["train"]
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
        )
        loader = self._loader("train", shuffle=True)
        best = {}
        best_score = -1.0
        for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
            loss = self._train_epoch(loader, optimizer)
            metrics = self.evaluate("valid")
            score = metrics.get("Recall@10", 0.0)
            if score >= best_score:
                best = metrics
                best_score = score
                self.save(self.run_dir / "best.pt")
            self._write_json(self.run_dir / f"epoch_{epoch}.json", {"loss": loss, "valid": metrics})
            print(f"epoch={epoch} loss={loss:.6f} valid={metrics}")
        self.save(self.run_dir / "last.pt")
        return best

    def _train_epoch(self, loader, optimizer) -> float:
        self.model.train()
        total = 0.0
        count = 0
        for batch in tqdm(loader, desc="baseline", leave=False):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            if self.target_only:
                mask = batch["domain"].eq(TARGET)
                if not bool(mask.any()):
                    continue
                batch = {key: value[mask] for key, value in batch.items()}
            out = self.model(batch)
            loss = bce_loss(out.logits, batch["label"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def evaluate(self, split: str = "test") -> dict[str, float]:
        self.model.eval()
        topk = list(self.cfg["train"].get("topk", [10, 20]))
        truth = self.data.target_ground_truth(split)
        seen = self.data.target_train_seen()
        rankings = {}
        for user in truth:
            scores = self.model.score_all_target_items(user, self.data.num_target_items, self.device).cpu()
            for item in seen.get(user, set()):
                scores[item] = -1e9
            rankings[user] = torch.argsort(scores, descending=True).tolist()
        return aggregate_ranking_metrics(rankings, truth, topk)

    def _loader(self, split: str, shuffle: bool):
        return DataLoader(
            self.data.dataset(split),
            batch_size=int(self.cfg["train"].get("batch_size", 256)),
            shuffle=shuffle,
            num_workers=int(self.cfg["data"].get("num_workers", 0)),
        )

    def save(self, path: str | Path) -> None:
        torch.save({"model": self.model.state_dict(), "config": self.cfg}, path)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

