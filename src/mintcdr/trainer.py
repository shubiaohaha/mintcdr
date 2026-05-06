from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mintcdr.config import get_device
from mintcdr.data import CDRDataModule, SOURCE, TARGET
from mintcdr.losses import bce_loss
from mintcdr.metrics import aggregate_ranking_metrics
from mintcdr.model import MINTCDR, min_max_normalize


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        set_seed(int(cfg.get("seed", 2026)))
        self.device = get_device(cfg.get("device", "auto"))
        data_cfg = cfg["data"]
        self.data = CDRDataModule(
            data_cfg["root"],
            max_seq_len=int(data_cfg.get("max_seq_len", 50)),
            neg_per_pos=int(data_cfg.get("neg_per_pos", 1)),
            seed=int(cfg.get("seed", 2026)),
        )
        model_cfg = cfg["model"]
        self.model = MINTCDR(
            num_users=self.data.num_users,
            num_source_items=self.data.num_source_items,
            num_target_items=self.data.num_target_items,
            max_seq_len=int(data_cfg.get("max_seq_len", 50)),
            embedding_dim=int(model_cfg.get("embedding_dim", 64)),
            semantic_dim=self.data.semantic_dim,
            num_heads=int(model_cfg.get("num_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            mlp_hidden=list(model_cfg.get("mlp_hidden", [256, 128])),
        ).to(self.device)
        self.run_dir = Path(cfg.get("run_dir", "runs/default"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0

    def fit(self) -> dict[str, float]:
        train_cfg = self.cfg["train"]
        loss_cfg = self.cfg["loss"]
        train_loader = self._loader("train", shuffle=True, batch_size=int(train_cfg["batch_size"]))
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        best = -1.0
        best_metrics: dict[str, float] = {}
        for epoch in range(1, int(train_cfg["epochs"]) + 1):
            logs = self._train_epoch(train_loader, optimizer, loss_cfg, int(train_cfg.get("anchor_steps", 1000)))
            if epoch % int(train_cfg.get("eval_every", 1)) == 0:
                metrics = self.evaluate("valid")
                score = metrics.get("Recall@10", next(iter(metrics.values()), 0.0))
                self._write_json(self.run_dir / f"epoch_{epoch}.json", {"train": logs, "valid": metrics})
                if score >= best:
                    best = score
                    best_metrics = metrics
                    self.save(self.run_dir / "best.pt")
                print(f"epoch={epoch} train={logs} valid={metrics}")
        if self.cfg.get("distill", {}).get("enabled", True):
            self.distill_utilities()
        self.save(self.run_dir / "last.pt")
        return best_metrics

    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_cfg: dict[str, Any],
        anchor_steps: int,
    ) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        count = 0
        for batch in tqdm(loader, desc="train", leave=False):
            batch = _to_device(batch, self.device)
            if self.global_step % anchor_steps == 0:
                self.model.dicn.sync_reference_encoder()

            pair, out = self.model.pair_features(batch)
            labels = batch["label"]
            source_mask = batch["domain"].eq(SOURCE)
            target_mask = batch["domain"].eq(TARGET)

            loss_src = _masked_bce(out.logits, labels, source_mask)
            loss_tgt = _masked_bce(out.logits, labels, target_mask)
            ref_logits = self.model.dicn.reference_logits(pair.detach())
            loss_ref = _masked_bce(ref_logits, labels, target_mask)

            ddan_loss, ddan_logs = self.model.ddan_losses(
                out, float(loss_cfg["temperature"]), float(loss_cfg["lambda_dis"])
            )
            cm_loss, cm_logs = self.model.cross_modal_losses(out, float(loss_cfg["temperature"]))
            total = (
                float(loss_cfg.get("lambda_src", 0.2)) * loss_src
                + loss_tgt
                + loss_ref
                + ddan_loss
                + float(loss_cfg.get("lambda_contrast", 1e-4)) * cm_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()

            logs = {
                "loss": float(total.detach()),
                "src_bce": float(loss_src.detach()),
                "tgt_bce": float(loss_tgt.detach()),
                "ref_bce": float(loss_ref.detach()),
                **ddan_logs,
                **cm_logs,
            }
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
            self.global_step += 1
        return {key: value / max(count, 1) for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, split: str = "test") -> dict[str, float]:
        self.model.eval()
        topk = list(self.cfg["train"].get("topk", [10, 20]))
        dataset = self.data.dataset("train")
        source_seq = dataset.source_seq
        target_seq = dataset.target_seq
        user_sem = dataset.user_sem
        target_item_sem = dataset.target_item_sem.to(self.device)
        truth = self.data.target_ground_truth(split)
        seen = self.data.target_train_seen()
        rankings: dict[int, list[int]] = {}
        for user in truth:
            scores = self.model.score_all_target_items(
                user,
                source_seq[user],
                target_seq[user],
                user_sem[user],
                target_item_sem,
            )
            scores = scores.detach().cpu()
            for item in seen.get(user, set()):
                scores[item] = -1e9
            rankings[user] = torch.argsort(scores, descending=True).tolist()
        return aggregate_ranking_metrics(rankings, truth, topk)

    def distill_utilities(self) -> None:
        distill_cfg = self.cfg["distill"]
        loss_cfg = self.cfg["loss"]
        full_dataset = self.data.dataset("train")
        source_indices = self._source_train_indices(full_dataset)
        if not source_indices:
            return
        train_count = max(1, int(len(source_indices) * 0.9))
        train_indices = source_indices[:train_count]
        valid_indices = source_indices[train_count:] or source_indices[: min(8, len(source_indices))]

        utilities = self.estimate_source_utilities(full_dataset, source_indices, float(loss_cfg["alpha_temperature"]))
        label_by_index = {idx: utilities[pos] for pos, idx in enumerate(source_indices)}

        optimizer = torch.optim.Adam(
            self.model.utility_distiller.parameters(),
            lr=float(distill_cfg.get("lr", 1e-5)),
            weight_decay=float(loss_cfg.get("utility_l2", 1e-4)),
        )
        best = float("inf")
        stale = 0
        for epoch in range(1, int(distill_cfg.get("epochs", 3)) + 1):
            train_loss = self._distill_epoch(full_dataset, train_indices, label_by_index, optimizer)
            valid_loss = self._distill_eval(full_dataset, valid_indices, label_by_index)
            print(f"distill_epoch={epoch} train_mse={train_loss:.6f} valid_mse={valid_loss:.6f}")
            if valid_loss < best:
                best = valid_loss
                stale = 0
                self.save(self.run_dir / "best_distilled.pt")
            else:
                stale += 1
                if stale >= int(distill_cfg.get("patience", 3)):
                    break

    @torch.no_grad()
    def estimate_source_utilities(self, dataset, source_indices: list[int], alpha_temperature: float) -> torch.Tensor:
        loader = DataLoader(Subset(dataset, source_indices), batch_size=int(self.cfg["train"]["batch_size"]), shuffle=False)
        losses = []
        gamma = self._global_domain_influence()
        for batch in loader:
            batch = _to_device(batch, self.device)
            out = self.model(batch)
            per_sample = F.binary_cross_entropy_with_logits(out.logits.view(-1), batch["label"].float(), reduction="none")
            losses.append(per_sample.detach().cpu())
        source_losses = torch.cat(losses)
        raw = torch.exp((torch.tensor(gamma) - source_losses) / alpha_temperature)
        return min_max_normalize(raw)

    @torch.no_grad()
    def _global_domain_influence(self) -> float:
        loader = self._loader("valid", shuffle=False, batch_size=int(self.cfg["train"]["batch_size"]))
        fused_losses = []
        ref_losses = []
        for batch in loader:
            batch = _to_device(batch, self.device)
            mask = batch["domain"].eq(TARGET)
            if not bool(mask.any()):
                continue
            pair, out = self.model.pair_features(batch)
            ref_logits = self.model.dicn.reference_logits(pair)
            fused_losses.append(F.binary_cross_entropy_with_logits(out.logits[mask], batch["label"][mask].float()))
            ref_losses.append(F.binary_cross_entropy_with_logits(ref_logits[mask], batch["label"][mask].float()))
        if not fused_losses:
            return 0.0
        return float(torch.stack(ref_losses).mean() - torch.stack(fused_losses).mean())

    def _distill_epoch(self, dataset, indices: list[int], labels: dict[int, torch.Tensor], optimizer) -> float:
        self.model.train()
        loader = DataLoader(Subset(dataset, indices), batch_size=int(self.cfg["distill"].get("batch_size", 64)), shuffle=False)
        total = 0.0
        count = 0
        for subset_batch, batch in _iter_with_indices(loader, indices):
            batch = _to_device(batch, self.device)
            y = torch.stack([labels[i] for i in subset_batch]).to(self.device).float()
            pred = self.model.predict_utility(batch["item_sem"])
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        return total / max(count, 1)

    @torch.no_grad()
    def _distill_eval(self, dataset, indices: list[int], labels: dict[int, torch.Tensor]) -> float:
        self.model.eval()
        loader = DataLoader(Subset(dataset, indices), batch_size=int(self.cfg["distill"].get("batch_size", 64)), shuffle=False)
        total = 0.0
        count = 0
        for subset_batch, batch in _iter_with_indices(loader, indices):
            batch = _to_device(batch, self.device)
            y = torch.stack([labels[i] for i in subset_batch]).to(self.device).float()
            pred = self.model.predict_utility(batch["item_sem"])
            total += float(F.mse_loss(pred, y))
            count += 1
        return total / max(count, 1)

    def _source_train_indices(self, dataset) -> list[int]:
        indices = [idx for idx, row in enumerate(dataset.rows) if row.domain == SOURCE and row.label > 0]
        rng = random.Random(int(self.cfg.get("seed", 2026)))
        rng.shuffle(indices)
        return indices

    def _loader(self, split: str, shuffle: bool, batch_size: int) -> DataLoader:
        return DataLoader(
            self.data.dataset(split),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=int(self.cfg["data"].get("num_workers", 0)),
        )

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": self.cfg,
                "mappings": {
                    "user2id": self.data.user2id,
                    "source_item2id": self.data.source_item2id,
                    "target_item2id": self.data.target_item2id,
                },
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


def _masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return bce_loss(logits[mask], labels[mask])


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _iter_with_indices(loader: DataLoader, original_indices: list[int]):
    cursor = 0
    for batch in loader:
        batch_size = next(iter(batch.values())).size(0)
        yield original_indices[cursor : cursor + batch_size], batch
        cursor += batch_size

