from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class BaselineOutput:
    logits: torch.Tensor


class TargetMF(nn.Module):
    """Target-only matrix factorization baseline."""

    def __init__(self, num_users: int, num_target_items: int, embedding_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_target_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_target_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        user = self.user_embedding(batch["user"])
        item = self.item_embedding(batch["item"].clamp_max(self.item_embedding.num_embeddings - 1))
        logits = (user * item).sum(dim=-1)
        logits = logits + self.user_bias(batch["user"]).squeeze(-1)
        logits = logits + self.item_bias(batch["item"].clamp_max(self.item_embedding.num_embeddings - 1)).squeeze(-1)
        return BaselineOutput(logits + self.global_bias)

    def score_all_target_items(self, user_id: int, num_items: int, device: torch.device) -> torch.Tensor:
        users = torch.full((num_items,), user_id, dtype=torch.long, device=device)
        items = torch.arange(num_items, dtype=torch.long, device=device)
        batch = {"user": users, "item": items}
        return torch.sigmoid(self(batch).logits)


class CrossDomainMF(nn.Module):
    """Dual-domain MF with shared user factors and domain-specific item factors."""

    def __init__(self, num_users: int, num_source_items: int, num_target_items: int, embedding_dim: int = 64):
        super().__init__()
        self.shared_user = nn.Embedding(num_users, embedding_dim)
        self.source_user = nn.Embedding(num_users, embedding_dim)
        self.target_user = nn.Embedding(num_users, embedding_dim)
        self.source_item = nn.Embedding(num_source_items, embedding_dim)
        self.target_item = nn.Embedding(num_target_items, embedding_dim)
        self.domain_bias = nn.Embedding(2, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> BaselineOutput:
        domain = batch["domain"]
        source_user = self.shared_user(batch["user"]) + self.source_user(batch["user"])
        target_user = self.shared_user(batch["user"]) + self.target_user(batch["user"])
        source_item = self.source_item(batch["item"].clamp_max(self.source_item.num_embeddings - 1))
        target_item = self.target_item(batch["item"].clamp_max(self.target_item.num_embeddings - 1))
        user = torch.where(domain.view(-1, 1).eq(0), source_user, target_user)
        item = torch.where(domain.view(-1, 1).eq(0), source_item, target_item)
        logits = (user * item).sum(dim=-1) + self.domain_bias(domain).squeeze(-1)
        return BaselineOutput(logits)

    def score_all_target_items(self, user_id: int, num_items: int, device: torch.device) -> torch.Tensor:
        users = torch.full((num_items,), user_id, dtype=torch.long, device=device)
        items = torch.arange(num_items, dtype=torch.long, device=device)
        domain = torch.ones(num_items, dtype=torch.long, device=device)
        batch = {"user": users, "item": items, "domain": domain}
        return torch.sigmoid(self(batch).logits)

