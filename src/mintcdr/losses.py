from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_logits(query: torch.Tensor, keys: torch.Tensor, temperature: float) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    keys = F.normalize(keys, dim=-1)
    return query @ keys.t() / temperature


def info_nce_in_batch(query: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = cosine_logits(query, positive, temperature)
    labels = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, labels)


def domain_disentangle_loss(source_specific: torch.Tensor, target_specific: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = cosine_logits(source_specific, target_specific, temperature)
    labels = torch.arange(source_specific.size(0), device=source_specific.device)
    # Minimize the same-user cross-domain matching probability to preserve domain-specific preferences.
    probs = F.softmax(logits, dim=1)
    return probs[torch.arange(source_specific.size(0), device=source_specific.device), labels].mean()


def bce_loss(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.view(-1), labels.float().view(-1), weight=weight)

