from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from mintcdr.losses import domain_disentangle_loss, info_nce_in_batch


@dataclass
class ModelOutput:
    logits: torch.Tensor
    user_fused: torch.Tensor
    item_fused: torch.Tensor
    inv_source: torch.Tensor
    inv_target: torch.Tensor
    spec_source_user: torch.Tensor
    spec_target_user: torch.Tensor
    item_cf: torch.Tensor
    user_sem_proj: torch.Tensor
    item_sem_proj: torch.Tensor


class SequenceEncoder(nn.Module):
    def __init__(self, num_items: int, dim: int, max_seq_len: int, num_heads: int, dropout: float):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        x = self.item_embedding(seq) + self.position_embedding(positions)
        key_padding_mask = seq.eq(0)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        mask = (~key_padding_mask).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden:
            layers.extend([nn.Linear(prev, dim), nn.GELU(), nn.Dropout(dropout)])
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DomainInfluenceCalibrationNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float):
        super().__init__()
        self.fusion_encoder = MLPEncoder(input_dim, hidden, dropout)
        self.reference_encoder = MLPEncoder(input_dim, hidden, dropout)
        self.sync_reference_encoder()

    @torch.no_grad()
    def sync_reference_encoder(self) -> None:
        self.reference_encoder.load_state_dict(self.fusion_encoder.state_dict())

    def fusion_logits(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.fusion_encoder(pair_features)

    def reference_logits(self, pair_features: torch.Tensor) -> torch.Tensor:
        return self.reference_encoder(pair_features)


class UtilityDistiller(nn.Module):
    def __init__(self, semantic_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(semantic_dim),
            nn.Linear(semantic_dim, semantic_dim),
            nn.GELU(),
            nn.Linear(semantic_dim, 1),
        )

    def forward(self, item_semantic: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(item_semantic).squeeze(-1))


class MINTCDR(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_source_items: int,
        num_target_items: int,
        max_seq_len: int,
        embedding_dim: int = 64,
        semantic_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        mlp_hidden: list[int] | None = None,
    ):
        super().__init__()
        mlp_hidden = mlp_hidden or [256, 128]
        self.embedding_dim = embedding_dim
        self.semantic_dim = semantic_dim

        self.source_user_id = nn.Embedding(num_users, embedding_dim)
        self.target_user_id = nn.Embedding(num_users, embedding_dim)
        self.source_item_id = nn.Embedding(num_source_items, embedding_dim)
        self.target_item_id = nn.Embedding(num_target_items, embedding_dim)

        shared_num_items = max(num_source_items, num_target_items)
        self.sequence_encoder = SequenceEncoder(shared_num_items, embedding_dim, max_seq_len, num_heads, dropout)
        self.user_semantic_projector = nn.Linear(semantic_dim, embedding_dim)
        self.item_semantic_projector = nn.Linear(semantic_dim, embedding_dim)

        user_dim = embedding_dim * 2 + semantic_dim
        item_dim = embedding_dim + semantic_dim
        self.dicn = DomainInfluenceCalibrationNetwork(user_dim + item_dim, mlp_hidden, dropout)
        self.utility_distiller = UtilityDistiller(semantic_dim)

    def encode_user(
        self,
        user: torch.Tensor,
        source_seq: torch.Tensor,
        target_seq: torch.Tensor,
        user_semantic: torch.Tensor,
        domain: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inv_source = self.sequence_encoder(source_seq)
        inv_target = self.sequence_encoder(target_seq)
        invariant = torch.where(domain.view(-1, 1).eq(0), inv_source, inv_target)
        spec_source = self.source_user_id(user)
        spec_target = self.target_user_id(user)
        specific = torch.where(domain.view(-1, 1).eq(0), spec_source, spec_target)
        fused = torch.cat([invariant, specific, user_semantic], dim=-1)
        return fused, inv_source, inv_target, spec_source, spec_target

    def encode_item(self, item: torch.Tensor, item_semantic: torch.Tensor, domain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        source_item = self.source_item_id(item.clamp_max(self.source_item_id.num_embeddings - 1))
        target_item = self.target_item_id(item.clamp_max(self.target_item_id.num_embeddings - 1))
        item_cf = torch.where(domain.view(-1, 1).eq(0), source_item, target_item)
        fused = torch.cat([item_cf, item_semantic], dim=-1)
        return fused, item_cf

    def pair_features(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ModelOutput]:
        user_fused, inv_s, inv_t, spec_s, spec_t = self.encode_user(
            batch["user"], batch["source_seq"], batch["target_seq"], batch["user_sem"], batch["domain"]
        )
        item_fused, item_cf = self.encode_item(batch["item"], batch["item_sem"], batch["domain"])
        pair = torch.cat([user_fused, item_fused], dim=-1)
        logits = self.dicn.fusion_logits(pair)
        out = ModelOutput(
            logits=logits,
            user_fused=user_fused,
            item_fused=item_fused,
            inv_source=inv_s,
            inv_target=inv_t,
            spec_source_user=spec_s,
            spec_target_user=spec_t,
            item_cf=item_cf,
            user_sem_proj=self.user_semantic_projector(batch["user_sem"]),
            item_sem_proj=self.item_semantic_projector(batch["item_sem"]),
        )
        return pair, out

    def forward(self, batch: dict[str, torch.Tensor]) -> ModelOutput:
        _, out = self.pair_features(batch)
        return out

    def ddan_losses(self, out: ModelOutput, temperature: float, lambda_dis: float) -> tuple[torch.Tensor, dict[str, float]]:
        align = info_nce_in_batch(out.inv_source, out.inv_target, temperature)
        dis = domain_disentangle_loss(out.spec_source_user, out.spec_target_user, temperature)
        loss = align + lambda_dis * dis
        return loss, {"ddan_align": float(align.detach()), "ddan_dis": float(dis.detach())}

    def cross_modal_losses(self, out: ModelOutput, temperature: float) -> tuple[torch.Tensor, dict[str, float]]:
        user_loss = info_nce_in_batch(out.inv_target, out.user_sem_proj, temperature)
        item_loss = info_nce_in_batch(out.item_cf, out.item_sem_proj, temperature)
        loss = user_loss + item_loss
        return loss, {"cm_user": float(user_loss.detach()), "cm_item": float(item_loss.detach())}

    def reference_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        pair, _ = self.pair_features(batch)
        return self.dicn.reference_logits(pair)

    def predict_utility(self, item_semantic: torch.Tensor) -> torch.Tensor:
        return self.utility_distiller(item_semantic)

    def score_all_target_items(
        self,
        user_id: int,
        source_seq: torch.Tensor,
        target_seq: torch.Tensor,
        user_semantic: torch.Tensor,
        target_item_semantics: torch.Tensor,
    ) -> torch.Tensor:
        device = target_item_semantics.device
        num_items = self.target_item_id.num_embeddings
        users = torch.full((num_items,), user_id, dtype=torch.long, device=device)
        items = torch.arange(num_items, dtype=torch.long, device=device)
        domain = torch.full((num_items,), 1, dtype=torch.long, device=device)
        batch = {
            "user": users,
            "item": items,
            "domain": domain,
            "source_seq": source_seq.to(device).unsqueeze(0).repeat(num_items, 1),
            "target_seq": target_seq.to(device).unsqueeze(0).repeat(num_items, 1),
            "user_sem": user_semantic.to(device).unsqueeze(0).repeat(num_items, 1),
            "item_sem": target_item_semantics.to(device),
        }
        return torch.sigmoid(self(batch).logits)


def min_max_normalize(values: torch.Tensor) -> torch.Tensor:
    vmin = values.min()
    vmax = values.max()
    return (values - vmin) / (vmax - vmin).clamp_min(1e-8)

