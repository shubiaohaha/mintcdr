from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from mintcdr.data import CDRDataModule
from mintcdr.model import MINTCDR


def test_forward_on_sample(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_sample_data.py", "--output", str(tmp_path), "--users", "8"],
        check=True,
    )
    data = CDRDataModule(tmp_path, max_seq_len=8, neg_per_pos=1)
    dataset = data.dataset("train")
    batch = {key: value.unsqueeze(0) for key, value in dataset[0].items()}
    model = MINTCDR(
        data.num_users,
        data.num_source_items,
        data.num_target_items,
        max_seq_len=8,
        embedding_dim=16,
        semantic_dim=data.semantic_dim,
        num_heads=4,
        mlp_hidden=[32],
    )
    out = model(batch)
    assert out.logits.shape == torch.Size([1])

