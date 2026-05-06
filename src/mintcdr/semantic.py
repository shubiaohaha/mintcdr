from __future__ import annotations

import hashlib

import numpy as np


class OfflineSemanticEncoder:
    """Deterministic fallback for experiments without an external LLM encoder.

    Real paper-scale experiments should replace this class with cached vectors
    produced by a selected LLM or Transformer encoder.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / max(norm, 1e-8)

