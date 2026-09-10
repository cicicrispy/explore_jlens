"""Linear centered kernel alignment between layers, on the geometry of the lens vectors."""
from __future__ import annotations

import torch

from .lens import Lens, lens_vectors


def cka_matrix(model, lens: Lens, n_tokens: int = 5000, seed: int = 0):
    """Returns (CKA: Tensor[L, L] on CPU, layers: list[int]).

    Grams stay float32 on the model's device. fp16 is avoided on purpose: Gram entries are dot
    products of lens vectors and can exceed fp16's max (65504), which would turn CKA into NaN.
    Double-centering H K H is computed without materializing H: with X column-centered, 1^T X = 0,
    so H (X X^T) H == X X^T exactly.
    """
    vocab_size = model.lm_head.weight.shape[0]
    g = torch.Generator().manual_seed(seed)
    sample = torch.randint(0, vocab_size, (n_tokens,), generator=g).tolist()

    layers = lens.layers
    flat = []
    for l in layers:
        X = lens_vectors(model, lens, sample, l)  # [n, d], float32, model device
        X = X - X.mean(dim=0, keepdim=True)
        flat.append((X @ X.T).flatten())
    flat = torch.stack(flat, dim=0)  # [L, n*n]
    norms = flat.norm(dim=1)
    gram = flat @ flat.T
    CKA = gram / (norms[:, None] * norms[None, :])
    return CKA.cpu(), layers
