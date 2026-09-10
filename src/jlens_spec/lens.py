"""J-lens loading and application (reading side): transport-then-unembed, lens vectors, loading probe.

Reference semantics (from the project spec, not re-derived here):
    z      = h.float() @ J[l].T
    logits = lm_head(final_norm(z))
    v_t^(l) = W_U[t] @ J[l]
The final layer is not covered by the lens artifact; its transport is the identity (J = I).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from . import model as model_mod


@dataclass
class Lens:
    J: dict[int, torch.Tensor]   # layer -> [d, d]
    d_model: int
    meta: dict = field(default_factory=dict)
    layers: list[int] = field(default_factory=list)
    dtype: torch.dtype = torch.float32
    file_size_bytes: int = 0


def load_lens(cfg: dict, device) -> Lens:
    """Load the J-lens .pt artifact.

    ASSUMPTION (unverified, flagged for review once the real .pt is downloaded at M1): the artifact
    is a dict at the top level. If it has a "J" key holding a dict of {layer: tensor}, that is used
    directly and every other top-level key is treated as metadata. Otherwise, every top-level entry
    whose key parses as an int (or matches "layer_<N>"/"J_<N>") and whose value is a tensor is taken
    to be a per-layer J matrix; everything else is metadata. This is a best-effort parse written
    without having seen the real file -- refactor this function against the actual keys at M1 step 0
    rather than trusting it blindly.

    Args:
        cfg: dict with "repo", "filename", and optionally "revision_sha" (configs/lens.yaml).
        device: torch device to move J matrices to after loading (loaded to CPU first, per spec).
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=cfg["repo"],
        filename=cfg["filename"],
        revision=cfg.get("revision_sha") or "main",
    )
    raw = torch.load(local_path, map_location="cpu", weights_only=True)

    def _layer_key(k):
        if isinstance(k, int):
            return k
        s = str(k)
        for prefix in ("layer_", "J_", "j_"):
            if s.startswith(prefix) and s[len(prefix):].isdigit():
                return int(s[len(prefix):])
        if s.isdigit():
            return int(s)
        return None

    raw_J: dict[int, torch.Tensor] = {}
    meta: dict = {}
    if isinstance(raw.get("J"), dict):
        for k, v in raw["J"].items():
            lk = _layer_key(k)
            assert lk is not None, f"unrecognized layer key in lens['J']: {k!r}"
            raw_J[lk] = v
        meta = {k: v for k, v in raw.items() if k != "J"}
    else:
        for k, v in raw.items():
            lk = _layer_key(k)
            if lk is not None and torch.is_tensor(v):
                raw_J[lk] = v
            else:
                meta[k] = v

    assert raw_J, "no per-layer J matrices found in the lens artifact -- parsing assumption is wrong"
    layers = sorted(raw_J.keys())
    d_model = raw_J[layers[0]].shape[0]
    dtype = raw_J[layers[0]].dtype  # stored dtype, kept for coverage_ratio
    for l in layers:
        assert raw_J[l].shape == (d_model, d_model), f"J[{l}] has shape {tuple(raw_J[l].shape)}, expected square"
    # Stored 16-bit (3.3 GB for 63 x 5120^2); cast to float32 once so h.float() @ J works (~6.7 GB).
    J = {l: raw_J[l].to(device=device, dtype=torch.float32) for l in layers}
    del raw_J, raw
    import os
    file_size_bytes = os.path.getsize(local_path)
    return Lens(J=J, d_model=d_model, meta=meta, layers=layers, dtype=dtype, file_size_bytes=file_size_bytes)


def random_lens(d: int, layers: list[int], seed: int) -> Lens:
    """Synthetic lens for tests: independent random J[l] per layer, deterministic given seed."""
    g = torch.Generator().manual_seed(seed)
    J = {l: torch.randn(d, d, generator=g) for l in layers}
    file_size_bytes = sum(t.numel() * t.element_size() for t in J.values())
    return Lens(J=J, d_model=d, meta={}, layers=list(layers), dtype=torch.float32, file_size_bytes=file_size_bytes)


def _token_ids(model, tokens: list) -> list[int]:
    ids = []
    for t in tokens:
        if isinstance(t, int):
            ids.append(t)
            continue
        encoded = model.tokenizer.encode(t, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"token {t!r} is not a single token under this tokenizer: {encoded}")
        ids.append(encoded[0])
    return ids


def _J(model, lens: Lens, layer: int, like: torch.Tensor) -> torch.Tensor | None:
    """J[layer] as float32 on `like`'s device, or None for the final layer (J = I). Any other layer
    the lens doesn't cover raises -- silently treating it as identity would turn a J-lens swap into
    a logit-lens swap with no warning."""
    if layer in lens.J:
        return lens.J[layer].to(device=like.device, dtype=torch.float32)
    final = model_mod.n_layers(model) - 1
    if layer == final:
        return None
    raise KeyError(
        f"layer {layer} is not covered by the lens (covered: {lens.layers[0]}..{lens.layers[-1]}, "
        f"{len(lens.layers)} layers) and is not the final layer {final}"
    )


def lens_vectors(model, lens: Lens, tokens: list, layer: int) -> torch.Tensor:
    """W_U[token_ids] @ J[layer] -> [len(tokens), d], float32. Raises if any string token is not
    single-token, or if `layer` is neither covered by the lens nor the final layer."""
    ids = _token_ids(model, tokens)
    rows = model.lm_head.weight[ids].float()
    J = _J(model, lens, layer, rows)
    return rows if J is None else rows @ J


def _transport_unembed(model, lens: Lens, h: torch.Tensor, layer: int) -> torch.Tensor:
    """h: [pos, d] -> full-vocab logits: [pos, vocab], via transport-then-unembed."""
    h = h.float()
    J = _J(model, lens, layer, h)
    z = h if J is None else h @ J.T
    return model_mod.unembed(model, z)


def readout(model, lens: Lens, h: torch.Tensor, layer: int, k: int = 10):
    """Top-k transport-then-unembed readout. h: [pos, d] -> (ids [pos, k], logits [pos, k])."""
    logits = _transport_unembed(model, lens, h, layer)
    top_logits, top_ids = torch.topk(logits, k, dim=-1)
    return top_ids, top_logits


def token_rank(model, lens: Lens, h: torch.Tensor, layer: int, token_id: int) -> torch.Tensor:
    """1-based rank of token_id in the full readout at every position. h: [pos, d] -> [pos]."""
    logits = _transport_unembed(model, lens, h, layer)
    target = logits[..., token_id : token_id + 1]
    return 1 + (logits > target).sum(dim=-1)


def coverage_ratio(lens: Lens) -> float:
    """file_size_bytes / (len(layers) * d_model**2 * bytes_per_elem)."""
    bytes_per_elem = torch.tensor([], dtype=lens.dtype).element_size()
    denom = len(lens.layers) * (lens.d_model ** 2) * bytes_per_elem
    return lens.file_size_bytes / denom
