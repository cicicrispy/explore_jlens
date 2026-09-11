"""Model loading and residual-stream access via nnsight.

Qwen3.6-27B is a vision-language checkpoint; depending on the transformers class that loads it, the
text decoder sits at `model.model` or `model.model.language_model`. `decoder()` finds whichever one
has `.layers`, and every other accessor goes through it, so the rest of the code never hard-codes
the path. Only text is ever fed in, so the vision encoder never runs.

`load_model` takes an optional `device` (not in the spec's contract) so tests can force CPU while
scripts keep `device_map="auto"`.
"""
from __future__ import annotations

import torch
from nnsight import LanguageModel


def load_model(cfg: dict, standin: bool = False, device: str | None = None):
    """Load the causal LM as an nnsight LanguageModel, with real weights (dispatch=True).

    Without dispatch=True nnsight keeps weights on the `meta` device until the first trace, so
    anything touching `lm_head.weight` outside a trace (lens_vectors, unembed, CKA) would silently
    get meta tensors.

    Args:
        cfg: configs/model.yaml contents.
        standin: load cfg["standin_hf_id"] (float32) instead of cfg["hf_id"] (cfg["dtype"]).
        device: None -> device_map="auto"; "cpu" forces CPU (used by tests).
    """
    if standin:
        hf_id, revision, dtype = cfg["standin_hf_id"], "main", torch.float32
    else:
        hf_id, revision = cfg["hf_id"], cfg["revision"]
        dtype = getattr(torch, cfg.get("dtype", "bfloat16"))

    # MPS: load on CPU (the path the tests exercise) and move afterwards; loading weights straight
    # onto MPS via accelerate's device_map was observed to stall at ~1% of weight loading.
    load_device = "cpu" if device == "mps" else device
    device_map = {"": load_device} if load_device is not None else "auto"
    model = LanguageModel(
        hf_id,
        revision=revision,
        dtype=dtype,  # transformers >=5 name (torch_dtype is the deprecated alias)
        device_map=device_map,
        dispatch=True,
    )
    if device is not None:
        # The transformers pipeline nnsight builds can move the model to an available accelerator
        # (MPS on a Mac) regardless of device_map, so enforce the requested device explicitly.
        model.to(device)
        actual = model.lm_head.weight.device.type
        assert actual == torch.device(device).type, f"requested device {device!r}, model is on {actual!r}"

    # Freeze weights (not global grad mode): lens-vector / readout ops on lm_head.weight then build no
    # autograd graph (which on the 27B model would waste GPU memory), while gradients w.r.t.
    # activations still work -- which is what fitting/verifying a Jacobian lens would need.
    for p in model.parameters():
        p.requires_grad_(False)

    tok_cls = type(model.tokenizer).__name__.lower()
    assert "qwen" in hf_id.lower() or "qwen" in tok_cls, (
        f"expected a Qwen-family tokenizer; got hf_id={hf_id!r}, tokenizer={type(model.tokenizer).__name__}"
    )
    return model


def decoder(model):
    """The text decoder module (the one holding `.layers` and the final `.norm`)."""
    inner = model.model
    if hasattr(inner, "layers"):
        return inner
    if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        return inner.language_model
    raise AttributeError("could not locate the text decoder (.layers) under model.model")


def device_of(model) -> torch.device:
    """Device of the unembedding weight -- where lens math and edits must live."""
    return model.lm_head.weight.device


def n_layers(model) -> int:
    return len(decoder(model).layers)


def d_model(model) -> int:
    cfg = model.config
    return getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size


def layer_output(model, l: int):
    """Envoy for decoder.layers[l].output: the residual stream after layer l, [batch, pos, d]."""
    return decoder(model).layers[l].output


def unembed(model, h: torch.Tensor) -> torch.Tensor:
    """The model's own final RMSNorm + lm_head. h: [..., d] -> logits: [..., vocab]."""
    w = model.lm_head.weight
    normed = decoder(model).norm(h.to(device=w.device, dtype=w.dtype))
    return model.lm_head(normed)
