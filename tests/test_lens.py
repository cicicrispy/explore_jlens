import pytest
import torch

from jlens_spec import lens as lens_mod
from jlens_spec import model as model_mod


def test_lens_vectors_shape(standin_model, random_lens):
    l = random_lens.layers[0]
    v = lens_mod.lens_vectors(standin_model, random_lens, ["the", "a"], l)
    assert v.shape == (2, random_lens.d_model)


def test_lens_vectors_raises_on_multi_token(standin_model, random_lens):
    tok = standin_model.tokenizer
    multi = "supercalifragilisticexpialidocious"
    ids = tok.encode(multi, add_special_tokens=False)
    if len(ids) < 2:
        pytest.skip("this string is a single token under the stand-in tokenizer")
    with pytest.raises(ValueError):
        lens_mod.lens_vectors(standin_model, random_lens, [multi], random_lens.layers[0])


def test_final_layer_readout_equals_logit_lens(standin_model, random_lens):
    n_layers = model_mod.n_layers(standin_model)
    final_layer = n_layers - 1
    assert final_layer not in random_lens.J  # excluded by construction: transport is identity there

    h = torch.randn(3, random_lens.d_model)
    ids, logits = lens_mod.readout(standin_model, random_lens, h, final_layer, k=5)

    expected_logits = model_mod.unembed(standin_model, h)
    expected_top = torch.topk(expected_logits, 5, dim=-1)

    assert torch.equal(ids, expected_top.indices)
    assert torch.allclose(logits, expected_top.values, atol=1e-4)


def test_token_rank_is_1_indexed(standin_model, random_lens):
    l = random_lens.layers[0]
    h = torch.randn(4, random_lens.d_model)
    tok_id = standin_model.tokenizer.encode("the", add_special_tokens=False)[0]
    rank = lens_mod.token_rank(standin_model, random_lens, h, l, tok_id)
    assert torch.all(rank >= 1)
