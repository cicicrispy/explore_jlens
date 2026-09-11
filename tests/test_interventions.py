import pytest
import torch

from jlens_spec import interventions as iv


def _rand(d=8, pos=6, batch=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, pos, d, generator=g)


def _vec_with_cosine(v_s: torch.Tensor, cos_target: float, seed: int = 1) -> torch.Tensor:
    d = v_s.shape[0]
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(d, generator=g)
    v_s_unit = v_s / v_s.norm()
    noise_orth = noise - (noise @ v_s_unit) * v_s_unit
    noise_orth = noise_orth / noise_orth.norm()
    v_t_unit = cos_target * v_s_unit + (1 - cos_target ** 2) ** 0.5 * noise_orth
    return v_t_unit * v_s.norm()


def test_swap_identity_when_source_equals_target():
    h = _rand()
    v = torch.randn(h.shape[-1])
    mask = torch.ones(h.shape[1], dtype=torch.bool)
    h_new, _ = iv.swap(h, mask, v, v, alpha=1.0)
    assert torch.equal(h_new, h)


def test_orthogonal_complement_unchanged():
    d = 8
    h = _rand(d=d)
    v_s = torch.zeros(d)
    v_s[0] = 1.0
    v_t = torch.zeros(d)
    v_t[1] = 1.0
    mask = torch.ones(h.shape[1], dtype=torch.bool)
    h_new, _ = iv.swap(h, mask, v_s, v_t, alpha=1.0)

    V, V_pinv = iv._basis(v_s, v_t)
    P = V @ V_pinv
    I = torch.eye(d)
    orth_before = torch.einsum("de,bpe->bpd", (I - P), h)
    orth_after = torch.einsum("de,bpe->bpd", (I - P), h_new)
    assert torch.allclose(orth_before, orth_after, atol=1e-5)


def test_correlated_vectors_project_onto_span():
    d = 8
    h = _rand(d=d)
    v_s = torch.randn(d)
    v_t = _vec_with_cosine(v_s, 0.7)
    assert abs(float(torch.cosine_similarity(v_s, v_t, dim=0)) - 0.7) < 1e-4

    V, V_pinv = iv._basis(v_s, v_t)
    c = torch.einsum("bpd,kd->bpk", h, V_pinv)
    proj_via_c = torch.einsum("bpk,dk->bpd", c, V)
    P = V @ V_pinv
    proj_direct = torch.einsum("de,bpe->bpd", P, h)
    assert torch.allclose(proj_via_c, proj_direct, atol=1e-4)


def test_alpha_scales_linearly():
    d = 8
    h = _rand(d=d)
    v_s, v_t = torch.randn(d), torch.randn(d)
    mask = torch.ones(h.shape[1], dtype=torch.bool)
    h1, _ = iv.swap(h, mask, v_s, v_t, alpha=1.0)
    h2, _ = iv.swap(h, mask, v_s, v_t, alpha=2.0)
    assert torch.allclose(h2 - h, 2 * (h1 - h), atol=1e-5)


def test_swap_is_not_steering():
    d = 8
    h = _rand(d=d)
    v_s, v_t = torch.randn(d), torch.randn(d)
    mask = torch.ones(h.shape[1], dtype=torch.bool)
    h_swap, _ = iv.swap(h, mask, v_s, v_t, alpha=1.0)
    h_steer = h + v_t
    assert not torch.allclose(h_swap, h_steer)


def test_all_false_mask_is_bitwise_identity():
    d = 8
    h = _rand(d=d)
    v_s, v_t = torch.randn(d), torch.randn(d)
    mask = torch.zeros(h.shape[1], dtype=torch.bool)
    h_new, logs = iv.swap(h, mask, v_s, v_t, alpha=1.0)
    assert torch.equal(h_new, h)
    assert logs == []


def test_masked_out_positions_bitwise_unchanged():
    d = 8
    h = _rand(d=d, pos=6)
    v_s, v_t = torch.randn(d), torch.randn(d)
    mask = torch.tensor([True, False, True, False, False, False])
    h_new, _ = iv.swap(h, mask, v_s, v_t, alpha=1.0)
    assert torch.equal(h_new[:, ~mask], h[:, ~mask])


def test_random_direction_and_big_nonlabel_hit_target_norms():
    d = 8
    h = _rand(d=d, pos=5)
    mask = torch.ones(5, dtype=torch.bool)
    target_norms = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5])

    h_rd, _ = iv.random_direction(h, mask, target_norms, seed=0, layer=0)
    got_rd = (h_rd - h)[0].norm(dim=-1)
    assert torch.allclose(got_rd, target_norms, atol=1e-3)

    v_a, v_b = torch.randn(d), torch.randn(d)
    h_bn, _ = iv.big_nonlabel(h, mask, v_a, v_b, target_norms, norm_scale=1.0, layer=0)
    got_bn = (h_bn - h)[0].norm(dim=-1)
    assert torch.allclose(got_bn, target_norms, atol=1e-3)


def test_big_nonlabel_leaves_label_coordinate_unchanged():
    d = 8
    h = _rand(d=d, pos=4)
    mask = torch.ones(4, dtype=torch.bool)
    v_a = torch.zeros(d)
    v_a[0] = 1.0
    v_b = torch.zeros(d)
    v_b[1] = 1.0
    v_label = torch.zeros(d)
    v_label[2] = 1.0  # orthogonal to span(v_a, v_b) by construction
    target_norms = torch.ones(4)

    h_new, _ = iv.big_nonlabel(h, mask, v_a, v_b, target_norms, norm_scale=1.0, layer=0)
    c_before = torch.einsum("d,bpd->bp", v_label, h)
    c_after = torch.einsum("d,bpd->bp", v_label, h_new)
    assert torch.allclose(c_before, c_after, atol=1e-5)


def test_label_to_present_logs_finite_delta():
    d = 8
    h = _rand(d=d, pos=3)
    mask = torch.ones(3, dtype=torch.bool)
    v_source = torch.randn(d)
    v_target = torch.randn(d) * 0.3
    _, logs = iv.label_to_present(h, mask, v_source, v_target, alpha=1.0, layer=0)
    assert len(logs) == 3
    assert all(torch.isfinite(torch.tensor(log.delta_h_norm)) for log in logs)


def _trace_setup(standin_model, random_lens):
    import json

    from jlens_spec import prompts as prompts_mod

    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)
    fmt = {"tokenizer": standin_model.tokenizer, "questions": stim["questions"]}
    p = prompts_mod.build_prompt(stim["passages"][0], "report", fmt)
    with standin_model.trace(p.input_ids):
        clean = standin_model.output.logits[0, p.metric_pos].float().save()
    mask = torch.ones(len(p.input_ids), dtype=torch.bool)
    mask[:4] = False
    return p, clean, mask, random_lens.layers[:4]


def test_apply_identity_matches_plain_forward(standin_model, random_lens):
    """The trace + in-place write path with delta = 0 must reproduce the plain forward pass."""
    p, clean, mask, layers = _trace_setup(standin_model, random_lens)
    logits, logs = iv.apply(standin_model, random_lens, p, "identity", layers, mask)
    assert logs
    assert torch.allclose(logits.cpu(), clean.cpu(), atol=1e-4)


def test_apply_edits_actually_propagate(standin_model, random_lens):
    """A large edit at every content position must change the logits at metric_pos. If nnsight's
    in-place write silently didn't propagate, every swap would return clean logits and M3 would
    show a clean-looking null -- this is the guard against that."""
    p, clean, mask, layers = _trace_setup(standin_model, random_lens)
    big = {l: torch.full((len(p.input_ids),), 1e3) for l in layers}
    logits, logs = iv.apply(standin_model, random_lens, p, "random_direction", layers, mask,
                            target_norms=big, seed=0)
    assert logs
    assert not torch.allclose(logits.cpu(), clean.cpu(), atol=1e-2)


def test_apply_measures_where_the_stream_actually_changed(standin_model, random_lens):
    """changes = the size of what was written at EVERY position: exactly zero outside the mask,
    non-zero at every planned position."""
    p, clean, mask, layers = _trace_setup(standin_model, random_lens)
    big = {l: torch.full((len(p.input_ids),), 5.0) for l in layers}
    _, _, changes = iv.apply(standin_model, random_lens, p, "random_direction", layers, mask,
                             return_changes=True, target_norms=big, seed=0)
    assert set(changes) == set(layers) and all(len(v) == len(mask) for v in changes.values())
    assert iv.edit_problems(changes, mask, "random_direction") == ([], [])
    for sizes in changes.values():
        assert all(sz == 0.0 for sz, m in zip(sizes, mask) if not m)
        assert all(sz > 0.0 for sz, m in zip(sizes, mask) if m)


def test_edit_problems_reports_spills_and_unchanged_positions():
    mask = torch.tensor([False, True, True])
    changes = {3: [0.0, 1.0, 0.0], 4: [0.1, 2.0, 5e-8]}
    outside, unchanged = iv.edit_problems(changes, mask, "swap")
    assert outside == [(4, 0, 0.1)]          # changed although not planned -> hard failure for callers
    assert unchanged == [(3, 2)]             # planned but unchanged -> reported
    assert iv.edit_problems({3: [0.0, 0.0, 0.0]}, mask, "identity") == ([], [])  # identity edits nothing


def test_apply_refuses_a_mask_that_does_not_match_the_sequence(standin_model, random_lens):
    p, clean, mask, layers = _trace_setup(standin_model, random_lens)
    with pytest.raises(Exception, match="positions but the mask has"):
        iv.apply(standin_model, random_lens, p, "identity", layers, mask[:-1])


def test_prompts_are_independent_forward_passes(standin_model):
    """Every prompt is its own single forward pass -- no chat history, no KV cache carried between
    traces. Running prompt B right after prompt A must give exactly what B gives on its own."""
    import json

    from jlens_spec import prompts as prompts_mod

    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)
    fmt = {"tokenizer": standin_model.tokenizer, "questions": stim["questions"]}
    a = prompts_mod.build_prompt(stim["passages"][0], "report", fmt)
    b = prompts_mod.build_prompt(stim["passages"][1], "anomaly", fmt)

    def last_logits(p):
        with standin_model.trace(p.input_ids):
            out = standin_model.output.logits[0, p.metric_pos].float().save()
        return out.cpu()

    b_alone = last_logits(b)
    last_logits(a)
    b_after_a = last_logits(b)
    assert torch.equal(b_alone, b_after_a)


@pytest.mark.skip(
    reason="requires the mini-paper's reference `run` implementation to compare against; "
    "not available in this repo -- add this test once the human supplies that reference."
)
def test_apply_matches_reference_run_on_standin():
    pass
