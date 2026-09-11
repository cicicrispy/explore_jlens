import pytest
import torch
import yaml

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


def _clean_and_edited(d=8, P=5, seed=3):
    """A clean state and the 'same' state after earlier band layers' edits and the model's own
    rewrites (here: clean + noise), plus two lens vectors."""
    g = torch.Generator().manual_seed(seed)
    h_clean = torch.randn(1, P, d, generator=g)
    h_now = h_clean + torch.randn(1, P, d, generator=g)
    v_s, v_t = torch.randn(d, generator=g), torch.randn(d, generator=g)
    return h_clean, h_now, v_s, v_t, torch.ones(P, dtype=torch.bool)


def test_a_second_swap_does_not_undo_the_first():
    """The M1 failure (run validate_20260911-201047): flipping the stream's CURRENT coordinates at
    every band layer undoes itself -- flip twice is back to clean. Clamped to the clean run, a
    second application changes nothing."""
    h_clean, _, v_s, v_t, mask = _clean_and_edited()
    once, _ = iv.swap(h_clean, mask, v_s, v_t, h_clean=h_clean)
    twice, _ = iv.swap(once, mask, v_s, v_t, h_clean=h_clean)
    assert torch.allclose(twice, once, atol=1e-5)
    assert not torch.allclose(twice, h_clean, atol=1e-3)


def test_swap_sets_the_clean_runs_swapped_coordinates_whatever_the_stream_holds():
    h_clean, h_now, v_s, v_t, mask = _clean_and_edited()
    V, V_pinv = iv._basis(v_s, v_t)
    P = V @ V_pinv
    c_clean = torch.einsum("bpd,kd->bpk", h_clean, V_pinv)
    for alpha in (1.0, 2.0):
        h_new, logs = iv.swap(h_now, mask, v_s, v_t, alpha=alpha, h_clean=h_clean)
        got = torch.einsum("bpd,kd->bpk", h_new, V_pinv)
        assert torch.allclose(got, c_clean + alpha * (c_clean[..., [1, 0]] - c_clean), atol=1e-4)
        # the rest of THIS stream (outside span(v_s, v_t)) is untouched
        orth = torch.eye(h_now.shape[-1]) - P
        assert torch.allclose(torch.einsum("de,bpe->bpd", orth, h_new),
                              torch.einsum("de,bpe->bpd", orth, h_now), atol=1e-5)
        # logged: the clamp's size on the clean run (not what this stream needed), and the
        # stream's own coordinates before the clamp
        _, on_clean = iv.swap(h_clean, mask, v_s, v_t, alpha=alpha)
        assert [lg.delta_h_norm for lg in logs] == pytest.approx([lg.delta_h_norm for lg in on_clean], rel=1e-5)
        assert [lg.delta_c_norm for lg in logs] == pytest.approx([lg.delta_c_norm for lg in on_clean], rel=1e-5)
        c_now = torch.einsum("bpd,kd->bpk", h_now, V_pinv)
        assert torch.allclose(torch.tensor([lg.c_stream for lg in logs]), c_now[0], atol=1e-5)


def test_the_controls_are_clamps_too():
    """label_to_present, big_nonlabel and random_direction: a second application changes nothing,
    and big_nonlabel / random_direction have the target size on the clean run."""
    h_clean, h_now, v_a, v_b, mask = _clean_and_edited()
    tn = torch.tensor([1.0, 2.0, 0.5, 3.0, 1.5])

    lp1, _ = iv.label_to_present(h_now, mask, v_a, v_b, h_clean=h_clean)
    lp2, _ = iv.label_to_present(lp1, mask, v_a, v_b, h_clean=h_clean)
    assert torch.allclose(lp1, lp2, atol=1e-5)

    bn1, logs = iv.big_nonlabel(h_now, mask, v_a, v_b, tn, h_clean=h_clean)
    bn2, _ = iv.big_nonlabel(bn1, mask, v_a, v_b, tn, h_clean=h_clean)
    assert torch.allclose(bn1, bn2, atol=1e-5)
    assert torch.allclose(torch.tensor([lg.delta_h_norm for lg in logs]), tn, rtol=1e-4)
    bn_clean, _ = iv.big_nonlabel(h_clean, mask, v_a, v_b, tn, h_clean=h_clean)
    assert torch.allclose((bn_clean - h_clean)[0].norm(dim=-1), tn, atol=1e-4)

    rd1, logs = iv.random_direction(h_now, mask, tn, seed=0, layer=0, h_clean=h_clean)
    rd2, _ = iv.random_direction(rd1, mask, tn, seed=0, layer=0, h_clean=h_clean)
    assert torch.allclose(rd1, rd2, atol=1e-5)
    assert torch.allclose(torch.tensor([lg.delta_h_norm for lg in logs]), tn)
    rd_clean, _ = iv.random_direction(h_clean, mask, tn, seed=0, layer=0, h_clean=h_clean)
    assert torch.allclose((rd_clean - h_clean)[0].norm(dim=-1), tn, atol=1e-4)


def _trace_setup(standin_model, random_lens):
    import json

    from jlens_spec import prompts as prompts_mod

    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)
    with open("configs/prompt_format.yaml") as f:
        fmt = prompts_mod.make_fmt(standin_model.tokenizer, stim, yaml.safe_load(f))
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


def test_apply_takes_its_targets_from_the_clean_run_across_the_band(standin_model, random_lens):
    """Inside the trace, every band layer's targets come from the UNEDITED run (not from the stream
    the earlier layers already edited): c_before at each layer = the clean run's coordinates there,
    c_after = those swapped. At the first layer nothing is edited yet, so the stream's own
    coordinates equal the clean ones; at the last, the earlier layers' edits have reached them."""
    from jlens_spec import lens as lens_mod

    p, _, mask, layers = _trace_setup(standin_model, random_lens)
    states = iv.clean_states(standin_model, p, layers)
    _, logs = iv.apply(standin_model, random_lens, p, "swap", layers, mask, s_token=" Spanish", t_token=" French")
    assert {lg.layer for lg in logs} == set(layers)
    for lg in logs:
        v = lens_mod.lens_vectors(standin_model, random_lens, [" Spanish", " French"], lg.layer)
        _, V_pinv = iv._basis(v[0], v[1])
        want = states[lg.layer][0, lg.pos] @ V_pinv.T
        assert torch.allclose(torch.tensor(lg.c_before), want.cpu(), rtol=1e-4, atol=1e-6)
        assert torch.allclose(torch.tensor(lg.c_after), want.cpu()[[1, 0]], rtol=1e-4, atol=1e-6)
        if lg.layer == layers[0]:
            assert torch.allclose(torch.tensor(lg.c_stream), torch.tensor(lg.c_before), rtol=1e-4, atol=1e-6)
    last = [lg for lg in logs if lg.layer == layers[-1]]
    gap = max(float((torch.tensor(lg.c_stream) - torch.tensor(lg.c_before)).abs().max()) for lg in last)
    scale = max(float(torch.tensor(lg.c_before).abs().max()) for lg in last)
    assert gap > 1e-3 * scale


def test_apply_records_the_stream_the_next_layer_receives(standin_model, random_lens):
    """`record`: with no edit, the recorded stream is the clean one; under a swap, the positions
    before the first edited one never change (the model is causal), and after the band the edited
    positions carry the edit on."""
    p, _, mask, layers = _trace_setup(standin_model, random_lens)
    rec = list(random_lens.layers[:6])  # the 4 edited layers and 2 after them
    clean = iv.clean_states(standin_model, p, rec)
    _, _, same = iv.apply(standin_model, random_lens, p, "identity", layers, mask, record=rec)
    assert sorted(same) == rec
    for l in rec:
        assert torch.allclose(same[l], clean[l][0], atol=1e-6)
    _, _, edited = iv.apply(standin_model, random_lens, p, "swap", layers, mask, record=rec,
                            s_token=" Spanish", t_token=" French")
    for l in rec:
        assert torch.allclose(edited[l][~mask], clean[l][0][~mask], atol=1e-6)
    after = rec[-1]
    assert not torch.allclose(edited[after][mask], clean[after][0][mask], atol=1e-3)


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
    with open("configs/prompt_format.yaml") as f:
        fmt = prompts_mod.make_fmt(standin_model.tokenizer, stim, yaml.safe_load(f))
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
