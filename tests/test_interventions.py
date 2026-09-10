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


@pytest.mark.skip(
    reason="requires the mini-paper's reference `run` implementation to compare against; "
    "not available in this repo -- add this test once the human supplies that reference."
)
def test_apply_matches_reference_run_on_standin():
    pass
