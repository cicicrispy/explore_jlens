"""Automatic control selection (jlens_spec.controls): the measures agree with what the swap itself
logs, and the tiers / "closest" / every-check rules pick what was agreed."""
import numpy as np
import pandas as pd
import pytest
import torch

from jlens_spec import controls
from jlens_spec import interventions as iv
from jlens_spec import runner

QUESTIONS = ("report", "anomaly")
CHECKS = [f"{q}_{r}" for q in QUESTIONS for r in controls.ROWS]


def test_treatment_rows_match_the_runners_direction_mapping():
    cfgs = {"tokens_raw": {"pairs": {"space": [" Spanish", " French"]}}}
    for row in controls.treatment_rows([" Spanish", " French"]):
        cell = runner.Cell(stimulus_id="x", question_key="report", direction=row["direction"], pair_name="space",
                           kind="swap", layers=[1], position_set={"question"}, alpha=1.0, seed=0)
        assert runner._resolve_pair_words(cfgs, cell, row["matrix_lang"]) == (row["removed"], row["swapped_in"])


def test_abs_dc_is_exactly_what_the_swap_logs():
    """|Δc| from dot products (used to pick controls) == delta_c_norm that interventions.swap logs."""
    g = torch.Generator().manual_seed(0)
    d, P = 16, 5
    h = torch.randn(1, P, d, generator=g)
    v_s, v_x = torch.randn(d, generator=g), torch.randn(d, generator=g)
    _, logs = iv.swap(h, torch.ones(P, dtype=torch.bool), v_s, v_x)
    got = controls._abs_dc(v_s @ v_s, v_s @ v_x, v_x @ v_x, h[0] @ v_s, h[0] @ v_x)
    assert torch.allclose(got, torch.tensor([lg.delta_c_norm for lg in logs]), rtol=1e-4, atol=1e-5)


def test_scaled_big_nonlabel_dc_is_the_lens_distance_ratio():
    """What the big_nonlabel rule relies on: once scaled to the treatment's ||Δh||, a pair's |Δc| is
    (||v_s - v_t|| / ||v_a - v_b||) x the treatment's, whatever state h it edits."""
    g = torch.Generator().manual_seed(1)
    d, P = 16, 6
    h, h_other = torch.randn(1, P, d, generator=g), torch.randn(1, P, d, generator=g)
    v_s, v_t, v_a, v_b = (torch.randn(d, generator=g) for _ in range(4))
    mask = torch.ones(P, dtype=torch.bool)
    _, sw = iv.swap(h, mask, v_s, v_t)
    _, bn = iv.big_nonlabel(h_other, mask, v_a, v_b, torch.tensor([lg.delta_h_norm for lg in sw]))
    got = torch.tensor([b.delta_c_norm / s.delta_c_norm for b, s in zip(bn, sw)])
    assert torch.allclose(got, ((v_s - v_t).norm() / (v_a - v_b).norm()).expand(P), rtol=1e-4)


def test_coverage_counts_band_layers_at_edited_positions_only():
    rows = []
    for sid, q in (("sp_01", "report"), ("fr_01", "report")):
        for layer in (0, 1, 2, 3):
            for pos, cls in ((3, "question"), (4, "question"), (9, "matrix")):
                ids = [7, 8] if cls == "question" else [5, 6]
                if layer == 2 and pos == 4:
                    ids = [9, 8]           # token 9 only once: layer 2, position 4
                rows.append({"stimulus_id": sid, "question_key": q, "pos": pos, "class": cls, "layer": layer,
                             "topk_ids": np.array(ids, dtype=np.int32)})
    cov = controls.coverage(pd.DataFrame(rows), {"question"}, band_layers=[1, 2, 3], skip_first=0, pool_k=100)
    got = {(r.stimulus_id, r.token_id): r.n_layers for r in cov.itertuples()}
    assert got[("sp_01", 7)] == 3 and got[("sp_01", 8)] == 3 and got[("sp_01", 9)] == 1
    assert ("sp_01", 5) not in got                     # matrix positions are not edited
    assert controls.coverage(pd.DataFrame(rows), {"question"}, [1, 2, 3], 0, pool_k=1) \
        .query("token_id == 8").empty                   # pool_k=1 keeps only the top token
    by_lang = controls.coverage_by_group(cov, {"es": [("sp_01", "report")], "fr": [("fr_01", "report")]}, 3)
    assert by_lang.loc[7, "cov_es"] == 1.0 and by_lang.loc[9, "cov_es"] == pytest.approx(1 / 3)


def test_special_and_non_roundtrip_tokens_are_kept_apart(standin_model):
    tok = standin_model.tokenizer
    special = tok.all_special_ids[0]
    odd = next(i for i in range(1000) if tok.encode(tok.decode([i]), add_special_tokens=False) != [i])
    ids = [special, odd] + [tok.encode(t, add_special_tokens=False)[0] for t in (" Spanish", " and", " house")]
    info = controls.classify_tokens(ids, tok, [" Spanish", "an"]).set_index("token_id")["excluded_reason"]
    assert info[special] == "special token"
    assert info[odd] == "does not re-tokenize to itself"
    assert info[ids[2]] == "control_ineligible"
    assert info[ids[3]] == "contains a language name"   # " and" contains the blocklisted "an"
    assert info[ids[4]] == ""


def _treat(cov_in=0.5, cov_out=0.5, dc=1.0):
    return {k: {"cov_swapped_in": cov_in, "cov_removed": cov_out, "dc": dc} for k in CHECKS}


def _cands(spec):
    """spec: {name: (coverage, |Δc|[, lowers_by]) in every check, or {"default": (...), check: (...)}}.
    lowers_by (mean c_label - c_token; > 0 = the swap lowers the label) defaults to 1.0."""
    rows = []
    for i, (name, v) in enumerate(spec.items()):
        per = {k: v.get(k, v["default"]) for k in CHECKS} if isinstance(v, dict) else {k: v for k in CHECKS}
        per = {k: (*t, 1.0) if len(t) == 2 else t for k, t in per.items()}
        rows.append({"token_id": i, "token": name, **{f"cov_{k}": per[k][0] for k in CHECKS},
                     **{f"dc_{k}": per[k][1] for k in CHECKS}, **{f"lower_{k}": per[k][2] for k in CHECKS}})
    return pd.DataFrame(rows)


def test_label_to_present_is_picked_per_check_among_tokens_that_lower_the_label():
    cands = _cands({
        "far": (0.6, 3.0),            # tier 1, |Δc| 3x the treatment
        "close": (0.6, 1.1),          # tier 1, closest
        "tier2": (0.4, 1.0),          # coverage 80% of the treatment
        "raises": (0.6, 1.0, -0.2),   # the best size match, but it RAISES the label: last resort only
        "fits_one_check": {"default": (0.6, 5.0), "anomaly_es_i2m": (0.6, 1.01)},
    })
    picks = controls.select_label_to_present(cands, _treat(), n=4, bars=(1.0, 0.75))
    assert set(picks) == set(CHECKS)                                    # every check gets its own
    assert list(picks["report_es_m2i"]["token"]) == ["close", "far", "fits_one_check", "tier2"]
    assert list(picks["report_es_m2i"]["tier"]) == [1, 1, 1, 2]
    assert list(picks["anomaly_es_i2m"]["token"]) == ["fits_one_check", "close", "far", "tier2"]
    five = controls.select_label_to_present(cands, _treat(), n=5, bars=(1.0, 0.75))["report_es_m2i"]
    assert five["token"].iloc[-1] == "raises" and five["tier_label"].iloc[-1] == "does not lower the label"


def test_big_nonlabel_bar_is_checked_for_each_question_not_on_their_average():
    """A big_nonlabel pair serves every check: |Δc| 1.3x the treatment's for report and 0.8x for
    anomaly averages to 1.05x -- but each check counts on its own, so it is tier 2."""
    cands = _cands({"a": (0.9, 1.0), "b": (0.8, 1.0)})
    treat = _treat(cov_in=0.5, cov_out=0.6)
    pool = controls.big_nonlabel_pool(cands, treat, pool_size=2, bars=(1.0, 0.75))
    ids = dict(zip(pool["token"], pool["token_id"]))
    pair_dc = pd.DataFrame([{"a_id": ids["a"], "b_id": ids["b"],
                             **{f"dc_{k}": 1.3 if k.startswith("report") else 0.8 for k in CHECKS}}])
    pick = controls.select_big_nonlabel(pool, pair_dc, treat, n=1, bars=(1.0, 0.75))
    assert pick["tier"].tolist() == [2] and pick["dc_ratio_worst"].tolist() == [pytest.approx(0.8)]


def test_zero_treatment_coverage_is_always_met():
    picks = controls.select_label_to_present(_cands({"x": (0.0, 1.0)}), _treat(cov_in=0.0), n=1, bars=(1.0, 0.75))
    assert all(p["tier"].tolist() == [1] and p["cov_ratio"].tolist() == [np.inf] for p in picks.values())


def test_big_nonlabel_pairs_share_no_token_and_prefer_the_closest():
    cands = _cands({"a": (0.9, 1.0), "b": (0.8, 1.0), "c": (0.7, 1.0), "d": (0.6, 1.0), "low": (0.1, 1.0)})
    treat = _treat(cov_in=0.5, cov_out=0.6)
    pool = controls.big_nonlabel_pool(cands, treat, pool_size=4, bars=(1.0, 0.75))
    assert list(pool["token"]) == ["a", "b", "c", "d"]          # widest coverage first; 'low' left out
    assert list(pool["member_tier"]) == [1, 1, 1, 1]
    ids = dict(zip(pool["token"], pool["token_id"]))
    # dc = the |Δc| each pair's swap will have AFTER scaling (with_delta_c); the treatment's is 1.0
    dcs = {("a", "b"): 1.05, ("a", "c"): 1.01, ("a", "d"): 4.0, ("b", "c"): 1.2, ("b", "d"): 1.02, ("c", "d"): 0.5}
    pair_dc = pd.DataFrame([{"a_id": ids[x], "b_id": ids[y], **{f"dc_{k}": v for k in CHECKS}}
                            for (x, y), v in dcs.items()])
    pick = controls.select_big_nonlabel(pool, pair_dc, treat, n=3, bars=(1.0, 0.75))
    assert [tuple(p) for p in pick[["a_token", "b_token"]].to_numpy()] == [("a", "c"), ("b", "d")]  # disjoint
    assert pick["tier"].tolist() == [1, 1]


def test_with_delta_c_averages_each_check_over_its_own_prompts():
    treat = {f"{q}_{r['row']}": {**r, "question": q, "group": controls.group_name(r["matrix_lang"], q),
                                  "cov_removed": 0.0, "cov_swapped_in": 0.0}
             for q in QUESTIONS for r in controls.treatment_rows(["S", "F"])}
    # 4 prompts (es/report, es/anomaly, fr/report, fr/anomaly), 2 band layers, 1 candidate, 1 pool pair
    est = {"ltp_removed_es": np.array([[1.0], [2.0], [3.0], [4.0]]),
           "ltp_removed_fr": np.array([[10.0], [20.0], [30.0], [40.0]]),
           "ltp_lowering_es": np.array([[0.5], [0.6], [0.7], [0.8]]),
           "ltp_lowering_fr": np.array([[-0.5], [-0.6], [-0.7], [-0.8]]),
           "treat_es_minus_fr": np.array([0.3, 0.3, -0.2, -0.2]),
           "treat_by_layer": np.array([[1.0, 3.0], [2.0, 2.0], [4.0, 0.0], [0.0, 4.0]]),
           "n_positions_x_layers": np.array([2.0, 2.0, 2.0, 2.0]), "treat": np.array([2.0, 2.0, 2.0, 2.0]),
           "treat_dist": np.array([1.0, 1.0]), "pairs": [(0, 1)],
           "pair_dist": np.array([[1.0, 2.0]])}                   # distance ratio per layer: 1.0, 0.5
    groups = ["es_report", "es_anomaly", "fr_report", "fr_anomaly"]
    cands, treat, pairs = controls.with_delta_c(pd.DataFrame({"token_id": [7], "token": ["x"]}), treat, est,
                                                groups, ["S", "F"], [5, 6], norm_scale=1.0)
    # es passage, m2i removes the es word: the candidate's |Δc| comes from the es/report prompt only
    assert cands.loc[0, "dc_report_es_m2i"] == 1.0 and cands.loc[0, "dc_anomaly_es_m2i"] == 2.0
    assert cands.loc[0, "dc_report_es_i2m"] == 10.0 and cands.loc[0, "dc_anomaly_fr_m2i"] == 40.0
    # does it lower the removed label? from the same prompts, for the label that check removes
    assert cands.loc[0, "lower_report_es_m2i"] == 0.5 and cands.loc[0, "lower_report_es_i2m"] == -0.5
    assert treat["report_es_m2i"]["lowers_by"] == pytest.approx(0.3)      # removes es: c_es - c_fr
    assert treat["report_fr_m2i"]["lowers_by"] == pytest.approx(0.2)      # removes fr: c_fr - c_es
    # the pair's scaled |Δc| = sum over layers of (distance ratio x treatment |Δc| there) / count
    assert pairs.loc[0, "dc_report_es_m2i"] == pytest.approx((1.0 * 1.0 + 3.0 * 0.5) / 2)
    assert pairs.loc[0, "dc_report_fr_m2i"] == pytest.approx((4.0 * 1.0 + 0.0 * 0.5) / 2)
    assert pairs.loc[0, "dc_anomaly_fr_i2m"] == pytest.approx((0.0 * 1.0 + 4.0 * 0.5) / 2)
    assert treat["report_es_m2i"]["dc"] == 2.0


def test_estimate_delta_c_matches_a_direct_pinv_computation(standin_model, random_lens):
    import json

    import yaml

    from jlens_spec import lens as lens_mod
    from jlens_spec import model as model_mod
    from jlens_spec import prompts as prompts_mod

    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)
    with open("configs/prompt_format.yaml") as f:
        fmt = prompts_mod.make_fmt(standin_model.tokenizer, stim, yaml.safe_load(f))
    p = prompts_mod.build_prompt(stim["passages"][0], "report", fmt)
    m = prompts_mod.mask(p, {"question"}, skip_first=0)
    tok = standin_model.tokenizer
    pair = [tok.encode(w, add_special_tokens=False)[0] for w in (" Spanish", " French")]
    cands = [tok.encode(w, add_special_tokens=False)[0] for w in (" the", " house")]
    pool = [tok.encode(w, add_special_tokens=False)[0] for w in (" a", " of", " to")]
    layers = random_lens.layers[2:4]
    est = controls.estimate_delta_c(standin_model, random_lens, [p], [m], layers, cands, pair, pool)

    pos = [i for i, on in enumerate(m.tolist()) if on]
    saved = {}
    with standin_model.trace(p.input_ids):
        for l in layers:
            saved[l] = model_mod.layer_output(standin_model, l).float()[0, pos].save()

    def direct(s_id, x_id, signed=False):
        vals = []
        for l in layers:
            V = lens_mod.lens_vectors(standin_model, random_lens, [s_id, x_id], l).T  # [d, 2]
            c = saved[l] @ torch.linalg.pinv(V).T                                   # [P, 2]
            vals.append(c[:, 0] - c[:, 1] if signed else (c[:, [1, 0]] - c).norm(dim=-1))
        return float(torch.cat(vals).mean())

    assert est["treat"][0] == pytest.approx(direct(*pair), rel=1e-3)
    scale = 1e-3 * direct(pair[0], cands[1])      # a signed mean can be near 0: absolute tolerance
    assert est["ltp_lowering_es"][0][1] == pytest.approx(direct(pair[0], cands[1], signed=True), abs=scale)
    assert est["treat_es_minus_fr"][0] == pytest.approx(direct(*pair, signed=True), abs=1e-3 * direct(*pair))
    assert est["treat_by_layer"][0].sum() / est["n_positions_x_layers"][0] == pytest.approx(est["treat"][0])
    assert est["ltp_removed_es"][0][1] == pytest.approx(direct(pair[0], cands[1]), rel=1e-3)
    assert est["ltp_removed_fr"][0][0] == pytest.approx(direct(pair[1], cands[0]), rel=1e-3)
    k = est["pairs"].index((0, 2))
    for li, l in enumerate(layers):
        Vq = lens_mod.lens_vectors(standin_model, random_lens, pool, l).float()
        Vp = lens_mod.lens_vectors(standin_model, random_lens, pair, l).float()
        assert est["pair_dist"][k][li] == pytest.approx(float((Vq[0] - Vq[2]).norm()), rel=1e-3)
        assert est["treat_dist"][li] == pytest.approx(float((Vp[0] - Vp[1]).norm()), rel=1e-3)
