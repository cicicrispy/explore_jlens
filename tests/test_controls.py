"""Automatic control selection (jlens_spec.controls): the measures agree with what the swap itself
logs, and the tiers / "closest" / shared-across-rows rules pick what was agreed."""
import numpy as np
import pandas as pd
import pytest
import torch

from jlens_spec import controls
from jlens_spec import interventions as iv
from jlens_spec import runner

ROWS = controls.ROWS


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
    by_lang = controls.coverage_by_language(cov, {"es": [("sp_01", "report")], "fr": [("fr_01", "report")]}, 3)
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
    return {r: {"cov_swapped_in": cov_in, "cov_removed": cov_out, "dc": dc} for r in ROWS}


def _cands(spec):
    """spec: {name: (coverage, |Δc|) for every row, or a {row: (coverage, |Δc|)} override dict}."""
    rows = []
    for i, (name, v) in enumerate(spec.items()):
        per_row = v if isinstance(v, dict) else {r: v for r in ROWS}
        rows.append({"token_id": i, "token": name, **{f"cov_{r}": per_row[r][0] for r in ROWS},
                     **{f"dc_{r}": per_row[r][1] for r in ROWS}})
    return pd.DataFrame(rows)


def test_label_to_present_tiers_and_closest():
    cands = _cands({
        "far": (0.6, 3.0),          # tier 1, |Δc| 3x the treatment
        "close": (0.6, 1.1),        # tier 1, closest
        "tier2": (0.4, 1.0),        # coverage 80% of the treatment
        "tier3_a": (0.1, 0.2),      # worst ratio 0.2
        "tier3_b": (0.3, 0.5),      # worst ratio 0.5 -> closer to the bar than tier3_a
        "one_bad_row": {**{r: (0.6, 1.0) for r in ROWS}, "es_i2m": (0.2, 1.0)},  # every row must pass
    })
    pick = controls.select_label_to_present(cands, _treat(), n=6, bars=(1.0, 0.75))
    assert list(pick["token"]) == ["close", "far", "tier2", "tier3_b", "one_bad_row", "tier3_a"]
    assert list(pick["tier"]) == [1, 1, 2, 3, 3, 3]
    assert list(pick["tier_label"][:3]) == [">=100%", ">=100%", ">=75%"]
    assert list(controls.select_label_to_present(cands, _treat(), n=3, bars=(1.0, 0.75))["token"]) == \
        ["close", "far", "tier2"]


def test_zero_treatment_coverage_is_always_met():
    pick = controls.select_label_to_present(_cands({"x": (0.0, 1.0)}), _treat(cov_in=0.0), n=1, bars=(1.0, 0.75))
    assert pick["tier"].tolist() == [1]


def test_big_nonlabel_pairs_share_no_token_and_prefer_the_closest():
    cands = _cands({"a": (0.9, 1.0), "b": (0.8, 1.0), "c": (0.7, 1.0), "d": (0.6, 1.0), "low": (0.1, 1.0)})
    treat = _treat(cov_in=0.5, cov_out=0.6)
    pool = controls.big_nonlabel_pool(cands, treat, pool_size=4, bars=(1.0, 0.75))
    assert list(pool["token"]) == ["a", "b", "c", "d"]          # widest coverage first; 'low' left out
    assert list(pool["member_tier"]) == [1, 1, 1, 1]
    ids = dict(zip(pool["token"], pool["token_id"]))
    dcs = {("a", "b"): 1.05, ("a", "c"): 1.01, ("a", "d"): 4.0, ("b", "c"): 1.2, ("b", "d"): 1.02, ("c", "d"): 2.0}
    pair_dc = pd.DataFrame([{"a_id": ids[x], "b_id": ids[y], **{f"dc_{r}": v for r in ROWS}}
                            for (x, y), v in dcs.items()])
    pick = controls.select_big_nonlabel(pool, pair_dc, treat, n=3)
    assert [tuple(p) for p in pick[["a_token", "b_token"]].to_numpy()] == [("a", "c"), ("b", "d")]  # disjoint


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

    def direct(s_id, x_id):
        vals = []
        for l in layers:
            V = lens_mod.lens_vectors(standin_model, random_lens, [s_id, x_id], l).T  # [d, 2]
            c = saved[l] @ torch.linalg.pinv(V).T                                   # [P, 2]
            vals.append((c[:, [1, 0]] - c).norm(dim=-1))
        return float(torch.cat(vals).mean())

    assert est["treat"][0] == pytest.approx(direct(*pair), rel=1e-3)
    assert est["ltp_removed_es"][0][1] == pytest.approx(direct(pair[0], cands[1]), rel=1e-3)
    assert est["ltp_removed_fr"][0][0] == pytest.approx(direct(pair[1], cands[0]), rel=1e-3)
    k = est["pairs"].index((0, 2))
    assert est["pair_dc"][0][k] == pytest.approx(direct(pool[0], pool[2]), rel=1e-3)
