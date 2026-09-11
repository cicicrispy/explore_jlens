"""M1 checks (jlens_spec.m1_checks) on the stand-in model with a random lens: the plumbing, plus the
parts whose right answer is known without the real lens."""
import json

import pandas as pd
import yaml

from jlens_spec import lens as lens_mod
from jlens_spec import m1_checks
from jlens_spec import model as model_mod
from jlens_spec import prompts as prompts_mod


def _prompts(standin_model):
    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)
    with open("configs/prompt_format.yaml") as f:
        fmt = prompts_mod.make_fmt(standin_model.tokenizer, stim, yaml.safe_load(f))
    return [prompts_mod.build_prompt(s, q, fmt) for s in stim["passages"] for q in stim["questions"]]


def _small_lens(standin_model, layers=(0, 5, 10)):
    return lens_mod.random_lens(model_mod.d_model(standin_model), list(layers), seed=1)


def test_check1_final_layer_readout_is_the_models_own_output(standin_model, random_lens):
    """At the final layer the lens is the identity, so its readout must BE the model's output:
    top-10 overlap 10/10 at every position, and (almost) identical logits."""
    rows = m1_checks.check1_final_layer(standin_model, random_lens, _prompts(standin_model)[:2],
                                        n_positions=6, seed=0, save_k=20)
    df = pd.DataFrame(rows)
    s = m1_checks.summarize_check1(df, k=10)
    assert s["n_positions"] == 6 and s["min"] == 10
    assert s["max_abs_logit_diff"] < 1e-3
    assert all(len(ids) == 20 for ids in df["lens_topk_ids"])


def test_check2_saves_topk_at_every_matrix_position_and_sampled_layer(standin_model):
    lens = _small_lens(standin_model, layers=(0, 3, 6, 9))
    p = _prompts(standin_model)[0]
    rows = m1_checks.check2_readout(standin_model, lens, p, n_layers=2, save_k=7)
    matrix = [i for i, c in enumerate(p.classes) if c == "matrix"]
    assert {r["layer"] for r in rows} == {0, 9}  # evenly spaced: first and last lens layer
    assert len(rows) == 2 * len(matrix)
    assert all(len(r["topk_ids"]) == 7 == len(r["topk_text"]) for r in rows)
    s = m1_checks.summarize_check2(pd.DataFrame(rows), lang_ids={rows[0]["topk_ids"][0]}, k=3)
    assert set(s["fraction_by_layer"]) == {0, 9}


def test_check3_runs_the_papers_control_at_every_position(standin_model, random_lens):
    with open("configs/experiments/m1_validate.yaml") as f:
        cfg = yaml.safe_load(f)["check3_positive_control"]
    r = m1_checks.check3_positive_control(standin_model, random_lens, cfg, save_k=10)
    info, n = r["info"], r["info"]["n_tokens"]

    assert "".join(info["tokens"]) == cfg["prompt"]  # the raw prompt, exactly
    assert r["results"][0]["condition"] == "clean"
    swaps = r["results"][1:]
    # alpha 2 runs only if alpha 1 failed
    assert len(swaps) == (1 if swaps[0]["top1_is_expected_swapped"] else 2)
    assert info["alphas_run"] == cfg["alphas"][:len(swaps)]
    # every position is planned and, with a random lens, every one actually changed
    assert len(r["masks"]) == n * len(swaps) and all(m["edited"] for m in r["masks"])
    assert len(r["changes"]) == len(swaps) * len(info["layers"]) * n
    depth = model_mod.n_layers(standin_model)
    assert all(int(0.25 * depth) <= l < int(0.75 * depth) for l in info["layers"])


def test_check4_ranks_and_kurtosis_per_position_and_layer(standin_model):
    lens = _small_lens(standin_model)
    p = _prompts(standin_model)[0]
    df = m1_checks.check4_positions(standin_model, lens, [p])
    content = [i for i, c in enumerate(p.classes) if c != "template"]
    assert len(df) == len(content) * 3
    assert (df["rank_of_actual"] >= 1).all() and df["kurtosis"].notna().all()
    cka = pd.DataFrame(m1_checks.cka_rows(standin_model, lens, n_tokens=50, seed=0))
    assert len(cka) == 9
    diag = cka[cka["layer_a"] == cka["layer_b"]]["cka"]
    assert ((diag - 1).abs() < 1e-4).all()


def test_band_signatures_hand_computed():
    pos = pd.DataFrame({"layer": [1, 1, 1, 1, 2, 2, 2, 2],
                        "rank_of_actual": [1, 2, 6, 1, 1, 1, 1, 9],
                        "kurtosis": [1.0, 3.0, 5.0, 7.0, 0.0, 0.0, 2.0, 2.0]})
    cka = pd.DataFrame({"layer_a": [1, 1, 2, 2], "layer_b": [1, 2, 1, 2], "cka": [1.0, 0.4, 0.4, 1.0]})
    df = m1_checks.band_signatures(pos, cka, agreement_k=5)
    assert list(df.columns) == ["layer", "cka_onset_score", "agreement_top1", "agreement_top5", "kurtosis"]
    assert df["agreement_top1"].tolist() == [0.5, 0.75]
    assert df["agreement_top5"].tolist() == [0.75, 0.75]
    assert df["kurtosis"].tolist() == [4.0, 1.0]
    assert df["cka_onset_score"].iloc[0] == 0.4 and pd.isna(df["cka_onset_score"].iloc[1])


def test_motor_onset_is_first_layer_above_the_midpoint():
    df = pd.DataFrame({"layer": list(range(10)),
                       "agreement_top1": [0, 0, 0, 0, 0, 0, 0, 0.5, 0.9, 1.0]})
    # median over layers 25-60% = 0, last = 1.0 -> threshold 0.5; first layer strictly above it: 8
    assert m1_checks.motor_onset(df, (0.25, 0.60)) == 8
    assert m1_checks.motor_onset(df.assign(agreement_top1=0.0), (0.25, 0.60)) is None
