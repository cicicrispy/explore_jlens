import dataclasses

import torch

from jlens_spec import interventions as iv
from jlens_spec import runner


def _make_swap_record(stimulus_id="sp_01", question_key="anomaly", direction="m2i", pair_name="plain"):
    logs = [
        iv.InterventionLog(
            pos=0, layer=5, c_before=(0.1, 0.2), c_after=(0.2, 0.1),
            delta_c_norm=0.1, delta_h_norm=1.5, alpha=1.0, kind="swap",
        ),
        iv.InterventionLog(
            pos=1, layer=5, c_before=(0.0, 0.0), c_after=(0.0, 0.0),
            delta_c_norm=0.0, delta_h_norm=2.5, alpha=1.0, kind="swap",
        ),
    ]
    return runner.CellRecord(
        stimulus_id=stimulus_id,
        question_key=question_key,
        direction=direction,
        pair_name=pair_name,
        kind="swap",
        layers=[5],
        position_set={"question"},
        alpha=1.0,
        seed=0,
        margin=0.3,
        clean_margin=0.1,
        flip=True,
        top1_changed=True,
        logprobs_fp16=torch.zeros(4, dtype=torch.float16).numpy(),
        topk=[("a", 0.1)],
        intervention_logs=logs,
        prompt_len=3,
        metric_pos=2,
        git_commit="abc",
        config_hash="x",
        lens_sha="y",
        model_revision="z",
        timestamp=0.0,
        answer_label="yes",
        top1_id=7,
    )


def _make_identity_record(**kw):
    r = _make_swap_record(**kw)
    r.kind = "identity"
    r.intervention_logs = []
    r.margin = 0.2
    r.answer_label = "no"
    r.top1_id = 3
    return r


_NORMS_KEY = ("sp_01", "anomaly", "m2i", "plain", (5,), 1.0)
_CLEAN_KEY = ("sp_01", "anomaly")


def test_target_norms_cache_from_records_dataclass():
    cache = runner.target_norms_cache_from_records([_make_swap_record()])
    assert _NORMS_KEY in cache
    assert torch.allclose(cache[_NORMS_KEY][5], torch.tensor([1.5, 2.5, 0.0]))


def test_target_norms_cache_from_records_dict_rows():
    rd = dataclasses.asdict(_make_swap_record())
    cache = runner.target_norms_cache_from_records([rd])
    assert torch.allclose(cache[_NORMS_KEY][5], torch.tensor([1.5, 2.5, 0.0]))


def test_clean_cache_from_records_dataclass_and_dict():
    rec = _make_identity_record()
    cache = runner.clean_cache_from_records([rec])
    assert cache[_CLEAN_KEY].margin == 0.2
    assert cache[_CLEAN_KEY].answer_label == "no"
    assert cache[_CLEAN_KEY].top1_id == 3

    rd = dataclasses.asdict(rec)
    cache2 = runner.clean_cache_from_records([rd])
    assert cache2[_CLEAN_KEY].margin == 0.2


def test_caches_ignore_non_matching_kind():
    assert runner.clean_cache_from_records([_make_swap_record()]) == {}
    assert runner.target_norms_cache_from_records([_make_identity_record()]) == {}


def test_norms_key_separates_layer_sets_and_alphas():
    a = _make_swap_record()
    b = _make_swap_record()
    b.layers = [5, 6]
    c = _make_swap_record()
    c.alpha = 2.0
    keys = {runner._norms_key(r) for r in (a, b, c)}
    assert len(keys) == 3


def test_clean_key_ignores_direction_pair_layers_alpha():
    a = _make_identity_record()
    b = _make_identity_record(direction="i2m", pair_name="space")
    b.layers = [1, 2]
    b.alpha = 2.0
    assert runner._clean_key(a) == runner._clean_key(b) == _CLEAN_KEY


# ----------------------------------------------------------- one prompt, every kind (stand-in)


def _cfgs_and_cells(standin_model, random_lens, n_layers=3):
    import json

    import yaml

    from jlens_spec import metrics
    from jlens_spec import prompts as prompts_mod

    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)
    with open("configs/tokens.yaml") as f:
        tokens_raw = yaml.safe_load(f)
    with open("configs/prompt_format.yaml") as f:
        fmt = prompts_mod.make_fmt(standin_model.tokenizer, stim, yaml.safe_load(f))
    tok = standin_model.tokenizer
    ids = [tok.encode(t, add_special_tokens=False)[0] for t in (" the", " a", " of")]
    cfgs = {"stimuli": {s["id"]: s for s in stim["passages"]}, "fmt": fmt, "tokens_raw": tokens_raw,
            "tokens_cfg": metrics.build_tokens_cfg(tok, tokens_raw), "skip_first": 0, "save_topk": 5,
            "norm_scale": 1.0, "run_id": "test"}
    layers = random_lens.layers[:n_layers]
    common = dict(stimulus_id="sp_01", question_key="report", pair_name="space", layers=layers,
                  position_set={"question"}, alpha=1.0, seed=0)
    cells = []
    for d in ("m2i", "i2m"):
        cells += [runner.Cell(kind=k, direction=d, **common) for k in ("identity", "swap", "random_direction")]
        cells.append(runner.Cell(kind="label_to_present", direction=d, control_index=0, control_tokens=[ids[0]],
                                 control_text=[" the"], control_tier=1, **common))
        cells.append(runner.Cell(kind="big_nonlabel", direction=d, control_index=0, control_tokens=[ids[1], ids[2]],
                                 control_text=[" a", " of"], control_tier=1, **common))
    return cfgs, cells, layers


def test_run_prompt_runs_every_kind_and_records_where_the_stream_changed(standin_model, random_lens):
    import numpy as np
    import pandas as pd

    cfgs, cells, layers = _cfgs_and_cells(standin_model, random_lens)
    records, details = runner.run_prompt(standin_model, random_lens, cells, cfgs)
    assert len(records) == len(cells) == 10
    d = pd.DataFrame(details)
    n = records[0].prompt_len
    assert len(d) == len(cells) * len(layers) * n  # every cell x layer x position
    # Only question positions are planned, and nothing changed anywhere else (else run_cell raises).
    assert set(d.loc[d["planned"], "class"]) == {"question"}
    assert (d.loc[~d["planned"], "change"] == 0).all()
    assert (d.loc[(d["kind"] == "identity"), "change"] == 0).all()

    by = {(r.kind, r.direction): r for r in records}
    # identity is the clean pass: same in both directions, bitwise; every other cell compares to it
    assert np.array_equal(by[("identity", "m2i")].logprobs_fp16, by[("identity", "i2m")].logprobs_fp16)
    assert all(r.clean_margin == by[("identity", "m2i")].margin for r in records)
    # the swap is symmetric in its two tokens: the two directions are the same edit. Stored in 16-bit,
    # they may land on neighbouring 16-bit steps (the 32-bit values differ by rounding, ~1e-4), so
    # every stored value may differ by at most ONE 16-bit step at its size (0.0005 near -0.5, 0.03
    # near -40) -- the storage precision; the strict 32-bit check is the next test.
    a, b = by[("swap", "m2i")].logprobs_fp16, by[("swap", "i2m")].logprobs_fp16
    step = np.spacing(np.maximum(np.abs(a), np.abs(b))).astype(np.float32)
    assert (np.abs(a.astype(np.float32) - b.astype(np.float32)) <= step).all()
    assert by[("swap", "m2i")].s_token == by[("swap", "i2m")].t_token == " Spanish"
    # big_nonlabel and random_direction are scaled to the treatment's ||delta h|| per position/layer
    planned = d[d["planned"]].set_index(["direction", "layer", "pos"])
    for kind in ("big_nonlabel", "random_direction"):
        for direction in ("m2i", "i2m"):
            got = planned[planned["kind"] == kind].loc[direction, "delta_h_norm"]
            want = planned[planned["kind"] == "swap"].loc[direction, "delta_h_norm"]
            assert np.allclose(got.sort_index().to_numpy(), want.sort_index().to_numpy(), rtol=1e-3, atol=1e-4)
    assert records[0].topk and len(records[0].topk) == 5


def test_the_two_swap_directions_are_the_same_edit(standin_model, random_lens):
    """Before any storage rounding: swapping (Spanish, French) and (French, Spanish) must give the
    same 32-bit log-probabilities up to float32 rounding (measured 2026-09-11: 7.3e-5; the tolerance
    is ~14x that)."""
    cfgs, cells, layers = _cfgs_and_cells(standin_model, random_lens)
    from jlens_spec import prompts as prompts_mod

    p = prompts_mod.build_prompt(cfgs["stimuli"]["sp_01"], "report", cfgs["fmt"])
    m = prompts_mod.mask(p, {"question"}, skip_first=0)
    a, _ = iv.apply(standin_model, random_lens, p, "swap", layers, m, s_token=" Spanish", t_token=" French")
    b, _ = iv.apply(standin_model, random_lens, p, "swap", layers, m, s_token=" French", t_token=" Spanish")
    la, lb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    assert float((la - lb).abs().max()) < 1e-3


def test_an_edit_outside_the_plan_is_a_hard_failure(standin_model, random_lens, monkeypatch):
    import pytest

    cfgs, cells, _ = _cfgs_and_cells(standin_model, random_lens, n_layers=1)
    monkeypatch.setattr(iv, "edit_problems", lambda changes, mask, kind: ([(0, 1, 0.5)], []))
    with pytest.raises(RuntimeError, match="unplanned positions"):
        runner.run_cell(standin_model, random_lens, cells[1], cfgs)
