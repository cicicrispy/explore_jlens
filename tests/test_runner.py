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
        top5=[("a", 0.1)],
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
