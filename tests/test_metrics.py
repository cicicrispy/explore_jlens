import torch

from jlens_spec import metrics


def test_margin_hand_built():
    logprobs = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
    m = metrics.margin(logprobs, pos_ids=[0, 1], neg_ids=[2, 3])
    expected = torch.logsumexp(logprobs[[0, 1]], dim=0) - torch.logsumexp(logprobs[[2, 3]], dim=0)
    assert abs(m - float(expected)) < 1e-6


def test_answer_ids_dedup(standin_model):
    ids = metrics.answer_ids(standin_model.tokenizer, ["Yes", "No"])
    assert len(ids) == len(set(ids))
    assert len(ids) > 0


_TOKENS_CFG = {
    "yes_ids": [1],
    "no_ids": [2],
    "lang_ids": {"es": [3], "fr": [4]},
    "hello_ids": {"es": [5], "fr": [6]},
}


def _logprobs(vocab=10, winner=1):
    lp = torch.full((vocab,), -10.0)
    lp[winner] = -0.1
    return lp


def test_sign_convention_anomaly_positive_is_yes():
    lp = _logprobs(winner=1)  # "Yes" id
    m = metrics.question_margin("anomaly", lp, _TOKENS_CFG, matrix_lang="es")
    assert m > 0


def test_sign_convention_report_positive_is_target_language():
    lp = _logprobs(winner=4)  # fr id, target when matrix_lang="es"
    m = metrics.question_margin("report", lp, _TOKENS_CFG, matrix_lang="es")
    assert m > 0


def test_flip_true_iff_argmax_changes():
    lp_yes = _logprobs(winner=1)
    lp_no = _logprobs(winner=2)

    label_yes = metrics.argmax_label("anomaly", lp_yes, _TOKENS_CFG, "es")
    label_no = metrics.argmax_label("anomaly", lp_no, _TOKENS_CFG, "es")
    assert label_yes == "yes"
    assert label_no == "no"
    assert label_yes != label_no  # direction 1: flip detected

    lp_no2 = _logprobs(winner=2)
    label_no2 = metrics.argmax_label("anomaly", lp_no2, _TOKENS_CFG, "es")
    assert label_no == label_no2  # no-change case: no flip

    # direction 2: report/hello dispatch
    lp_source = _logprobs(winner=3)  # es id == matrix_lang -> "source"
    lp_target = _logprobs(winner=4)  # fr id == other lang -> "target"
    lbl_source = metrics.argmax_label("report", lp_source, _TOKENS_CFG, "es")
    lbl_target = metrics.argmax_label("report", lp_target, _TOKENS_CFG, "es")
    assert lbl_source == "source"
    assert lbl_target == "target"
    assert lbl_source != lbl_target
