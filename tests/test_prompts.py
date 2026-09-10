import json

from jlens_spec import prompts as prompts_mod


def _load_stimuli():
    with open("stimuli/stimuli.json") as f:
        return json.load(f)


def _fmt(standin_model, stim):
    return {"tokenizer": standin_model.tokenizer, "questions": stim["questions"]}


def test_classes_cover_all_tokens(standin_model):
    stim = _load_stimuli()
    stimulus = stim["passages"][0]
    p = prompts_mod.build_prompt(stimulus, "report", _fmt(standin_model, stim))
    assert len(p.classes) == len(p.input_ids)
    assert set(p.classes) <= {"template", "question", "matrix", "intrusion"}


def test_spans_reconstruct_sentences(standin_model):
    stim = _load_stimuli()
    stimulus = stim["passages"][0]
    p = prompts_mod.build_prompt(stimulus, "report", _fmt(standin_model, stim))
    p_start = p.spans["passage"][0]
    for orig, s in zip(stimulus["sentences"], p.spans["sentences"]):
        assert s["char_start"] == p_start + orig["char_start"]
        assert s["char_end"] == p_start + orig["char_end"]
        got = p.text[s["char_start"] : s["char_end"]]
        expected = stimulus["text"][orig["char_start"] : orig["char_end"]]
        assert got == expected


def test_question_span_matches_question_text(standin_model):
    stim = _load_stimuli()
    stimulus = stim["passages"][0]
    p = prompts_mod.build_prompt(stimulus, "hello", _fmt(standin_model, stim))
    q0, q1 = p.spans["question"]
    assert p.text[q0:q1] == stim["questions"]["hello"]


def test_metric_pos_after_prefill(standin_model):
    stim = _load_stimuli()
    stimulus = stim["passages"][0]
    p = prompts_mod.build_prompt(stimulus, "report", _fmt(standin_model, stim))
    assert p.metric_pos == len(p.input_ids) - 1
    assert p.classes[p.metric_pos] == "template"
    assert p.text.endswith(prompts_mod.REQUIRED_SUFFIX)


def test_all_stimuli_all_questions_build(standin_model):
    stim = _load_stimuli()
    fmt = _fmt(standin_model, stim)
    results = []
    for stimulus in stim["passages"]:
        for qkey in stim["questions"]:
            p = prompts_mod.build_prompt(stimulus, qkey, fmt)
            results.append((stimulus["id"], qkey, p.flags))
    assert len(results) == len(stim["passages"]) * len(stim["questions"])
    flagged = [r for r in results if r[2]]
    # Per spec: "builds without flags ... (or flags are listed)" -- not asserted empty here; the
    # smoke script and summary.md are where flags get surfaced to the human. Just confirm we know
    # exactly which ones, if any.
    print(f"{len(flagged)}/{len(results)} stimulus x question builds carried flags: {flagged}")


def test_mask_skips_first_n_positions(standin_model):
    stim = _load_stimuli()
    stimulus = stim["passages"][0]
    p = prompts_mod.build_prompt(stimulus, "report", _fmt(standin_model, stim))
    m = prompts_mod.mask(p, {"question", "matrix", "intrusion", "template"}, skip_first=4)
    assert not bool(m[:4].any())
