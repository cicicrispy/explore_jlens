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


def test_mask_rows_carry_everything_the_mask_figure_draws(standin_model):
    """masks.parquet is the only input to the mask figures, so each row must reproduce the prompt:
    one row per token, in order, with its text, class, edited flag, and exactly one metric_pos."""
    stim = _load_stimuli()
    p = prompts_mod.build_prompt(stim["passages"][0], "report", _fmt(standin_model, stim))
    m = prompts_mod.mask(p, {"question"}, skip_first=4)
    rows = prompts_mod.mask_rows(p, m)

    assert [r["pos"] for r in rows] == list(range(len(p.input_ids)))
    assert [r["token_id"] for r in rows] == p.input_ids
    assert [r["class"] for r in rows] == p.classes
    assert [r["edited"] for r in rows] == m.tolist()
    assert [r["token_text"] for r in rows] == [p.text[s:e] for s, e in p.offsets]
    assert [r["pos"] for r in rows if r["is_metric_pos"]] == [p.metric_pos]


def test_region_tokens_spell_their_text_exactly(standin_model):
    """For every prompt, the tokens labelled "question" must spell EXACTLY the question text, and
    the tokens labelled with a sentence's role (matrix/intrusion) that overlap that sentence must
    spell EXACTLY that sentence -- nothing stripped, no extra characters. A token that also carries
    characters from outside its region (a separator newline, the space before a sentence, template
    text) would be edited along with the region, so every such token must be known. All 64 prompts
    are checked and every mismatch is listed, not just the first."""
    stim = _load_stimuli()
    fmt = _fmt(standin_model, stim)
    problems = []
    for stimulus in stim["passages"]:
        for qkey in stim["questions"]:
            p = prompts_mod.build_prompt(stimulus, qkey, fmt)
            regions = [("question", *p.spans["question"])] + [
                (s["role"], s["char_start"], s["char_end"]) for s in p.spans["sentences"]]
            for cls, a, b in regions:
                toks = [(s, e) for i, (s, e) in enumerate(p.offsets)
                        if p.classes[i] == cls and s < b and e > a]
                got = "".join(p.text[s:e] for s, e in toks)
                want = p.text[a:b]
                if got != want:
                    problems.append(
                        f"{stimulus['id']}/{qkey} {cls} chars [{a}:{b}]: first token "
                        f"{p.text[slice(*toks[0])]!r} (region starts {want[:8]!r}), last token "
                        f"{p.text[slice(*toks[-1])]!r} (region ends {want[-8:]!r})"
                        if toks else f"{stimulus['id']}/{qkey} {cls} chars [{a}:{b}]: no tokens")
    assert not problems, f"{len(problems)} region(s) whose tokens don't spell exactly their text:\n" + \
        "\n".join(problems)
