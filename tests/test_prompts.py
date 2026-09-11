import json

import pytest
import yaml

from jlens_spec import prompts as prompts_mod


def _load_stimuli():
    with open("stimuli/stimuli.json") as f:
        return json.load(f)


def _prompt_format():
    with open("configs/prompt_format.yaml") as f:
        return yaml.safe_load(f)


def _fmt(standin_model, stim):
    return prompts_mod.make_fmt(standin_model.tokenizer, stim, _prompt_format())


def test_classes_cover_all_tokens(standin_model):
    stim = _load_stimuli()
    stimulus = stim["passages"][0]
    p = prompts_mod.build_prompt(stimulus, "report", _fmt(standin_model, stim))
    assert len(p.classes) == len(p.input_ids)
    assert set(p.classes) <= {"template", "instruction", "question", "matrix", "intrusion"}


def test_user_message_is_the_papers_wrapper_around_question_and_passage(standin_model):
    """The user turn is exactly configs/prompt_format.yaml's user_message with the question and
    passage filled in (the paper's Figure 20 wrapper), and everything else in it is 'instruction'
    text: the two instruction sentences and the blank lines between the parts."""
    stim = _load_stimuli()
    stimulus = stim["passages"][0]
    p = prompts_mod.build_prompt(stimulus, "anomaly", _fmt(standin_model, stim))
    q = stim["questions"]["anomaly"]
    message = ("I will show you a passage of text. After reading it, answer the following.\n\n"
               f"{q}\n\nHere is the passage:\n\n{stimulus['text']}")
    assert message in p.text
    start = p.text.index(message)
    q0, q1 = p.spans["question"]
    p0, p1 = p.spans["passage"]
    assert p.text[q0:q1] == q and p.text[p0:p1] == stimulus["text"]
    assert p.spans["instruction"] == [[start, q0], [q1, p0]]  # nothing after the passage
    assert p.text[start:q0] == "I will show you a passage of text. After reading it, answer the following.\n\n"
    assert p.text[q1:p0] == "\n\nHere is the passage:\n\n"


def test_fill_user_message_rejects_bad_templates():
    assert prompts_mod.fill_user_message("A {question} B {passage} C", "q", "p") == ("A q B p C", 2, 6)
    for bad in ("{question} only", "{passage} {question}", "{question} {question} {passage}"):
        with pytest.raises(ValueError):
            prompts_mod.fill_user_message(bad, "q", "p")


def test_position_sets_never_include_template_tokens(standin_model):
    """M3's two position sets: 'question' = exactly the question-class tokens; 'message' = every
    token of the user message -- and neither ever includes a chat-template token."""
    stim = _load_stimuli()
    p = prompts_mod.build_prompt(stim["passages"][0], "report", _fmt(standin_model, stim))
    for name, classes in prompts_mod.POSITION_SETS.items():
        m = prompts_mod.mask(p, set(classes), skip_first=0)
        assert all(p.classes[i] != "template" for i in range(len(m)) if m[i]), name
        assert all(bool(m[i]) == (p.classes[i] in classes) for i in range(len(m))), name
    q = prompts_mod.mask(p, set(prompts_mod.POSITION_SETS["question"]), skip_first=0)
    assert q.sum() == sum(c == "question" for c in p.classes) > 0


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


# Characters a tokenizer is KNOWN to add to a region, beyond the region's exact text, as
# (before, after) -- decided with the human; anything else is a failure. Regions: the question, each
# sentence, and (since the paper's wrapper, 2026-09-11) each piece of instruction text. The stand-in
# (M0) merged the question's final period with the blank line after it into one token ".\n\n" -- with
# the wrapper that blank line is instruction text, so the stand-in case may now fail differently; the
# human decided (2026-09-11) that the stand-in case may fail: only the real tokenizer matters from M1
# on. The real model's tokenizer ends the question with a plain ".". Sentence spans in stimuli.json
# include the one space before sentences 2-5 (the tokenizers attach it to the next word).
KNOWN_EXTRA = {
    "standin": {"question": ("", "\n\n")},
    "real": {},
}


@pytest.mark.parametrize("which", ["standin", "real"])
def test_region_tokens_spell_their_text_exactly(which, request):
    """For every prompt, the tokens labelled "question" must spell EXACTLY the question text, and
    the tokens labelled with a sentence's role (matrix/intrusion) that overlap that sentence must
    spell EXACTLY that sentence -- nothing stripped, no extra characters beyond KNOWN_EXTRA. A token
    that also carries characters from outside its region would be edited along with the region, so
    every such token must be known. Run for the stand-in tokenizer (M0) and the real one (M1). All
    64 prompts are checked and every mismatch is listed, not just the first."""
    tok = (request.getfixturevalue("standin_model").tokenizer if which == "standin"
           else request.getfixturevalue("real_tokenizer"))
    stim = _load_stimuli()
    fmt = prompts_mod.make_fmt(tok, stim, _prompt_format())
    problems = []
    for stimulus in stim["passages"]:
        for qkey in stim["questions"]:
            p = prompts_mod.build_prompt(stimulus, qkey, fmt)
            for r in prompts_mod.region_check(p):
                before, after = KNOWN_EXTRA[which].get(r["region"], ("", ""))
                if r["tokens_text"] != before + r["region_text"] + after:
                    problems.append(
                        f"{r['stimulus_id']}/{r['question_key']} {r['region']} chars "
                        f"[{r['char_start']}:{r['char_end']}]: first token {r['first_token']!r} "
                        f"(region starts {r['region_text'][:8]!r}), last token {r['last_token']!r} "
                        f"(region ends {r['region_text'][-8:]!r})"
                        if r["n_tokens"] else
                        f"{r['stimulus_id']}/{r['question_key']} {r['region']} chars "
                        f"[{r['char_start']}:{r['char_end']}]: no tokens")
    assert not problems, f"{len(problems)} region(s) whose tokens don't spell exactly their text:\n" + \
        "\n".join(problems)
