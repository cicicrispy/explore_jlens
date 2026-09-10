"""M2 (Phase B; clean pass, no interventions). Workspace loading + rule-based control selection.
Refuses to run while configs/bands.yaml's `workspace` is null.

NOT executed yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from jlens_spec import env

env.bootstrap()

from jlens_spec import figures
from jlens_spec import io as io_mod
from jlens_spec import lens as lens_mod
from jlens_spec import loading as loading_mod
from jlens_spec import metrics
from jlens_spec import model as model_mod
from jlens_spec import prompts as prompts_mod

RUN_DIR = Path("runs/M2")


def _load_yaml(name: str):
    with open(Path("configs") / name) as f:
        return yaml.safe_load(f)


def _expected_label(question_key: str, stimulus: dict) -> str:
    """Ground truth for the accuracy table, derived from stimuli.json (every passage here has
    exactly one intrusion sentence, so `anomaly`'s correct answer is always 'yes'; `report`/`hello`
    should reflect the matrix language, i.e. metrics.argmax_label's 'source'; `content` reflects
    the passage's `outdoors` field)."""
    if question_key in ("report", "hello"):
        return "source"
    if question_key == "anomaly":
        return "yes"
    if question_key == "content":
        return "yes" if stimulus["outdoors"] else "no"
    raise ValueError(f"unknown question_key {question_key!r}")


def main() -> None:
    env.require_env("HF_TOKEN")
    model_cfg = _load_yaml("model.yaml")
    lens_cfg = _load_yaml("lens.yaml")
    tokens_raw = _load_yaml("tokens.yaml")
    bands = _load_yaml("bands.yaml")

    workspace = bands.get("workspace")
    assert workspace is not None, "configs/bands.yaml workspace is null -- M2 refuses to run"
    layers = env.expand_band(workspace)

    model = model_mod.load_model(model_cfg, standin=False)
    assert lens_cfg.get("revision_sha"), (
        "configs/lens.yaml revision_sha is null -- copy it from runs/M1/lens_resolved.yaml (M1 step 0)"
    )
    lens = lens_mod.load_lens(lens_cfg, device=env.get_device())
    tokenizer = model.tokenizer
    tokens_cfg = metrics.build_tokens_cfg(tokenizer, tokens_raw)

    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)
    fmt = {"tokenizer": tokenizer, "questions": stim["questions"]}

    all_lang_tokens = sorted({t for forms in tokens_raw["language_tokens"].values() for t in forms})
    lang_token_ids = [
        tokenizer.encode(t, add_special_tokens=False)[0]
        for t in all_lang_tokens
        if len(tokenizer.encode(t, add_special_tokens=False)) == 1
    ]

    pairs, dropped_pairs = loading_mod.single_token_pairs(tokenizer, tokens_raw["pairs"])
    assert pairs, f"no pair in configs/tokens.yaml is single-token under this tokenizer (dropped: {dropped_pairs})"

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    loadings_path = RUN_DIR / "loadings.parquet"
    clean_path = RUN_DIR / "clean.parquet"

    # Syncs loadings.parquet to HF as it grows, without blocking the loop below on the network --
    # see io.BackgroundUploader / the README's "Background uploads" section.
    uploader = io_mod.BackgroundUploader(RUN_DIR)

    clean_rows = []
    prompts_built = []
    topk_by_prompt: dict[tuple, set] = {}
    for stimulus in stim["passages"]:
        for qkey in stim["questions"]:
            p = prompts_mod.build_prompt(stimulus, qkey, fmt)
            prompts_built.append(p)

            df = loading_mod.loading(model, lens, p, lang_token_ids, layers)
            io_mod.append_records(df.to_dict("records"), loadings_path)
            uploader.trigger()

            # Control-selection step 1: the lens readout's real top-25 at question positions.
            topk_by_prompt[(p.stimulus_id, qkey)] = loading_mod.question_topk_ids(model, lens, p, layers, k=25)

            with model.trace(p.input_ids):
                logits = model.output.logits[0, p.metric_pos].float().save()
            logprobs = torch.log_softmax(logits, dim=-1)
            margin = metrics.question_margin(qkey, logprobs, tokens_cfg, stimulus["matrix_lang"])
            label = metrics.argmax_label(qkey, logprobs, tokens_cfg, stimulus["matrix_lang"])
            top5_vals, top5_ids = torch.topk(logits, 5)
            top5 = [{"token": tokenizer.decode([int(i)]), "logit": float(v)} for v, i in zip(top5_vals, top5_ids)]

            clean_rows.append(
                {
                    "stimulus_id": stimulus["id"],
                    "question_key": qkey,
                    "matrix_lang": stimulus["matrix_lang"],
                    "prompt_len": len(p.input_ids),
                    "margin": margin,
                    "answer_label": label,
                    "expected_label": _expected_label(qkey, stimulus),
                    "correct": label == _expected_label(qkey, stimulus),
                    "top5": top5,
                }
            )

    io_mod.append_records(clean_rows, clean_path)
    uploader.trigger()
    _, bg_error = uploader.flush()
    uploader.shutdown()
    bg_upload_note = (f"- Background uploads: {uploader.n_uploads} ok, {uploader.n_failures} failed"
                      + (f"; last error: {bg_error}" if bg_error else ""))
    loadings_df = pd.read_parquet(loadings_path)

    (RUN_DIR / "figures").mkdir(parents=True, exist_ok=True)
    for prefix, best_lang in (("sp_", "es"), ("fr_", "fr")):
        example_stim = next(s["id"] for s in stim["passages"] if s["id"].startswith(prefix))
        best_variant = tokens_raw["language_tokens"][best_lang][0]
        figures.loading_heatmap(
            loadings_df, example_stim, best_variant, RUN_DIR / "figures" / f"loading_heatmap_{example_stim}.png"
        )
    figures.loading_summary_bars(loadings_df, layers, RUN_DIR / "figures" / "loading_summary_bars.png")

    pair_scores = {
        name: loading_mod.pair_score(loadings_df, tuple(pair), layers)
        for name, pair in pairs.items()
    }
    # NaN means a pair had no matching loading rows; never let it win (or lose) the argmax silently.
    nan_pairs = sorted(n for n, s in pair_scores.items() if s != s)
    valid_scores = {n: s for n, s in pair_scores.items() if s == s}
    assert valid_scores, f"every pair_score is NaN ({nan_pairs}) -- loading rows don't match pair tokens"
    best_pair = max(valid_scores, key=valid_scores.get)

    # --- Control token selection (rule-based; see the spec's M2 section, steps 1-6) ---
    # Step 1: pool the real top-25 readout at question positions (collected in the loop above), then
    # measure mean cos / mean rank of every pooled candidate at every prompt's question positions.
    n_prompts = len(topk_by_prompt)
    candidate_ids = sorted(set().union(*topk_by_prompt.values()))
    cand_stats = pd.concat(
        [loading_mod.question_token_stats(model, lens, p, candidate_ids, layers) for p in prompts_built],
        ignore_index=True,
    )
    cand_stats.to_parquet(RUN_DIR / "control_candidates.parquet")
    agg = cand_stats.groupby("token_id").agg(token=("token", "first"), mean_cos=("mean_cos", "mean"),
                                             mean_rank=("mean_rank", "mean")).reset_index()
    agg["freq"] = agg["token_id"].map(
        lambda tid: sum(tid in s for s in topk_by_prompt.values()) / n_prompts
    )
    freq_df = agg[["token", "token_id", "freq", "mean_cos", "mean_rank"]]

    def is_single_token(t):
        return len(tokenizer.encode(t, add_special_tokens=False)) == 1

    ineligible = set(tokens_raw["controls"]["control_ineligible"])

    def contains_lang_name(t):
        tl = t.lower()
        return any(name.strip().lower() in tl for name in ineligible if name.strip())

    # Step 2
    eligible_mask = (
        (~freq_df["token"].isin(ineligible))
        & freq_df["token"].apply(is_single_token)
        & (freq_df["freq"] >= 0.5)
        & (~freq_df["token"].apply(contains_lang_name))
    )
    eligible_df = freq_df[eligible_mask].copy()
    excluded_df = freq_df[~eligible_mask].copy()

    # Step 3: label_cos = mean cos of the treatment *target* token at question positions -- the
    # intrusion-language member of the selected pair (" French" on es-matrix prompts, " Spanish" on
    # fr-matrix prompts, for the space pair). "Pair A" is read as the pair chosen by pair_score.
    es_word, fr_word = pairs[best_pair]
    q_df = loadings_df[loadings_df["class"] == "question"].copy()
    q_df["matrix_lang"] = q_df["stimulus_id"].map(loading_mod._matrix_lang_of)
    label_rows = q_df[
        ((q_df["matrix_lang"] == "es") & (q_df["token"] == fr_word))
        | ((q_df["matrix_lang"] == "fr") & (q_df["token"] == es_word))
    ]
    assert len(label_rows), f"no question-position loading rows for pair {best_pair!r}'s target tokens"
    label_cos = float(label_rows["cos"].mean())

    lt_candidates = eligible_df[eligible_df["mean_cos"] < label_cos].sort_values("freq", ascending=False)
    label_to_present_target = str(lt_candidates.iloc[0]["token"]) if len(lt_candidates) else None

    bn_candidates = eligible_df.sort_values("mean_cos", ascending=False)
    big_nonlabel_pair = (
        [str(bn_candidates.iloc[0]["token"]), str(bn_candidates.iloc[1]["token"])]
        if len(bn_candidates) >= 2
        else None
    )

    proposed = {
        "label_to_present_target": label_to_present_target,
        "big_nonlabel_pair": big_nonlabel_pair,
        "label_cos_reference": label_cos,
        "eligible": eligible_df.to_dict("records"),
        "excluded": excluded_df.to_dict("records"),
        "pair_scores": {n: (None if s != s else s) for n, s in pair_scores.items()},
        "pair_score_argmax": best_pair,
        "pairs_nan_score": nan_pairs,
        "pairs_dropped_not_single_token": dropped_pairs,
    }
    with open(RUN_DIR / "controls_proposed.yaml", "w") as f:
        yaml.safe_dump(proposed, f, sort_keys=False, allow_unicode=True)

    accuracy_df = pd.DataFrame(clean_rows)
    accuracy_by_question = accuracy_df.groupby("question_key")["correct"].mean().to_dict()

    summary_lines = [
        "# M2 summary",
        "",
        "## 1. Environment",
        "- Environment: cuda (Phase B)",
        f"- git commit: {env.git_commit()}",
        f"- workspace band: {workspace}",
        "",
        "## 2. What passed by assertion / checked by eye / not checked",
        f"- Computed loading for {len(lang_token_ids)} language tokens x {len(layers)} layers over "
        f"{len(clean_rows)} (stimulus, question) prompts.",
        f"- Accuracy by question: {accuracy_by_question}",
        f"- pair_score table: {pair_scores}; argmax = {best_pair!r}.",
        f"- Control selection: label_to_present.target proposed = {label_to_present_target!r}; "
        f"big_nonlabel.pair proposed = {big_nonlabel_pair!r}."
        + (" NULL for at least one -- human decides." if not (label_to_present_target and big_nonlabel_pair) else ""),
        "- NOT checked: whether the proposed controls are semantically sensible -- that is the "
        "human review step over runs/M2/controls_proposed.yaml.",
        "",
        "## 3. Figures",
        "- runs/M2/figures/loading_heatmap_<stimulus>.png (best es/fr variant)",
        "- runs/M2/figures/loading_summary_bars.png",
        "",
        "## 4. Anomalies / open questions",
        "- This script has not been executed; all numbers above are placeholders pending a real run.",
        "",
        "## 5. Artifact URL, parquet sha256s",
        bg_upload_note,
        f"- runs/M2/loadings.parquet sha256: {io_mod.sha256_of(loadings_path)}",
        f"- runs/M2/clean.parquet sha256: {io_mod.sha256_of(clean_path)}",
    ]

    upload_url = None
    try:
        upload_url = io_mod.upload_run(RUN_DIR)
        summary_lines.append(f"- HF dataset upload: {upload_url}")
    except Exception as e:  # noqa: BLE001
        summary_lines.append(f"- HF dataset upload FAILED: {e!r}")

    with open(RUN_DIR / "summary.md", "w") as f:
        f.write("\n".join(summary_lines))

    io_mod.write_manifest(RUN_DIR, milestone="M2", environment="cuda", upload_url=upload_url)
    print("M2 loading pass complete. See runs/M2/summary.md")


if __name__ == "__main__":
    main()
