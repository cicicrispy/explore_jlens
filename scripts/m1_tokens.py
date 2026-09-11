"""M1 token checks -- run LOCALLY (Mac), before the GPU. Loads only the REAL model's tokenizer and
chat template (small files; no weights) and records, in a new run folder runs/M1/tokens_<time>/:

  - region_check.parquet: for all 64 prompts, whether each region's tokens (question, each sentence)
    spell exactly that region's text (prompts.region_check) -- the check the stand-in test runs;
  - template_string.txt and template_diff.txt: the real chat template vs the M0 run's (stand-in) one;
  - single_token_check.yaml: which configs/tokens.yaml strings are single tokens (spec M1 step 5);
  - antonym_tokens.parquet + antonym_check.yaml: how the positive-control prompt splits into tokens
    (all of which the swap edits) and whether its swap/answer tokens are single tokens.

Settings come from --experiment (default configs/experiments/m1_tokens.yaml)."""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import pandas as pd
import yaml

from jlens_spec import env

env.bootstrap()

from jlens_spec import io as io_mod  # noqa: E402
from jlens_spec import m1_checks  # noqa: E402
from jlens_spec import metrics  # noqa: E402
from jlens_spec import prompts as prompts_mod  # noqa: E402
from jlens_spec import runs as runs_mod  # noqa: E402

DEFAULT_EXPERIMENT = "configs/experiments/m1_tokens.yaml"
SETTINGS_FILES = ["configs/model.yaml", "configs/tokens.yaml", "configs/prompt_format.yaml",
                  "stimuli/stimuli.json"]


def _m0_template(m0_run: str) -> tuple[str, str]:
    """(text, where it came from): the M0 run's template_string.txt, from this machine if present,
    otherwise from the HF dataset."""
    rel = f"runs/M0/{m0_run}/template_string.txt"
    if Path(rel).exists():
        return Path(rel).read_text(), rel
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(io_mod.HF_DATASET, rel, repo_type="dataset")
    return Path(local).read_text(), f"{rel} (downloaded from the HF dataset)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help=f"experiment file (default {DEFAULT_EXPERIMENT})")
    args = ap.parse_args()
    with open(args.experiment) as f:
        validate_path = yaml.safe_load(f)["validate_experiment"]

    run = runs_mod.start_run("M1", args.experiment, SETTINGS_FILES + [validate_path])
    print(f"Run folder: {run.dir}", flush=True)
    exp = run.experiment
    model_cfg = run.load("model.yaml")
    tokens_raw = run.load("tokens.yaml")
    stim = run.load("stimuli.json")
    pc = run.load(Path(validate_path).name)["check3_positive_control"]

    from transformers import AutoTokenizer

    hf_id, revision = model_cfg["hf_id"], model_cfg["revision"]
    print(f"[1/5] Loading the tokenizer of {hf_id} @ {revision} (tokenizer files only, no weights) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    assert "qwen" in hf_id.lower() or "qwen" in type(tok).__name__.lower(), f"not a Qwen tokenizer: {type(tok).__name__}"

    print("[2/5] Building the 64 prompts and checking every region's tokens ...", flush=True)
    fmt = prompts_mod.make_fmt(tok, stim, run.load("prompt_format.yaml"))
    prompts = [prompts_mod.build_prompt(s, q, fmt) for s in stim["passages"] for q in stim["questions"]]
    flags = {f"{p.stimulus_id}/{p.question_key}": p.flags for p in prompts if p.flags}
    rows = [{**r, "run_id": run.run_id} for p in prompts for r in prompts_mod.region_check(p)]
    region = pd.DataFrame(rows)
    region.to_parquet(run.dir / "region_check.parquet")
    bad = region[~region["match"]]
    question_nl = bad[(bad["region"] == "question") & (bad["tokens_text"] == bad["region_text"] + "\n\n")]
    lead_space = bad[(bad["region"] != "question") & (bad["tokens_text"] == " " + bad["region_text"])]
    other = bad.drop(question_nl.index).drop(lead_space.index)

    print("[3/5] Template string vs the M0 run ...", flush=True)
    example = next(p for p in prompts if p.stimulus_id == "sp_01" and p.question_key == "report")
    (run.dir / "template_string.txt").write_text(example.text)
    m0_text, m0_source = _m0_template(exp["m0_run"])
    diff = "".join(difflib.unified_diff(m0_text.splitlines(True), example.text.splitlines(True),
                                        f"M0 {exp['m0_run']} (stand-in)", f"{hf_id} (real)"))
    (run.dir / "template_diff.txt").write_text(diff)

    print("[4/5] Single-token table on the real tokenizer ...", flush=True)
    table, dropped_pairs, kept_pairs = metrics.single_token_table(tok, tokens_raw)
    with open(run.dir / "single_token_check.yaml", "w") as f:
        yaml.safe_dump({"tokenizer": hf_id, "table": table, "pairs_kept": kept_pairs,
                        "pairs_dropped": dropped_pairs}, f, sort_keys=False, allow_unicode=True)

    print("[5/5] Positive-control prompt tokens ...", flush=True)
    enc = tok(pc["prompt"], add_special_tokens=False, return_offsets_mapping=True)
    ant = pd.DataFrame([{"pos": i, "token_id": int(t), "token_text": pc["prompt"][s:e], "run_id": run.run_id}
                        for i, (t, (s, e)) in enumerate(zip(enc["input_ids"], enc["offset_mapping"]))])
    ant.to_parquet(run.dir / "antonym_tokens.parquet")
    n_tok = {v: len(tok.encode(v, add_special_tokens=False))
             for v in [pc["expect_clean"], pc["expect_swapped"]]
             + [v for pr in pc["pairs"] for side in ("source", "target") for v in pr[side]]}
    chosen = [[m1_checks.first_single_token(tok, pr["source"]), m1_checks.first_single_token(tok, pr["target"])]
              for pr in pc["pairs"]]
    with open(run.dir / "antonym_check.yaml", "w") as f:
        yaml.safe_dump({"prompt": pc["prompt"], "n_tokens_by_string": n_tok, "pairs_used": chosen},
                       f, sort_keys=False, allow_unicode=True)

    def listing(df):
        return [f"  - {r.stimulus_id}/{r.question_key} {r.region}: first token {r.first_token!r}, "
                f"last token {r.last_token!r}" for r in df.itertuples()]

    summary_lines = [
        f"# M1 token checks -- run {run.run_id}", "",
        "## 1. Environment",
        "- Environment: local (tokenizer only, no model weights)",
        f"- tokenizer: {hf_id} @ {revision} ({type(tok).__name__})",
        f"- git commit: {run.manifest()['git_commit']}",
        f"- experiment file: {args.experiment}; config_hash (every file in settings/): {run.config_hash()}", "",
        "## 2. Region check (every region's tokens must spell exactly its text)",
        f"- {len(prompts)} prompts, {len(region)} regions; {len(bad)} don't match exactly "
        "(full table: region_check.parquet).",
        f"- {len(question_nl)} are the question's last token carrying the blank line before the passage "
        "(tokens = question + '\\n\\n').",
        f"- {len(lead_space)} are a sentence's first token carrying the space before it (tokens = ' ' + sentence).",
        f"- {len(other)} are something else" + (":" if len(other) else "."),
        *listing(other),
        "- Expected: 0. Known tokenizer boundaries are documented in README 'Known open items' and "
        "stimuli/notes.md.",
        f"- Prompts carrying build flags: {flags or 'none'}", "",
        "## 3. Template string (real tokenizer) vs M0",
        f"- Compared with {m0_source}: " + ("identical." if not diff else "DIFFERENT -- see template_diff.txt:"),
        *(["```diff", diff, "```"] if diff else []), "",
        "## 4. Single-token table (spec M1 step 5)",
        f"- Pairs NOT single-token under the real tokenizer: {dropped_pairs or 'none'} "
        "(configs/tokens.yaml is not modified; see single_token_check.yaml).", "",
        "## 5. Positive-control prompt (swap applied at every one of these positions)",
        f"- `{pc['prompt']}` -> {len(ant)} tokens: " + " | ".join(repr(t) for t in ant["token_text"]),
        f"- tokens per string: {n_tok}",
        f"- pairs used (first single-token variant): {chosen}", "",
        "## 6. Artifact URL, parquet sha256s",
        *[f"- {n} sha256: {io_mod.sha256_of(run.dir / n)}" for n in ("region_check.parquet", "antonym_tokens.parquet")],
    ]
    url = io_mod.finalize_run(run.dir, summary_lines)
    if url is None:
        print(f"Token checks computed, but the upload FAILED, so this run counts as unfinished "
              f"(see {run.dir}/upload_error.txt).")
    else:
        print(f"M1 token checks complete. See {run.dir}/summary.md")


if __name__ == "__main__":
    main()
