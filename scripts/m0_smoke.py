"""M0 (Phase A, Mac): build all 64 prompts on the stand-in, save their position masks (and draw
them from the saved file), run one `swap` and one `identity` cell with a random lens, and write the
run's summary.md. No GPU, no real model/lens download -- only configs/model.yaml's standin_hf_id (a
small, real Qwen3-family model).

Every run gets its own folder, runs/M0/<experiment name>_<UTC start time>/, holding a copy of the
settings it used (see jlens_spec/runs.py). Settings come from --experiment (default
configs/experiments/m0_smoke.yaml)."""
from __future__ import annotations

import argparse
import json

import torch
import yaml
from tqdm import tqdm

from jlens_spec import env

env.bootstrap()

from jlens_spec import figures
from jlens_spec import interventions as iv
from jlens_spec import io as io_mod
from jlens_spec import lens as lens_mod
from jlens_spec import metrics
from jlens_spec import model as model_mod
from jlens_spec import prompts as prompts_mod
from jlens_spec import runs as runs_mod

DEFAULT_EXPERIMENT = "configs/experiments/m0_smoke.yaml"
# Every settings file M0 reads; copied into the run folder's settings/ when the run starts.
SETTINGS_FILES = ["configs/model.yaml", "configs/tokens.yaml", "configs/prompt_format.yaml",
                  "stimuli/stimuli.json"]


def _step(n: int, msg: str) -> None:
    print(f"[{n}/5] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help=f"experiment file (default {DEFAULT_EXPERIMENT})")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                    help="mps (default on Apple Silicon: loads on CPU, then moves to MPS) or cpu")
    args = ap.parse_args()

    run = runs_mod.start_run("M0", args.experiment, SETTINGS_FILES)
    print(f"Run folder: {run.dir}", flush=True)
    exp = run.experiment
    model_cfg = run.load("model.yaml")
    tokens_raw = run.load("tokens.yaml")
    stim = run.load("stimuli.json")
    position_set, skip_first = set(exp["mask"]["position_set"]), exp["mask"]["skip_first"]
    sc = exp["smoke_cells"]

    _step(1, f"Loading stand-in model {model_cfg['standin_hf_id']} (device={args.device}) ...")
    model = model_mod.load_model(model_cfg, standin=True, device=args.device)
    print(f"      loaded on {model_mod.device_of(model)}", flush=True)
    tokenizer = model.tokenizer
    fmt = {"tokenizer": tokenizer, "questions": stim["questions"]}

    all_prompts = []
    all_flags = {}
    example_prompt = None
    mask_rows = []
    _step(2, "Building 64 prompts and their position masks ...")
    jobs = [(s, q) for s in stim["passages"] for q in stim["questions"]]
    for stimulus, qkey in tqdm(jobs, desc="prompts", unit="prompt"):
        p = prompts_mod.build_prompt(stimulus, qkey, fmt)
        all_prompts.append(p)
        if p.flags:
            all_flags[f"{stimulus['id']}/{qkey}"] = p.flags
        if stimulus["id"] == sc["stimulus"] and qkey == sc["question"]:
            example_prompt = p
        m = prompts_mod.mask(p, position_set, skip_first=skip_first)
        mask_rows += [{**r, "run_id": run.run_id} for r in prompts_mod.mask_rows(p, m)]

    assert len(all_prompts) == 64, f"expected 64 prompts, built {len(all_prompts)}"
    assert example_prompt is not None, f"smoke_cells prompt {sc['stimulus']}/{sc['question']} not found"
    io_mod.append_records(mask_rows, run.dir / "masks.parquet")
    # Built from masks.parquet (not the in-memory prompts) -- same call as scripts/make_figures.py.
    mask_figs = figures.make_figures(run.dir)

    # Nothing under configs/ is ever written by code. Observed values go to the run folder; the
    # human copies anything that should become canonical into configs/ by hand.
    (run.dir / "template_string.txt").write_text(example_prompt.text)

    _step(3, "Single-token check on the stand-in tokenizer ...")
    table, dropped_pairs, kept_pairs = metrics.single_token_table(tokenizer, tokens_raw)
    with open(run.dir / "single_token_check.yaml", "w") as f:
        yaml.safe_dump({"tokenizer": model_cfg["standin_hf_id"], "table": table,
                        "pairs_kept": kept_pairs, "pairs_dropped": dropped_pairs},
                       f, sort_keys=False, allow_unicode=True)

    d = model_mod.d_model(model)
    n_layers = model_mod.n_layers(model)
    layers = list(range(n_layers - 1))
    lens = lens_mod.random_lens(d, layers, seed=sc["lens_seed"])

    p = example_prompt
    mask = prompts_mod.mask(p, position_set, skip_first=skip_first)
    s_ids = tokenizer.encode(sc["swap"][0], add_special_tokens=False)
    t_ids = tokenizer.encode(sc["swap"][1], add_special_tokens=False)
    smoke_layers = layers[: max(1, int(len(layers) * sc["layer_fraction"]))]

    _step(4, f"Running swap + identity cells on {len(smoke_layers)} layers ...")
    records = []
    if s_ids and t_ids:
        logits_swap, logs_swap = iv.apply(
            model, lens, p, "swap", smoke_layers, mask, s_token=s_ids[0], t_token=t_ids[0], alpha=sc["alpha"]
        )
        records.append(("swap", logits_swap, logs_swap))
    logits_id, logs_id = iv.apply(model, lens, p, "identity", smoke_layers, mask)
    records.append(("identity", logits_id, logs_id))

    git_commit = env.git_commit()
    row_dicts = [
        {
            "run_id": run.run_id,
            "stimulus_id": p.stimulus_id,
            "question_key": p.question_key,
            "kind": kind,
            "layers": smoke_layers,
            "prompt_len": len(p.input_ids),
            "metric_pos": p.metric_pos,
            "top1_id": int(torch.argmax(logits)),
            "topk": metrics.topk_tokens(logits, tokenizer, exp["save_topk"]),
            "n_logs": len(logs),
            "git_commit": git_commit,
            "config_hash": run.config_hash(),
        }
        for kind, logits, logs in records
    ]
    io_mod.append_records(row_dicts, run.dir / "m0_smoke.parquet")

    _step(5, f"Writing {run.dir}/summary.md and uploading to the HF dataset (figures excluded) ...")
    summary_lines = [
        f"# M0 summary -- run {run.run_id}",
        "",
        "## 1. Environment",
        "- Environment: mac (Phase A)",
        f"- git commit: {git_commit}",
        f"- experiment file: {args.experiment} (copied to settings/{run.manifest()['experiment_file']})",
        f"- config_hash (every file in settings/): {run.config_hash()}",
        "",
        "## 2. What passed by assertion / checked by eye / not checked",
        f"- Built and structurally asserted {len(all_prompts)}/64 stimulus x question prompts "
        "(classes cover all tokens, metric_pos lands on the required suffix).",
        f"- Prompts carrying flags: {len(all_flags)} (see section 4).",
        f"- Pairs that are not single-token under the stand-in tokenizer: {dropped_pairs or 'none'} "
        "(configs/tokens.yaml is NOT modified; see single_token_check.yaml in this folder).",
        f"- Ran one `swap` cell and one `identity` cell ({p.stimulus_id}/{p.question_key}) on the stand-in "
        f"with a random lens; wrote m0_smoke.parquet ({len(row_dicts)} rows, top-{exp['save_topk']} "
        "tokens each). NOT checked against any reference numeric output -- no reference implementation "
        "is available in this repo to diff against.",
        "- NOT checked: real model/lens behavior (Phase A does not download either).",
        "",
        "## 3. Figures",
        f"- {len(mask_figs)} mask figures in {run.dir}/figures/png/masks/, drawn from masks.parquet. "
        "Figures are NOT uploaded for M0. Redraw them anywhere without the model, e.g. as PDFs "
        f"(written to figures/pdf/): `python scripts/make_figures.py {run.dir} --format pdf`",
        "",
        "## 4. Anomalies / open questions",
        f"- Flags by stimulus/question: {json.dumps(all_flags, indent=2)}",
        "- prompts.build_prompt's tokenizer/questions call convention (folded into the `fmt` arg) "
        "is a documented assumption, not explicit in the module contract -- see prompts.py's "
        "module docstring; flag for review.",
        "",
        f"## Exact templated prompt string ({p.stimulus_id} / {p.question_key}, stand-in tokenizer)",
        "```",
        example_prompt.text,
        "```",
        "",
        "## Single-token table (stand-in tokenizer)",
        "| label | text | n_tokens | single_token |",
        "|---|---|---|---|",
    ]
    for row in table:
        summary_lines.append(
            f"| {row['label']} | `{row['text']!r}` | {row['n_tokens']} | {row['single_token']} |"
        )
    summary_lines += ["", "## 5. Artifact URL, parquet sha256s", ""]
    for name in ("m0_smoke.parquet", "masks.parquet"):
        summary_lines.append(f"- {name} sha256: {io_mod.sha256_of(run.dir / name)}")

    url = io_mod.finalize_run(run.dir, summary_lines, exclude_figures=True)
    if url is None:
        print(f"M0 computed everything, but the upload FAILED, so this run counts as unfinished "
              f"(see {run.dir}/upload_error.txt).")
    else:
        print(f"M0 smoke test complete. See {run.dir}/summary.md")


if __name__ == "__main__":
    main()
