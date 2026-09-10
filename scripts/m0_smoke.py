"""M0 (Phase A, Mac): build all 64 prompts on the stand-in, render mask PNGs, run one `swap` and
one `identity` cell with a random lens, and write runs/M0/summary.md. No GPU, no real model/lens
download -- only configs/model.yaml's standin_hf_id (a small, real Qwen3-family model)."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from jlens_spec import env

env.bootstrap()

from jlens_spec import interventions as iv
from jlens_spec import io as io_mod
from jlens_spec import lens as lens_mod
from jlens_spec import model as model_mod
from jlens_spec import prompts as prompts_mod

RUN_DIR = Path("runs/M0")
MASKS_DIR = RUN_DIR / "figures" / "masks"


def _load_yaml(name: str):
    with open(Path("configs") / name) as f:
        return yaml.safe_load(f)


def _single_token_table(tokenizer, tokens_raw: dict):
    table = []

    def check(label, s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        table.append({"label": label, "text": s, "n_tokens": len(ids), "single_token": len(ids) == 1})
        return len(ids) == 1

    for lang, forms in tokens_raw["language_tokens"].items():
        for f in forms:
            check(f"language_tokens.{lang}", f)

    dropped_pairs, kept_pairs = [], {}
    for name, (a, b) in tokens_raw["pairs"].items():
        ok_a = check(f"pairs.{name}[0]", a)
        ok_b = check(f"pairs.{name}[1]", b)
        if ok_a and ok_b:
            kept_pairs[name] = [a, b]
        else:
            dropped_pairs.append(name)

    for label, forms in tokens_raw["answers"].items():
        if isinstance(forms, dict):
            for lang, fs in forms.items():
                for f in fs:
                    check(f"answers.{label}.{lang}", f)
        else:
            for f in forms:
                check(f"answers.{label}", f)

    return table, dropped_pairs, kept_pairs


def main() -> None:
    model_cfg = _load_yaml("model.yaml")
    tokens_raw = _load_yaml("tokens.yaml")

    model = model_mod.load_model(model_cfg, standin=True)  # device=None -> opportunistic MPS/CPU
    tokenizer = model.tokenizer

    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)

    fmt = {"tokenizer": tokenizer, "questions": stim["questions"]}

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    all_prompts = []
    all_flags = {}
    example_prompt = None
    for stimulus in stim["passages"]:
        for qkey in stim["questions"]:
            p = prompts_mod.build_prompt(stimulus, qkey, fmt)
            all_prompts.append(p)
            if p.flags:
                all_flags[f"{stimulus['id']}/{qkey}"] = p.flags
            if example_prompt is None and stimulus["id"] == "sp_01" and qkey == "report":
                example_prompt = p
            m = prompts_mod.mask(p, {"question"}, skip_first=4)
            prompts_mod.render_mask(p, m, MASKS_DIR / f"{stimulus['id']}_{qkey}.png")

    assert len(all_prompts) == 64, f"expected 64 prompts, built {len(all_prompts)}"
    assert example_prompt is not None

    # Nothing under configs/ is ever written by code. Observed values go to runs/M0/; the human
    # copies anything that should become canonical into configs/ by hand.
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "template_string.txt").write_text(example_prompt.text)

    table, dropped_pairs, kept_pairs = _single_token_table(tokenizer, tokens_raw)
    with open(RUN_DIR / "single_token_check.yaml", "w") as f:
        yaml.safe_dump({"tokenizer": model_cfg["standin_hf_id"], "table": table,
                        "pairs_kept": kept_pairs, "pairs_dropped": dropped_pairs},
                       f, sort_keys=False, allow_unicode=True)

    d = model_mod.d_model(model)
    n_layers = model_mod.n_layers(model)
    layers = list(range(n_layers - 1))
    lens = lens_mod.random_lens(d, layers, seed=0)

    p = example_prompt
    mask = prompts_mod.mask(p, {"question"}, skip_first=4)
    s_ids = tokenizer.encode(" the", add_special_tokens=False)
    t_ids = tokenizer.encode(" a", add_special_tokens=False)
    smoke_layers = layers[: max(1, len(layers) // 4)]

    records = []
    if s_ids and t_ids:
        logits_swap, logs_swap = iv.apply(
            model, lens, p, "swap", smoke_layers, mask, s_token=s_ids[0], t_token=t_ids[0], alpha=1.0
        )
        records.append(("swap", logits_swap, logs_swap))
    logits_id, logs_id = iv.apply(model, lens, p, "identity", smoke_layers, mask)
    records.append(("identity", logits_id, logs_id))

    row_dicts = [
        {
            "stimulus_id": p.stimulus_id,
            "question_key": p.question_key,
            "kind": kind,
            "layers": smoke_layers,
            "prompt_len": len(p.input_ids),
            "metric_pos": p.metric_pos,
            "top1_id": int(torch.argmax(logits)),
            "n_logs": len(logs),
            "git_commit": env.git_commit(),
        }
        for kind, logits, logs in records
    ]
    io_mod.append_records(row_dicts, RUN_DIR / "m0_smoke.parquet")

    summary_lines = [
        "# M0 summary",
        "",
        "## 1. Environment",
        "- Environment: mac (Phase A)",
        f"- git commit: {env.git_commit()}",
        f"- config_hash(model.yaml, tokens.yaml, prompt_format.yaml): "
        f"{env.config_hash('configs/model.yaml', 'configs/tokens.yaml', 'configs/prompt_format.yaml')}",
        "",
        "## 2. What passed by assertion / checked by eye / not checked",
        f"- Built and structurally asserted {len(all_prompts)}/64 stimulus x question prompts "
        "(classes cover all tokens, metric_pos lands on the required suffix).",
        f"- Prompts carrying flags: {len(all_flags)} (see section 4).",
        f"- Pairs that are not single-token under the stand-in tokenizer: {dropped_pairs or 'none'} "
        "(configs/tokens.yaml is NOT modified; see runs/M0/single_token_check.yaml).",
        "- Ran one `swap` cell and one `identity` cell on the stand-in with a random lens; wrote "
        "runs/M0/m0_smoke.parquet (2 rows). NOT checked against any reference numeric output -- no "
        "reference implementation is available in this repo to diff against.",
        "- NOT checked: real model/lens behavior (Phase A does not download either).",
        "",
        "## 3. Figures",
        f"- {len(all_prompts)} mask PNGs written to runs/M0/figures/masks/.",
        "",
        "## 4. Anomalies / open questions",
        f"- Flags by stimulus/question: {json.dumps(all_flags, indent=2)}",
        "- prompts.build_prompt's tokenizer/questions call convention (folded into the `fmt` arg) "
        "is a documented assumption, not explicit in the module contract -- see prompts.py's "
        "module docstring; flag for review.",
        "",
        "## 5. Exact templated prompt string (sp_01 / report, stand-in tokenizer)",
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

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = RUN_DIR / "m0_smoke.parquet"
    sha = io_mod.sha256_of(parquet_path) if parquet_path.exists() else "N/A"
    summary_lines.append(f"- runs/M0/m0_smoke.parquet sha256: {sha}")

    upload_url = None
    try:
        upload_url = io_mod.upload_run(RUN_DIR)
        summary_lines.append(f"- HF dataset upload: {upload_url}")
    except Exception as e:  # noqa: BLE001 -- report, do not hide
        summary_lines.append(f"- HF dataset upload FAILED: {e!r} (HF_TOKEN must be set; see .env)")

    with open(RUN_DIR / "summary.md", "w") as f:
        f.write("\n".join(summary_lines))

    io_mod.write_manifest(
        RUN_DIR, milestone="M0", environment="mac", n_prompts=len(all_prompts), upload_url=upload_url
    )
    print("M0 smoke test complete. See runs/M0/summary.md")


if __name__ == "__main__":
    main()
