"""M1 (Phase B, GPU box; after M0 sign-off and scripts/download.py). Lens validation:
final-layer agreement, readout reproduction, the causal positive control, band signatures, and the
real-tokenizer single-token check. Writes only under runs/M1/ -- nothing under configs/.

NOT executed yet.
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from jlens_spec import env

env.bootstrap()

from jlens_spec import cka as cka_mod  # noqa: E402
from jlens_spec import figures  # noqa: E402
from jlens_spec import interventions as iv  # noqa: E402
from jlens_spec import io as io_mod  # noqa: E402
from jlens_spec import lens as lens_mod  # noqa: E402
from jlens_spec import model as model_mod  # noqa: E402
from jlens_spec import prompts as prompts_mod  # noqa: E402

sys.path.insert(0, "scripts")
from m0_smoke import _single_token_table  # noqa: E402

RUN_DIR = Path("runs/M1")

# TODO(human): replace with the paper's exact antonym prompt if it differs. The lead-in exists so
# that 小 is not among the first 4 positions, which skip_first=4 always leaves unedited.
ANTONYM_USER_TEXT = "请回答下面的问题。小的反义词是什么？只用一个字回答。"


def _load_yaml(name: str):
    with open(Path("configs") / name) as f:
        return yaml.safe_load(f)


def check1_final_layer_agreement(model, lens, prompts, n_positions: int = 50):
    """Top-10 overlap between the lens readout at the final layer (J = I) and the model's ACTUAL
    output logits at the same positions. This catches a wrong/missing final norm or a broken
    envoy call in unembed(); comparing against unembed() itself would pass by construction."""
    final_layer = model_mod.n_layers(model) - 1
    g = torch.Generator().manual_seed(0)
    overlaps, max_abs = [], []
    per_prompt = max(1, n_positions // len(prompts))
    for p in prompts:
        pos = torch.randperm(len(p.input_ids), generator=g)[:per_prompt].tolist()
        with model.trace(p.input_ids):
            h = model_mod.layer_output(model, final_layer).float()[0, pos].save()
            real = model.output.logits[0, pos].float().save()
        lens_ids, _ = lens_mod.readout(model, lens, h, final_layer, k=10)
        lens_logits = lens_mod._transport_unembed(model, lens, h, final_layer).float()
        real_ids = torch.topk(real, 10, dim=-1).indices
        for i in range(len(pos)):
            overlaps.append(len(set(lens_ids[i].tolist()) & set(real_ids[i].tolist())))
        max_abs.append(float((lens_logits - real).abs().max()))
    return overlaps, max(max_abs)


def check2_readout_reproduction(model, lens, stimulus, fmt, lang_ids: set, n_layers_sample: int = 8):
    p = prompts_mod.build_prompt(stimulus, "report", fmt)
    matrix_positions = [i for i, c in enumerate(p.classes) if c == "matrix"]
    idx = np.linspace(0, len(lens.layers) - 1, n_layers_sample).astype(int)
    sampled_layers = [lens.layers[i] for i in idx]

    saved = {}
    with model.trace(p.input_ids):
        for l in sampled_layers:
            saved[l] = model_mod.layer_output(model, l).float()[0, matrix_positions].save()

    tok = model.tokenizer
    grid = np.empty((len(matrix_positions), len(sampled_layers)), dtype=object)
    es_top3_frac = {}
    for j, l in enumerate(sampled_layers):
        ids, _ = lens_mod.readout(model, lens, saved[l], l, k=5)
        for i in range(len(matrix_positions)):
            grid[i, j] = tok.decode([int(ids[i, 0])])
        es_top3_frac[l] = float(np.mean([bool(set(ids[i, :3].tolist()) & lang_ids) for i in range(len(matrix_positions))]))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.3 * len(sampled_layers) + 2, 0.28 * len(matrix_positions) + 1))
    ax.axis("off")
    table = ax.table(cellText=grid, rowLabels=matrix_positions, colLabels=sampled_layers, loc="center")
    table.set_fontsize(6)
    ax.set_title("sp_01/report: lens top-1 at matrix positions (rows=pos, cols=layer)")
    (RUN_DIR / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RUN_DIR / "figures" / "readout_top1_sp01.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return sampled_layers, es_top3_frac


def check3_causal_positive_control(model, lens):
    """Antonym of 小 should be 大; swapping big->long and bigger->longer at the user-content
    positions across layers 25-75% of depth should make 长 top-1."""
    tok = model.tokenizer
    templated = tok.apply_chat_template(
        [{"role": "user", "content": ANTONYM_USER_TEXT}],
        add_generation_prompt=True, enable_thinking=False, tokenize=False,
    )
    if not templated.endswith(prompts_mod.REQUIRED_SUFFIX):
        templated += prompts_mod.MANUAL_THINK_PREFILL
    enc = tok(templated, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    c0 = templated.find(ANTONYM_USER_TEXT)
    c1 = c0 + len(ANTONYM_USER_TEXT)
    content = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > c0 and s < c1]
    mask = torch.zeros(len(input_ids), dtype=torch.bool)
    mask[content] = True
    mask[:4] = False
    assert mask.any(), "positive control has no editable positions -- the swap would be a no-op"
    small_id = tok.encode("小", add_special_tokens=False)[0]
    small_pos = [i for i in content if input_ids[i] == small_id]
    assert small_pos and all(mask[i] for i in small_pos), "小 is not inside the edited positions"

    def single(options):
        return next((w for w in options if len(tok.encode(w, add_special_tokens=False)) == 1), None)

    pairs = [(single([" big", "big"]), single([" long", "long"])),
             (single([" bigger", "bigger"]), single([" longer", "longer"]))]
    pairs = [pr for pr in pairs if pr[0] and pr[1]]
    assert pairs, "neither big->long nor bigger->longer has single-token variants"

    n = model_mod.n_layers(model)
    layers = [l for l in lens.layers if int(0.25 * n) <= l < int(0.75 * n)]

    class _P:
        pass

    p = _P()
    p.input_ids, p.metric_pos = input_ids, len(input_ids) - 1

    def top5(logits):
        t = torch.topk(logits, 5)
        return [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(t.values, t.indices)]

    with model.trace(input_ids):
        clean_logits = model.output.logits[0, p.metric_pos].float().save()
    da_id = tok.encode("大", add_special_tokens=False)[0]
    chang_id = tok.encode("长", add_special_tokens=False)[0]

    results = {}
    for alpha in (1.0, 2.0):
        logits, _ = iv.apply(model, lens, p, "swap", layers, mask, pairs=pairs, alpha=alpha)
        results[alpha] = {"top5": top5(logits), "top1_is_chang": int(torch.argmax(logits)) == chang_id}
    info = {
        "prompt": templated, "pairs": pairs, "layers": [layers[0], layers[-1]],
        "edited_positions": mask.nonzero().flatten().tolist(),
        "clean_top5": top5(clean_logits), "clean_top1_is_da": int(torch.argmax(clean_logits)) == da_id,
    }
    passed = results[1.0]["top1_is_chang"] or results[2.0]["top1_is_chang"]
    return info, results, passed


def check4_band_signatures(model, lens, prompts):
    C, layers = cka_mod.cka_matrix(model, lens, n_tokens=5000, seed=0)
    (RUN_DIR / "figures").mkdir(parents=True, exist_ok=True)
    figures.cka_heatmap(C, layers, RUN_DIR / "figures" / "cka_heatmap.png")
    # cka_onset_score[l] = mean CKA between layer l and every later covered layer (how much l
    # already shares the geometry of the block above it). A heuristic summary; the heatmap is primary.
    Cn = C.numpy()
    onset_score = [float(Cn[i, i + 1:].mean()) if i + 1 < len(layers) else float("nan") for i in range(len(layers))]

    agree1 = {l: [] for l in layers}
    agree5 = {l: [] for l in layers}
    kurt = {l: [] for l in layers}
    for p in prompts:
        pos = [i for i, c in enumerate(p.classes) if c != "template"]
        with model.trace(p.input_ids):
            saved = {l: model_mod.layer_output(model, l).float()[0, pos].save() for l in layers}
            actual = model.output.logits[0, pos].float().save()
        actual_top1 = torch.argmax(actual, dim=-1)
        for l in layers:
            logits = lens_mod._transport_unembed(model, lens, saved[l], l).float()  # [P, vocab]
            top5 = torch.topk(logits, 5, dim=-1).indices
            agree1[l].extend((top5[:, 0] == actual_top1).tolist())
            agree5[l].extend((top5 == actual_top1[:, None]).any(-1).tolist())
            # Excess kurtosis of the readout LOGITS across the vocab (not softmax probabilities,
            # which are heavy-tailed at every layer and would hide the signature).
            z = (logits - logits.mean(-1, keepdim=True)) / logits.std(-1, unbiased=False, keepdim=True)
            kurt[l].extend(((z ** 4).mean(-1) - 3.0).tolist())

    df = pd.DataFrame({
        "layer": layers,
        "cka_onset_score": onset_score,
        "agreement_top1": [float(np.mean(agree1[l])) for l in layers],
        "agreement_top5": [float(np.mean(agree5[l])) for l in layers],
        "kurtosis": [float(np.mean(kurt[l])) for l in layers],
    })

    n = len(layers)
    mid = df.iloc[int(0.25 * n): int(0.60 * n) + 1]["agreement_top1"]
    threshold = (mid.median() + df["agreement_top1"].iloc[-1]) / 2
    above = df[df["agreement_top1"] > threshold]["layer"]
    motor_onset = int(above.iloc[0]) if len(above) else None

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for ax, col in zip(axes, ["cka_onset_score", "agreement_top1", "agreement_top5", "kurtosis"]):
        ax.plot(df["layer"], df[col], marker=".")
        ax.set_ylabel(col, fontsize=8)
        if motor_onset is not None:
            ax.axvline(motor_onset, color="red", linestyle="--", linewidth=0.8)
    axes[-1].set_xlabel("layer")
    axes[0].set_title(f"band signatures (red = candidate motor onset {motor_onset}; human fills configs/bands.yaml)")
    fig.savefig(RUN_DIR / "band_signatures.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    df.to_parquet(RUN_DIR / "band_signatures.parquet")
    return df, motor_onset


def main() -> None:
    env.require_env("HF_TOKEN")
    model_cfg = _load_yaml("model.yaml")
    lens_cfg = _load_yaml("lens.yaml")
    tokens_raw = _load_yaml("tokens.yaml")
    if not lens_cfg.get("revision_sha"):
        with open(RUN_DIR / "lens_resolved.yaml") as f:  # written by scripts/download.py
            lens_cfg = {**lens_cfg, "revision_sha": yaml.safe_load(f)["revision_sha"]}

    model = model_mod.load_model(model_cfg, standin=False)
    lens = lens_mod.load_lens(lens_cfg, device=model_mod.device_of(model))
    tok = model.tokenizer

    with open("stimuli/stimuli.json") as f:
        stim = json.load(f)
    fmt = {"tokenizer": tok, "questions": stim["questions"]}
    all_prompts = [prompts_mod.build_prompt(s, q, fmt) for s in stim["passages"] for q in stim["questions"]]
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    overlaps, max_abs = check1_final_layer_agreement(model, lens, all_prompts[:5])

    es_ids = {tok.encode(t, add_special_tokens=False)[0] for t in tokens_raw["language_tokens"]["es"]}
    sp_01 = next(s for s in stim["passages"] if s["id"] == "sp_01")
    sampled_layers, es_top3_frac = check2_readout_reproduction(model, lens, sp_01, fmt, es_ids)

    ctrl_info, ctrl_results, ctrl_passed = check3_causal_positive_control(model, lens)
    with open(RUN_DIR / "positive_control.json", "w") as f:
        json.dump({"info": ctrl_info, "results": {str(k): v for k, v in ctrl_results.items()},
                   "passed": ctrl_passed}, f, indent=2, ensure_ascii=False)
    if not ctrl_passed:
        (RUN_DIR / "summary.md").write_text(
            "# M1 summary -- STOPPED\n\nCausal positive control failed at alpha=1 and alpha=2. "
            "Do not run M2. Details: runs/M1/positive_control.json\n"
        )
        print("STOP: causal positive control failed at both alphas. Do not run M2.", file=sys.stderr)
        sys.exit(1)

    band_df, motor_onset = check4_band_signatures(model, lens, all_prompts)

    table, dropped_pairs, kept_pairs = _single_token_table(tok, tokens_raw)
    with open(RUN_DIR / "single_token_check.yaml", "w") as f:
        yaml.safe_dump({"tokenizer": model_cfg["hf_id"], "table": table, "pairs_kept": kept_pairs,
                        "pairs_dropped": dropped_pairs}, f, sort_keys=False, allow_unicode=True)

    template = all_prompts[0].text
    (RUN_DIR / "template_string.txt").write_text(template)
    m0_path = Path("runs/M0/template_string.txt")
    if m0_path.exists():
        diff = "".join(difflib.unified_diff(m0_path.read_text().splitlines(True), template.splitlines(True),
                                            "M0 (stand-in)", "M1 (real, canonical)"))
        template_note = f"```diff\n{diff}```" if diff else "identical to M0's stand-in template"
    else:
        template_note = "runs/M0/template_string.txt not present on this machine -- no diff computed"

    credit = Path(RUN_DIR / "CREDIT.md").read_text() if (RUN_DIR / "CREDIT.md").exists() else "(missing)"
    summary_lines = [
        "# M1 summary", "",
        "## 1. Environment",
        "- Environment: cuda (Phase B)",
        f"- git commit: {env.git_commit()}",
        f"- lens_sha: {lens_cfg['revision_sha']}",
        f"- model: {model_cfg['hf_id']} @ {model_cfg['revision']}", "",
        "## 2. Checks",
        f"- Check 1 (final-layer lens readout vs actual model logits, top-10 overlap, {len(overlaps)} positions): "
        f"min={min(overlaps)}, median={int(np.median(overlaps))}, max={max(overlaps)}; max |logit diff|={max_abs:.4g}.",
        f"- Check 2 (sp_01 matrix positions): fraction with an es variant in top-3, by layer: {es_top3_frac}. "
        "Figure: runs/M1/figures/readout_top1_sp01.png",
        f"- Check 3 (causal positive control): passed={ctrl_passed}; clean top-1 is 大: {ctrl_info['clean_top1_is_da']}; "
        f"clean top5 {ctrl_info['clean_top5']}; alpha=1 {ctrl_results[1.0]['top5']}; alpha=2 {ctrl_results[2.0]['top5']}. "
        "Prompt is a placeholder unless replaced with the paper's (see ANTONYM_USER_TEXT).",
        f"- Check 4 (band signatures): candidate motor onset = {motor_onset}. The human fills configs/bands.yaml.",
        f"- Check 5 (single-token, real tokenizer): pairs not single-token: {dropped_pairs or 'none'} "
        "(configs/tokens.yaml not modified; M2 filters at runtime).", "",
        "## 3. Figures",
        "- runs/M1/figures/cka_heatmap.png (+ .npy), runs/M1/band_signatures.png, runs/M1/figures/readout_top1_sp01.png", "",
        "## 4. Anomalies / open questions",
        f"- Template (M1 is canonical): {template_note}", "",
        "## 5. Artifact URL, parquet sha256s",
        f"- runs/M1/band_signatures.parquet sha256: {io_mod.sha256_of(RUN_DIR / 'band_signatures.parquet')}", "",
        "## Lens CREDIT.md", credit,
    ]
    upload_url = None
    try:
        upload_url = io_mod.upload_run(RUN_DIR)
        summary_lines.append(f"\n- HF dataset upload: {upload_url}")
    except Exception as e:  # noqa: BLE001
        summary_lines.append(f"\n- HF dataset upload FAILED: {e}")
    (RUN_DIR / "summary.md").write_text("\n".join(summary_lines))
    io_mod.write_manifest(RUN_DIR, milestone="M1", environment="cuda", upload_url=upload_url)
    print("M1 validation complete. See runs/M1/summary.md")


if __name__ == "__main__":
    main()
