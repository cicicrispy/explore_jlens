"""M1 step 0 (Phase B / GPU box only): download the real model + lens, resolve the lens repo's
current version (sha), and record what was found -- in a new run folder
runs/M1/download_<model>_<UTC start time>/, uploaded to the HF dataset like every run.

Nothing under configs/ is written. After this runs, copy `revision_sha` from the run's
lens_resolved.yaml into configs/lens.yaml by hand (M2/M3 refuse to run without it), and put the
run's folder name into configs/experiments/m1_validate.yaml's `download_run`.
Requires HF_TOKEN in .env (loaded by env.bootstrap(), never sourced in a shell).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from jlens_spec import env

env.bootstrap()

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from jlens_spec import io as io_mod  # noqa: E402
from jlens_spec import lens as lens_mod  # noqa: E402
from jlens_spec import runs as runs_mod  # noqa: E402

DEFAULT_EXPERIMENT = "configs/experiments/m1_download.yaml"
SETTINGS_FILES = ["configs/model.yaml", "configs/lens.yaml"]


def _yaml_safe(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_yaml_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _yaml_safe(x) for k, x in v.items()}
    return repr(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help=f"experiment file (default {DEFAULT_EXPERIMENT})")
    args = ap.parse_args()
    env.require_env("HF_TOKEN")
    with open("configs/model.yaml") as f:
        model_name = Path(yaml.safe_load(f)["hf_id"]).name  # e.g. Qwen3.6-27B

    run = runs_mod.start_run("M1", args.experiment, SETTINGS_FILES, name_suffix=model_name)
    print(f"Run folder: {run.dir}", flush=True)
    model_cfg = run.load("model.yaml")
    lens_cfg = run.load("lens.yaml")

    # Model weights: fetch into HF_HOME without loading them (M1 loads them).
    snapshot_download(model_cfg["hf_id"], revision=model_cfg["revision"])
    mcfg = AutoConfig.from_pretrained(model_cfg["hf_id"], revision=model_cfg["revision"])
    d = getattr(mcfg, "hidden_size", None) or mcfg.text_config.hidden_size

    lens_sha = HfApi().repo_info(repo_id=lens_cfg["repo"]).sha
    resolved = {"repo": lens_cfg["repo"], "filename": lens_cfg["filename"], "revision_sha": lens_sha}
    loaded = lens_mod.load_lens(resolved, device="cpu")

    credit_dir = str(Path(lens_cfg["filename"]).parent)
    try:
        credit = Path(hf_hub_download(lens_cfg["repo"], f"{credit_dir}/CREDIT.md", revision=lens_sha)).read_text()
        (run.dir / "CREDIT.md").write_text(credit)
    except Exception as e:  # noqa: BLE001 -- reported in the summary
        credit = f"(could not fetch CREDIT.md: {e!r})"

    one = loaded.layers[0]
    out = {
        **resolved,
        "layers": loaded.layers,
        "n_layers_covered": len(loaded.layers),
        "stored_dtype": str(loaded.dtype),
        "one_matrix_shape": list(loaded.J[one].shape),
        "coverage_ratio": lens_mod.coverage_ratio(loaded),
        "file_size_bytes": loaded.file_size_bytes,
        "model_hidden_size": d,
        "meta": _yaml_safe(loaded.meta),
    }
    with open(run.dir / "lens_resolved.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

    ok = loaded.d_model == d
    summary_lines = [
        f"# M1 download -- run {run.run_id}" + ("" if ok else " -- STOPPED"), "",
        f"- model: {model_cfg['hf_id']} @ {model_cfg['revision']} (weights downloaded into HF_HOME, not loaded)",
        f"- lens: {lens_cfg['repo']} / {lens_cfg['filename']}",
        f"- lens revision_sha (current `main`): {lens_sha}",
        f"- layers covered: {len(loaded.layers)} ({loaded.layers[0]}..{loaded.layers[-1]}); stored dtype "
        f"{loaded.dtype}; one matrix {list(loaded.J[one].shape)}; coverage_ratio {out['coverage_ratio']:.4f}",
        f"- lens d_model {loaded.d_model} vs model hidden size {d}: " + ("MATCH" if ok else "MISMATCH -- STOP, do not run M1"),
        f"- git commit: {run.manifest()['git_commit']}", "",
        "## Next (by hand)",
        f"- copy revision_sha into configs/lens.yaml: {lens_sha}",
        f"- set download_run in configs/experiments/m1_validate.yaml: {run.run_id}", "",
        "## Lens CREDIT.md", credit, "",
        "## Artifact URL",
    ]
    io_mod.finalize_run(run.dir, summary_lines)
    if not ok:
        print(f"ERROR: lens d_model ({loaded.d_model}) != model hidden size ({d}). STOP.", file=sys.stderr)
        sys.exit(1)
    print(f"lens d_model == model hidden size == {d}")
    print(f"lens_sha={lens_sha}  layers covered: {len(loaded.layers)} ({loaded.layers[0]}..{loaded.layers[-1]})")
    print(f"NEXT: copy revision_sha into configs/lens.yaml, and set download_run: {run.run_id} "
          "in configs/experiments/m1_validate.yaml.")


if __name__ == "__main__":
    main()
