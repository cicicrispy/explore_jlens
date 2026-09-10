"""M1 step 0 (Phase B / GPU box only): download the real model + lens, resolve the lens repo's
current sha, and record what was found in runs/M1/lens_resolved.yaml.

Nothing under configs/ is written. After this runs, copy `revision_sha` from
runs/M1/lens_resolved.yaml into configs/lens.yaml by hand (M2/M3 refuse to run without it).
Requires HF_TOKEN in the environment (source .env first).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from jlens_spec import env

env.bootstrap()

from huggingface_hub import HfApi, hf_hub_download, snapshot_download  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from jlens_spec import lens as lens_mod  # noqa: E402

RUN_DIR = Path("runs/M1")


def _yaml_safe(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_yaml_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _yaml_safe(x) for k, x in v.items()}
    return repr(v)


def main() -> None:
    env.require_env("HF_TOKEN")
    with open("configs/model.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/lens.yaml") as f:
        lens_cfg = yaml.safe_load(f)

    # Model weights: fetch into HF_HOME without loading them (M1 loads them).
    snapshot_download(model_cfg["hf_id"], revision=model_cfg["revision"])
    mcfg = AutoConfig.from_pretrained(model_cfg["hf_id"], revision=model_cfg["revision"])
    d = getattr(mcfg, "hidden_size", None) or mcfg.text_config.hidden_size

    lens_sha = HfApi().repo_info(repo_id=lens_cfg["repo"]).sha
    resolved = {"repo": lens_cfg["repo"], "filename": lens_cfg["filename"], "revision_sha": lens_sha}
    loaded = lens_mod.load_lens(resolved, device="cpu")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    credit_dir = str(Path(lens_cfg["filename"]).parent)
    try:
        credit = Path(hf_hub_download(lens_cfg["repo"], f"{credit_dir}/CREDIT.md", revision=lens_sha)).read_text()
        (RUN_DIR / "CREDIT.md").write_text(credit)
    except Exception as e:  # noqa: BLE001
        credit = f"(could not fetch CREDIT.md: {e})"

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
    with open(RUN_DIR / "lens_resolved.yaml", "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)

    if loaded.d_model != d:
        print(f"ERROR: lens d_model ({loaded.d_model}) != model hidden size ({d}). STOP.", file=sys.stderr)
        sys.exit(1)

    print(f"lens d_model == model hidden size == {d}")
    print(f"lens_sha={lens_sha}  layers covered: {len(loaded.layers)} ({loaded.layers[0]}..{loaded.layers[-1]})")
    print(f"coverage_ratio={out['coverage_ratio']:.4f}  stored dtype={loaded.dtype}")
    print("Wrote runs/M1/lens_resolved.yaml and runs/M1/CREDIT.md.")
    print("NEXT: copy revision_sha into configs/lens.yaml by hand.")


if __name__ == "__main__":
    main()
