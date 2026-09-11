"""Setup shared by the long milestone scripts (M2, M3): real run vs Mac dry run, where files are
uploaded, loading the model and lens, and the checks every such run makes before it starts.

A DRY RUN (experiment file with a `dryrun:` block, in configs/experiments/dryrun/) runs the whole
pipeline on the Mac with the small stand-in model and a random lens -- no GPU, nothing uploaded:
  - run folders go under runs/dryrun/<milestone>/ instead of runs/<milestone>/;
  - a local folder (`dryrun.store`) stands in for the HF dataset, so resume, "finished = summary.md
    in the store" and every check work exactly as for real;
  - the bands come from `dryrun.bands_file` (configs/bands.yaml stays untouched for the real runs);
  - the "M1 validation passed" check is skipped (the stand-in has no M1 run).
Its numbers mean nothing (random lens); it only shows that the code runs end to end.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import io as io_mod
from . import lens as lens_mod
from . import model as model_mod

BASE_SETTINGS = ["configs/model.yaml", "configs/lens.yaml", "configs/tokens.yaml",
                 "configs/prompt_format.yaml", "stimuli/stimuli.json"]


def read_experiment(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def is_dryrun(exp: dict) -> bool:
    return bool(exp.get("dryrun"))


def settings_files(exp: dict) -> list[str]:
    """Every settings file the run copies into settings/ (the bands file is the dry run's own)."""
    bands = exp["dryrun"]["bands_file"] if is_dryrun(exp) else "configs/bands.yaml"
    return BASE_SETTINGS + [bands]


def runs_root(exp: dict) -> Path:
    return Path("runs/dryrun") if is_dryrun(exp) else Path("runs")


def store_for(exp: dict):
    return io_mod.LocalStore(exp["dryrun"]["store"]) if is_dryrun(exp) else io_mod.HFStore()


def environment(exp: dict) -> str:
    return "mac dry run (stand-in model, random lens -- numbers are meaningless)" if is_dryrun(exp) else "cuda (Phase B)"


def load_model_and_lens(exp: dict, model_cfg: dict, lens_cfg: dict, device: str | None):
    """Real run: the real model (device_map="auto") and the J-lens at configs/lens.yaml's
    revision_sha (required). Dry run: the stand-in on `device` and a random lens over every layer
    but the last."""
    if is_dryrun(exp):
        model = model_mod.load_model(model_cfg, standin=True, device=device)
        n = model_mod.n_layers(model)
        lens = lens_mod.random_lens(model_mod.d_model(model), list(range(n - 1)), seed=exp["dryrun"]["lens_seed"])
        return model, lens
    if not lens_cfg.get("revision_sha"):
        raise SystemExit("configs/lens.yaml revision_sha is empty -- copy it from the lens_resolved.yaml of "
                         "your runs/M1/download_* run (M1 step 0)")
    model = model_mod.load_model(model_cfg, standin=False)
    lens = lens_mod.load_lens(lens_cfg, device=model_mod.device_of(model))
    return model, lens


def fetch(path, store) -> Path:
    """A file of another run: from this machine if present, else downloaded from the store."""
    path = Path(path)
    if not path.exists():
        if not store.exists(path):
            raise SystemExit(f"{path} is neither on this machine nor in the {store.name}")
        store.download(path)
    return path


def resolve_run(ref: str, milestone: str, exp: dict, store) -> str:
    """A run named in an experiment file: a folder name, or -- in DRY-RUN files only --
    "latest:<experiment name>", the most recent FINISHED run of that experiment (so the dry-run
    chain needs no hand edits between steps). Real runs must name their folders explicitly."""
    from . import runs as runs_mod

    if not ref.startswith("latest:"):
        return ref
    if not is_dryrun(exp):
        raise SystemExit(f"'{ref}': 'latest:' is only allowed in dry-run experiment files -- name the run folder")
    name = ref.split(":", 1)[1]
    root = runs_root(exp)
    for run_id in reversed(runs_mod.run_ids(milestone, name, root, store)):
        if runs_mod.is_finished(root / milestone / run_id, store):
            return run_id
    raise SystemExit(f"'{ref}': no finished {milestone} run of '{name}' under {root}/{milestone}")


def require_finished(run_dir, store, what: str) -> None:
    from . import runs as runs_mod

    if not runs_mod.is_finished(run_dir, store):
        raise SystemExit(f"{what} {run_dir} is not finished (no summary.md in the {store.name}) -- "
                         "finish it first, or name a finished run in the experiment file")


def check_m1_passed(exp: dict, store) -> str:
    """Invariant 1 (lens before science) and the spec's "if the positive control fails, do not run
    M2": the M1 validation run named in the experiment file must be finished and its causal
    positive control must have passed. Skipped for dry runs. Returns a line for the summary."""
    if is_dryrun(exp):
        return "M1 validation check: skipped (dry run -- the stand-in has no M1 run)"
    name = exp.get("m1_validate_run")
    if not name:
        raise SystemExit("set m1_validate_run in the experiment file to the folder name of your finished "
                         "M1 validation run (runs/M1/validate_<time>)")
    run_dir = Path("runs/M1") / name
    require_finished(run_dir, store, "M1 validation run")
    info = yaml.safe_load(fetch(run_dir / "check3_positive_control.yaml", store).read_text())
    if not info.get("passed"):
        raise SystemExit(f"{run_dir}: the causal positive control did NOT pass -- per the spec, M2/M3 must not run")
    return f"M1 validation run {name}: finished, causal positive control passed (alphas run {info.get('alphas_run')})"
