"""Run folders: one folder per run, with a copy of the settings it used.

Every milestone run lives in its own folder, never shared with another run and never overwritten:

    runs/<milestone>/<experiment name>_<UTC start time YYYYmmdd-HHMMSS>/
        settings/       copies of every settings file the run uses, taken when the run starts
        manifest.json   fixed facts about the run, written once at the start and never changed:
                        run_id, milestone, experiment file, settings sources, config_hash, git
                        commit, start time
        ...             the milestone's data files, figures/<format>/
        summary.md      written LAST, only after the run folder is uploaded (io.finalize_run): a run
                        is finished if and only if its summary.md is on HF

A run reads its settings only from its own settings/ folder, never from configs/. So a run that is
resumed later continues with exactly the settings it started with, even if configs/ has been edited
since (a new run -- `--fresh` -- picks those edits up). Code is not copied: every record carries the
git commit that produced it.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import env
from . import io as io_mod

RUNS_ROOT = Path("runs")


@dataclass
class Run:
    milestone: str
    run_id: str
    dir: Path

    def settings_path(self, name: str) -> Path:
        return self.dir / "settings" / name

    def load(self, name: str):
        """A settings file from this run's own copy (.yaml/.yml or .json, by suffix)."""
        path = self.settings_path(name)
        with open(path) as f:
            return json.load(f) if path.suffix == ".json" else yaml.safe_load(f)

    @property
    def experiment(self) -> dict:
        return self.load(self.manifest()["experiment_file"])

    def manifest(self) -> dict:
        return json.loads((self.dir / "manifest.json").read_text())

    def config_hash(self) -> str:
        """sha256 over every file in settings/ (sorted by name, name included), i.e. over exactly
        the settings this run used."""
        h = hashlib.sha256()
        for p in sorted((self.dir / "settings").iterdir()):
            h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
        return h.hexdigest()


def start_run(milestone: str, experiment_path, settings_files: list, root=RUNS_ROOT,
              name_suffix: str | None = None) -> Run:
    """Create a new run folder and copy the experiment file plus `settings_files` (e.g.
    "configs/model.yaml", "stimuli/stimuli.json") into its settings/ folder. The folder is named
    <experiment name>[_<name_suffix>]_<UTC start time>, e.g. download_Qwen3.6-27B_20260912-031000."""
    experiment_path = Path(experiment_path)
    with open(experiment_path) as f:
        exp = yaml.safe_load(f)
    if exp.get("milestone") != milestone:
        raise ValueError(f"{experiment_path} is for milestone {exp.get('milestone')!r}, not {milestone!r}")

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    prefix = f"{exp['name']}_{name_suffix}" if name_suffix else exp["name"]
    run_id = f"{prefix}_{stamp}"
    run_dir = Path(root) / milestone / run_id
    n = 2
    while run_dir.exists():  # two runs started within the same second
        run_id = f"{prefix}_{stamp}-{n}"
        run_dir = Path(root) / milestone / run_id
        n += 1
    settings = run_dir / "settings"
    settings.mkdir(parents=True)

    sources = {experiment_path.name: str(experiment_path)}
    for src in settings_files:
        src = Path(src)
        if src.name in sources:
            raise ValueError(f"two settings files are both named {src.name!r}: {sources[src.name]}, {src}")
        sources[src.name] = str(src)
    for name, src in sources.items():
        shutil.copy2(src, settings / name)

    run = Run(milestone, run_id, run_dir)
    io_mod.write_manifest(  # written once; never changed afterwards
        run_dir,
        run_id=run_id,
        milestone=milestone,
        experiment_file=experiment_path.name,
        settings_sources=sources,
        config_hash=run.config_hash(),
        git_commit=env.git_commit(),
        started_at=time.time(),
    )
    return run


def open_run(run_dir) -> Run:
    """An existing run folder (reads its manifest.json)."""
    run_dir = Path(run_dir)
    m = json.loads((run_dir / "manifest.json").read_text())
    return Run(m["milestone"], m["run_id"], run_dir)


# ------------------------------------------------------------------------------------ resume
#
# A long run (M2, M3) saves one set of files per prompt and uploads them as each prompt finishes.
# Starting the same experiment again RESUMES it if its most recent run is unfinished (no
# summary.md in the store): prompts whose files are in the store are done; anything that exists only
# on this machine is recomputed (a file on one machine only does not count); files that are in the
# store but not here are downloaded at the end. A resumed run keeps using its own settings copy --
# edits to configs/ since it started are printed, not applied -- and code changes are allowed (every
# row carries the git commit that produced it).

_RUN_ID = re.compile(r"^(?P<name>.+)_(?P<stamp>\d{8}-\d{6})(?:-(?P<n>\d+))?$")


def _order(run_id: str):
    m = _RUN_ID.match(run_id)
    return (m["stamp"], int(m["n"] or 1))


def run_ids(milestone: str, name: str, root=RUNS_ROOT, store=None) -> list[str]:
    """Every run of the experiment called `name` -- folders on this machine and, if `store` is
    given, folders in the store -- oldest first."""
    found = set()
    local = Path(root) / milestone
    if local.exists():
        found |= {p.name for p in local.iterdir() if p.is_dir()}
    if store is not None:
        found |= store.list_dirs(f"{Path(root).as_posix()}/{milestone}")
    return sorted((i for i in found if (m := _RUN_ID.match(i)) and m["name"] == name), key=_order)


def is_finished(run_dir, store) -> bool:
    """A run is finished if and only if its summary.md is in the store."""
    return store.exists(f"{Path(run_dir).as_posix()}/summary.md")


def fetch_settings(run_dir, store) -> None:
    """Download a run's manifest.json and settings/ from the store if this machine lacks them
    (resuming on a new machine)."""
    run_dir = Path(run_dir)
    for f in sorted(store.list_files(run_dir)):
        rel = Path(f).relative_to(run_dir)
        if (rel.name == "manifest.json" and len(rel.parts) == 1) or rel.parts[0] == "settings":
            if not Path(f).exists():
                store.download(f)


def settings_diff(run: Run) -> list[str]:
    """Names of the settings files whose current source (the path recorded in the manifest) now
    differs from the run's own copy, or no longer exists. The run keeps using its copy."""
    changed = []
    for name, src in run.manifest()["settings_sources"].items():
        src = Path(src)
        if not src.exists() or src.read_bytes() != run.settings_path(name).read_bytes():
            changed.append(name)
    return changed


def uploaded_files(run_dir, store) -> set[str]:
    """Paths (relative to the run folder) of every file of this run that is in the store."""
    run_dir = Path(run_dir)
    return {Path(f).relative_to(run_dir).as_posix() for f in store.list_files(run_dir)}


def sync_down(run_dir, store) -> list[str]:
    """Download every file of this run that is in the store but not on this machine (e.g. prompts
    finished on another machine before a resume). Returns the paths downloaded."""
    run_dir = Path(run_dir)
    got = []
    for rel in sorted(uploaded_files(run_dir, store)):
        if not (run_dir / rel).exists():
            store.download(run_dir / rel)
            got.append(rel)
    return got


def start_or_resume(milestone: str, experiment_path, settings_files: list, store, root=RUNS_ROOT,
                    fresh: bool = False, resume=None) -> tuple[Run, bool]:
    """The run a long script should work in, and whether it is a resumed one:
    - `resume` (a run folder): that run. Refuses if it is already finished.
    - `fresh`: always a new run folder.
    - otherwise: the experiment's MOST RECENT run if it is unfinished, else a new run. An older
      unfinished run is never picked up by default (pass its folder with `resume`).
    Prints which, and -- when resuming -- which settings files have changed in configs/ since (the
    run keeps using its own copy)."""
    if resume is not None and fresh:
        raise ValueError("pass either --resume or --fresh, not both")
    if resume is not None:
        run_dir = Path(resume)
    elif fresh:
        run_dir = None
    else:
        with open(experiment_path) as f:
            name = yaml.safe_load(f)["name"]
        ids = run_ids(milestone, name, root, store)
        run_dir = Path(root) / milestone / ids[-1] if ids else None
        if run_dir is not None and is_finished(run_dir, store):
            print(f"The most recent run of '{name}' ({run_dir.name}) is finished -- starting a new run.", flush=True)
            run_dir = None

    if run_dir is None:
        run = start_run(milestone, experiment_path, settings_files, root=root)
        print(f"New run folder: {run.dir}", flush=True)
        return run, False

    if is_finished(run_dir, store):
        raise SystemExit(f"{run_dir} is already finished (its summary.md is in the {store.name}); "
                         "a finished run is never changed. Start a new one with --fresh.")
    fetch_settings(run_dir, store)
    if not (run_dir / "manifest.json").exists():
        raise SystemExit(f"{run_dir}: no manifest.json on this machine or in the {store.name}")
    run = open_run(run_dir)
    if run.milestone != milestone:
        raise SystemExit(f"{run_dir} is a {run.milestone} run, not {milestone}")
    changed = settings_diff(run)
    print(f"RESUMING unfinished run {run.dir} (settings from its own copy in settings/).", flush=True)
    if changed:
        print(f"  NOTE: these settings files changed in configs/ since the run started; the run keeps "
              f"its copy, the changes apply to a new run (--fresh): {', '.join(changed)}", flush=True)
    return run, True
