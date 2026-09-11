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
