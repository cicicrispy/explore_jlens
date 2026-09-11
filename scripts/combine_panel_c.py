"""Combined figures over several finished M3 runs -- e.g. a position set's positives run and its
anomaly run, so all four questions are in one figure -- drawn from their saved records only (no
model, no GPU). Written to a NEW folder next to the runs, runs/M3/combined_<position set>_<UTC
time>/ (runs/dryrun/M3/... for dry runs):
    sources.json               which runs went in, and what each used (pair, band, position set,
                               lens, model, git commits)
    figures/<format>/panel_c_combined.<format>
    figures/<format>/flip_heatmap_combined.<format>     flip rate per question x kind (k/n in each box)
    figures/<format>/margin_change_combined.<format>    how far each edit moved the answer
Then, if every run that went in is finished, the folder is uploaded to the same path in the runs'
store: the HF dataset for real runs, the dry run's local stand-in (runs/dryrun/_hf/) for dry runs --
a dry run never touches HF. If one is unfinished, the folder stays local and the script says so.
Refuses runs that differ in pair, band, position set, lens or model, and a mix of dry and real runs;
different git commits are allowed and listed.

    python scripts/combine_panel_c.py runs/M3/positives_question_20260912-031000 runs/M3/anomaly_question_20260912-041000
    python scripts/combine_panel_c.py runs/M3/positives_question_20260912-031000 runs/M3/anomaly_question_20260912-041000 --format pdf png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from jlens_spec import env

env.bootstrap()

from jlens_spec import figures  # noqa: E402
from jlens_spec import pipeline  # noqa: E402
from jlens_spec import runs as runs_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="two or more M3 run folders on this machine")
    ap.add_argument("--format", nargs="+", default=["png"], choices=list(figures.FORMATS),
                    help="one or more of png/pdf/svg (default png)")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.run_dirs]
    if len(dirs) < 2:
        raise SystemExit("give at least two M3 run folders")
    for d in dirs:
        needed = [d / "records", d / "manifest.json", d / "settings"]
        if not all(p.exists() for p in needed):
            raise SystemExit(f"{d}: records/, manifest.json or settings/ not found on this machine. Download them "
                             f"first, e.g.\n  hf download orbitsoferis/jlens-specificity --repo-type dataset "
                             f"--include '{d.as_posix()}/records/*' --include '{d.as_posix()}/figure_params.json' "
                             f"--include '{d.as_posix()}/manifest.json' --include '{d.as_posix()}/settings/*' "
                             "--local-dir .")
    # Which store the runs live in (their own settings copy says): HF for real runs, the local
    # stand-in for dry runs.
    exps = [runs_mod.open_run(d).experiment for d in dirs]
    if len({pipeline.is_dryrun(e) for e in exps}) != 1:
        raise SystemExit("these runs mix dry runs and real runs -- combine only runs of one kind")
    store = pipeline.store_for(exps[0])
    if not pipeline.is_dryrun(exps[0]):
        env.require_env("HF_TOKEN")
    info, problems = figures.combine_check(dirs)
    if problems:
        raise SystemExit("these runs can't be combined into one panel c -- they differ in:\n  " + "\n  ".join(problems))

    # The folder is named after the position set's name (question, message) from the runs'
    # figure_params.json; runs without one fall back to their position classes.
    names = set()
    for d in dirs:
        p = d / "figure_params.json"
        names.add(json.loads(p.read_text()).get("position_set") if p.exists() else None)
    position_set = names.pop() if len(names) == 1 and None not in names else \
        "+".join(sorted({"-".join(p) for v in info.values() for p in v["position_set"]}))
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = dirs[0].parent / f"combined_{position_set}_{stamp}"
    n = 2
    while out.exists():
        out = dirs[0].parent / f"combined_{position_set}_{stamp}-{n}"
        n += 1
    written = []
    for fmt in args.format:
        paths, _ = figures.combined_panel_c(dirs, formats=(fmt,), out_dir=out / "figures" / fmt)
        written += paths + figures.combined_summary_figures(dirs, formats=(fmt,), out_dir=out / "figures" / fmt)
    (out / "sources.json").write_text(json.dumps({"runs": [d.as_posix() for d in dirs], "what_each_used": info,
                                                  "made_at_utc": stamp}, indent=2, ensure_ascii=False))
    print(f"wrote {len(written)} figure file(s) and sources.json to {out}", flush=True)

    unfinished = [d.name for d in dirs if not runs_mod.is_finished(d, store)]
    if unfinished:
        print(f"NOT uploaded: these runs are unfinished (no summary.md in the {store.name}): {unfinished}. "
              "The folder stays local; run the same command again once they are finished.")
        return
    try:
        url = store.upload_path(out)
    except Exception as e:  # noqa: BLE001 -- reported; the local folder is complete
        raise SystemExit(f"the upload to the {store.name} FAILED ({e!r}); the figures are in {out}. Run the same "
                         "command again to make and upload a new combined folder.")
    print(f"uploaded to the {store.name}: {url}")


if __name__ == "__main__":
    main()
