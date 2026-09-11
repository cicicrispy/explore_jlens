"""One panel c over several finished M3 runs -- e.g. a position set's positives run and its anomaly
run -- drawn from their saved records only (no model, no GPU). Written to a NEW local folder next to
the runs, runs/M3/combined_<position set>_<UTC time>/ (runs/dryrun/M3/... for dry runs):
    sources.json               which runs went in, and what each used (pair, band, position set,
                               lens, model, git commits)
    figures/<format>/panel_c_combined.<format>
It is never uploaded -- copy it off the machine yourself (e.g. sftp). Refuses runs that differ in
pair, band, position set, lens or model; different git commits are allowed and listed.

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
        if not (d / "records").is_dir():
            raise SystemExit(f"{d}/records not found on this machine. Download the run's records first, e.g.\n"
                             f"  hf download orbitsoferis/jlens-specificity --repo-type dataset "
                             f"--include '{d.as_posix()}/records/*' --include '{d.as_posix()}/figure_params.json' "
                             "--local-dir .")
    info, problems = figures.combine_check(dirs)
    if problems:
        raise SystemExit("these runs can't be combined into one panel c -- they differ in:\n  " + "\n  ".join(problems))
    position_set = "+".join(sorted({"-".join(p) for v in info.values() for p in v["position_set"]}))
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = dirs[0].parent / f"combined_{position_set}_{stamp}"
    n = 2
    while out.exists():
        out = dirs[0].parent / f"combined_{position_set}_{stamp}-{n}"
        n += 1
    written = []
    for fmt in args.format:
        paths, _ = figures.combined_panel_c(dirs, formats=(fmt,), out_dir=out / "figures" / fmt)
        written += paths
    (out / "sources.json").write_text(json.dumps({"runs": [d.as_posix() for d in dirs], "what_each_used": info,
                                                  "made_at_utc": stamp}, indent=2, ensure_ascii=False))
    print(f"wrote {len(written)} figure file(s) and sources.json to {out} (local only -- not uploaded)")


if __name__ == "__main__":
    main()
