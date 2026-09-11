"""Redraw one run's figures from its saved files only -- no model, no GPU.

    python scripts/make_figures.py runs/M0/smoke_20260912-031000 --format pdf
    python scripts/make_figures.py runs/M3/stage1_20260912-031000 --format pdf svg
    python scripts/make_figures.py runs/M3/stage1_20260912-031000 --format pdf --out writeup/figs

By default writes to <run folder>/figures/<format>/ -- the same place, and the same code
(figures.make_figures), the milestone script used for its PNGs at the end of the run. With --out,
writes straight into that folder instead (<out>/<figure>.<format>). Which files each milestone's
figures read is listed at the top of src/jlens_spec/figures.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from jlens_spec import env

env.bootstrap()

from jlens_spec import figures  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="a run folder, e.g. runs/M0/smoke_20260912-031000")
    ap.add_argument("--format", nargs="+", default=["png"], choices=list(figures.FORMATS),
                    help="one or more of png/pdf/svg (default png)")
    ap.add_argument("--out", default=None,
                    help="write straight into this folder instead of <run folder>/figures/<format>/")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "manifest.json").exists():
        runs = sorted(p.name for p in run_dir.iterdir() if (p / "manifest.json").exists()) if run_dir.is_dir() else []
        hint = f" Run folders in {run_dir}: {', '.join(runs)}" if runs else ""
        raise SystemExit(f"{run_dir} is not a run folder (no manifest.json).{hint}")

    written = figures.make_figures(run_dir, args.format, out_dir=args.out)
    print(f"wrote {len(written)} figure file(s) under {args.out or run_dir / 'figures'}")


if __name__ == "__main__":
    main()
