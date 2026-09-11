"""Figures: plotting only, from saved run files.

Every figure is a pure function of files in one run folder (runs/<M>/<run_id>/, see runs.py) -- no
model, no GPU, no configs/. Each plotting function takes DataFrames (exactly as read back from
parquet) and returns a matplotlib Figure; `make_figures(run_dir, formats)` reads one run's saved
files and writes all of its figures to <run_dir>/figures/<format>/ (or, with `out_dir`, straight
into a folder of your choice). Milestone scripts write their
data files first and then call `make_figures` (so the figure you see at the end of a run is built
from what was saved, not from in-memory state), and `scripts/make_figures.py` calls the same
function later -- e.g. `--format pdf` for the writeup. The two paths therefore cannot drift, and
each format gets its own folder.

Per-milestone inputs (all inside the run folder):
    M0: masks.parquet                        -> masks/<stimulus>_<question>
    M1: check2_readout.parquet               -> readout_top1_<stimulus>
        check3_masks.parquet                 -> masks/check3_<condition>  (positive control; the
                                                red outline = where the stream ACTUALLY changed)
        cka.parquet                          -> cka_heatmap
        band_signatures.parquet
          + figure_params.json[motor_onset]  -> band_signatures
        (M1's tokens_* and download_* runs have no figures.)
    M2: loadings.parquet
          + figure_params.json[band, loading_heatmaps]
                                             -> loading_heatmap_<stimulus>, loading_summary_bars
    M3: records.parquet                      -> panel_c, margin_vs_deltac_{anomaly,report}
`figure_params.json` holds the few scalars a figure needs that aren't a table (e.g. which band was
used), written by the milestone script so figures never read configs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from . import loading as loading_mod

__all__ = [
    "FORMATS",
    "write",
    "save",
    "make_figures",
    "mask_figure",
    "readout_grid",
    "cka_heatmap",
    "band_signatures",
    "loading_heatmap",
    "loading_summary_bars",
    "panel_c",
    "margin_vs_deltac",
]

FORMATS = ("png", "pdf", "svg")
DPI = 150  # raster formats only

CLASS_COLORS = {
    "template": "#cccccc",
    "question": "#7fb3ff",
    "matrix": "#7fdc7f",
    "intrusion": "#ffb37f",
    "prompt": "#e6d7ff",  # M1 positive control: a raw prompt, no question/matrix/intrusion classes
}

# Columns read from M3's records.parquet -- logprobs_fp16 (a full-vocab vector per row) is skipped.
_M3_COLUMNS = ["stimulus_id", "question_key", "direction", "kind", "flip", "margin", "intervention_logs"]


def _plt():
    import matplotlib.pyplot as plt

    return plt


def write(fig, path) -> Path:
    """Save `fig` to exactly `path` (format from its suffix) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    _plt().close(fig)
    return path


def save(fig, name: str, formats, run_dir, out_dir=None) -> list[Path]:
    """Save `fig` once per format, then close it. Default: <run_dir>/figures/<fmt>/<name>.<fmt>
    (each format in its own folder). With `out_dir`: <out_dir>/<name>.<fmt>, straight into that
    folder. `name` may contain a subfolder (e.g. "masks/sp_01_report")."""
    for fmt in formats:
        if fmt not in FORMATS:
            raise ValueError(f"unknown figure format {fmt!r}; expected one of {FORMATS}")
    plt = _plt()
    paths = []
    for fmt in formats:
        base = Path(out_dir) if out_dir is not None else Path(run_dir) / "figures" / fmt
        p = base / f"{name}.{fmt}"
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _visible(text: str, max_len: int = 12) -> str:
    """Token text with whitespace made visible (space -> '·', newline -> '\\n'), truncated with '…'."""
    t = text.replace(" ", "·").replace("\n", "\\n").replace("\t", "\\t")
    return t if len(t) <= max_len else t[: max_len - 1] + "…"


# A token with no text (empty offset span) is labelled with its id; several such tokens can occur in
# one prompt, so the id keeps them distinguishable. Every figure that uses this label explains it.
NO_TEXT_NOTE = "#<token id> = token with no text"


def _token_label(text: str, token_id, max_len: int = 12) -> str:
    return _visible(text, max_len) if text else f"#{int(token_id)}"


# --------------------------------------------------------------------------------------- M0


def mask_figure(rows: pd.DataFrame):
    """One prompt's position mask: one box per token showing its text and position index, colored
    by class, red outline where edited, ★ on metric_pos, legend at the bottom. `rows` = that
    prompt's rows of masks.parquet (see prompts.mask_rows)."""
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    rows = rows.sort_values("pos")
    n = len(rows)
    cols = 12
    n_rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(cols * 1.35, n_rows * 0.62 + 1.2))

    # `class` is a Python keyword, which itertuples can't use as an attribute name.
    for r in rows.rename(columns={"class": "cls"}).itertuples(index=False):
        i = int(r.pos)
        x, y = i % cols, n_rows - 1 - i // cols
        edited = bool(r.edited)
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=CLASS_COLORS.get(r.cls, "#ffffff"),
                               edgecolor="red" if edited else "black", linewidth=2.5 if edited else 0.6))
        ax.text(x + 0.5, y + 0.45, _token_label(r.token_text, r.token_id), ha="center", va="center",
                fontsize=8, clip_on=True)
        ax.text(x + 0.05, y + 0.93, str(i), ha="left", va="top", fontsize=5, color="#444444")
        if bool(r.is_metric_pos):
            ax.plot(x + 0.88, y + 0.8, marker="*", color="black", markersize=11)

    ax.set_xlim(0, cols)
    ax.set_ylim(-0.05, n_rows + 0.05)
    ax.axis("off")
    first = rows.iloc[0]
    ax.set_title(f"{first['stimulus_id']} / {first['question_key']}   "
                 f"({n} tokens, {int(rows['edited'].sum())} edited)", fontsize=11)
    present = set(rows["class"])
    legend = [Patch(facecolor=c, edgecolor="black", label=k) for k, c in CLASS_COLORS.items() if k in present]
    legend += [Patch(facecolor="white", edgecolor="red", linewidth=2.5, label="edited (red outline)"),
               Line2D([], [], marker="*", color="black", linestyle="", markersize=11,
                      label="metric_pos (answer read here)"),
               Patch(facecolor="none", edgecolor="none", label=NO_TEXT_NOTE)]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=7, fontsize=8,
              frameon=False)
    return fig


# --------------------------------------------------------------------------------------- M1


def readout_grid(df: pd.DataFrame):
    """Lens top-1 token at each (matrix position x sampled layer). `df` = check2_readout.parquet
    (top-k lists per row; the top-1 is the first entry)."""
    plt = _plt()

    df = df.assign(top1_text=[str(t[0]) for t in df["topk_text"]], top1_id=[int(t[0]) for t in df["topk_ids"]])
    text = df.pivot(index="pos", columns="layer", values="top1_text").sort_index()
    ids = df.pivot(index="pos", columns="layer", values="top1_id").loc[text.index, text.columns]
    grid = text
    cells = [[_token_label(str(t), i, 14) for t, i in zip(trow, irow)]
             for trow, irow in zip(text.values, ids.values)]
    fig, ax = plt.subplots(figsize=(1.3 * grid.shape[1] + 2, 0.28 * grid.shape[0] + 1))
    ax.axis("off")
    table = ax.table(cellText=cells, rowLabels=list(grid.index), colLabels=list(grid.columns), loc="center")
    table.set_fontsize(6)
    first = df.iloc[0]
    ax.set_title(f"{first['stimulus_id']}/{first['question_key']}: lens top-1 at matrix positions "
                 "(rows=pos, cols=layer)")
    fig.text(0.5, 0.0, NO_TEXT_NOTE, ha="center", va="top", fontsize=7)
    return fig


def cka_heatmap(df: pd.DataFrame):
    """Layer x layer CKA. `df` = cka.parquet (long form: layer_a, layer_b, cka)."""
    plt = _plt()

    C = df.pivot(index="layer_a", columns="layer_b", values="cka").sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(C.columns)))
    ax.set_xticklabels(C.columns, fontsize=5, rotation=90)
    ax.set_yticks(range(len(C.index)))
    ax.set_yticklabels(C.index, fontsize=5)
    fig.colorbar(im, ax=ax, label="CKA")
    return fig


def band_signatures(df: pd.DataFrame, motor_onset: int | None):
    """Stacked per-layer signatures, one panel per column of band_signatures.parquet (cka_onset_score,
    agreement_top1, agreement_top<k>, kurtosis)."""
    plt = _plt()

    cols = [c for c in df.columns if c != "layer"]
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 2.5 * len(cols)), sharex=True)
    for ax, col in zip(axes, cols):
        ax.plot(df["layer"], df[col], marker=".")
        ax.set_ylabel(col, fontsize=8)
        if motor_onset is not None:
            ax.axvline(motor_onset, color="red", linestyle="--", linewidth=0.8)
    axes[-1].set_xlabel("layer")
    axes[0].set_title(f"band signatures (red = candidate motor onset {motor_onset}; human fills configs/bands.yaml)")
    return fig


# --------------------------------------------------------------------------------------- M2


def loading_heatmap(df: pd.DataFrame, stimulus_id: str, token: str, question_key: str | None = None):
    """(position x layer) cos heatmap for one prompt. A stimulus appears under 4 questions, so
    `question_key` selects which prompt; if None, the first question present is used."""
    plt = _plt()

    sub = df[(df["stimulus_id"] == stimulus_id) & (df["token"] == token)]
    if question_key is None:
        question_key = sorted(sub["question_key"].unique())[0]
    sub = sub[sub["question_key"] == question_key]
    pivot = sub.pivot(index="pos", columns="layer", values="cos")
    fig, ax = plt.subplots(figsize=(max(6, 0.15 * pivot.shape[1]), max(4, 0.15 * pivot.shape[0])))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=6, rotation=90)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=5)
    ax.set_xlabel("layer")
    ax.set_ylabel("position")
    ax.set_title(f"{stimulus_id} / {question_key} / {token!r}")
    fig.colorbar(im, ax=ax, label="cos")
    return fig


def loading_summary_bars(df: pd.DataFrame, band: list[int]):
    plt = _plt()

    d = df[df["layer"].isin(band)].copy()
    d["matrix_lang"] = d["stimulus_id"].map(loading_mod._matrix_lang_of)
    grouped = d.groupby(["class", "matrix_lang", "token"])["cos"].mean().reset_index()

    tokens = sorted(grouped["token"].unique())
    classes = sorted(grouped["class"].unique())
    mls = sorted(grouped["matrix_lang"].dropna().unique())

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.6 * len(classes) * len(mls)), 5))
    width = 0.8 / max(len(tokens), 1)
    x_base = 0.0
    xticks, xticklabels = [], []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for cls in classes:
        for ml in mls:
            for i, tok in enumerate(tokens):
                row = grouped[
                    (grouped["class"] == cls) & (grouped["matrix_lang"] == ml) & (grouped["token"] == tok)
                ]
                val = float(row["cos"].iloc[0]) if len(row) else 0.0
                ax.bar(x_base + i * width, val, width=width, color=colors[i % len(colors)])
            xticks.append(x_base + width * len(tokens) / 2)
            xticklabels.append(f"{cls}/{ml}")
            x_base += width * len(tokens) + 0.4
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("mean cos over band")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i % len(colors)]) for i in range(len(tokens))]
    ax.legend(handles, tokens, fontsize=6, ncol=min(len(tokens), 4))
    return fig


# --------------------------------------------------------------------------------------- M3


def _wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def panel_c(df: pd.DataFrame):
    """Two rows: top = flip rate with Wilson 95% CI per question, treatment vs 4 controls, split by
    direction x matrix language; bottom = mean margin +/- SE for the same cells. `df` = records.parquet."""
    plt = _plt()

    df = df.copy()
    df["matrix_lang"] = df["stimulus_id"].map(loading_mod._matrix_lang_of)

    questions = sorted(df["question_key"].unique())
    kinds = [k for k in ["swap", "label_to_present", "big_nonlabel", "random_direction", "identity"]
             if k in set(df["kind"])]
    directions = sorted(df["direction"].unique())
    mls = sorted(df["matrix_lang"].dropna().unique())

    labels, rates, err_lo, err_hi = [], [], [], []
    means, ses = [], []
    for q in questions:
        for kind in kinds:
            for direction in directions:
                for ml in mls:
                    sub = df[
                        (df["question_key"] == q)
                        & (df["kind"] == kind)
                        & (df["direction"] == direction)
                        & (df["matrix_lang"] == ml)
                    ]
                    flips = sub["flip"].dropna()  # None = no clean baseline; excluded, not counted as 0
                    if len(flips) == 0:
                        continue
                    k = int(flips.astype(bool).sum())
                    n = len(flips)
                    phat, lo, hi = _wilson_ci(k, n)
                    labels.append(f"{q}|{kind}|{direction}|{ml}")
                    rates.append(phat)
                    err_lo.append(phat - lo)
                    err_hi.append(hi - phat)
                    means.append(float(sub["margin"].mean()))
                    ses.append(float(sub["margin"].std(ddof=1) / (len(sub) ** 0.5)) if len(sub) > 1 else 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(labels) * 0.25), 9))
    x = range(len(labels))
    axes[0].bar(x, rates, yerr=[err_lo, err_hi], capsize=2)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=90, fontsize=5)
    axes[0].set_ylabel("flip rate")
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, means, yerr=ses, capsize=2)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=90, fontsize=5)
    axes[1].set_ylabel("mean margin (SE)")

    fig.tight_layout()
    return fig


def margin_vs_deltac(df: pd.DataFrame, question_key: str):
    """Margin vs mean |delta_c| over intervened positions, one point per cell. `df` = records.parquet;
    `intervention_logs` comes back from parquet as an array of dicts (possibly empty)."""
    plt = _plt()

    points = []
    for r in df[df["question_key"] == question_key].itertuples(index=False):
        logs = r.intervention_logs
        if logs is None or len(logs) == 0:
            continue
        deltas = [lg["delta_c_norm"] for lg in logs]
        points.append((sum(deltas) / len(deltas), r.margin, r.kind))

    fig, ax = plt.subplots(figsize=(6, 5))
    for kind in sorted(set(p[2] for p in points)):
        ax.scatter([p[0] for p in points if p[2] == kind], [p[1] for p in points if p[2] == kind],
                   label=kind, alpha=0.7, s=18)
    ax.set_xlabel("mean |delta_c| over intervened positions")
    ax.set_ylabel("margin")
    ax.set_title(question_key)
    ax.legend(fontsize=7)
    return fig


# ----------------------------------------------------------------------- milestone registry


def _params(run_dir: Path) -> dict:
    p = run_dir / "figure_params.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _missing(run_dir: Path, *names: str) -> bool:
    """True (and says so on stderr) if any input file is absent -- that figure is skipped."""
    absent = [n for n in names if not (run_dir / n).exists()]
    if absent:
        print(f"[figures] skipping: {run_dir}/{', '.join(absent)} not found", file=sys.stderr)
    return bool(absent)


def _m0(run_dir: Path, formats, out_dir) -> list[Path]:
    if _missing(run_dir, "masks.parquet"):
        return []
    df = pd.read_parquet(run_dir / "masks.parquet")
    written = []
    for (sid, q), rows in df.groupby(["stimulus_id", "question_key"], sort=False):
        written += save(mask_figure(rows), f"masks/{sid}_{q}", formats, run_dir, out_dir)
    return written


def _m1(run_dir: Path, formats, out_dir) -> list[Path]:
    written = []
    if not _missing(run_dir, "check2_readout.parquet"):
        df = pd.read_parquet(run_dir / "check2_readout.parquet")
        written += save(readout_grid(df), f"readout_top1_{df['stimulus_id'].iloc[0]}", formats, run_dir, out_dir)
    if not _missing(run_dir, "check3_masks.parquet"):
        masks = pd.read_parquet(run_dir / "check3_masks.parquet")
        for cond, rows in masks.groupby("question_key", sort=False):
            written += save(mask_figure(rows), f"masks/check3_{cond}", formats, run_dir, out_dir)
    if not _missing(run_dir, "cka.parquet"):
        written += save(cka_heatmap(pd.read_parquet(run_dir / "cka.parquet")), "cka_heatmap", formats, run_dir, out_dir)
    if not _missing(run_dir, "band_signatures.parquet", "figure_params.json"):
        df = pd.read_parquet(run_dir / "band_signatures.parquet")
        fig = band_signatures(df, _params(run_dir).get("motor_onset"))
        written += save(fig, "band_signatures", formats, run_dir, out_dir)
    return written


def _m2(run_dir: Path, formats, out_dir) -> list[Path]:
    if _missing(run_dir, "loadings.parquet", "figure_params.json"):
        return []
    df = pd.read_parquet(run_dir / "loadings.parquet")
    params = _params(run_dir)
    written = []
    for h in params["loading_heatmaps"]:
        fig = loading_heatmap(df, h["stimulus_id"], h["token"], h.get("question_key"))
        written += save(fig, f"loading_heatmap_{h['stimulus_id']}", formats, run_dir, out_dir)
    written += save(loading_summary_bars(df, params["band"]), "loading_summary_bars", formats, run_dir, out_dir)
    return written


def _m3(run_dir: Path, formats, out_dir) -> list[Path]:
    if _missing(run_dir, "records.parquet"):
        return []
    df = pd.read_parquet(run_dir / "records.parquet", columns=_M3_COLUMNS)
    written = save(panel_c(df), "panel_c", formats, run_dir, out_dir)
    for q in ("anomaly", "report"):
        written += save(margin_vs_deltac(df, q), f"margin_vs_deltac_{q}", formats, run_dir, out_dir)
    return written


_MILESTONES = {"M0": _m0, "M1": _m1, "M2": _m2, "M3": _m3}


def make_figures(run_dir, formats=("png",), milestone: str | None = None, out_dir=None) -> list[Path]:
    """Build every figure for the run in `run_dir` from its saved files, into
    <run_dir>/figures/<format>/ -- or, if `out_dir` is given, straight into that folder. The
    milestone comes from the run's manifest.json unless given. Returns the paths written; missing
    inputs are reported on stderr and that figure is skipped."""
    run_dir = Path(run_dir)
    if milestone is None:
        manifest = run_dir / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"{run_dir} has no manifest.json -- pass a run folder such as runs/M0/<run_id>/")
        milestone = json.loads(manifest.read_text())["milestone"]
    if milestone not in _MILESTONES:
        raise ValueError(f"unknown milestone {milestone!r}; expected one of {sorted(_MILESTONES)}")
    return _MILESTONES[milestone](run_dir, tuple(formats), out_dir)
