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
    M2: loadings/<stimulus>_<question>.parquet
          + figure_params.json[band, pairs]  -> loadings/<stimulus>_<question> (cos; 4 pairs x es/fr,
                                                every layer, band dashed), ranks/<stimulus>_<question>
                                                (the same layout, rank in the lens readout), both on
                                                one color scale per run; loading_summary_bars
    M3: records/<stimulus>_<question>.parquet -> panel_c, margin_vs_deltac_<question>, flip_heatmap,
                                                margin_change
        details/ + tokens/ (same file names)  -> masks/<stimulus>_<question> (5 kinds x 2
                                                directions; red = where the stream ACTUALLY changed)
    Several M3 runs (scripts/combine_panel_c.py): combined_panel_c, combined_summary_figures --
        refuse runs that differ in pair, band, position set, lens or model.
`figure_params.json` holds the few values a figure needs that aren't a table (e.g. which band was
used), written by the milestone script so figures never read configs/. The data behind every figure
is uploaded, so any figure can be redrawn from it; the only figures uploaded themselves are an M3
run's summary figures (everything above except its mask figures -- scripts/m3_grid.py) and the
combined figures (scripts/combine_panel_c.py, once every run in them is finished).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import io as io_mod
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
    "loading_heatmap_grid",
    "rank_heatmap_grid",
    "loading_summary_bars",
    "layers_text",
    "panel_c",
    "flip_heatmap",
    "margin_change",
    "margin_vs_deltac",
    "mask_grid",
    "combined_panel_c",
    "combined_summary_figures",
]

FORMATS = ("png", "pdf", "svg")
DPI = 150  # raster formats only

CLASS_COLORS = {
    "template": "#cccccc",
    "instruction": "#f5e6a8",  # the paper's wrapper text in the user message
    "question": "#7fb3ff",
    "matrix": "#7fdc7f",
    "intrusion": "#ffb37f",
    "prompt": "#e6d7ff",  # M1 positive control: a raw prompt, no question/matrix/intrusion classes
}

# Columns read from M3's records.parquet -- logprobs_fp16 (a full-vocab vector per row) is skipped.
_M3_COLUMNS = ["stimulus_id", "question_key", "direction", "kind", "control_index", "flip", "margin",
               "clean_margin", "intervention_logs"]


# Chinese (CJK) characters: matplotlib's default font (DejaVu Sans) has none and would draw empty
# boxes. Figures therefore fall back, character by character, to Noto Sans SC (SIL Open Font
# License), downloaded once from Google Fonts' GitHub repo -- pinned to one commit and checked
# against its sha256 -- and cached in $HF_HOME/fonts/. If Google moves the file, update CJK_FONT_URL
# and CJK_FONT_SHA256 (README: "Chinese characters in figures"). If it can't be fetched, this
# machine's own Chinese font is used if it has one, with a warning; no figure is ever skipped over it.
CJK_FONT_URL = ("https://raw.githubusercontent.com/google/fonts/8e44913e4ff26fc997e6856c1ec40ff4791c98c5/"
                "ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf")
CJK_FONT_SHA256 = "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da"
_SYSTEM_CJK_FONTS = ["Hiragino Sans GB", "Heiti SC", "Noto Sans CJK SC", "Arial Unicode MS"]
_fonts_ready = False


def cjk_font_file() -> Path:
    """The cached Noto Sans SC file, downloading (and sha256-checking) it the first time."""
    cache = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "fonts"
    path = cache / "NotoSansSC.ttf"
    if path.exists() and io_mod.sha256_of(path) == CJK_FONT_SHA256:
        return path
    cache.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".download")
    urllib.request.urlretrieve(CJK_FONT_URL, tmp)
    got = io_mod.sha256_of(tmp)
    if got != CJK_FONT_SHA256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"the downloaded font's sha256 is {got}, expected {CJK_FONT_SHA256} -- not using it")
    os.replace(tmp, path)
    return path


def _regular_cjk_font(path: Path) -> Path:
    """A static Regular (weight 400) copy of the variable-weight Noto Sans SC file, made once next to
    it with fontTools (which comes with matplotlib). matplotlib can't use a variable font's weight
    axis: it registers the file at its default weight, 100 (Thin) -- printing "findfont: Failed to
    find font weight normal, now using 100" -- and draws Chinese characters Thin."""
    out = path.with_name("NotoSansSC-Regular.ttf")
    if out.exists():
        return out
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    print("[figures] making a Regular-weight copy of the Chinese font (once; can take a minute) ...",
          file=sys.stderr, flush=True)
    font = instancer.instantiateVariableFont(TTFont(str(path)), {"wght": 400})
    font["OS/2"].usWeightClass = 400  # what matplotlib reads the weight from
    tmp = out.with_name(out.name + ".tmp")
    font.save(str(tmp))
    tmp.replace(out)
    return out


def _setup_fonts() -> None:
    """Once per process: DejaVu Sans first, then a Chinese font for the characters it lacks."""
    global _fonts_ready
    if _fonts_ready:
        return
    _fonts_ready = True
    import matplotlib
    from matplotlib import font_manager as fm

    families = ["DejaVu Sans"]
    try:
        path = cjk_font_file()
        try:
            path = _regular_cjk_font(path)
        except Exception as e:  # noqa: BLE001 -- reported; the variable-weight file still draws them
            print(f"[figures] WARNING: could not make a Regular-weight copy of the Chinese font ({e!r}); "
                  "Chinese characters will be drawn in its Thin weight.", file=sys.stderr)
        fm.fontManager.addfont(str(path))
        families.append(fm.FontProperties(fname=str(path)).get_name())
    except Exception as e:  # noqa: BLE001 -- reported; figures are still drawn
        have = {f.name for f in fm.fontManager.ttflist}
        system = [n for n in _SYSTEM_CJK_FONTS if n in have]
        families += system
        print(f"[figures] WARNING: could not get the Chinese font Noto Sans SC ({e!r}); "
              + (f"using this machine's {system[0]} instead." if system
                 else "Chinese characters in figures will show as empty boxes."), file=sys.stderr)
    # font.family (not font.sans-serif) is what matplotlib's per-character fallback walks through.
    matplotlib.rcParams["font.family"] = families


def _plt():
    import matplotlib.pyplot as plt

    _setup_fonts()
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


def _draw_token_boxes(ax, rows: pd.DataFrame, cols: int = 12, fontsize: int = 8) -> None:
    """One box per token (text + position index), colored by class, red outline where `edited`, ★ on
    metric_pos. `rows`: pos, token_id, token_text, class, edited, is_metric_pos."""
    from matplotlib.patches import Rectangle

    rows = rows.sort_values("pos")
    n_rows = (len(rows) + cols - 1) // cols
    # `class` is a Python keyword, which itertuples can't use as an attribute name.
    for r in rows.rename(columns={"class": "cls"}).itertuples(index=False):
        i = int(r.pos)
        x, y = i % cols, n_rows - 1 - i // cols
        edited = bool(r.edited)
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=CLASS_COLORS.get(r.cls, "#ffffff"),
                               edgecolor="red" if edited else "black", linewidth=2.5 if edited else 0.6))
        ax.text(x + 0.5, y + 0.45, _token_label(r.token_text, r.token_id), ha="center", va="center",
                fontsize=fontsize, clip_on=True)
        ax.text(x + 0.05, y + 0.93, str(i), ha="left", va="top", fontsize=max(4, fontsize - 3), color="#444444")
        if bool(r.is_metric_pos):
            ax.plot(x + 0.88, y + 0.8, marker="*", color="black", markersize=fontsize + 3)
    ax.set_xlim(0, cols)
    ax.set_ylim(-0.05, n_rows + 0.05)
    ax.axis("off")


def _mask_legend(present_classes, edited_label: str = "edited (red outline)"):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend = [Patch(facecolor=c, edgecolor="black", label=k) for k, c in CLASS_COLORS.items() if k in present_classes]
    legend += [Patch(facecolor="white", edgecolor="red", linewidth=2.5, label=edited_label),
               Line2D([], [], marker="*", color="black", linestyle="", markersize=11,
                      label="metric_pos (answer read here)"),
               Patch(facecolor="none", edgecolor="none", label=NO_TEXT_NOTE)]
    return legend


def mask_figure(rows: pd.DataFrame):
    """One prompt's position mask: one box per token showing its text and position index, colored
    by class, red outline where edited, ★ on metric_pos, legend at the bottom. `rows` = that
    prompt's rows of masks.parquet (see prompts.mask_rows)."""
    plt = _plt()

    rows = rows.sort_values("pos")
    n = len(rows)
    cols = 12
    n_rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(figsize=(cols * 1.35, n_rows * 0.62 + 1.2))
    _draw_token_boxes(ax, rows, cols)
    first = rows.iloc[0]
    ax.set_title(f"{first['stimulus_id']} / {first['question_key']}   "
                 f"({n} tokens, {int(rows['edited'].sum())} edited)", fontsize=11)
    ax.legend(handles=_mask_legend(set(rows["class"])), loc="upper center", bbox_to_anchor=(0.5, 0.0),
              ncol=7, fontsize=8, frameon=False)
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


def _blocks(layers) -> list[tuple[int, int]]:
    """Contiguous [start, end] blocks of a sorted layer list (a band may have several)."""
    layers = sorted(int(l) for l in layers)
    blocks, start = [], layers[0]
    for a, b in zip(layers, layers[1:]):
        if b != a + 1:
            blocks.append((start, a))
            start = b
    blocks.append((start, layers[-1]))
    return blocks


def layers_text(layers) -> str:
    """A band's layers as text: [9, ..., 18] -> '9–18'; a split band -> '9–13 and 16–18'."""
    parts = [f"{a}–{b}" if b > a else str(a) for a, b in _blocks(layers)]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


# Position classes in the order they appear in a prompt (M2 summary bars).
CLASS_ORDER = ("template", "instruction", "question", "matrix", "intrusion")


def _pair_grid(df: pd.DataFrame, pairs: dict, band: list[int], column: str, imshow_kw: dict,
               colorbar_label: str, title: str, overlay=None):
    """The layout shared by M2's per-prompt figures: rows = the pairs in `pairs` ({name: [es word,
    fr word]}), columns = each pair's Spanish and French member; each panel a (position x layer)
    heatmap of `column` over every layer saved, the band's edges dashed, the position classes marked
    on the right. `overlay(ax, grid, layers)` may draw on each panel."""
    plt = _plt()

    layers = sorted(df["layer"].unique())
    n_pos = int(df["pos"].max()) + 1
    cls = df.drop_duplicates("pos").set_index("pos")["class"].reindex(range(n_pos)).tolist()
    names = list(pairs)
    fig, axes = plt.subplots(len(names), 2, figsize=(13, 3.5 * len(names) + 1), squeeze=False)
    fig.subplots_adjust(hspace=0.45)  # room between a panel's "layer" label and the title below it
    extent = [layers[0] - 0.5, layers[-1] + 0.5, n_pos - 0.5, -0.5]
    im = None
    for i, name in enumerate(names):
        for j, (lang, word) in enumerate(zip(("es", "fr"), pairs[name])):
            ax = axes[i][j]
            sub = df[df["token"] == word]
            if sub.empty:
                ax.set_title(f"{name}: {word!r} -- no rows", fontsize=8)
                ax.axis("off")
                continue
            grid = sub.pivot(index="pos", columns="layer", values=column).reindex(index=range(n_pos), columns=layers)
            im = ax.imshow(grid.values, aspect="auto", extent=extent, interpolation="nearest", **imshow_kw)
            if overlay is not None:
                overlay(ax, grid, layers)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
            for a, b in _blocks(band):
                for x in (a - 0.5, b + 0.5):
                    ax.axvline(x, color="black", linestyle="--", linewidth=0.8)
            for p in range(1, n_pos):  # class boundaries
                if cls[p] != cls[p - 1]:
                    ax.axhline(p - 0.5, color="#555555", linewidth=0.4)
            starts = [p for p in range(n_pos) if p == 0 or cls[p] != cls[p - 1]]
            for k, p in enumerate(starts):
                end = starts[k + 1] if k + 1 < len(starts) else n_pos
                ax.text(layers[-1] + 1.0, (p + end - 1) / 2, cls[p], fontsize=5, va="center", clip_on=False)
            ax.set_title(f"{name} / {lang}: {word!r}", fontsize=8)
            ax.set_xlabel("layer", fontsize=7)
            ax.set_ylabel("position", fontsize=7)
            ax.tick_params(labelsize=6)
    if im is not None:
        fig.colorbar(im, ax=axes, label=colorbar_label, shrink=0.6)
    fig.suptitle(title, fontsize=10)
    return fig


def loading_heatmap_grid(df: pd.DataFrame, pairs: dict, band: list[int], vmax: float | None = None,
                         band_name: str = "workspace"):
    """One prompt: a (position x layer) heatmap of cos(h, v_token) for every pair member (layout:
    _pair_grid). Symmetric color scale -+`vmax`: M2 passes one value for all its prompts (the run's
    largest |cos|), so a color means the same in every figure; without it, this prompt's largest.
    `df` = that prompt's loadings rows (M2's loadings/<stimulus>_<question>.parquet)."""
    first = df.iloc[0]
    one_scale = vmax is not None
    vmax = vmax if one_scale else max(float(df["cos"].abs().max()), 1e-6)
    return _pair_grid(
        df, pairs, band, "cos", {"cmap": "RdBu_r", "vmin": -vmax, "vmax": vmax},
        "cos(h, v_token)" + (" -- one scale for every prompt of this run" if one_scale else ""),
        f"{first['stimulus_id']} / {first['question_key']} -- loading (cos) of each pair member "
        f"(dashed = {band_name} layers {layers_text(band)})")


def _rank_contour(ax, grid: pd.DataFrame, layers) -> None:
    """A white line around the cells where the rank is at most 100 (in the top-100 readout)."""
    vals = np.ma.masked_invalid(grid.to_numpy(dtype=float))
    if vals.count() and vals.min() <= 100 < vals.max():
        ax.contour(np.asarray(layers, dtype=float), np.arange(vals.shape[0], dtype=float), vals,
                   levels=[100.5], colors="white", linewidths=0.6)


def rank_heatmap_grid(df: pd.DataFrame, pairs: dict, band: list[int], vmax: int | None = None,
                      band_name: str = "workspace"):
    """One prompt: the rank of every pair member in the full lens readout (1 = the top token) at
    each (position x layer), in the same layout as loading_heatmap_grid. Log color scale from rank 1
    (darkest) to `vmax` -- M2 passes the run's largest rank, so one scale serves every prompt; a
    white line marks rank 100 (the top-100 readout that M3's control coverage counts)."""
    from matplotlib.colors import LogNorm

    first = df.iloc[0]
    one_scale = vmax is not None
    vmax = max(int(vmax if one_scale else df["rank"].max()), 2)
    return _pair_grid(
        df, pairs, band, "rank", {"cmap": "magma", "norm": LogNorm(vmin=1, vmax=vmax)},
        "rank in the lens readout (1 = top; log scale" + ("; one scale for every prompt of this run)" if one_scale
                                                          else ")"),
        f"{first['stimulus_id']} / {first['question_key']} -- rank of each pair member in the lens readout "
        f"(dashed = {band_name} layers {layers_text(band)}; white line = rank 100)",
        overlay=_rank_contour)


def loading_summary_bars(df: pd.DataFrame, band: list[int], band_name: str = "workspace"):
    """Mean cos(h, v_token) per position class and language token, averaged over the band's layers,
    every position of the class and the prompts: three rows on one y scale -- all prompts, the
    Spanish passages' prompts, the French passages' prompts. Classes in prompt order."""
    plt = _plt()

    d = df[df["layer"].isin(band)].copy()
    d["matrix_lang"] = d["stimulus_id"].map(loading_mod._matrix_lang_of)
    tokens = sorted(d["token"].unique())
    present = set(d["class"])
    classes = [c for c in CLASS_ORDER if c in present] + sorted(present - set(CLASS_ORDER))
    rows = [("all prompts", d), ("Spanish passages", d[d["matrix_lang"] == "es"]),
            ("French passages", d[d["matrix_lang"] == "fr"])]

    fig, axes = plt.subplots(3, 1, figsize=(max(9.0, 0.3 * len(tokens) * len(classes) + 3.5), 9.5),
                             sharex=True, sharey=True)
    width = 0.8 / max(len(tokens), 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(classes))
    for ax, (label, sub) in zip(axes, rows):
        means = sub.groupby(["class", "token"])["cos"].mean()
        for i, tok in enumerate(tokens):
            vals = [float(means.get((c, tok), np.nan)) for c in classes]
            ax.bar(x + (i - (len(tokens) - 1) / 2) * width, vals, width=width, color=colors[i % len(colors)],
                   label=repr(tok))
        ax.axhline(0, color="black", linewidth=0.6)
        n_prompts = len(sub[["stimulus_id", "question_key"]].drop_duplicates())
        ax.set_title(f"{label} ({n_prompts} prompts)", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(classes, fontsize=9)
    axes[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="token", title_fontsize=8)
    fig.supylabel(f"mean cos(h, v_token), averaged over the {band_name} layers ({layers_text(band)})", fontsize=9)
    fig.suptitle("Loading of each language token, by position class (clean pass)", fontsize=10)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.96))  # tight_layout ignores supylabel / suptitle: leave room
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


QUESTION_ORDER = ("report", "hello", "anomaly", "content")
KIND_LABELS = {"identity": "identity", "swap": "swap", "label_to_present": "label→present",
               "big_nonlabel": "big non-label", "random_direction": "random"}
# The swap is symmetric in its two tokens, so these kinds make the same edit in both directions:
# the presentation figures count each passage once (its m2i cell). label_to_present differs by
# direction (it removes a different label), so both directions count.
SAME_EDIT_BOTH_DIRECTIONS = ("identity", "swap", "big_nonlabel", "random_direction")
# Each question's margin, in words (metrics.question_margin): the y label of its margin_change panel.
MARGIN_LABELS = {
    "anomaly": "log P(Yes) − log P(No)",
    "content": "log P(Yes) − log P(No)",
    "report": "log P(intrusion language) − log P(passage language)",
    "hello": "log P(intrusion-language hello) − log P(passage-language hello)",
}


def _questions_in(df: pd.DataFrame) -> list[str]:
    present = set(df["question_key"])
    return [q for q in QUESTION_ORDER if q in present] + sorted(present - set(QUESTION_ORDER))


def _one_edit_each(df: pd.DataFrame) -> pd.DataFrame:
    """Records with each kind in SAME_EDIT_BOTH_DIRECTIONS kept once per passage (the m2i cell)."""
    d = df.assign(_o=(df["direction"] != "m2i").astype(int)).sort_values("_o", kind="stable")
    dup = d["kind"].isin(SAME_EDIT_BOTH_DIRECTIONS) & d.duplicated(
        ["stimulus_id", "question_key", "kind", "control_index"])
    return d[~dup].drop(columns="_o")


def title_suffix(params: dict) -> str:
    """' -- position set 'question', workspace layers 9–18' from an M3 run's figure_params.json."""
    ps, layers = params.get("position_set"), params.get("band_layers")
    return (f" -- position set '{ps}'" if ps else "") + \
        (f", {params.get('band', 'band')} layers {layers_text(layers)}" if layers else "")


def flip_heatmap(df: pd.DataFrame, suffix: str = ""):
    """How often each edit flipped the answer: rows = questions, columns = the five kinds; the shade
    is the flip rate (fixed 0-1 scale, so a shade means the same in every figure) and each box says
    flipped/total. Each passage counts once for the kinds that make the same edit in both directions;
    label_to_present counts both directions; the pooled controls count each of their cells. Cells
    without a clean baseline (flip = None) are left out. `df` = M3 records."""
    plt = _plt()

    d = _one_edit_each(df)
    questions = _questions_in(d)
    kinds = [k for k in MASK_KINDS if k in set(d["kind"])]
    rate = np.full((len(questions), len(kinds)), np.nan)
    text = {}
    for i, q in enumerate(questions):
        for j, k in enumerate(kinds):
            flips = d.loc[(d["question_key"] == q) & (d["kind"] == k), "flip"].dropna()
            if len(flips):
                n_flip = int(flips.astype(bool).sum())
                rate[i, j] = n_flip / len(flips)
                text[i, j] = f"{n_flip}/{len(flips)}"
    fig, ax = plt.subplots(figsize=(1.55 * len(kinds) + 2.6, 0.75 * len(questions) + 2.0))
    im = ax.imshow(rate, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, j), t in text.items():
        ax.text(j, i, t, ha="center", va="center", fontsize=10, color="white" if rate[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([KIND_LABELS.get(k, k) for k in kinds], fontsize=9)
    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels(questions, fontsize=9)
    fig.colorbar(im, ax=ax, label="flip rate", fraction=0.046, pad=0.04)
    ax.set_title(f"Flip rate: flipped / total{suffix}\nswap, big non-label, random, identity: each passage once "
                 "(both directions are the same edit); label→present: both directions", fontsize=9)
    return fig


def _per_passage_delta(df: pd.DataFrame) -> pd.DataFrame:
    """Every non-identity cell with `delta` (margin − the same prompt's clean margin) and `lang`,
    one edit each. What margin_change's dots are averaged from."""
    d = _one_edit_each(df[df["kind"] != "identity"])
    return d.assign(delta=d["margin"] - d["clean_margin"], lang=d["stimulus_id"].map(loading_mod._matrix_lang_of))


def delta_limits(df: pd.DataFrame, pad: float = 0.05) -> tuple[float, float]:
    """The y range margin_change needs for `df`, from the per-passage means it draws as dots. Pass it
    back as `ylim` when the same scale must hold across several figures (m3_by_language)."""
    dots = _per_passage_delta(df).groupby(["question_key", "kind", "stimulus_id", "lang"])["delta"].mean()
    lo, hi = float(dots.min()), float(dots.max())
    margin = pad * (hi - lo) or 1.0
    return lo - margin, hi + margin


def margin_change(df: pd.DataFrame, suffix: str = "", ylim: tuple[float, float] | None = None):
    """How far each edit moved the answer: one panel per question; for swap and the three controls,
    Δ = margin with the edit − the clean margin of the same prompt, in that question's own units
    (MARGIN_LABELS; natural log). Bar = the mean over passages; dots = the passages (a pooled kind's
    dot averages that passage's cells), colored by passage language. Identity is the zero line. One
    y scale for every panel -- `ylim` fixes it (delta_limits), so figures drawn from different subsets
    of the same run can be read against each other. `df` = M3 records."""
    from matplotlib.lines import Line2D

    plt = _plt()

    d = _per_passage_delta(df)
    kinds = [k for k in ("swap", "label_to_present", "big_nonlabel", "random_direction") if k in set(d["kind"])]
    questions = _questions_in(d)
    bar_colors = {"swap": "#c0392b", "label_to_present": "#8a8a8a", "big_nonlabel": "#a9a9a9",
                  "random_direction": "#c8c8c8"}
    dot_colors = {"es": "#e69f00", "fr": "#0072b2"}
    rng = np.random.default_rng(0)  # the dots' sideways jitter, the same every time
    fig, axes = plt.subplots(1, len(questions), figsize=(3.4 * len(questions) + 0.8, 5.2), sharey=True,
                             squeeze=False)
    for ax, q in zip(axes[0], questions):
        per_passage = d[d["question_key"] == q].groupby(["kind", "stimulus_id", "lang"])["delta"].mean().reset_index()
        for x, k in enumerate(kinds):
            pts = per_passage[per_passage["kind"] == k]
            if pts.empty:
                continue
            ax.bar(x, pts["delta"].mean(), width=0.7, color=bar_colors[k], zorder=1)
            ax.scatter(x + rng.uniform(-0.2, 0.2, len(pts)), pts["delta"], s=16, zorder=2,
                       c=[dot_colors.get(lang, "black") for lang in pts["lang"]], edgecolors="white", linewidths=0.4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([KIND_LABELS[k] for k in kinds], rotation=30, ha="right", fontsize=8)
        ax.set_title(q, fontsize=10)
        ax.set_ylabel(f"Δ [{MARGIN_LABELS.get(q, 'margin')}]", fontsize=8)
        ax.tick_params(axis="y", labelleft=True, labelsize=8)  # every panel has its own units
        if ylim is not None:
            ax.set_ylim(*ylim)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=dot_colors["es"], label="Spanish passages"),
                        Line2D([], [], marker="o", linestyle="", color=dot_colors["fr"], label="French passages")],
               loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"How far each edit moved the answer{suffix}\nΔ = margin with the edit − clean margin (same prompt); "
                 "bar = mean over passages, dots = passages", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    return fig


def deltac_points(df: pd.DataFrame, question_key: str) -> list[tuple[float, float, str]]:
    """(mean |delta_c| over the intervened positions, margin, kind) for every cell of this question
    that recorded a log. `intervention_logs` comes back from parquet as an array of dicts."""
    points = []
    for r in df[df["question_key"] == question_key].itertuples(index=False):
        logs = r.intervention_logs
        if logs is None or len(logs) == 0:
            continue
        deltas = [lg["delta_c_norm"] for lg in logs]
        points.append((sum(deltas) / len(deltas), r.margin, r.kind))
    return points


def margin_vs_deltac(df: pd.DataFrame, question_key: str, suffix: str = "", xlim=None, ylim=None):
    """Margin vs mean |delta_c| over intervened positions, one point per cell. `xlim`/`ylim` fix the
    axes so figures drawn from different subsets of one run can be read against each other
    (m3_by_language). `df` = records.parquet."""
    plt = _plt()

    points = deltac_points(df, question_key)

    fig, ax = plt.subplots(figsize=(6, 5))
    for kind in sorted(set(p[2] for p in points)):
        ax.scatter([p[0] for p in points if p[2] == kind], [p[1] for p in points if p[2] == kind],
                   label=kind, alpha=0.7, s=18)
    ax.set_xlabel("mean |delta_c| over intervened positions")
    ax.set_ylabel("margin")
    ax.set_title(f"{question_key}{suffix}", fontsize=10)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


MASK_KINDS = ("identity", "swap", "label_to_present", "big_nonlabel", "random_direction")


def mask_grid(tokens: pd.DataFrame, details: pd.DataFrame):
    """One prompt's M3 mask sanity figure, drawn from where the stream ACTUALLY changed: a 5 x 2
    grid -- rows = kinds, columns = directions. A token is outlined red if it changed at any layer
    in any cell of that kind and direction (label_to_present and big_nonlabel: their 3 cells
    combined -- they plan the same positions; details/ has every cell separately). `tokens` = the
    prompt's rows of tokens/ (pos, token_id, token_text, class, is_metric_pos); `details` = its
    rows of details/ (kind, direction, control_index, layer, pos, planned, change)."""
    plt = _plt()

    tokens = tokens.sort_values("pos")
    directions = sorted(details["direction"].unique(), key=lambda d: (d != "m2i", d))  # m2i on the left
    cols = 16
    n_rows = (len(tokens) + cols - 1) // cols
    panel_h = n_rows * 0.42 + 0.6
    height = panel_h * len(MASK_KINDS) + 1
    fig, axes = plt.subplots(len(MASK_KINDS), len(directions), squeeze=False,
                             figsize=(cols * 0.8 * len(directions), height))
    # margins in inches, not fractions: room for the title above and the legend below, no more
    fig.subplots_adjust(top=1 - 0.6 / height, bottom=0.5 / height, left=0.01, right=0.99,
                        hspace=0.35 / panel_h, wspace=0.04)
    changed_at = details[details["change"] != 0].groupby(["kind", "direction"])["pos"].apply(set).to_dict()
    planned = set(details.loc[details["planned"], "pos"])
    for i, kind in enumerate(MASK_KINDS):
        for j, direction in enumerate(directions):
            ax = axes[i][j]
            changed = changed_at.get((kind, direction), set())
            _draw_token_boxes(ax, tokens.assign(edited=tokens["pos"].isin(changed)), cols, fontsize=6)
            n_cells = details[(details["kind"] == kind) & (details["direction"] == direction)][
                ["control_index"]].drop_duplicates().shape[0]
            combined = f", {n_cells} cells combined" if n_cells > 1 else ""
            ax.set_title(f"{kind} / {direction}{combined}: {len(changed)} changed, {len(planned)} planned",
                         fontsize=8)
    first = tokens.iloc[0]
    fig.suptitle(f"{first['stimulus_id']} / {first['question_key']} -- where the stream actually changed",
                 fontsize=11, y=1 - 0.1 / height)
    fig.legend(handles=_mask_legend(set(tokens["class"]), "changed at any layer (red outline)"),
               loc="lower center", ncol=8, fontsize=8, frameon=False)
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
    if _missing(run_dir, "loadings", "figure_params.json"):
        return []
    params = _params(run_dir)
    band, band_name, pairs = params["band"], params.get("band_name", "workspace"), params["pairs"]
    files = sorted((run_dir / "loadings").glob("*.parquet"))
    # One color scale per figure kind for the whole run (a color means the same in every prompt's
    # figure): the largest |cos| and the largest rank of any pair member, anywhere in the run.
    words = {w for pair in pairs.values() for w in pair}
    vmax_cos, vmax_rank = 1e-6, 2
    for f in files:
        s = pd.read_parquet(f, columns=["token", "cos", "rank"])
        s = s[s["token"].isin(words)]
        if len(s):
            vmax_cos, vmax_rank = max(vmax_cos, float(s["cos"].abs().max())), max(vmax_rank, int(s["rank"].max()))
    written = []
    parts = []
    for f in files:
        df = pd.read_parquet(f)
        written += save(loading_heatmap_grid(df, pairs, band, vmax=vmax_cos, band_name=band_name),
                        f"loadings/{f.stem}", formats, run_dir, out_dir)
        written += save(rank_heatmap_grid(df, pairs, band, vmax=vmax_rank, band_name=band_name),
                        f"ranks/{f.stem}", formats, run_dir, out_dir)
        parts.append(df[df["layer"].isin(band)])
    if parts:
        written += save(loading_summary_bars(pd.concat(parts, ignore_index=True), band, band_name),
                        "loading_summary_bars", formats, run_dir, out_dir)
    return written


def read_records(run_dir, columns=_M3_COLUMNS) -> pd.DataFrame:
    """An M3 run's records (one file per prompt in records/), without the full-vocabulary vectors
    unless asked for."""
    return pd.read_parquet(Path(run_dir) / "records", columns=list(columns))


def _m3(run_dir: Path, formats, out_dir) -> list[Path]:
    if _missing(run_dir, "records"):
        return []
    df = read_records(run_dir)
    suffix = title_suffix(_params(run_dir))
    written = save(panel_c(df), "panel_c", formats, run_dir, out_dir)
    written += save(flip_heatmap(df, suffix), "flip_heatmap", formats, run_dir, out_dir)
    written += save(margin_change(df, suffix), "margin_change", formats, run_dir, out_dir)
    for q in sorted(df["question_key"].unique()):
        written += save(margin_vs_deltac(df, q), f"margin_vs_deltac_{q}", formats, run_dir, out_dir)
    if not _missing(run_dir, "details", "tokens"):
        for f in sorted((run_dir / "details").glob("*.parquet")):
            tok = run_dir / "tokens" / f.name
            if not tok.exists():
                print(f"[figures] skipping mask figure {f.stem}: tokens/{f.name} not found", file=sys.stderr)
                continue
            fig = mask_grid(pd.read_parquet(tok), pd.read_parquet(
                f, columns=["kind", "direction", "control_index", "layer", "pos", "planned", "change"]))
            written += save(fig, f"masks/{f.stem}", formats, run_dir, out_dir)
    return written


LANG_NAMES = {"es": "Spanish passages", "fr": "French passages"}
LANG_COLORS = {"es": "#e69f00", "fr": "#0072b2"}


def move_rows(df: pd.DataFrame, x_field: str = "delta_c_norm") -> pd.DataFrame:
    """One row per non-identity cell: where it started (its prompt's clean margin, at size 0) and
    where it ended (the cell's own margin, at its mean `x_field` over the intervened positions).
    `x_field` is "delta_c_norm" (the size in the swap's own 2-D plane -- 0 for random_direction,
    which moves along a direction outside that plane) or "delta_h_norm" (the size in the residual
    stream, which every kind is matched on).

    One edit each, as in every presentation figure: swap, big_nonlabel and random_direction are the
    same computation in both directions (the paper's swap exchanges the two coordinates, so naming
    either token the source gives the same target), so their i2m cell is left out rather than counted
    twice; label_to_present removes a different label in each direction, so both of its cells stay.
    Nothing is averaged. Each row carries the clean margin of its own prompt -- the one control value,
    repeated so that every experimental cell has its pair."""
    rows = []
    for r in _one_edit_each(df[df["kind"] != "identity"]).itertuples(index=False):
        logs = r.intervention_logs
        if logs is None or len(logs) == 0:
            continue
        rows.append({"stimulus_id": r.stimulus_id, "question_key": r.question_key, "kind": r.kind,
                     "direction": r.direction, "control_index": r.control_index,
                     "lang": loading_mod._matrix_lang_of(r.stimulus_id),
                     "size": sum(lg[x_field] for lg in logs) / len(logs),
                     "margin": r.margin, "clean_margin": r.clean_margin})
    return pd.DataFrame(rows)


def margin_moves(df: pd.DataFrame, question_key: str, suffix: str = "", x_field: str = "delta_c_norm",
                 xlim=None, ylim=None):
    """Where each edit moved the answer, one panel per intervention kind: every cell is a line from
    its prompt's clean margin at size 0 (what identity gives -- identity changes nothing) out to the
    margin that cell ended with, at that cell's mean edit size. Colored by passage language; the
    dashed line at 0 is the flip boundary, so a line crossing it is a flipped answer.

    A pooled kind draws one line per cell (label_to_present: 3 tokens x 2 directions per passage;
    big_nonlabel: 3 pairs), so the spread within a kind is visible. All panels share both axes.
    `df` = M3 records; see move_rows for `x_field`."""
    from matplotlib.lines import Line2D

    plt = _plt()

    rows = move_rows(df[df["question_key"] == question_key], x_field)
    kinds = [k for k in ("swap", "label_to_present", "big_nonlabel", "random_direction")
             if k in set(rows["kind"])]
    fig, axes = plt.subplots(1, len(kinds) or 1, figsize=(2.9 * (len(kinds) or 1) + 1.0, 4.6),
                             sharey=True, sharex=True, squeeze=False)
    for ax, k in zip(axes[0], kinds):
        for r in rows[rows["kind"] == k].itertuples(index=False):
            color = LANG_COLORS.get(r.lang, "black")
            ax.plot([0.0, r.size], [r.clean_margin, r.margin], color=color, linewidth=0.9, alpha=0.55, zorder=1)
            ax.scatter([r.size], [r.margin], s=14, color=color, edgecolors="white", linewidths=0.4, zorder=2)
            ax.scatter([0.0], [r.clean_margin], s=9, color="black", alpha=0.5, zorder=2)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_title(KIND_LABELS.get(k, k), fontsize=9)
        ax.set_xlabel(f"mean |{'Δc' if x_field == 'delta_c_norm' else 'Δh'}| over intervened positions", fontsize=8)
        ax.tick_params(labelsize=8)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0][0].set_ylabel(MARGIN_LABELS.get(question_key, "margin"), fontsize=8)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="-", color=LANG_COLORS["es"], label="Spanish passages"),
                        Line2D([], [], marker="o", linestyle="-", color=LANG_COLORS["fr"], label="French passages"),
                        Line2D([], [], marker="o", linestyle="", color="black", alpha=0.5, label="clean (identity)")],
               loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"Where each edit moved the answer -- {question_key}{suffix}\none line per cell, from its prompt's "
                 "clean margin to the edited margin; dashed line = the flip boundary", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    return fig


def m3_moves(run_dir, formats=("png",), out_dir=None, x_field: str = "delta_c_norm") -> list[Path]:
    """`margin_moves_<question>` for every question of an M3 run -- both languages together, and then
    `_es` and `_fr` on their own -- every one of them on the WHOLE run's axes, so the three can be
    read against each other (`python scripts/make_figures.py <run folder> --moves`). Nothing is
    recomputed from the model."""
    run_dir = Path(run_dir)
    if _missing(run_dir, "records"):
        return []
    df = read_records(run_dir)
    df = df.assign(lang=df["stimulus_id"].map(loading_mod._matrix_lang_of))
    suffix = title_suffix(_params(run_dir))
    rows = move_rows(df, x_field)
    xlim = _pad((0.0, float(rows["size"].max()))) if len(rows) else None
    ylim = _pad((float(min(rows["margin"].min(), rows["clean_margin"].min())),
                 float(max(rows["margin"].max(), rows["clean_margin"].max())))) if len(rows) else None
    subsets = [("", df, "")] + [(f"_{lang}", df[df["lang"] == lang], f" -- {LANG_NAMES.get(lang, lang)}")
                                for lang in sorted(df["lang"].dropna().unique())]
    written = []
    for q in sorted(df["question_key"].unique()):
        for tag, d, extra in subsets:
            written += save(margin_moves(d, q, suffix + extra, x_field, xlim=xlim, ylim=ylim),
                            f"margin_moves_{q}{tag}", formats, run_dir, out_dir)
    return written


def m3_by_language(run_dir, formats=("png",), out_dir=None) -> list[Path]:
    """The M3 summary figures again, drawn for one passage language at a time: `flip_heatmap_<lang>`,
    `margin_change_<lang>`, `margin_vs_deltac_<question>_<lang>`. Same figures, same code, the records
    filtered by the passage's matrix language -- so a one-sided effect (the workspace runs flip the
    French passages and not the Spanish ones) is visible per language instead of pooled.

    The axes are fixed to the WHOLE run's range (delta_limits, deltac_points), so the two languages'
    figures can be read against each other; the flip heatmap is already on a fixed 0-1 scale. panel_c
    is not repeated -- it splits by language already (one bar per direction x language).

    Nothing is recomputed from the model: `python scripts/make_figures.py <run folder> --by-language`.
    """
    run_dir = Path(run_dir)
    if _missing(run_dir, "records"):
        return []
    df = read_records(run_dir)
    df = df.assign(lang=df["stimulus_id"].map(loading_mod._matrix_lang_of))
    base = title_suffix(_params(run_dir))
    ylim = delta_limits(df)
    questions = sorted(df["question_key"].unique())
    limits = {}
    for q in questions:
        pts = deltac_points(df, q)
        limits[q] = ((min(p[0] for p in pts), max(p[0] for p in pts)),
                     (min(p[1] for p in pts), max(p[1] for p in pts))) if pts else (None, None)
    written = []
    for lang in sorted(df["lang"].dropna().unique()):
        d = df[df["lang"] == lang]
        suffix = f"{base} -- {LANG_NAMES.get(lang, lang)}"
        written += save(flip_heatmap(d, suffix), f"flip_heatmap_{lang}", formats, run_dir, out_dir)
        written += save(margin_change(d, suffix, ylim=ylim), f"margin_change_{lang}", formats, run_dir, out_dir)
        for q in questions:
            xlim, ylim_q = limits[q]
            written += save(margin_vs_deltac(d, q, suffix=f" -- {LANG_NAMES.get(lang, lang)}", xlim=_pad(xlim),
                                             ylim=_pad(ylim_q)),
                            f"margin_vs_deltac_{q}_{lang}", formats, run_dir, out_dir)
    return written


def _pad(lim, frac: float = 0.05):
    """A (lo, hi) range widened by `frac` of its width, so points don't sit on the frame."""
    if lim is None:
        return None
    lo, hi = lim
    pad = frac * (hi - lo) or 1.0
    return lo - pad, hi + pad


# Settings two M3 runs must share to be drawn in one combined panel c.
COMBINE_KEYS = ("pair_name", "band_layers", "position_set", "lens_sha", "model_revision")


def combine_check(run_dirs) -> tuple[dict, list[str]]:
    """What each run used (from its figure_params.json and records) and every mismatch in
    COMBINE_KEYS. Git commits are returned for the record but never a reason to refuse."""
    info = {}
    for d in run_dirs:
        d = Path(d)
        params = _params(d)
        rec = read_records(d, ["pair_name", "lens_sha", "model_revision", "git_commit", "position_set"])
        info[d.name] = {
            "pair_name": sorted(rec["pair_name"].unique().tolist()),
            "band_layers": params.get("band_layers"),
            "position_set": sorted({tuple(sorted(p)) for p in rec["position_set"]}),
            "lens_sha": sorted(rec["lens_sha"].unique().tolist()),
            "model_revision": sorted(rec["model_revision"].unique().tolist()),
            "git_commits": sorted(rec["git_commit"].unique().tolist()),
        }
    problems = []
    first = next(iter(info))
    for key in COMBINE_KEYS:
        values = {name: v[key] for name, v in info.items()}
        # band_layers is one list per run; the others list every distinct value found in the run's
        # records, which must be exactly one.
        mixed = key != "band_layers" and any(len(v) != 1 for v in values.values())
        if mixed or any(v != values[first] for v in values.values()):
            problems.append(f"{key}: {values}")
    return info, problems


def combined_panel_c(run_dirs, formats=("png",), out_dir=None):
    """One panel c over several M3 runs (e.g. positives + anomaly for one position set). Refuses
    (ValueError) if the runs differ in pair, band, position set, lens or model. Returns the paths
    written into `out_dir`."""
    info, problems = combine_check(run_dirs)
    if problems:
        raise ValueError("these runs can't be combined into one panel c -- they differ in:\n  "
                         + "\n  ".join(problems))
    df = pd.concat([read_records(d) for d in run_dirs], ignore_index=True)
    return save(panel_c(df), "panel_c_combined", formats, run_dir=None, out_dir=out_dir), info


def combined_summary_figures(run_dirs, formats=("png",), out_dir=None) -> list[Path]:
    """flip_heatmap and margin_change over several M3 runs -- e.g. a position set's positives and
    anomaly runs, i.e. all four questions in one figure. Refuses (ValueError) runs that differ, like
    combined_panel_c. Returns the paths written into `out_dir`."""
    _, problems = combine_check(run_dirs)
    if problems:
        raise ValueError("these runs can't be combined -- they differ in:\n  " + "\n  ".join(problems))
    df = pd.concat([read_records(d) for d in run_dirs], ignore_index=True)
    suffix = title_suffix(_params(Path(run_dirs[0])))
    return (save(flip_heatmap(df, suffix), "flip_heatmap_combined", formats, run_dir=None, out_dir=out_dir)
            + save(margin_change(df, suffix), "margin_change_combined", formats, run_dir=None, out_dir=out_dir))


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
