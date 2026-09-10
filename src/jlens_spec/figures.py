"""Figures. Every function takes an explicit `path` to save to (same convention as
prompts.render_mask, which the spec gives a literal `path` parameter for -- extended here to the
other figure functions, whose listed signatures omit it but must write a PNG somewhere)."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import loading as loading_mod
from .prompts import render_mask as mask_figure  # re-exported per spec: "mask_figure (delegates to prompts.render_mask)"

__all__ = [
    "mask_figure",
    "loading_heatmap",
    "loading_summary_bars",
    "cka_heatmap",
    "panel_c",
    "margin_vs_deltac",
]


def _as_dict(r):
    return r if isinstance(r, dict) else dataclasses.asdict(r)


def loading_heatmap(df: pd.DataFrame, stimulus_id: str, token: str, path, question_key: str | None = None) -> None:
    """(position x layer) cos heatmap for one prompt. A stimulus appears under 4 questions, so
    `question_key` selects which prompt; if None, the first question present is used."""
    import matplotlib.pyplot as plt

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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def loading_summary_bars(df: pd.DataFrame, band: list[int], path) -> None:
    import matplotlib.pyplot as plt

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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def cka_heatmap(C, layers: list[int], path) -> None:
    import matplotlib.pyplot as plt

    C_np = C.detach().cpu().numpy() if torch.is_tensor(C) else np.asarray(C)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".npy"), C_np)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C_np, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=5, rotation=90)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=5)
    fig.colorbar(im, ax=ax, label="CKA")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half = z * ((phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / denom
    return phat, max(0.0, center - half), min(1.0, center + half)


def panel_c(records, path) -> None:
    """Two rows: top = flip rate with Wilson 95% CI per question, treatment vs 4 controls, split by
    direction x matrix language; bottom = mean margin +/- SE for the same cells."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame([_as_dict(r) for r in records])
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def margin_vs_deltac(records, question_key: str, path) -> None:
    import matplotlib.pyplot as plt

    points = []
    for r in records:
        rd = _as_dict(r)
        if rd["question_key"] != question_key:
            continue
        logs = rd["intervention_logs"]
        if not logs:
            continue
        deltas = [l["delta_c_norm"] if isinstance(l, dict) else l.delta_c_norm for l in logs]
        points.append((sum(deltas) / len(deltas), rd["margin"], rd["kind"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    kinds = sorted(set(p[2] for p in points))
    for kind in kinds:
        xs = [p[0] for p in points if p[2] == kind]
        ys = [p[1] for p in points if p[2] == kind]
        ax.scatter(xs, ys, label=kind, alpha=0.7, s=18)
    ax.set_xlabel("mean |delta_c| over intervened positions")
    ax.set_ylabel("margin")
    ax.set_title(question_key)
    ax.legend(fontsize=7)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
