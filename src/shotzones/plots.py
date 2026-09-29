"""Figures produced by the pipeline. One function per figure, each returns a Figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Validated categorical order (blue, orange, aqua, yellow, magenta, green, violet, red)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEG, POS = "#e34948", "#2a78d6"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})


def _field(ax):
    ax.add_patch(plt.Circle((175, 175), 165, fill=False, color=MUTED, ls="--", lw=1, alpha=0.6))
    ax.set_xlim(0, 350)
    ax.set_ylim(350, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("FieldX")
    ax.set_ylabel("FieldY")
    ax.grid(False)


def zone_map(df: pd.DataFrame, centres: np.ndarray, sample: int = 4000):
    fig, ax = plt.subplots(figsize=(6, 6))
    d = df.sample(min(sample, len(df)), random_state=42)
    for z in sorted(d["shot_zone"].unique()):
        s = d[d["shot_zone"] == z]
        ax.scatter(s["FieldX"], s["FieldY"], s=6, alpha=0.35, color=PALETTE[z % 8], lw=0)
    for z, (cx, cy) in enumerate(centres):
        ax.scatter(cx, cy, s=120, color=PALETTE[z % 8], edgecolor="white", lw=2, zorder=5)
        ax.annotate(f"Z{z}", (cx, cy), xytext=(7, 5), textcoords="offset points",
                    fontweight="bold", color=INK)
    _field(ax)
    ax.set_title("Shot zones from k-means on landing position", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def confusion(y_true, y_pred, k: int):
    cm = confusion_matrix(y_true, y_pred, labels=range(k))
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=max(0.6, norm.max()))
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.4 else INK)
    ax.set_xticks(range(k), [f"Z{i}" for i in range(k)])
    ax.set_yticks(range(k), [f"Z{i}" for i in range(k)])
    ax.set_xlabel("Predicted zone")
    ax.set_ylabel("Actual zone")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Share of actual zone")
    ax.set_title("Confusion matrix, row-normalised (LOMO)", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def importance(imp: pd.DataFrame, top: int = 20):
    colours = {"game state": PALETTE[0], "ball tracking": PALETTE[1],
               "delivery type": PALETTE[2], "fielder": PALETTE[6]}
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(d["feature"], d["importance"], color=[colours[g] for g in d["group"]], height=0.7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
    ax.legend(handles, colours.keys(), frameon=False, loc="lower right")
    ax.set_xlabel("XGBoost gain importance")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Top {top} features", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def uplift_by_group(tbl: pd.DataFrame, title: str):
    """Point estimate and CI of DR uplift per group, zero line for reference."""
    d = tbl.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(d) + 1.4))
    y = np.arange(len(d))
    colours = [POS if v > 0 else NEG for v in d["DR_uplift"]]
    ax.hlines(y, d["uplift_ci_low"], d["uplift_ci_high"], color=colours, lw=2)
    ax.scatter(d["DR_uplift"], y, color=colours, s=60, zorder=3, edgecolor="white", lw=2)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, d["group"])
    ax.set_xlabel("Doubly-robust uplift vs batters' own choices (runs per ball, 95% CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontweight="bold")
    fig.tight_layout()
    return fig
