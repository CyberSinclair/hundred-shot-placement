"""Draw the README figures from the numbers published in the dissertation.

The real data cannot be shared, so these charts are built only from aggregate
results reported in the dissertation's tables (cited inline below). Nothing here
reads delivery-level data. Each figure is drawn twice, for GitHub's light and
dark themes.

    python scripts/make_readme_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  s1="#2a78d6", s2="#eb6834", neutral="#b8b7b1",
                  ramp=["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]),
    "dark": dict(surface="#0d1117", ink="#ffffff", muted="#c3c2b7", grid="#30363d",
                 s1="#3987e5", s2="#d95926", neutral="#6e6d68",
                 ramp=["#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]),
}

# --- Published results ------------------------------------------------------
# Table 5.4
ACCURACY = [("Random guess (1 in 8)", 12.5), ("Always predict the\nlargest zone", 18.2),
            ("XGBoost, leave-one-\nmatch-out", 36.7)]

# Table 5.12 (naive runs difference, followed minus not) and Table 5.17 (DR uplift)
# Pooled figures from Table 5.9 and Section 5.7.2.
PHASES = ["All deliveries", "Opening (balls 1-49)", "Middle (balls 50-79)", "Death (balls 80-99)"]
NAIVE = [1.5373 - 1.6455, 1.322 - 1.630, 1.615 - 1.562, 2.115 - 1.837]
DR = [-0.016, -0.100, 0.058, 0.108]
DR_CI = [(-0.048, 0.017), (-0.147, -0.054), (-0.006, 0.122), (0.010, 0.205)]

# Table 5.2: zone centres (0-350 grid) and mean runs per ball
ZONES = {0: (100.4, 284.4, 2.301), 1: (261.1, 77.0, 2.295), 2: (307.9, 188.1, 1.951),
         3: (135.9, 163.6, 0.374), 4: (37.9, 176.0, 1.930), 5: (246.8, 291.1, 2.465),
         6: (97.0, 47.2, 2.891), 7: (204.6, 189.7, 0.393)}


def _style(ax, t):
    fig = ax.figure
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["muted"])
    ax.tick_params(colors=t["muted"], labelcolor=t["ink"])
    ax.xaxis.label.set_color(t["ink"])
    ax.yaxis.label.set_color(t["ink"])
    ax.title.set_color(t["ink"])


def accuracy(t, name):
    fig, ax = plt.subplots(figsize=(8, 3.1), dpi=200)
    _style(ax, t)
    labels = [a for a, _ in ACCURACY]
    vals = [v for _, v in ACCURACY]
    colours = [t["neutral"], t["neutral"], t["s1"]]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colours, height=0.55)
    for yi, v in zip(y, vals):
        ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", color=t["ink"], fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 45)
    ax.set_xlabel("Accuracy on held-out matches, 8 zones (%)")
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("The model doubles the majority-class baseline", loc="left",
                 fontweight="bold", fontsize=12, color=t["ink"])
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=t["surface"])
    plt.close(fig)


def phase_reversal(t, name):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    _style(ax, t)
    y = np.arange(len(PHASES))[::-1]
    off = 0.14
    for yi, (lo, hi) in zip(y, DR_CI):
        ax.plot([lo, hi], [yi - off] * 2, color=t["s1"], lw=2.2, solid_capstyle="round")
    ax.scatter(DR, y - off, s=70, color=t["s1"], edgecolor=t["surface"], lw=2, zorder=3,
               label="Doubly-robust estimate, 95% CI (execution and selection corrected)")
    ax.scatter(NAIVE, y + off, s=70, marker="D", color=t["s2"], edgecolor=t["surface"], lw=2,
               zorder=3, label="Naive comparison (followed minus not followed)")
    ax.axvline(0, color=t["muted"], lw=1)
    ax.axhline(y[0] - 0.5, color=t["grid"], lw=1)
    ax.set_yticks(y, PHASES)
    ax.set_xlim(-0.36, 0.33)
    ax.set_xlabel("Change in runs per ball from following the recommendation")
    ax.grid(axis="x", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(-0.35, -0.95, "worse than batters' own choices", color=t["muted"], fontsize=8)
    ax.text(0.32, -0.95, "better", color=t["muted"], fontsize=8, ha="right")
    ax.set_ylim(-1.1, len(PHASES) - 0.4)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8.5,
                    ncol=1, labelcolor=t["ink"])
    ax.set_title("The effect reverses across the innings", loc="left", fontweight="bold",
                 fontsize=12, color=t["ink"])
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=t["surface"], bbox_extra_artists=[leg], bbox_inches="tight")
    plt.close(fig)


def zone_map(t, name):
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=200)
    _style(ax, t)
    n = 700
    xs = np.linspace(0, 350, n)
    X, Y = np.meshgrid(xs, xs)
    cent = np.array([[v[0], v[1]] for v in ZONES.values()])
    runs = np.array([v[2] for v in ZONES.values()])
    d = (X[..., None] - cent[:, 0]) ** 2 + (Y[..., None] - cent[:, 1]) ** 2
    lab = d.argmin(axis=2)
    inside = (X - 175) ** 2 + (Y - 175) ** 2 <= 165 ** 2
    val = np.where(inside, runs[lab], np.nan)
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    im = ax.imshow(val, extent=(0, 350, 350, 0), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    # Zone edges
    edge = np.zeros_like(lab, dtype=bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    ey, ex = np.where(edge & inside)
    ax.scatter(xs[ex], xs[ey], s=0.15, color=t["surface"], lw=0)
    ax.add_patch(plt.Circle((175, 175), 165, fill=False, color=t["muted"], lw=1.2))
    for z, (cx, cy, r) in ZONES.items():
        dark_bg = r > 1.5 if t is THEMES["light"] else r < 1.5
        c = "#ffffff" if dark_bg else "#0b0b0b"
        ax.text(cx, cy, f"Z{z}\n{r:.2f}", ha="center", va="center", color=c,
                fontweight="bold", fontsize=9)
    ax.set_xlim(0, 350)
    ax.set_ylim(350, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Mean runs per ball in zone", color=t["ink"])
    cb.ax.tick_params(colors=t["muted"], labelcolor=t["ink"])
    cb.outline.set_visible(False)
    ax.set_title("Eight shot zones learned from landing positions", loc="left",
                 fontweight="bold", fontsize=12, color=t["ink"])
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=t["surface"])
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        accuracy(t, f"accuracy-{mode}.png")
        phase_reversal(t, f"phase-reversal-{mode}.png")
        zone_map(t, f"zone-map-{mode}.png")
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
