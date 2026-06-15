import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rt_panel_layout import (
    AX_LEFT_VIOLIN,
    FIG_H_CM,
    LABEL_FS,
    TICK_FS_Y_VIOLIN,
    X_LABEL_PAD,
    apply_axes_margins,
)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

CSV_PATH = Path(__file__).resolve().parent / "rt_exports" / "rt_exclusion_ratio_by_subject.csv"
OUT_PATH = Path(__file__).resolve().parent / "rt_exports" / "rt_excluded_trials_violin.svg"

FIG_W_CM = 7.5

with CSV_PATH.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

values = np.array([int(r["excluded"]) for r in rows], dtype=float)
vmax = float(values.max()) if values.size else 0.0
ylim_top = max(5.0, vmax + 1.0)

fig, ax = plt.subplots(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54))

violin = ax.violinplot(values, showmeans=True, showmedians=True, widths=0.55)
for body in violin["bodies"]:
    body.set_facecolor("#A5A4C9")
    body.set_edgecolor("black")
    body.set_linewidth(1.2)
    body.set_alpha(0.75)

for key in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
    if key in violin and violin[key] is not None:
        violin[key].set_color("#666666")

ax.set_xticks([1])
ax.set_xticklabels([])
ax.set_xlabel(
    "Excluded trials",
    fontsize=LABEL_FS,
    fontname="Arial",
    fontweight="normal",
    color="black",
    labelpad=X_LABEL_PAD,
)
ax.set_ylabel("Number")
ax.grid(axis="y", linestyle="--", alpha=0.35, color="#8C8C8C")
for spine in ax.spines.values():
    spine.set_color("black")
ax.tick_params(axis="both", colors="black")
ax.set_xlim(0.6, 1.4)
ax.set_ylim(0, ylim_top)

for label in ax.get_yticklabels():
    label.set_fontname("Arial")
    label.set_fontsize(TICK_FS_Y_VIOLIN)
    label.set_fontweight("normal")
    label.set_color("black")

ax.yaxis.label.set_fontname("Arial")
ax.yaxis.label.set_fontsize(LABEL_FS)
ax.yaxis.label.set_fontweight("normal")
ax.yaxis.label.set_color("black")

apply_axes_margins(fig, left=AX_LEFT_VIOLIN)
fig.savefig(OUT_PATH, dpi=200)
plt.close()

print(f"Saved: {OUT_PATH}")
