"""
Three-bar chart with error bars: BACC under three CV splitting schemes.
Same 23 participants (paired). Saves PDF + PNG.
"""
import matplotlib.pyplot as plt
import numpy as np

random_bacc = np.array(
    [
        0.6262,
        0.6539,
        0.8442,
        0.5820,
        0.6271,
        0.8837,
        0.6088,
        0.6317,
        0.6443,
        0.6924,
        0.7121,
        0.7261,
        0.8297,
        0.5988,
        0.6122,
        0.7458,
        0.6936,
        0.7089,
        0.6172,
        0.8044,
        0.6194,
        0.6916,
        0.5733,
    ]
)

mirror_bacc = np.array(
    [
        0.61151344,
        0.67451730,
        0.83393591,
        0.59201243,
        0.65082180,
        0.88710097,
        0.63635204,
        0.61571469,
        0.61070810,
        0.71280890,
        0.68461624,
        0.72086283,
        0.8275390,
        0.5714622,
        0.6179135,
        0.7456459,
        0.6764455,
        0.6878601,
        0.6478262,
        0.7941763,
        0.6148597,
        0.6866558,
        0.5908546,
    ]
)

heldout_bacc = np.array(
    [
        0.6183014,
        0.7101329,
        0.8475977,
        0.5690534,
        0.6408501,
        0.8816832,
        0.6350246,
        0.6375540,
        0.6098156,
        0.7146297,
        0.7174137,
        0.7303804,
        0.8141231,
        0.5984628,
        0.6417975,
        0.7623298,
        0.7015718,
        0.7141197,
        0.6379124,
        0.7940965,
        0.6315174,
        0.7143245,
        0.5843158,
    ]
)

assert len(random_bacc) == len(mirror_bacc) == len(heldout_bacc) == 23

data = [random_bacc, mirror_bacc, heldout_bacc]
labels = [
    "Trial-random",
    "Mirror-constrained",
    "Pairing-type held-out",
]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams["axes.titleweight"] = "normal"
plt.rcParams["font.size"] = 18

means = np.array([np.mean(v) for v in data], dtype=float)
stds = np.array([np.std(v, ddof=1) for v in data], dtype=float)
x = np.arange(len(labels))
colors = ["#E7796E", "#9594C0", "#6AC2D2"]

"""
# 旧版柱状图（保留，暂不删除）
fig, ax = plt.subplots(figsize=(7, 5.6), dpi=120)
ax.bar(
    x,
    means,
    yerr=stds,
    width=0.42,
    color=colors,
    edgecolor="black",
    linewidth=1.2,
    error_kw=dict(ecolor="black", lw=1.5, capsize=5, capthick=1.5),
)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=30, fontweight="bold")
ax.set_ylabel("BACC", fontsize=30, fontweight="bold")
ax.set_ylim(0.00, 1.00)
ax.set_yticks(np.arange(0.0, 1.01, 0.1))
ax.grid(axis="y", linestyle=":", alpha=0.5)
for tick in ax.get_yticklabels():
    tick.set_fontsize(30)
    tick.set_fontweight("bold")
ax.set_title("")

plt.tight_layout(rect=(0, 0, 1, 0.97))
out_base = "bacc_three_strategies_boxplot"
plt.savefig(f"{out_base}.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
print(f"Saved: {out_base}.pdf, {out_base}.png")
"""

# 新版散点图：S1~S23，每种策略一种点形 + 颜色
subjects = np.arange(1, 24)
jitter = 0.18

width_cm, height_cm = 41.39, 12.0  # 宽度较上一版（40.39 cm）再增加 1 cm
width_in, height_in = width_cm / 2.54, height_cm / 2.54
fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=140)
ax.scatter(subjects - jitter, random_bacc, s=130, marker="s", color=colors[0], edgecolor="black", linewidth=0.7, label=labels[0], alpha=0.9)
ax.scatter(subjects, mirror_bacc, s=130, marker="^", color=colors[1], edgecolor="black", linewidth=0.7, label=labels[1], alpha=0.9)
ax.scatter(subjects + jitter, heldout_bacc, s=130, marker="o", color=colors[2], edgecolor="black", linewidth=0.7, label=labels[2], alpha=0.9)

ax.axhline(means[0], color=colors[0], linestyle="--", linewidth=1.5, alpha=0.8)
ax.axhline(means[1], color=colors[1], linestyle="--", linewidth=1.5, alpha=0.8)
ax.axhline(means[2], color=colors[2], linestyle="--", linewidth=1.5, alpha=0.8)

ax.set_xlim(0.4, 23.6)
ax.set_ylim(0.40, 1.00)
ax.set_ylabel("BACC", fontsize=18, fontweight="normal")
ax.set_xlabel("Subjects", fontsize=18, fontweight="normal", labelpad=18)
ax.set_xticks(subjects)
ax.set_xticklabels([str(i) for i in subjects], rotation=0, fontsize=18, fontweight="normal")
ax.set_yticks(np.arange(0.4, 1.01, 0.2))
ax.tick_params(axis="x", pad=10)
for tick in ax.get_yticklabels():
    tick.set_fontsize(18)
    tick.set_fontweight("normal")
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Reference line at 0.5 (fine + transparent)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.25)

_x_data_for_transform = float(np.mean(subjects))
_y_data_for_legend_bottom = 1.0
_y_axes = ax.transAxes.inverted().transform(ax.transData.transform((_x_data_for_transform, _y_data_for_legend_bottom)))[
    1
]

ax.legend(
    # Use lower-center so bbox_to_anchor y corresponds to the legend bottom edge.
    loc="lower center",
    bbox_to_anchor=(0.5, _y_axes),
    bbox_transform=ax.transAxes,
    ncol=3,
    frameon=True,
    fontsize=18,
    edgecolor="#BEBEBE",
    facecolor="#F5F5F5",
    framealpha=1.0,
    fancybox=False,
)
ax.set_title("")

plt.tight_layout()
out_base = "bacc_three_strategies_scatter"
plt.savefig(f"{out_base}.pdf", dpi=300, bbox_inches="tight")
plt.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{out_base}.svg", bbox_inches="tight")
print(f"Saved: {out_base}.pdf, {out_base}.png, {out_base}.svg")
print(
    "Group mean BACC:",
    float(np.mean(random_bacc)),
    float(np.mean(mirror_bacc)),
    float(np.mean(heldout_bacc)),
)
plt.close(fig)
