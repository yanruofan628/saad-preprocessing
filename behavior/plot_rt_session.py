import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from rt_panel_layout import (
    AX_LEFT_RIDGELINE,
    FIG_H_CM,
    LABEL_FS,
    X_LABEL_PAD,
    apply_axes_margins,
)


CSV_PATH = r"d:\PycharmProjects\Attention_switch\rt_exports\rt_violin_3session_plot_data.csv"
OUT_DIR = r"d:\PycharmProjects\Attention_switch\rt_exports"

# 山峦图参数（与 ggridges 风格一致：scale 越大层间重叠越多）
RIDGE_SCALE = 1.35
RIDGE_ALPHA = 0.45
RIDGE_REL_MIN_HEIGHT = 0.012
RT_XMAX_DEFAULT = 2500.0


def _trim_kde_tails(x, dens, rel_min_height):
    dens = np.asarray(dens, dtype=float)
    if dens.size == 0 or np.nanmax(dens) <= 0:
        return x, dens
    thresh = float(rel_min_height) * float(np.nanmax(dens))
    above = dens >= thresh
    if not np.any(above):
        return x, dens
    idx = np.where(above)[0]
    i0, i1 = int(idx[0]), int(idx[-1])
    return x[i0 : i1 + 1], dens[i0 : i1 + 1]


def _kde_on_grid(values, x_grid, rel_min_height):
    """在固定 x_grid 上评估 KDE，返回与 x_grid 同长的归一化密度 [0,1]（截尾后可能变短）。"""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    kde = gaussian_kde(v)
    d = kde(x_grid)
    d = np.asarray(d, dtype=float)
    dmax = float(np.nanmax(d))
    if dmax <= 0:
        return None
    d = d / dmax
    return _trim_kde_tails(x_grid, d, rel_min_height)


def _load_subject_medians():
    df = pd.read_csv(CSV_PATH)
    df = df[df["RT_ms"].notna()].copy()
    df["RT_ms"] = pd.to_numeric(df["RT_ms"], errors="coerce")
    df = df[df["RT_ms"].notna()]

    med = (
        df.groupby(["subject_name", "Run"], as_index=False)["RT_ms"]
        .median()
        .rename(columns={"RT_ms": "median_rt"})
    )
    piv = med.pivot(index="subject_name", columns="Run", values="median_rt")
    piv = piv[[1, 2, 3]].dropna()
    return piv, df


def plot_slope(piv):
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=130)
    x = np.array([1, 2, 3])
    for _, row in piv.iterrows():
        ax.plot(x, row.values, color="0.65", linewidth=1.0, alpha=0.8, zorder=1)
        ax.scatter(x, row.values, color="0.45", s=14, alpha=0.8, zorder=2)

    mean_vals = piv.mean(axis=0).values
    ax.plot(x, mean_vals, color="#d62728", linewidth=2.6, marker="o", markersize=6, zorder=3, label="Group mean")

    ax.set_xticks([1, 2, 3], labels=["Session 1", "Session 2", "Session 3"])
    ax.set_ylabel("Median RT per subject (ms)")
    ax.set_title("Session trend: subject-wise median RT")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}\\rt_session_slopeplot.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}\\rt_session_slopeplot.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_raindot(piv):
    fig, ax = plt.subplots(figsize=(6.8, 4.6), dpi=130)
    rng = np.random.default_rng(123)
    colors = ["#4c78a8", "#59a14f", "#f28e2b"]
    sessions = [1, 2, 3]

    for i, run in enumerate(sessions, start=1):
        vals = piv[run].values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals, s=26, alpha=0.70, color=colors[i - 1], edgecolors="none")

        mean = np.mean(vals)
        sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
        ci95 = 1.96 * sem
        ax.errorbar(i, mean, yerr=ci95, fmt="o", color="black", capsize=4, markersize=5, linewidth=1.3, zorder=5)

    ax.set_xticks([1, 2, 3], labels=["Session 1", "Session 2", "Session 3"])
    ax.set_ylabel("Median RT per subject (ms)")
    ax.set_title("Session trend: rain-dot with mean ±95% CI")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}\\rt_session_raindot.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}\\rt_session_raindot.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ridgeline(df):
    """
    各 Session 的 RT 核密度山峦图：纵轴从上到下为 Session 1 → Session 2 → …（Run 编号升序），
    横轴为反应时（ms）。
    """
    fig_w_cm = 11.59
    fig, ax = plt.subplots(figsize=(fig_w_cm / 2.54, FIG_H_CM / 2.54), dpi=130)
    plt.rcParams["font.family"] = "Arial"
    sessions = sorted(df["Run"].dropna().astype(int).unique())
    n = len(sessions)
    # 按“从上到下”指定色系：Session 1, Session 2, Session 3
    palette = ["#92D5D0", "#BCE1A4", "#D16552"]
    colors = [palette[i % len(palette)] for i in range(n)]

    # 自上而下：列表第一个 session 在图上方（y 最大）
    y_base = {run: float(n - 1 - i) for i, run in enumerate(sessions)}

    # 按需求固定横轴到 3000 ms
    x_hi = RT_XMAX_DEFAULT
    x_grid = np.linspace(0.0, x_hi, 600)

    for run, color in zip(sessions, colors):
        vals = df.loc[df["Run"] == run, "RT_ms"].dropna().to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals >= 0) & (vals <= x_hi * 1.5)]
        curve = _kde_on_grid(vals, x_grid, RIDGE_REL_MIN_HEIGHT)
        if curve is None:
            continue
        xg, dn = curve
        base = y_base[run]
        ridge_top = base + RIDGE_SCALE * dn
        ax.fill_between(xg, base, ridge_top, color=color, alpha=RIDGE_ALPHA, linewidth=0)
        ax.plot(xg, ridge_top, color=color, linewidth=1.15, alpha=min(1.0, RIDGE_ALPHA + 0.35))

        if vals.size:
            qx = float(np.median(vals))
            j = int(np.searchsorted(xg, qx))
            j = np.clip(j, 0, len(xg) - 1)
            y_tip = float(base + RIDGE_SCALE * dn[j])
            ax.plot(
                [qx, qx],
                [base, y_tip],
                color="0.25",
                linestyle="--",
                linewidth=0.9,
                alpha=0.85,
            )

    ax.set_xlim(0, x_hi)
    # 顶部留足空间，避免最上方山峦被裁切
    upper = (n - 1) + RIDGE_SCALE * 1.15
    ax.set_ylim(-0.35, upper)
    ax.set_yticks(list(range(n)))
    # y 轴自下而上标注：底部为最大 Run，顶部为 Session 1
    ax.set_yticklabels(
        [f"Session {sessions[n - 1 - k]}" for k in range(n)],
        fontsize=LABEL_FS,
        fontname="Arial",
    )
    ax.set_xlabel(
        "Reaction time (ms)",
        fontsize=LABEL_FS,
        fontname="Arial",
        fontweight="normal",
        color="black",
        labelpad=X_LABEL_PAD,
    )
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="x", labelsize=LABEL_FS, colors="black")
    ax.tick_params(axis="y", length=0, labelsize=LABEL_FS, colors="black")
    apply_axes_margins(fig, left=AX_LEFT_RIDGELINE)
    fig.savefig(f"{OUT_DIR}\\rt_session_ridgeline.png", dpi=300)
    fig.savefig(f"{OUT_DIR}\\rt_session_ridgeline.pdf", dpi=300)
    fig.savefig(f"{OUT_DIR}\\rt_session_ridgeline.svg")
    plt.close(fig)


def main():
    piv, df = _load_subject_medians()
    plot_slope(piv)
    plot_raindot(piv)
    plot_ridgeline(df)
    print("Saved:")
    print(f"{OUT_DIR}\\rt_session_slopeplot.png/.pdf")
    print(f"{OUT_DIR}\\rt_session_raindot.png/.pdf")
    print(f"{OUT_DIR}\\rt_session_ridgeline.png/.pdf")


if __name__ == "__main__":
    main()
