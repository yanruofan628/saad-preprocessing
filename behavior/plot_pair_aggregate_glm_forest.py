#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 pair_aggregate_ols_from_trial_table.py 输出的 glm_coefficients.csv 绘制森林图。

默认：logit 系数 + 95% CI，参考线 x=0；排除 const（可用 --include-intercept）。
亦可选 --scale or（比值比，参考线 x=1）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GLM_CSV = _SCRIPT_DIR / "pair_aggregate_ols_out" / "glm_coefficients.csv"


def prettify_term(term: str) -> str:
    t = str(term)
    if t == "const":
        return "Intercept"
    if t.startswith("d_") and t.endswith("_mean"):
        return t[2:-5].replace("_", " ")
    return t.replace("_", " ")


def p_to_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot_forest(
    df: pd.DataFrame,
    scale: str,
    include_intercept: bool,
    title: str,
    fig_out: Path,
    dpi: int,
) -> None:
    d = df.copy()
    if not include_intercept:
        d = d[d["term"].astype(str) != "const"].copy()
    if d.empty:
        raise SystemExit("无可用行：检查 CSV 或 --include-intercept")

    d["_label"] = d["term"].map(prettify_term)

    if scale == "logit":
        est = d["coef_logit"].to_numpy(dtype=float)
        lo = d["ci_lower_logit"].to_numpy(dtype=float)
        hi = d["ci_upper_logit"].to_numpy(dtype=float)
        xlab = "Log-odds (β) with 95% CI"
        ref = 0.0
    else:
        est = d["or"].to_numpy(dtype=float)
        lo = d["or_ci_lower"].to_numpy(dtype=float)
        hi = d["or_ci_upper"].to_numpy(dtype=float)
        xlab = "Odds ratio with 95% CI"
        ref = 1.0

    d["_est"] = est
    d = d.sort_values("_est", ascending=True).reset_index(drop=True)
    est = d["_est"].to_numpy()
    lo = d["ci_lower_logit" if scale == "logit" else "or_ci_lower"].to_numpy(dtype=float)
    hi = d["ci_upper_logit" if scale == "logit" else "or_ci_upper"].to_numpy(dtype=float)
    labels = d["_label"].tolist()
    pvals = d["pvalue"].to_numpy(dtype=float)

    n = len(d)
    fig_h = max(3.0, 0.38 * n + 1.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_h), dpi=dpi)

    y = np.arange(n)
    xerr_lo = est - lo
    xerr_hi = hi - est
    colors = np.where(pvals < 0.05, "#1f77b4", "#555555")

    for i in range(n):
        ax.errorbar(
            est[i],
            y[i],
            xerr=[[xerr_lo[i]], [xerr_hi[i]]],
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            capsize=3,
            markersize=6,
            linewidth=1.6,
        )

    ax.axvline(ref, color="0.35", linestyle="--", linewidth=1, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)

    span = float(np.nanmax(hi) - np.nanmin(lo))
    pad = 0.035 * span if span > 1e-12 else 0.05
    for i, (_, row) in enumerate(d.iterrows()):
        st = p_to_star(float(row["pvalue"]))
        if st:
            ax.text(hi[i] + pad, y[i], st, va="center", fontsize=10, color="#1f77b4")

    ax.tick_params(axis="x", labelsize=10)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("已保存:", fig_out)


def main() -> None:
    ap = argparse.ArgumentParser(description="GLM 系数森林图（pair_aggregate 输出）")
    ap.add_argument(
        "--glm-csv",
        type=str,
        default="",
        help="glm_coefficients.csv 路径；默认项目内 pair_aggregate_ols_out/",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="pair_aggregate 输出目录（将使用该目录下 glm_coefficients.csv）",
    )
    ap.add_argument(
        "--scale",
        choices=("logit", "or"),
        default="logit",
        help="横轴：logit 系数（参考 0）或 OR（参考 1）",
    )
    ap.add_argument(
        "--include-intercept",
        action="store_true",
        help="图中包含 const（默认仅预测变量）",
    )
    ap.add_argument(
        "--fig-out",
        type=str,
        default="",
        help="输出图路径；默认与 CSV 同目录 glm_forest_logit.png / glm_forest_or.png",
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--title",
        type=str,
        default="Binomial GLM (logit): feature Δ → group choice",
        help="图标题",
    )
    args = ap.parse_args()

    if args.glm_csv:
        csv_path = Path(args.glm_csv)
        if not csv_path.is_absolute():
            csv_path = _SCRIPT_DIR / csv_path
    elif args.out_dir:
        od = Path(args.out_dir)
        if not od.is_absolute():
            od = _SCRIPT_DIR / od
        csv_path = od / "glm_coefficients.csv"
    else:
        csv_path = Path(DEFAULT_GLM_CSV)

    if not csv_path.is_file():
        raise SystemExit(f"找不到: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    need = {
        "term",
        "coef_logit",
        "pvalue",
        "ci_lower_logit",
        "ci_upper_logit",
        "or",
        "or_ci_lower",
        "or_ci_upper",
    }
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"CSV 缺列: {miss}")

    if args.fig_out:
        fig_out = Path(args.fig_out)
        if not fig_out.is_absolute():
            fig_out = _SCRIPT_DIR / fig_out
    else:
        suffix = "logit" if args.scale == "logit" else "or"
        fig_out = csv_path.parent / f"glm_forest_{suffix}.png"

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    plot_forest(
        df,
        scale=args.scale,
        include_intercept=args.include_intercept,
        title=args.title,
        fig_out=fig_out,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
