"""
预处理：插值坏道数、ICA 剔除分量数 —— 各画一张与 plot_rt_excluded_violin 版式一致的小提琴图。

数据默认读 preproc_violin_export/preproc_summary_per_subject.csv（可先运行
export_preproc_violin_for_ngplot.py 并设 PREPROC_DATA_ROOT）。

输出：
- preproc_violin_export/interpolated_channels_violin.svg
- preproc_violin_export/ica_removed_components_violin.svg
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_CSV = SCRIPT_DIR / "preproc_violin_export" / "preproc_summary_per_subject.csv"

# 插值通道图 / ICA 图 分别指定画布（宽 × 高，单位 cm）
INTERP_W_CM = 7.7
INTERP_H_CM = 7.72
ICA_W_CM = 6.33
ICA_H_CM = 7.72

TICK_FS_X = 18
TICK_FS_Y = 16
EDGE = "black"


def _style_violin(violin, face: str) -> None:
    for body in violin["bodies"]:
        body.set_facecolor(face)
        body.set_edgecolor(EDGE)
        body.set_linewidth(1.2)
        body.set_alpha(0.75)
    for key in ("cbars", "cmins", "cmaxes", "cmedians", "cmeans"):
        if key in violin and violin[key] is not None:
            violin[key].set_color(EDGE)


def _style_axes(ax, *, ylabel: str | None, ylim: tuple[float, float], yticks: list[float]) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#8C8C8C")
    for spine in ax.spines.values():
        spine.set_color(EDGE)
    ax.tick_params(axis="both", colors="black")
    ax.set_xlim(0.6, 1.4)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    if ylabel:
        ax.set_ylabel(ylabel)
        ax.yaxis.label.set_fontname("Arial")
        ax.yaxis.label.set_fontsize(18)
        ax.yaxis.label.set_fontweight("normal")
        ax.yaxis.label.set_color("black")
    else:
        ax.set_ylabel("")
        ax.yaxis.get_label().set_visible(False)

    for label in ax.get_xticklabels():
        label.set_fontname("Arial")
        label.set_fontsize(TICK_FS_X)
        label.set_fontweight("normal")
        label.set_color("black")
    for label in ax.get_yticklabels():
        label.set_fontname("Arial")
        label.set_fontsize(TICK_FS_Y)
        label.set_fontweight("normal")
        label.set_color("black")


def _plot_one(
    values: np.ndarray,
    *,
    fig_w_cm: float,
    fig_h_cm: float,
    xtick: str,
    ylabel: str | None,
    ylim: tuple[float, float],
    yticks: list[float],
    face: str,
    out_path: Path,
) -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54))
    violin = ax.violinplot(values, showmeans=True, showmedians=True, widths=0.55)
    _style_violin(violin, face)

    ax.set_xticks([1])
    ax.set_xticklabels([xtick])
    _style_axes(ax, ylabel=ylabel, ylim=ylim, yticks=yticks)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved: {out_path}")


def load_values(summary_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    with summary_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    interp: list[float] = []
    ica: list[float] = []
    for r in rows:
        if str(r.get("trial_info_ok", "True")).lower() not in ("true", "1", "yes"):
            continue
        ib = r.get("Interpolated_channels", "").strip()
        ic = r.get("Rejected_ICA_components", "").strip()
        if ib:
            interp.append(float(ib))
        if ic:
            ica.append(float(ic))
    if not interp:
        raise SystemExit(f"无插值通道数据，请先运行 export_preproc_violin_for_ngplot.py 或检查: {summary_csv}")
    if not ica:
        raise SystemExit(f"无 ICA 剔除数据，请检查: {summary_csv}")
    return np.array(interp, dtype=float), np.array(ica, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="preproc_summary_per_subject.csv 路径",
    )
    args = ap.parse_args()

    vals_interp, vals_ica = load_values(args.summary_csv)
    out_dir = SCRIPT_DIR / "preproc_violin_export"

    _plot_one(
        vals_interp,
        fig_w_cm=INTERP_W_CM,
        fig_h_cm=INTERP_H_CM,
        xtick="Interpolated Channels",
        ylabel="Number",
        ylim=(0.0, 32.0),
        yticks=[0.0, 8.0, 16.0, 24.0, 32.0],
        face="#D16552",
        out_path=out_dir / "interpolated_channels_violin.svg",
    )
    _plot_one(
        vals_ica,
        fig_w_cm=ICA_W_CM,
        fig_h_cm=ICA_H_CM,
        xtick="ICA Removed",
        ylabel=None,
        ylim=(0.0, 16.0),
        yticks=[0.0, 4.0, 8.0, 12.0, 16.0],
        face="#78B3AE",
        out_path=out_dir / "ica_removed_components_violin.svg",
    )


if __name__ == "__main__":
    main()
