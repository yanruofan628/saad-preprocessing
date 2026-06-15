import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

try:
    # 优先交互后端，便于“保存失败时直接展示”
    matplotlib.use("Qt5Agg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import mne
from mne.preprocessing import ICA

from extract_channel_info import make_montage
from subject_mff_benchmark_config import SUBJECTS_CONFIG

try:
    from mne_icalabel import label_components

    IC_LABEL_AVAILABLE = True
except ImportError:
    IC_LABEL_AVAILABLE = False
    label_components = None

try:
    from pyprep import PrepPipeline

    PYPREP_AVAILABLE = True
except ImportError:
    PYPREP_AVAILABLE = False


# 与主预处理脚本保持一致
BASE_DATA_DIRS = [r"A:\\", r"A:/standard_data_noica"]

# 单被试预览：直接点运行时只画这一个人（名字须与 SUBJECTS_CONFIG 里 subject_name 一致）
PREVIEW_SUBJECT_NAME = "zhouyu"
PREVIEW_FILE_INDEX = 1
PREVIEW_OUTPUT_ROOT = "A:/standard_data_interp_ica"

# 会话缓存文件名前缀（fif + json），用于 --replot-only 不重算 ICA
ICA_SESSION_STEM = "ica_visual"
REPLOT_ICA_ONLY = False
# 默认是否自动使用缓存重画（建议 False：点击运行时默认走完整预处理+ICA）
AUTO_USE_REPLOT_CACHE = False

MANUAL_BAD_ELECTRODES = [
    "E48",
    "E119",
    "E49",
    "E113",
    "E44",
    "E114",
    "E131",
    "E132",
    "E56",
    "E107",
    "E126",
    "E127",
    "E43",
    "E120",
    "E45",
    "E108",
    "E57",
    "E100",
    "E99",
    "E63",
]

# 波形 y 轴固定范围（uV）：-1000 ~ +1000
WAVE_YLIM_UV = 1000

# 原始波形多通道图：电极编号（与重命名后 EEG### 一致）
# 当前目标：EEG019、EEG011、EEG004
WAVE_PLOT_EEG_NUMBERS = [19, 11, 4]
# 对应显示标签（按 WAVE_PLOT_EEG_NUMBERS 顺序）
WAVE_PLOT_DISPLAY_LABELS = ["FC1", "FCZ", "FC2"]

# 是否仅保留前 128 个 EEG 通道（旧流程默认 True）。
# 画 EEG207 时必须为 False。
KEEP_FIRST_128_EEG_ONLY = False

# Raw 波形展示时间段（秒），闭开区间语义与采样对齐：[tmin, tmax) 对应 get_data(start, stop)
WAVE_PLOT_TMIN_SEC = 0.0
WAVE_PLOT_TMAX_SEC = 5.0

# Raw 波形图横轴主刻度间隔（秒）
WAVE_PLOT_XTICK_STEP_SEC = 1.0

# 通道标签整体上移量（厘米）
CHANNEL_LABEL_UP_CM = 0.8

# 仅导出“pre/post ICA 合并波形图”（不再额外导出两张单独 raw 波形图）
WAVE_PLOT_ONLY_STACKED = True

# PSD Welch 上界（Hz）。略高于 50 便于看工频；低通（如 40/45）以上主要是滚降，作 QC/对比仍有意义
PSD_FMAX_HZ = 60.0


def _wave_plot_channel_names():
    return [f"EEG{i:03d}" for i in WAVE_PLOT_EEG_NUMBERS]


def _wave_plot_display_labels(ch_names):
    if len(WAVE_PLOT_DISPLAY_LABELS) == len(ch_names):
        return list(WAVE_PLOT_DISPLAY_LABELS)
    return list(ch_names)


def _wave_plot_channels_ready(wave_ch_names, wave_picks) -> bool:
    return bool(wave_ch_names) and len(wave_picks) == len(wave_ch_names)

# 全局绘图字体：Arial，字号 42（统一且不加粗）
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 42
plt.rcParams["axes.labelsize"] = 42
plt.rcParams["xtick.labelsize"] = 42
plt.rcParams["ytick.labelsize"] = 42
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "normal"


def _find_subject_cfg(subject_name: str):
    return next((cfg for cfg in SUBJECTS_CONFIG if cfg.get("subject_name") == subject_name), None)


def _find_existing_mff_path(mff_rel_path: str):
    for base_dir in BASE_DATA_DIRS:
        candidate_path = os.path.join(base_dir, mff_rel_path)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def _convert_bad_e_to_eeg(bad_list):
    out = []
    for name in bad_list:
        if name.startswith("E") and name[1:].isdigit():
            out.append(f"EEG{int(name[1:]):03d}")
        else:
            out.append(name)
    return out


def _save_figure_or_list(fig_obj, output_stem: Path):
    if fig_obj is None:
        return

    objs = fig_obj if isinstance(fig_obj, list) else [fig_obj]
    for i, obj in enumerate(objs):
        out_path = output_stem.parent / (
            f"{output_stem.name}_{i + 1:02d}.svg" if isinstance(fig_obj, list) else f"{output_stem.name}.svg"
        )
        saved = False

        # 1) matplotlib Figure
        if hasattr(obj, "savefig"):
            obj.savefig(out_path, dpi=220, bbox_inches="tight")
            saved = True
        # 2) MNEQtBrowser with embedded figure handles
        elif hasattr(obj, "figure") and hasattr(obj.figure, "savefig"):
            obj.figure.savefig(out_path, dpi=220, bbox_inches="tight")
            saved = True
        elif hasattr(obj, "fig") and hasattr(obj.fig, "savefig"):
            obj.fig.savefig(out_path, dpi=220, bbox_inches="tight")
            saved = True
        # 3) browser screenshot fallback
        elif hasattr(obj, "screenshot"):
            try:
                img = obj.screenshot()
                plt.figure(figsize=(16, 9))
                plt.imshow(img)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out_path, dpi=220, bbox_inches="tight")
                plt.close()
                saved = True
            except Exception:
                saved = False

        # 4) if still not saved, just display and continue (do not crash)
        if not saved:
            print(f"警告: 无法保存图像 {out_path}，将尝试直接展示。")
            try:
                if hasattr(obj, "show"):
                    obj.show()
                else:
                    plt.show()
            except Exception as exc:
                print(f"警告: 图像展示也失败，已跳过。原因: {exc}")

        if hasattr(obj, "close"):
            try:
                obj.close()
            except Exception:
                pass
        elif hasattr(plt, "close"):
            try:
                plt.close(obj)
            except Exception:
                pass


def _normalize_channels_and_set_montage(raw, montage):
    # VREF/参考通道不参与 EEG 流程，避免 montage 与插值阶段异常
    ref_like = [ch for ch in raw.ch_names if str(ch).upper() in {"VREF", "REF", "CZREF"}]
    if ref_like:
        try:
            raw.set_channel_types({ch: "misc" for ch in ref_like}, verbose=False)
        except Exception:
            pass

    # 可选：仅保留前 128 个 EEG 通道（旧流程与主流程一致）
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if KEEP_FIRST_128_EEG_ONLY and len(eeg_picks) > 128:
        eeg_picks = eeg_picks[:128]
    raw.pick(eeg_picks)

    # E### -> EEG###（与主流程一致）
    rename_dict = {}
    for ch_name in raw.ch_names:
        if ch_name.startswith("E") and ch_name[1:].isdigit():
            rename_dict[ch_name] = f"EEG{int(ch_name[1:]):03d}"
    if rename_dict:
        raw.rename_channels(rename_dict)

    if montage is not None:
        # 统一 montage 命名到 EEG###，避免与 raw 的 EEG001 风格不匹配
        try:
            ch_pos = montage.get_positions().get("ch_pos", {})
            m_rename = {}
            for name in ch_pos.keys():
                if name.startswith("EEG") and name[3:].isdigit():
                    m_rename[name] = f"EEG{int(name[3:]):03d}"
                elif name.startswith("E") and name[1:].isdigit():
                    m_rename[name] = f"EEG{int(name[1:]):03d}"
            if m_rename:
                montage = montage.copy()
                montage.rename_channels(m_rename)
        except Exception:
            pass
        raw.set_montage(montage, on_missing="warn")


def _detect_bad_channels(raw_for_detection, montage):
    if not PYPREP_AVAILABLE:
        print("PyPREP 不可用，使用手工坏道列表作为后备。")
        return _convert_bad_e_to_eeg(MANUAL_BAD_ELECTRODES)

    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": [50],
        "max_iterations": 4,
    }
    try:
        prep = PrepPipeline(raw_for_detection, prep_params, montage, ransac=True)
        prep.fit()
        bads = [str(ch) for ch in prep.noisy_channels_original["bad_all"]]
        return bads
    except Exception as exc:
        print(f"PyPREP 坏道检测失败，改用手工坏道列表。原因: {exc}")
        return _convert_bad_e_to_eeg(MANUAL_BAD_ELECTRODES)


def _plot_psd_svg(raw_obj, save_path: Path, title: str):
    """保存 PSD 图（SVG）。"""
    fig = raw_obj.plot_psd(fmax=PSD_FMAX_HZ, show=False)
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _compute_psd_db(raw_obj, fmin=0.1, fmax=None):
    """返回 (freqs, db[n_channels, n_freqs], picks_name)。优先 EEG，失败则退回 data。"""
    if fmax is None:
        fmax = PSD_FMAX_HZ
    try:
        psd = raw_obj.compute_psd(method="welch", fmin=fmin, fmax=fmax, picks="eeg", verbose=False)
        picks_name = "eeg"
    except Exception:
        psd = raw_obj.compute_psd(method="welch", fmin=fmin, fmax=fmax, picks="data", verbose=False)
        picks_name = "data"
    data = psd.get_data()
    freqs = psd.freqs
    db = 10 * np.log10(data + np.finfo(float).eps)
    return freqs, db, picks_name


def _style_psd_xaxis(ax, fmin: float, fmax: float, pad_hz: float = 3.0) -> None:
    """横轴略超出 fmax（留白），主刻度只标到 fmax。"""
    ax.set_xlim(float(fmin), float(fmax) + pad_hz)
    fm = int(round(float(fmax)))
    majors = np.arange(0, fm + 1, 10, dtype=float)
    if majors.size == 0:
        majors = np.array([0.0, float(fmax)])
    ax.set_xticks(majors)


def _plot_psd_matplotlib(raw_obj, save_path: Path, title: str, fmin=0.1, fmax=None):
    """全通道 PSD（SVG）；纵轴使用 matplotlib 默认自动范围（不手动锁 ylim）。"""
    if fmax is None:
        fmax = PSD_FMAX_HZ
    freqs, db, picks_name = _compute_psd_db(raw_obj, fmin=fmin, fmax=fmax)
    db = db[np.all(np.isfinite(db), axis=1)]
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    if db.size == 0:
        ax.set_title(f"{title} (no finite PSD)")
    else:
        for i in range(db.shape[0]):
            ax.plot(freqs, db[i], linewidth=0.8, alpha=0.35)
        ax.set_title(f"{title} ({db.shape[0]} channels, picks={picks_name})")
    _style_psd_xaxis(ax, fmin, fmax)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density (dB)")
    ax.tick_params(labelsize=18)
    ax.grid(alpha=0.25)
    ax.autoscale(enable=True, axis="y", tight=False)
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _wave_plot_sample_range(
    raw_a,
    raw_b,
    tmin=None,
    tmax=None,
):
    """前后 raw 在 [tmin, tmax) 秒内的公共采样区间，返回 (start_samp, stop_samp)，stop 为不包含。"""
    if tmin is None:
        tmin = WAVE_PLOT_TMIN_SEC
    if tmax is None:
        tmax = WAVE_PLOT_TMAX_SEC
    sfreq = float(raw_a.info["sfreq"])
    n = min(raw_a.n_times, raw_b.n_times)
    i0 = int(round(float(tmin) * sfreq))
    i1 = int(round(float(tmax) * sfreq))
    i0 = max(0, min(i0, n))
    i1 = max(i0, min(i1, n))
    if i1 <= i0:
        return None, None
    return i0, i1


def _wave_row_ylim_uv(
    row_uv,
    pct_low=2.0,
    pct_high=98.0,
    margin_ratio=0.12,
    min_margin_uv=10.0,
):
    """单通道一段波形（µV）的稳健纵轴范围；避免多道共用全局 y 轴时弱道被压成直线。"""
    row = np.asarray(row_uv, dtype=float).ravel()
    row = row[np.isfinite(row)]
    if row.size == 0:
        return -float(WAVE_YLIM_UV), float(WAVE_YLIM_UV)
    lo = float(np.percentile(row, pct_low))
    hi = float(np.percentile(row, pct_high))
    if hi <= lo:
        lo, hi = float(np.min(row)), float(np.max(row))
    span = max(hi - lo, 1e-6)
    margin = max(span * margin_ratio, min_margin_uv)
    return lo - margin, hi + margin


def _smooth_ylim_pair(y0: float, y1: float):
    """将 y 轴上下限圆整到更平滑的刻度。"""
    lo = float(min(y0, y1))
    hi = float(max(y0, y1))
    span = max(hi - lo, 1e-6)
    # 根据量级选择步长，避免出现零碎刻度范围
    if span <= 60:
        step = 5.0
    elif span <= 150:
        step = 10.0
    elif span <= 300:
        step = 20.0
    elif span <= 600:
        step = 50.0
    else:
        step = 100.0
    lo_s = np.floor(lo / step) * step
    hi_s = np.ceil(hi / step) * step
    if hi_s <= lo_s:
        hi_s = lo_s + step
    return float(lo_s), float(hi_s)


def _plot_selected_waveforms(
    raw_obj,
    picks,
    save_path: Path,
    title: str,
    start_samp: int,
    stop_samp: int,
    tmin_sec: float,
    tmax_sec: float,
):
    """多通道波形（SVG）；每通道单独设 y 轴；时间轴为 [tmin_sec, tmax_sec) 对应数据段。"""
    data_uv = raw_obj.get_data(picks=picks, start=start_samp, stop=stop_samp) * 1e6
    times = raw_obj.times[start_samp:stop_samp]
    ch_names = [raw_obj.ch_names[p] for p in picks]

    n = len(picks)
    fig_h = max(6.0, min(28.0, 0.72 * n + 2.5))
    fig, axes = plt.subplots(n, 1, figsize=(14, fig_h), sharex=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(times, data_uv[i], linewidth=1.5)
        y0, y1 = _wave_row_ylim_uv(data_uv[i])
        ax.set_ylim(y0, y1)
        ax.set_xlim(float(tmin_sec), float(tmax_sec))
        ax.set_ylabel(ch_names[i])
        ax.grid(alpha=0.25)
        xticks = np.arange(float(tmin_sec), float(tmax_sec) + 1e-9, float(WAVE_PLOT_XTICK_STEP_SEC))
        if xticks.size == 0 or not np.isclose(xticks[-1], float(tmax_sec)):
            xticks = np.append(xticks, float(tmax_sec))
        ax.set_xticks(xticks)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout(pad=0.08, h_pad=0.05)
    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _plot_pre_post_stacked_waveforms(
    raw_pre,
    raw_post,
    picks,
    save_path: Path,
    start_samp: int,
    stop_samp: int,
    tmin_sec: float,
    tmax_sec: float,
):
    """单图展示：上半部分 pre-ICA，下半部分 post-ICA；仅保留波形与底部时间轴。"""
    data_pre_uv = raw_pre.get_data(picks=picks, start=start_samp, stop=stop_samp) * 1e6
    data_post_uv = raw_post.get_data(picks=picks, start=start_samp, stop=stop_samp) * 1e6
    times = raw_pre.times[start_samp:stop_samp]
    ch_names = [raw_pre.ch_names[p] for p in picks]
    disp_labels = _wave_plot_display_labels(ch_names)

    n = len(picks)
    total_rows = 2 * n
    fig = plt.figure(figsize=(8.0, 7.95))
    # 组内紧凑，组间插入一行“空白间隔”
    gap_ratio = 0.65
    height_ratios = [0.85] * n + [gap_ratio] + [0.85] * n
    gs = fig.add_gridspec(total_rows + 1, 1, height_ratios=height_ratios)
    axes = []
    shared_ax = None
    for i in range(total_rows):
        gs_row = i if i < n else i + 1
        ax = fig.add_subplot(gs[gs_row, 0], sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax
        axes.append(ax)

    for i in range(n):
        ax_pre = axes[i]
        ax_post = axes[n + i]

        line_pre = ax_pre.plot(times, data_pre_uv[i], linewidth=8.0, color="#D16552")[0]
        y0_pre, y1_pre = _wave_row_ylim_uv(data_pre_uv[i])
        y0_post, y1_post = _wave_row_ylim_uv(data_post_uv[i])
        # 每个通道 pre/post 共用一套平滑后的纵轴范围，视觉更稳定
        y_lo, y_hi = _smooth_ylim_pair(min(y0_pre, y0_post), max(y1_pre, y1_post))
        ax_pre.set_ylim(y_lo, y_hi)
        ax_pre.set_xlim(float(tmin_sec), float(tmax_sec))
        ax_pre.set_yticks([])
        ax_pre.grid(False)
        for s in ax_pre.spines.values():
            s.set_visible(False)
        ax_pre.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # 仅第一条线（最上方 red 线）在显示坐标中向下平移 0.1 cm
        if i == 0:
            dy_cm = 0.1
            dy_points = dy_cm * 72.0 / 2.54
            dy_pixels = dy_points * fig.dpi / 72.0
            x0 = float(times[0]) if len(times) else 0.0
            y0 = 0.5 * float(y_lo + y_hi)
            x_pix, y_pix = ax_pre.transData.transform((x0, y0))
            # display y 往下是“增大像素”，但 Matplotlib transData 的 y 递增方向与像素坐标一致性不保证；
            # 这里用经验：向下平移 -> y_pix 减少
            y_pix_shifted = float(y_pix) - float(dy_pixels)
            y_shifted = ax_pre.transData.inverted().transform((x_pix, y_pix_shifted))[1]
            delta_uV = float(y_shifted) - float(y0)
            line_pre.set_ydata(data_pre_uv[i] + delta_uV)

        ax_post.plot(times, data_post_uv[i], linewidth=8.0, color="#78B3AE")
        ax_post.set_ylim(y_lo, y_hi)
        ax_post.set_xlim(float(tmin_sec), float(tmax_sec))
        ax_post.set_yticks([])
        ax_post.grid(False)
        for s in ax_post.spines.values():
            s.set_visible(False)
        ax_post.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    ax_bottom = axes[-1]
    xticks = np.arange(float(tmin_sec), float(tmax_sec) + 1e-9, float(WAVE_PLOT_XTICK_STEP_SEC))
    if xticks.size == 0 or not np.isclose(xticks[-1], float(tmax_sec)):
        xticks = np.append(xticks, float(tmax_sec))
    ax_bottom.set_xticks(xticks)
    ax_bottom.tick_params(axis="x", which="both", bottom=True, labelbottom=True, length=6, width=1.2)
    ax_bottom.spines["bottom"].set_visible(True)
    ax_bottom.set_xlabel("Time (s)")

    fig.tight_layout(pad=0.02, h_pad=0.0)
    fig.savefig(save_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _annotate_component_figures(figs, picks, comp_labels, comp_proba):
    """给 ICA component 图标题加上 IC 编号、标签和概率。"""
    if figs is None:
        return figs
    fig_list = figs if isinstance(figs, list) else [figs]
    for fig in fig_list:
        axes = fig.axes if hasattr(fig, "axes") else []
        for ax in axes:
            title = ax.get_title()
            if not title:
                continue
            digits = "".join(ch for ch in title if ch.isdigit())
            if not digits:
                continue
            ic_idx = int(digits)
            if ic_idx not in picks:
                continue
            label = str(comp_labels[ic_idx])
            proba = float(comp_proba[ic_idx])
            ax.set_title(f"IC {ic_idx}\n{label} ({proba:.2f})", fontsize=20)
    return figs


def _label_to_category(label_name: str):
    lab = str(label_name).lower()
    if "eye" in lab:
        return "eye"
    if "muscle" in lab:
        return "muscle"
    if "heart" in lab:
        return "heart"
    if "line" in lab and "noise" in lab:
        return "line_noise"
    return None


def _is_iclabel_artifact(label_name: str):
    """仅将伪迹类判为可剔除，保留 brain 和 other。"""
    lab = str(label_name).lower().strip()
    artifact_labels = {
        "eye blink",
        "muscle artifact",
        "heart beat",
        "line noise",
        "channel noise",
    }
    return lab in artifact_labels


def _save_ica_visual_session(
    out_dir: Path,
    subject_name: str,
    file_index: int,
    mff_rel_path: str,
    mff_dir: str,
    proba_threshold: float,
    bad_channels,
    comp_labels,
    comp_proba,
    excluded_all_artifact,
    brain_picks,
    rejected_by_category,
    rejected_all_in_categories,
    psd_ymin,
    psd_ymax,
    wave_meta,
    raw_before,
    raw_pp,
    ica,
):
    stem = ICA_SESSION_STEM
    raw_before.save(str(out_dir / f"{stem}_raw_before.fif"), overwrite=True)
    raw_pp.save(str(out_dir / f"{stem}_raw_pp.fif"), overwrite=True)
    ica.save(str(out_dir / f"{stem}_ica.fif"), overwrite=True)
    session = {
        "subject_name": subject_name,
        "file_index": file_index,
        "mff_rel_path": mff_rel_path,
        "mff_dir": mff_dir,
        "proba_threshold": float(proba_threshold),
        "bad_channels": bad_channels,
        "comp_labels": [str(x) for x in comp_labels],
        "comp_proba": [float(x) for x in comp_proba],
        "excluded_all_artifact": list(excluded_all_artifact),
        "brain_picks": list(brain_picks),
        "rejected_by_category": {k: list(v) for k, v in rejected_by_category.items()},
        "rejected_all_in_categories": list(rejected_all_in_categories),
        "psd_ymin": None if psd_ymin is None else float(psd_ymin),
        "psd_ymax": None if psd_ymax is None else float(psd_ymax),
        "wave_meta": wave_meta,
    }
    with open(out_dir / f"{stem}_session.json", "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=2)
    print(f"已写入 ICA 可视化会话缓存: {out_dir / (stem + '_session.json')}")


def _render_all_figs_and_summary(
    raw_pp,
    ica,
    subject_name: str,
    out_dir: Path,
    file_index: int,
    mff_rel_path: str,
    mff_dir: str,
    proba_threshold: float,
    bad_channels,
    comp_labels,
    comp_proba,
    excluded_all_artifact,
    brain_picks,
    rejected_by_category,
    rejected_all_in_categories,
    wave_ch_names,
    wave_picks,
    wave_start_samp,
    wave_stop_samp,
    wave_tmin_sec,
    wave_tmax_sec,
):
    raw_ica = raw_pp.copy()
    if excluded_all_artifact:
        ica.apply(raw_ica, exclude=excluded_all_artifact)

    _plot_psd_matplotlib(
        raw_pp,
        out_dir / f"{subject_name}_psd_after_preprocess_pre_ica.svg",
        "PSD after preprocess (pre-ICA)",
        fmin=0.1,
        fmax=PSD_FMAX_HZ,
    )
    _plot_psd_matplotlib(
        raw_ica,
        out_dir / f"{subject_name}_psd_after_ica.svg",
        "PSD after ICA",
        fmin=0.1,
        fmax=PSD_FMAX_HZ,
    )

    if (
        _wave_plot_channels_ready(wave_ch_names, wave_picks)
        and wave_start_samp is not None
        and wave_stop_samp is not None
    ):
        eeg_tag = "eeg" + "_".join(f"{i:03d}" for i in WAVE_PLOT_EEG_NUMBERS)
        win = f"{wave_tmin_sec:g}-{wave_tmax_sec:g}s"
        ch_tag = ", ".join(wave_ch_names)
        _plot_pre_post_stacked_waveforms(
            raw_pp,
            raw_ica,
            wave_picks,
            out_dir / f"{subject_name}_raw_pre_post_ica_stacked_{eeg_tag}.svg",
            wave_start_samp,
            wave_stop_samp,
            wave_tmin_sec,
            wave_tmax_sec,
        )
        if not WAVE_PLOT_ONLY_STACKED:
            wtitle = f"Raw after preprocess, pre-ICA ({ch_tag}, {win})"
            atitle = f"Raw after ICA ({ch_tag}, {win})"
            _plot_selected_waveforms(
                raw_pp,
                wave_picks,
                out_dir / f"{subject_name}_raw_after_preprocess_pre_ica_{eeg_tag}.svg",
                wtitle,
                wave_start_samp,
                wave_stop_samp,
                wave_tmin_sec,
                wave_tmax_sec,
            )
            _plot_selected_waveforms(
                raw_ica,
                wave_picks,
                out_dir / f"{subject_name}_raw_after_ica_{eeg_tag}.svg",
                atitle,
                wave_start_samp,
                wave_stop_samp,
                wave_tmin_sec,
                wave_tmax_sec,
            )

    if brain_picks:
        brain_comp_figs = ica.plot_components(picks=brain_picks, show=False)
        brain_comp_figs = _annotate_component_figures(
            brain_comp_figs,
            brain_picks,
            comp_labels,
            comp_proba,
        )
        _save_figure_or_list(
            brain_comp_figs,
            out_dir / f"{subject_name}_brain_components_annotated",
        )
        brain_props_figs = ica.plot_properties(raw_pp, picks=brain_picks, show=False)
        _save_figure_or_list(
            brain_props_figs,
            out_dir / f"{subject_name}_brain_properties",
        )
        _save_figure_or_list(
            ica.plot_sources(raw_pp, picks=brain_picks, show=False),
            out_dir / f"{subject_name}_brain_sources",
        )

    if excluded_all_artifact:
        all_comp_figs = ica.plot_components(picks=excluded_all_artifact, show=False)
        all_comp_figs = _annotate_component_figures(
            all_comp_figs,
            excluded_all_artifact,
            comp_labels,
            comp_proba,
        )
        _save_figure_or_list(
            all_comp_figs,
            out_dir / f"{subject_name}_all_rejected_components_annotated",
        )

    for cat, indices in rejected_by_category.items():
        if not indices:
            continue
        _save_figure_or_list(
            _annotate_component_figures(
                ica.plot_components(picks=indices, show=False),
                indices,
                comp_labels,
                comp_proba,
            ),
            out_dir / f"{subject_name}_{cat}_rejected_components",
        )
        _save_figure_or_list(
            ica.plot_properties(raw_pp, picks=indices, show=False),
            out_dir / f"{subject_name}_{cat}_rejected_properties",
        )

    if excluded_all_artifact:
        _save_figure_or_list(
            ica.plot_sources(raw_pp, picks=excluded_all_artifact, show=False),
            out_dir / f"{subject_name}_all_rejected_sources",
        )
    else:
        print("没有满足阈值的伪迹分量，跳过 source 图。")

    summary = {
        "subject_name": subject_name,
        "file_index": file_index,
        "mff_rel_path": mff_rel_path,
        "mff_dir": mff_dir,
        "proba_threshold": proba_threshold,
        "plot_format": "svg",
        "font_family": "Times New Roman",
        "font_size": 28,
        "preprocess_params": {
            "pick_first_128_eeg": True,
            "bad_channel_detection": "PyPREP (fallback MANUAL_BAD_ELECTRODES)",
            "interpolate_bads": True,
            "filter_l_freq": 0.1,
            "filter_h_freq": 45.0,
            "notch_freqs": [50],
            "reference": "average",
            "ica_method": "infomax",
            "ica_random_state": 97,
            "ica_max_iter": 500,
        },
        "plot_params": {
            "psd": {
                "style": "matplotlib multi-line PSD (Welch dB), y-axis autoscale",
                "fmin": 0.1,
                "fmax": PSD_FMAX_HZ,
                "ylim_db": "autoscale",
            },
            "waveform": {
                "channels": wave_ch_names,
                "ylim_uv": "per_channel (2–98% + margin)",
                "time_window_sec": [WAVE_PLOT_TMIN_SEC, WAVE_PLOT_TMAX_SEC],
            },
            "ica_components": {
                "all_artifact_picks": excluded_all_artifact,
                "annotated_titles": True,
                "title_format": "IC <idx> | <label> (<proba>)",
            },
        },
        "bad_channels_used": bad_channels,
        "rejected_by_category": rejected_by_category,
        "rejected_all_in_categories": rejected_all_in_categories,
        "excluded_all_artifact": excluded_all_artifact,
        "brain_picks": brain_picks,
        "component_details": [
            {"ic": i, "label": str(lab), "proba": float(p)}
            for i, (lab, p) in enumerate(zip(comp_labels, comp_proba))
        ],
        "replot_from_session": str(out_dir / f"{ICA_SESSION_STEM}_session.json"),
    }
    summary_path = out_dir / f"{subject_name}_ica_reject_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"全部图像与摘要已保存到: {out_dir}")


def _replot_ica_session(out_dir: Path, subject_name: str):
    stem = ICA_SESSION_STEM
    jp = out_dir / f"{stem}_session.json"
    if not jp.is_file():
        raise FileNotFoundError(
            f"缺少会话文件 {jp}，请先完整运行一次以生成 {stem}_*.fif 与 session.json"
        )
    with open(jp, "r", encoding="utf-8") as f:
        session = json.load(f)
    raw_pp = mne.io.read_raw_fif(out_dir / f"{stem}_raw_pp.fif", preload=True, verbose=False)
    ica = mne.preprocessing.read_ica(out_dir / f"{stem}_ica.fif", verbose=False)

    comp_labels = session["comp_labels"]
    comp_proba = session["comp_proba"]
    excluded_all_artifact = session["excluded_all_artifact"]
    brain_picks = session["brain_picks"]
    rejected_by_category = {k: list(v) for k, v in session["rejected_by_category"].items()}
    rejected_all_in_categories = session["rejected_all_in_categories"]
    proba_threshold = float(session["proba_threshold"])
    bad_channels = session["bad_channels"]
    file_index = int(session["file_index"])
    mff_rel_path = session["mff_rel_path"]
    mff_dir = session["mff_dir"]

    wave_meta = session.get("wave_meta")
    wave_ch_names = _wave_plot_channel_names()
    wave_picks = [raw_pp.ch_names.index(ch) for ch in wave_ch_names if ch in raw_pp.ch_names]
    # 重画时始终采用当前脚本配置的时间窗，避免被历史缓存窗口覆盖
    wt0 = WAVE_PLOT_TMIN_SEC
    wt1 = WAVE_PLOT_TMAX_SEC

    wave_start_samp, wave_stop_samp = (None, None)
    if _wave_plot_channels_ready(wave_ch_names, wave_picks):
        wave_start_samp, wave_stop_samp = _wave_plot_sample_range(raw_pp, raw_pp, wt0, wt1)

    print(f"[replot-only] 从会话缓存重画: {out_dir}")
    _render_all_figs_and_summary(
        raw_pp,
        ica,
        subject_name,
        out_dir,
        file_index,
        mff_rel_path,
        mff_dir,
        proba_threshold,
        bad_channels,
        comp_labels,
        comp_proba,
        excluded_all_artifact,
        brain_picks,
        rejected_by_category,
        rejected_all_in_categories,
        wave_ch_names,
        wave_picks,
        wave_start_samp,
        wave_stop_samp,
        wt0,
        wt1,
    )


def _has_replot_session(out_dir: Path) -> bool:
    return (out_dir / f"{ICA_SESSION_STEM}_session.json").is_file()


def _session_has_required_wave_channels(out_dir: Path, required_channels) -> bool:
    """缓存会话中是否包含当前要画的通道。"""
    jp = out_dir / f"{ICA_SESSION_STEM}_session.json"
    raw_pp_path = out_dir / f"{ICA_SESSION_STEM}_raw_pp.fif"
    if (not jp.is_file()) or (not raw_pp_path.is_file()):
        return False
    try:
        raw_pp_cached = mne.io.read_raw_fif(raw_pp_path, preload=False, verbose=False)
    except Exception:
        return False
    cached_set = set(raw_pp_cached.ch_names)
    req = [str(ch) for ch in required_channels]
    return all(ch in cached_set for ch in req)


def run_one_subject(subject_name: str, file_index: int, output_root: str, proba_threshold: float, replot_only: bool = False):
    subject_cfg = _find_subject_cfg(subject_name)
    if subject_cfg is None:
        raise ValueError(f"找不到被试: {subject_name}")

    mff_files = subject_cfg.get("mff_files", [])
    if not mff_files:
        raise ValueError(f"被试 {subject_name} 没有 mff_files 配置")
    if file_index < 1 or file_index > len(mff_files):
        raise ValueError(f"file_index 超出范围: 1~{len(mff_files)}")

    out_dir = Path(output_root) / subject_name / f"ica_reject_plots_file{file_index}_new"
    out_dir.mkdir(parents=True, exist_ok=True)

    if replot_only:
        _replot_ica_session(out_dir, subject_name)
        return

    if not (IC_LABEL_AVAILABLE and label_components is not None):
        raise RuntimeError("缺少 mne-icalabel，请先安装: pip install mne-icalabel")

    mff_rel_path = mff_files[file_index - 1]["mff"]
    mff_dir = _find_existing_mff_path(mff_rel_path)
    if mff_dir is None:
        raise FileNotFoundError(f"未找到 MFF 目录: {mff_rel_path}，搜索路径: {BASE_DATA_DIRS}")

    print(f"读取 MFF: {mff_dir}")
    raw = mne.io.read_raw_egi(mff_dir, preload=True, verbose=False)

    # 准备 montage
    montage = None
    coord_xml = os.path.join(mff_dir, "coordinates.xml")
    if os.path.exists(coord_xml):
        montage_path = out_dir / "electrode_montage.fif"
        make_montage(coord_xml, str(montage_path))
        montage = mne.channels.read_dig_fif(str(montage_path))
        print("montage 设置完成")
    else:
        print("未找到 coordinates.xml，继续执行（topomap 可能受影响）")

    # 原始波形图（预处理前）
    raw_before = raw.copy()
    _normalize_channels_and_set_montage(raw_before, montage)
    # 坏道检测
    raw_for_detection = raw.copy()
    _normalize_channels_and_set_montage(raw_for_detection, montage)
    bad_channels = _detect_bad_channels(raw_for_detection, montage)
    bad_channels = [ch for ch in bad_channels if ch in raw_for_detection.ch_names]
    print(f"坏道检测结果: {bad_channels}")

    # 正式预处理（不保存预处理数据）
    raw_pp = raw.copy()
    _normalize_channels_and_set_montage(raw_pp, montage)
    if bad_channels:
        raw_pp.info["bads"] = bad_channels
        try:
            raw_pp.interpolate_bads(reset_bads=True, verbose=False)
        except Exception as exc:
            # 坐标异常时不要中断整条流程；继续后续滤波/ICA/绘图
            print(f"警告: interpolate_bads 失败，已跳过插值继续执行。原因: {exc}")
            raw_pp.info["bads"] = []
    raw_pp.filter(l_freq=0.1, h_freq=45.0, method="fir", fir_window="hamming", verbose=False)
    raw_pp.notch_filter(freqs=[50], verbose=False)
    raw_pp.set_eeg_reference("average", verbose=False)

    wave_ch_names = _wave_plot_channel_names()
    wave_picks = [raw_pp.ch_names.index(ch) for ch in wave_ch_names if ch in raw_pp.ch_names]
    wave_start_samp, wave_stop_samp = (None, None)
    if _wave_plot_channels_ready(wave_ch_names, wave_picks):
        wave_start_samp, wave_stop_samp = _wave_plot_sample_range(
            raw_pp,
            raw_pp,
            WAVE_PLOT_TMIN_SEC,
            WAVE_PLOT_TMAX_SEC,
        )
        if wave_start_samp is None:
            print(
                f"警告: 波形时间窗 [{WAVE_PLOT_TMIN_SEC}, {WAVE_PLOT_TMAX_SEC}) s "
                "与数据长度无有效重叠，跳过 Raw 多道图。"
            )
    else:
        miss = [ch for ch in wave_ch_names if ch not in raw_pp.ch_names]
        print(
            f"警告: 波形图需要完整指定通道 {wave_ch_names}（共 {len(wave_ch_names)} 道），"
            f"缺失: {miss}；已匹配 {len(wave_picks)} 道"
        )

    # ICA + ICLabel
    ica = ICA(
        n_components=0.99,
        random_state=97,
        method="infomax",
        max_iter=500,
    )
    print("开始 ICA 拟合...")
    try:
        ica.fit(raw_pp, verbose=False)
    except RuntimeError as exc:
        msg = str(exc)
        if "One PCA component captures most of the explained variance" not in msg:
            raise
        # 兜底：当方差阈值导致只保留 1 个成分时，改为固定成分数再拟合
        # 经验上 20~30 对 EEG artifact 分离足够，且更稳定
        fallback_n = min(30, max(2, len(raw_pp.ch_names) - 1))
        print(
            "检测到 n_components=0.99 导致仅 1 个 PCA 成分，"
            f"自动切换为固定 n_components={fallback_n} 重试 ICA。"
        )
        ica = ICA(
            n_components=fallback_n,
            random_state=97,
            method="infomax",
            max_iter=500,
        )
        ica.fit(raw_pp, verbose=False)
    print("ICA 拟合完成，开始 ICLabel...")

    try:
        ic_labels = label_components(raw_pp, ica, method="iclabel")
        comp_labels = ic_labels["labels"]
        comp_proba = ic_labels["y_pred_proba"]
    except Exception as exc:
        # 某些数据可能因通道位置信息不完整导致 ICLabel 失败；不阻断 raw/PSD 出图
        n_comp = int(getattr(ica, "n_components_", 0) or 0)
        comp_labels = ["other"] * n_comp
        comp_proba = [0.0] * n_comp
        print(f"警告: ICLabel 失败，已跳过标签分类并继续绘图。原因: {exc}")

    rejected_by_category = {
        "eye": [],
        "muscle": [],
        "heart": [],
        "line_noise": [],
    }

    # 与 Preprocess_eeg.py 保持一致：仅剔除伪迹类，保留 brain 和 other
    excluded_all_artifact = [
        i
        for i, (lab, p) in enumerate(zip(comp_labels, comp_proba))
        if _is_iclabel_artifact(lab) and float(p) >= float(proba_threshold)
    ]
    excluded_all_artifact = sorted(excluded_all_artifact)

    # 也把 brain 成分画出来（规则与非 brain 保持一致：proba >= 阈值）
    brain_picks = [
        i
        for i, (lab, p) in enumerate(zip(comp_labels, comp_proba))
        if str(lab).lower() == "brain" and float(p) >= float(proba_threshold)
    ]
    brain_picks = sorted(brain_picks)

    # 仅用于“按 eye/muscle/heart/line_noise 分类”的列表
    rejected_all_in_categories = []
    for idx in excluded_all_artifact:
        category = _label_to_category(comp_labels[idx])
        if category is None:
            continue
        rejected_by_category[category].append(idx)
        rejected_all_in_categories.append(idx)
    rejected_all_in_categories = sorted(rejected_all_in_categories)

    print(f"被剔除分量总数（伪迹类全部）：{len(excluded_all_artifact)}")
    print(f"被剔除分量总数（四类：eye/muscle/heart/line_noise）：{len(rejected_all_in_categories)}")
    print(f"brain 成分数（label=brain & proba>=阈值）：{len(brain_picks)}")
    for cat, indices in rejected_by_category.items():
        print(f"  {cat}: {len(indices)} -> {indices}")

    wave_meta = None
    if _wave_plot_channels_ready(wave_ch_names, wave_picks) and wave_start_samp is not None:
        wave_meta = {
            "ch_names": wave_ch_names,
            "tmin_sec": float(WAVE_PLOT_TMIN_SEC),
            "tmax_sec": float(WAVE_PLOT_TMAX_SEC),
        }

    _save_ica_visual_session(
        out_dir,
        subject_name,
        file_index,
        mff_rel_path,
        mff_dir,
        proba_threshold,
        bad_channels,
        comp_labels,
        comp_proba,
        excluded_all_artifact,
        brain_picks,
        rejected_by_category,
        rejected_all_in_categories,
        None,
        None,
        wave_meta,
        raw_before,
        raw_pp,
        ica,
    )
    _render_all_figs_and_summary(
        raw_pp,
        ica,
        subject_name,
        out_dir,
        file_index,
        mff_rel_path,
        mff_dir,
        proba_threshold,
        bad_channels,
        comp_labels,
        comp_proba,
        excluded_all_artifact,
        brain_picks,
        rejected_by_category,
        rejected_all_in_categories,
        wave_ch_names,
        wave_picks,
        wave_start_samp,
        wave_stop_samp,
        WAVE_PLOT_TMIN_SEC,
        WAVE_PLOT_TMAX_SEC,
    )


def main():
    parser = argparse.ArgumentParser(description="单被试 ICA 被剔除分量可视化（按 eye/muscle/heart/line_noise 分类）")
    parser.add_argument(
        "--subject-name",
        type=str,
        default=PREVIEW_SUBJECT_NAME,
        help=f"被试名（来自 SUBJECTS_CONFIG），默认 PREVIEW_SUBJECT_NAME={PREVIEW_SUBJECT_NAME!r}",
    )
    parser.add_argument(
        "--file-index",
        type=int,
        default=PREVIEW_FILE_INDEX,
        help=f"该被试第几个 mff 文件（从 1 开始），默认 PREVIEW_FILE_INDEX={PREVIEW_FILE_INDEX}",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=PREVIEW_OUTPUT_ROOT,
        help=f"图像输出根目录（只保存图，不保存预处理数据），默认 PREVIEW_OUTPUT_ROOT={PREVIEW_OUTPUT_ROOT!r}",
    )
    parser.add_argument(
        "--proba-threshold",
        type=float,
        default=0.7,
        help="ICLabel 置信度阈值，满足条件的伪迹类成分（eye/muscle/heart/line/channel noise）会被判为剔除",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="仅根据 ica_visual_*.fif 与 ica_visual_session.json 重画（不读 MFF、不拟合 ICA）",
    )
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="强制完整重跑（读 MFF + 预处理 + ICA），忽略可用的重画缓存",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_root) / args.subject_name / f"ica_reject_plots_file{args.file_index}_new"
    requested_replot = args.replot_only or REPLOT_ICA_ONLY
    wave_ch_names = _wave_plot_channel_names()
    cache_ok_for_current_wave = _session_has_required_wave_channels(out_dir, wave_ch_names)
    auto_replot = AUTO_USE_REPLOT_CACHE and (not requested_replot) and (not args.full_run) and cache_ok_for_current_wave
    replot_only = requested_replot or auto_replot

    if auto_replot:
        print(
            "[AUTO] 检测到会话缓存且包含当前目标通道，"
            f"自动使用 replot-only: {out_dir / f'{ICA_SESSION_STEM}_session.json'}"
        )
    elif (not requested_replot) and (not args.full_run) and _has_replot_session(out_dir):
        print(
            "[AUTO] 检测到会话缓存，但缓存不包含当前目标通道，"
            "将自动完整重跑并刷新全通道缓存（读 MFF + 预处理 + ICA）。"
        )

    run_one_subject(
        subject_name=args.subject_name,
        file_index=args.file_index,
        output_root=args.output_root,
        proba_threshold=args.proba_threshold,
        replot_only=replot_only,
    )


if __name__ == "__main__":
    # 直接点击运行：只处理 PREVIEW_SUBJECT_NAME；设 REPLOT_ICA_ONLY=True 可只重画
    main()
