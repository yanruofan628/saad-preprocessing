# ============================================================
# 全被试群体拓扑（Left-Right Delta + All）
# 目标：每个频带（含全频段宽带）输出两张 topomap：
#   1) left-right delta：对每个被试先算百分比变化（% change）的 left/right，再 delta=left-right
#      群体层：对“被试级 delta”做聚合（默认 median）并做显著性检验
#   2) all：不区分标签，把 label=0/1 的 trials 合并后先算百分比变化，再被试聚合、群体聚合
#
# 百分比变化公式：
#   percent change = (P_stim - P_base) / P_base * 100
#   P_base：0-3s；P_stim：3-5s；频带内功率：Welch PSD（n_fft/n_per_seg=256，overlap=128）
#   在该频带内对频率维取均值（与全局 PSD 曲线同一 Welch 设定）。
#
# 与 global PSD 对齐的输出（文件名含 OUTPUT_ALIGNED_TAG）：
#   - 单被试 PSD 曲线：trial median → channel median；群体曲线：被试维 GROUP_AGG（默认 median）。
#   - task-minus-baseline 拓扑：10*log10(median_sbj P_task) - 10*log10(median_sbj P_base)。
#
# 显著性检验（每个通道独立）：
#   使用被试级 delta 与 0 的 Wilcoxon signed-rank test（可 FDR 校正）
#   输出显著通道列表 + 在 topomap 上用 mask 加粗标记
#
# 输出：
#   <DATA_ROOT>/all_subjects_topomap_delta_results/
#   - 每个频带 / 全频段：*_delta_*.svg（含显著点）
#   - 每个频带 / 全频段：*_all_*.svg（不分标签）
#   - npy_cache/median_topo_psd_unified/：topo + global PSD 统一管线缓存（不写 npy_cache/ 根目录，避免覆盖旧 .npy）
# ============================================================

import os
import json
import warnings
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")  # 保证无 GUI 环境也能保存图、不阻塞
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

from mne.channels import make_dig_montage
from subject_mff_benchmark_config import SUBJECTS_CONFIG


# ----------------------------
# 全局配置
# ----------------------------
DATA_ROOT = "A:/standard_data_interp_ica_99"
OUTPUT_DIR = os.path.join(DATA_ROOT, "all_subjects_topomap_delta_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Topo 与 global PSD 共用一套 Welch + trial/channel median；缓存单独子目录，不覆盖 npy_cache/ 根目录旧文件
UNIFIED_NPY_CACHE_SUBDIR = "median_topo_psd_unified"


def get_active_npy_cache_dir():
    return os.path.join(OUTPUT_DIR, "npy_cache", UNIFIED_NPY_CACHE_SUBDIR)


BASE_WINDOW = (0.0, 3.0)
STIM_WINDOW = (3.0, 5.0)
SAMPLING_RATE = 250

FREQ_BANDS = {
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    # 全频段（不分 Theta/Alpha/Beta）：在 [fmin,fmax] 内对 Welch PSD 做频率维平均，再算 % change
    # 1–45 Hz 与 extract 预处理上界一致，并略抬高下限以减少近直流影响
    "Broadband": (1, 45),
}

# 群体层聚合：topo 与 global PSD（group_aggregate_psd_freq）固定 median，勿改为 mean
GROUP_AGG = "median"

# 显著性
ALPHA = 0.05
USE_FDR = True

# 是否输出 left-right delta 图（及 consistency、delta 相关 npy/显著性）
DRAW_LEFT_RIGHT_DELTA = True

# False（默认）：完整重算 trial/Welch + 写入 median_topo_psd_unified 并出图——直接点运行即可。
# True：仅从统一缓存重绘 SVG（加快迭代）；需该子目录内已有完整 .npy。
REPLOT_ONLY = False
AUTO_FULL_RUN_IF_CACHE_MISSING = True

# 计算 Welch 的分块（避免大内存）
CHUNK_SIZE = 256
PSD_CURVE_FMIN = 1.0
PSD_CURVE_FMAX = 45.0

# Welch 参数：频段功率与全局 PSD 曲线共用，便于与 task-minus-baseline 拓扑对齐
WELCH_N_FFT = 256
WELCH_N_PER_SEG = 256
WELCH_N_OVERLAP = 128

# SVG 后缀（与 UNIFIED_NPY_CACHE_SUBDIR 对应）；缓存文件名放在子目录内故不用长后缀防冲突。
OUTPUT_ALIGNED_TAG = "_median_topo_psd_unified"
PSD_CURVE_CACHE_SUBJECTS_BASE = "psd_curve_subjects_base.npy"
PSD_CURVE_CACHE_SUBJECTS_TASK = "psd_curve_subjects_task.npy"
PSD_CURVE_CACHE_FREQS = "psd_curve_freqs.npy"

# 全局 PSD：False = 只画平滑双线（无 SEM / 无 IQR 阴影），与旧版 smoothed_sem 主曲线同款观感但不画不确定性带
DRAW_GLOBAL_PSD_UNCERTAINTY = False

# Global PSD 作图：谱线在 PSD_DISPLAY_LINE_MAX_HZ 截止；横坐标上限与之对齐；横轴主刻度间隔（Hz）
PSD_DISPLAY_LINE_MAX_HZ = 40.0
PSD_DISPLAY_AXIS_X_HI_HZ = PSD_DISPLAY_LINE_MAX_HZ
PSD_DISPLAY_X_TICK_STEP_HZ = 8.0
PSD_DISPLAY_Y_TICK_STEP_DB = 10.0
# 纵轴显示范围下限（ylim）；刻度数字从 PSD_DISPLAY_Y_TICK_START_DB 起按步长标注
PSD_DISPLAY_YLIM_AXIS_LO_DB = -125.0
PSD_DISPLAY_Y_TICK_START_DB = -120.0
PSD_DISPLAY_YLIM_HI_DB = -90.0
PSD_DISPLAY_FONT_SIZE = 38
PSD_DISPLAY_SPINE_LINEWIDTH = 0.8
# 画布宽高（inch）；宽度在原先 12.89 in 基础上净增 2 cm（先前 +5 cm 后再减 3 cm）
PSD_DISPLAY_FIGSIZE_INCHES = (12.89 + 2.0 / 2.54, 8.44)

PLOT_PARAMS = {
    "font_family": "Times New Roman",
    "font_size": 28,
    "cmap": "RdBu_r",
    "mask_marker": "o",
    "mask_edgecolor": "k",
    "mask_facecolor": "none",
    "mask_markersize": 12,
    "mask_markeredgewidth": 2,
}

# 电极蒙太奇文件名、仅 ch_pos 的 montage 会触发无害警告；过滤后避免 PowerShell 把 stderr 当致命错误
warnings.filterwarnings(
    "ignore",
    message=r".*does not conform to MNE naming conventions.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Fiducial point nasion not found.*",
    category=RuntimeWarning,
)


def _write_measurement_info(fname, info):
    """写入 measurement info；兼容无 write_info(overwrite=) 的旧版 MNE。"""
    fname = os.fspath(fname)
    if os.path.isfile(fname):
        os.remove(fname)
    save = getattr(info, "save", None)
    if callable(save):
        save(fname)
        return
    if hasattr(mne.io, "write_info"):
        mne.io.write_info(fname, info)
        return
    raise RuntimeError("当前环境无法写入 Info（缺少 mne.io.write_info 与 info.save）")


def _read_measurement_info(fname):
    """读取缓存的 measurement info；兼容旧版 API。"""
    fname = os.fspath(fname)
    if hasattr(mne.io, "read_info"):
        return mne.io.read_info(fname)
    reader = getattr(mne, "read_meas_info", None)
    if reader is not None:
        return reader(fname)
    raise RuntimeError("当前环境无法读取 Info（缺少 mne.io.read_info）")


def fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR 控制，返回显著mask。"""
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    if n == 0:
        return np.zeros((0,), dtype=bool), np.nan
    order = np.argsort(pvals)
    ranked = np.arange(1, n + 1)
    thresh = alpha * ranked / n
    passed = pvals[order] <= thresh
    if not np.any(passed):
        return np.zeros_like(pvals, dtype=bool), 0.0
    k = np.max(np.where(passed)[0])
    crit_p = pvals[order][k]
    return pvals <= crit_p, float(crit_p)


def add_fiducials_to_info(info, montage):
    """把 nasion/LPA/RPA 从 montage.dig 补回 info['dig']，让 plot_topomap 能显示尖尖与耳朵。"""
    try:
        from mne._fiff.constants import FIFF

        fid_kinds = {FIFF.FIFFV_POINT_NASION, FIFF.FIFFV_POINT_LPA, FIFF.FIFFV_POINT_RPA}
        fiducials = [d for d in montage.dig if d.get("kind") in fid_kinds]
        if not fiducials:
            return info

        current_dig = list(info.get("dig", [])) if info.get("dig", None) is not None else []
        current_dig = [d for d in current_dig if d.get("kind") not in fid_kinds]
        info["dig"] = current_dig + fiducials
    except Exception:
        return info
    return info


def load_subject_entry(subject_name):
    subject_dir = os.path.join(DATA_ROOT, subject_name)
    npy_path = os.path.join(subject_dir, f"{subject_name}_trials.npy")
    info_path = os.path.join(subject_dir, f"{subject_name}_trial_info.json")
    labels_path = os.path.join(subject_dir, f"{subject_name}_labels.csv")
    montage_path = os.path.join(subject_dir, "electrode_montage.fif")

    required = [npy_path, info_path, labels_path, montage_path]
    if not all(os.path.exists(p) for p in required):
        missing = [p for p in required if not os.path.exists(p)]
        print(f"[跳过] {subject_name} 缺少文件: {missing}")
        return None

    trials = np.load(npy_path, allow_pickle=True)
    with open(info_path, "r", encoding="utf-8") as f:
        trial_info = json.load(f)
    labels_df = pd.read_csv(labels_path)
    montage = mne.channels.read_dig_fif(montage_path)
    ch_labels = trial_info["channels"]

    # 兼容 reshape（如果之前导出成 2D flatten）
    if trials.ndim == 2:
        n_trials = trials.shape[0]
        n_channels = len(ch_labels)
        samples_per_trial = trials.shape[1] // n_channels
        trials = trials.reshape(n_trials, n_channels, samples_per_trial)

    if trials.shape[0] == 0:
        print(f"[跳过] {subject_name} 无 trial")
        return None

    return {
        "subject_name": subject_name,
        "trials": trials,
        "labels_df": labels_df,
        "montage": montage,
        "ch_labels": ch_labels,
    }


def _collect_label_indices(labels_df, n_trials):
    """返回 label0/label1 的 0-based int 索引列表。"""
    label_0_indices, label_1_indices = [], []
    for _, row in labels_df.iterrows():
        trial_num = row.get("Trial")
        label = row.get("Label")
        if pd.isna(trial_num) or pd.isna(label):
            continue
        try:
            trial_idx = int(float(trial_num)) - 1
            label_int = int(float(label))
        except (TypeError, ValueError):
            continue
        if trial_idx < 0 or trial_idx >= n_trials:
            continue
        if label_int == 0:
            label_0_indices.append(trial_idx)
        elif label_int == 1:
            label_1_indices.append(trial_idx)
    return label_0_indices, label_1_indices


def get_common_channels(entries):
    common = set(entries[0]["ch_labels"])
    for e in entries[1:]:
        common &= set(e["ch_labels"])
    common = [ch for ch in entries[0]["ch_labels"] if ch in common]
    if not common:
        raise RuntimeError("各被试无公共通道")
    return common


def build_info_and_channel_order(entries, common_channels):
    """用第一个被试的 montage 构建 info，并返回 info + topomap channel order（=valid_channels）。"""
    montage0 = entries[0]["montage"]
    ch_pos_dict = montage0.get_positions()["ch_pos"]

    common_ch_pos = {}
    for ch in common_channels:
        if ch in ch_pos_dict:
            common_ch_pos[ch] = ch_pos_dict[ch]
            continue
        if ch.startswith("EEG") and ch[3:].isdigit():
            num = int(ch[3:])
            e_key = f"E{num}"
            eeg_key = f"EEG{num:03d}"
            if e_key in ch_pos_dict:
                common_ch_pos[ch] = ch_pos_dict[e_key]
            elif eeg_key in ch_pos_dict:
                common_ch_pos[ch] = ch_pos_dict[eeg_key]
        elif ch.startswith("E") and ch[1:].isdigit():
            num = int(ch[1:])
            eeg_key = f"EEG{num:03d}"
            if eeg_key in ch_pos_dict:
                common_ch_pos[ch] = ch_pos_dict[eeg_key]

    valid_channels = [ch for ch in common_channels if ch in common_ch_pos]
    if not valid_channels:
        raise RuntimeError("公共通道在 montage 中均未找到位置")

    pos_all = montage0.get_positions()
    dig_kw = {"ch_pos": common_ch_pos}
    for key in ("nasion", "lpa", "rpa"):
        v = pos_all.get(key)
        if v is not None:
            dig_kw[key] = np.asarray(v, dtype=float)

    info = mne.create_info(ch_names=valid_channels, sfreq=SAMPLING_RATE, ch_types="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        info.set_montage(make_dig_montage(**dig_kw))
    info = add_fiducials_to_info(info, montage0)
    return info, valid_channels


def _compute_trial_band_power_with_epochs_compute_psd(seg_3d, fmin, fmax):
    """用 Epochs.compute_psd（Welch 参数与全局 PSD 曲线一致）计算 trial×channel 频带功率均值，返回 (n_trials, n_ch)。"""
    if seg_3d.shape[0] == 0:
        return np.zeros((0, seg_3d.shape[1]), dtype=np.float32)
    n_ch = int(seg_3d.shape[1])
    ch_names = [f"EEG{i:03d}" for i in range(1, n_ch + 1)]
    info_tmp = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types="eeg")
    epochs = mne.EpochsArray(np.asarray(seg_3d, dtype=np.float32), info_tmp, tmin=0.0, verbose=False)
    spec = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=WELCH_N_FFT,
        n_per_seg=WELCH_N_PER_SEG,
        n_overlap=WELCH_N_OVERLAP,
        verbose=False,
    )
    return np.mean(spec.get_data(), axis=2).astype(np.float32)


def compute_percent_change_per_trial(trials_3d, fmin, fmax, base_window, stim_window, chunk_size=CHUNK_SIZE):
    """对每个 trial/通道计算 percent change：((P_stim-P_base)/P_base)*100，返回 (n_trials, n_ch)。"""
    if trials_3d.shape[0] == 0:
        return np.zeros((0, trials_3d.shape[1]), dtype=np.float32)

    base_start = int(round(base_window[0] * SAMPLING_RATE))
    base_end = int(round(base_window[1] * SAMPLING_RATE))
    stim_start = int(round(stim_window[0] * SAMPLING_RATE))
    stim_end = int(round(stim_window[1] * SAMPLING_RATE))
    n_times = trials_3d.shape[-1]
    if base_start < 0 or stim_end > n_times:
        raise ValueError("基线/刺激时间窗超出 trial 长度")

    eps = 1e-20
    data = np.asarray(trials_3d, dtype=np.float32)
    out_chunks = []

    for start in range(0, data.shape[0], chunk_size):
        chunk = data[start:start + chunk_size]
        base_seg = chunk[:, :, base_start:base_end]
        stim_seg = chunk[:, :, stim_start:stim_end]
        p_base = _compute_trial_band_power_with_epochs_compute_psd(base_seg, fmin=fmin, fmax=fmax)
        p_stim = _compute_trial_band_power_with_epochs_compute_psd(stim_seg, fmin=fmin, fmax=fmax)
        pct = ((p_stim - p_base) / (p_base + eps)) * 100.0
        out_chunks.append(pct.astype(np.float32))

    return np.vstack(out_chunks)


def compute_task_minus_baseline_per_trial(trials_3d, fmin, fmax, base_window, stim_window, chunk_size=CHUNK_SIZE):
    """对每个 trial/通道计算绝对差：P_stim - P_base，返回 (n_trials, n_ch)。"""
    if trials_3d.shape[0] == 0:
        return np.zeros((0, trials_3d.shape[1]), dtype=np.float32)

    base_start = int(round(base_window[0] * SAMPLING_RATE))
    base_end = int(round(base_window[1] * SAMPLING_RATE))
    stim_start = int(round(stim_window[0] * SAMPLING_RATE))
    stim_end = int(round(stim_window[1] * SAMPLING_RATE))
    n_times = trials_3d.shape[-1]
    if base_start < 0 or stim_end > n_times:
        raise ValueError("基线/刺激时间窗超出 trial 长度")

    data = np.asarray(trials_3d, dtype=np.float32)
    out_chunks = []

    for start in range(0, data.shape[0], chunk_size):
        chunk = data[start:start + chunk_size]
        base_seg = chunk[:, :, base_start:base_end]
        stim_seg = chunk[:, :, stim_start:stim_end]
        p_base = _compute_trial_band_power_with_epochs_compute_psd(base_seg, fmin=fmin, fmax=fmax)
        p_stim = _compute_trial_band_power_with_epochs_compute_psd(stim_seg, fmin=fmin, fmax=fmax)
        diff = p_stim - p_base
        out_chunks.append(diff.astype(np.float32))

    return np.vstack(out_chunks)


def compute_baseline_and_task_power_per_trial(trials_3d, fmin, fmax, base_window, stim_window, chunk_size=CHUNK_SIZE):
    """返回每 trial/通道的 (P_base, P_stim)，shape 均为 (n_trials, n_ch)。"""
    if trials_3d.shape[0] == 0:
        z = np.zeros((0, trials_3d.shape[1]), dtype=np.float32)
        return z, z

    base_start = int(round(base_window[0] * SAMPLING_RATE))
    base_end = int(round(base_window[1] * SAMPLING_RATE))
    stim_start = int(round(stim_window[0] * SAMPLING_RATE))
    stim_end = int(round(stim_window[1] * SAMPLING_RATE))
    n_times = trials_3d.shape[-1]
    if base_start < 0 or stim_end > n_times:
        raise ValueError("基线/刺激时间窗超出 trial 长度")

    data = np.asarray(trials_3d, dtype=np.float32)
    base_chunks, stim_chunks = [], []
    for start in range(0, data.shape[0], chunk_size):
        chunk = data[start:start + chunk_size]
        base_seg = chunk[:, :, base_start:base_end]
        stim_seg = chunk[:, :, stim_start:stim_end]
        p_base = _compute_trial_band_power_with_epochs_compute_psd(base_seg, fmin=fmin, fmax=fmax)
        p_stim = _compute_trial_band_power_with_epochs_compute_psd(stim_seg, fmin=fmin, fmax=fmax)
        base_chunks.append(p_base.astype(np.float32))
        stim_chunks.append(p_stim.astype(np.float32))
    return np.vstack(base_chunks), np.vstack(stim_chunks)


def aggregate_subject_median_percent(trials_subset, fmin, fmax):
    """先算每 trial 的 percent change，再对 trial 取 median => shape (n_ch,)."""
    pct = compute_percent_change_per_trial(
        trials_subset,
        fmin=fmin,
        fmax=fmax,
        base_window=BASE_WINDOW,
        stim_window=STIM_WINDOW,
        chunk_size=CHUNK_SIZE,
    )
    if pct.shape[0] == 0:
        return None
    return np.median(pct, axis=0)


def aggregate_subject_median_task_minus_baseline(trials_subset, fmin, fmax):
    """先算每 trial 的 task-baseline 绝对差，再对 trial 取 median => shape (n_ch,)."""
    diff = compute_task_minus_baseline_per_trial(
        trials_subset,
        fmin=fmin,
        fmax=fmax,
        base_window=BASE_WINDOW,
        stim_window=STIM_WINDOW,
        chunk_size=CHUNK_SIZE,
    )
    if diff.shape[0] == 0:
        return None
    return np.median(diff, axis=0)


def aggregate_subject_median_base_and_task_power(trials_subset, fmin, fmax):
    """先算每 trial 的 (P_base, P_stim)，再对 trial 取 median。"""
    p_base, p_stim = compute_baseline_and_task_power_per_trial(
        trials_subset,
        fmin=fmin,
        fmax=fmax,
        base_window=BASE_WINDOW,
        stim_window=STIM_WINDOW,
        chunk_size=CHUNK_SIZE,
    )
    if p_base.shape[0] == 0:
        return None, None
    return np.median(p_base, axis=0), np.median(p_stim, axis=0)


def compute_subject_mean_psd_curve(trials_3d, t_window, fmin=PSD_CURVE_FMIN, fmax=PSD_CURVE_FMAX):
    """基于 Epochs.compute_psd 计算单被试 PSD 曲线（trial median → channel median，与频段拓扑 trial median 对齐）。"""
    if trials_3d.shape[0] == 0:
        return None, None
    start = int(round(t_window[0] * SAMPLING_RATE))
    stop = int(round(t_window[1] * SAMPLING_RATE))
    if start < 0 or stop > trials_3d.shape[-1] or stop <= start:
        return None, None
    seg = np.asarray(trials_3d[:, :, start:stop], dtype=np.float32)
    n_ch = int(seg.shape[1])
    ch_names = [f"EEG{i:03d}" for i in range(1, n_ch + 1)]
    info_tmp = mne.create_info(ch_names=ch_names, sfreq=SAMPLING_RATE, ch_types="eeg")
    epochs = mne.EpochsArray(seg, info_tmp, tmin=0.0, verbose=False)
    spec = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        n_fft=WELCH_N_FFT,
        n_per_seg=WELCH_N_PER_SEG,
        n_overlap=WELCH_N_OVERLAP,
        verbose=False,
    )
    data = spec.get_data()  # (n_trials, n_ch, n_freqs)
    curve = np.median(np.median(data, axis=0), axis=0).astype(np.float32)
    return curve, np.asarray(spec.freqs, dtype=np.float32)


def _hide_top_right_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    """谱线展示：保留 f≤上限；若末采样 <40 Hz，则在 PSD_DISPLAY_LINE_MAX_HZ 处线性插值追加端点。"""
    freqs = np.asarray(freqs, dtype=float)
    series = [np.asarray(s, dtype=float) for s in series]
    cap = float(PSD_DISPLAY_LINE_MAX_HZ)
    m = freqs <= cap
    if not np.any(m):
        raise ValueError(f"没有频点满足 f <= {cap} Hz")
    f_out = freqs[m].copy()
    outs = [s[m].copy() for s in series]
    if float(f_out[-1]) < cap - 1e-6:
        f_out = np.append(f_out, cap)
        for i, s in enumerate(series):
            outs[i] = np.append(outs[i], float(np.interp(cap, freqs, s)))
    return (f_out,) + tuple(outs)


def _apply_global_psd_xy_axes(ax, y_values=None):
    """横轴 [0, PSD_DISPLAY_AXIS_X_HI_HZ]，主刻度间隔 PSD_DISPLAY_X_TICK_STEP_HZ；
    纵轴范围 [PSD_DISPLAY_YLIM_AXIS_LO_DB, PSD_DISPLAY_YLIM_HI_DB]，刻度从 PSD_DISPLAY_Y_TICK_START_DB 起按步长标。"""
    import matplotlib as mpl

    ax.set_xlim(0.0, float(PSD_DISPLAY_AXIS_X_HI_HZ))
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(float(PSD_DISPLAY_X_TICK_STEP_HZ)))
    ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))

    y_lo_axis = float(PSD_DISPLAY_YLIM_AXIS_LO_DB)
    y_hi = float(PSD_DISPLAY_YLIM_HI_DB)
    ax.set_ylim(y_lo_axis, y_hi)
    y_step = float(PSD_DISPLAY_Y_TICK_STEP_DB)
    y_first = float(PSD_DISPLAY_Y_TICK_START_DB)
    y_ticks = np.arange(y_first, y_hi + 0.5 * y_step, y_step)
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0f"))
    lw = float(PSD_DISPLAY_SPINE_LINEWIDTH)
    ax.spines["left"].set_linewidth(lw)
    ax.spines["bottom"].set_linewidth(lw)
    ax.tick_params(axis="both", which="major", width=lw, length=max(6.0, lw * 2.5))


def plot_psd_task_vs_baseline_svg(freqs, base_curve, task_curve, save_path):
    """画 baseline vs task 的群体 PSD 对比曲线（dB）。"""
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    fs = int(PSD_DISPLAY_FONT_SIZE)
    mpl.rcParams["font.size"] = fs
    mpl.rcParams["axes.titlesize"] = fs
    mpl.rcParams["axes.labelsize"] = fs
    mpl.rcParams["xtick.labelsize"] = fs
    mpl.rcParams["ytick.labelsize"] = fs
    mpl.rcParams["legend.fontsize"] = fs
    mpl.rcParams["font.weight"] = "normal"
    mpl.rcParams["axes.titleweight"] = "normal"
    mpl.rcParams["axes.labelweight"] = "normal"

    freqs = np.asarray(freqs, dtype=float)
    base_curve = np.asarray(base_curve, dtype=float)
    task_curve = np.asarray(task_curve, dtype=float)
    # 兜底：若缓存里残留旧数据导致长度不一致，先截到共同长度避免崩溃
    if not (freqs.shape[0] == base_curve.shape[0] == task_curve.shape[0]):
        n = int(min(freqs.shape[0], base_curve.shape[0], task_curve.shape[0]))
        freqs = freqs[:n]
        base_curve = base_curve[:n]
        task_curve = task_curve[:n]

    packed = _cap_psd_freq_series_at_display_limit(freqs, base_curve, task_curve)
    freqs = packed[0]
    base_curve = packed[1]
    task_curve = packed[2]

    eps = np.finfo(float).eps
    base_db = 10.0 * np.log10(base_curve + eps)
    task_db = 10.0 * np.log10(task_curve + eps)

    fig, ax = plt.subplots(figsize=PSD_DISPLAY_FIGSIZE_INCHES)
    ax.plot(freqs, base_db, label="Baseline", color="grey", linestyle="--", linewidth=2.0)
    ax.plot(freqs, task_db, label="Task", color="red", linewidth=2.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.grid(True, alpha=0.3)
    _apply_global_psd_xy_axes(ax, np.concatenate([base_db, task_db]))
    _hide_top_right_spines(ax)
    ax.legend()
    fig.tight_layout()
    # 不使用 bbox_inches="tight"，避免裁剪导致整体宽高比偏离 figsize 设定
    fig.savefig(save_path, dpi=300, format="svg")
    plt.close(fig)
    print(f"已保存: {save_path}")


def _moving_average_1d(x, window=3):
    """一维移动平均（edge padding 保持长度不变）。"""
    x = np.asarray(x, dtype=float)
    w = int(max(1, window))
    if w <= 1 or x.size == 0:
        return x.copy()
    if w % 2 == 0:
        w += 1
    pad = w // 2
    kernel = np.ones((w,), dtype=float) / float(w)
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, kernel, mode="valid")


def plot_psd_task_vs_baseline_with_uncertainty_svg(
    freqs,
    base_subject_mat,
    task_subject_mat,
    save_path,
    smooth_window=3,
):
    """画 task/baseline PSD（dB）：群体趋势线对被试维取 median（或 GROUP_AGG=mean），阴影为各频点 25–75% 分位。"""
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    fs = int(PSD_DISPLAY_FONT_SIZE)
    mpl.rcParams["font.size"] = fs
    mpl.rcParams["axes.titlesize"] = fs
    mpl.rcParams["axes.labelsize"] = fs
    mpl.rcParams["xtick.labelsize"] = fs
    mpl.rcParams["ytick.labelsize"] = fs
    mpl.rcParams["legend.fontsize"] = fs
    mpl.rcParams["font.weight"] = "normal"
    mpl.rcParams["axes.titleweight"] = "normal"
    mpl.rcParams["axes.labelweight"] = "normal"

    freqs = np.asarray(freqs, dtype=float)
    base_subject_mat = np.asarray(base_subject_mat, dtype=float)
    task_subject_mat = np.asarray(task_subject_mat, dtype=float)

    if base_subject_mat.ndim != 2 or task_subject_mat.ndim != 2:
        raise ValueError("base_subject_mat/task_subject_mat 必须是 2D (n_subj, n_freq)")

    n_subj = int(min(base_subject_mat.shape[0], task_subject_mat.shape[0]))
    n_freq = int(min(freqs.shape[0], base_subject_mat.shape[1], task_subject_mat.shape[1]))
    if n_subj <= 0 or n_freq <= 0:
        raise ValueError("PSD 数据为空，无法绘图")

    freqs = freqs[:n_freq]
    base_subject_mat = base_subject_mat[:n_subj, :n_freq]
    task_subject_mat = task_subject_mat[:n_subj, :n_freq]

    eps = np.finfo(float).eps
    base_db_subj = 10.0 * np.log10(np.maximum(base_subject_mat, eps))
    task_db_subj = 10.0 * np.log10(np.maximum(task_subject_mat, eps))

    if GROUP_AGG == "mean":
        base_cent = np.mean(base_db_subj, axis=0)
        task_cent = np.mean(task_db_subj, axis=0)
        denom = float(np.sqrt(max(n_subj, 1)))
        base_lo = base_cent - np.std(base_db_subj, axis=0, ddof=1 if n_subj > 1 else 0) / denom
        base_hi = base_cent + np.std(base_db_subj, axis=0, ddof=1 if n_subj > 1 else 0) / denom
        task_lo = task_cent - np.std(task_db_subj, axis=0, ddof=1 if n_subj > 1 else 0) / denom
        task_hi = task_cent + np.std(task_db_subj, axis=0, ddof=1 if n_subj > 1 else 0) / denom
    else:
        base_cent = np.median(base_db_subj, axis=0)
        task_cent = np.median(task_db_subj, axis=0)
        base_lo = np.percentile(base_db_subj, 25, axis=0)
        base_hi = np.percentile(base_db_subj, 75, axis=0)
        task_lo = np.percentile(task_db_subj, 25, axis=0)
        task_hi = np.percentile(task_db_subj, 75, axis=0)

    base_cent_s = _moving_average_1d(base_cent, window=smooth_window)
    task_cent_s = _moving_average_1d(task_cent, window=smooth_window)
    base_lo_s = _moving_average_1d(base_lo, window=smooth_window)
    base_hi_s = _moving_average_1d(base_hi, window=smooth_window)
    task_lo_s = _moving_average_1d(task_lo, window=smooth_window)
    task_hi_s = _moving_average_1d(task_hi, window=smooth_window)

    packed = _cap_psd_freq_series_at_display_limit(
        freqs,
        base_cent_s,
        task_cent_s,
        base_lo_s,
        base_hi_s,
        task_lo_s,
        task_hi_s,
    )
    freqs = packed[0]
    base_cent_s, task_cent_s, base_lo_s, base_hi_s, task_lo_s, task_hi_s = packed[1:]

    # 画布尺寸：宽高（单位=inch）
    fig, ax = plt.subplots(figsize=PSD_DISPLAY_FIGSIZE_INCHES)
    # 先画阴影再画线，避免遮挡主趋势线；同时降低透明度减少重叠糊感
    ax.fill_between(
        freqs,
        base_lo_s,
        base_hi_s,
        color="#B0B0B0",
        alpha=0.12,
        linewidth=0,
        zorder=1,
    )
    ax.fill_between(
        freqs,
        task_lo_s,
        task_hi_s,
        color="#F4A3A3",
        alpha=0.10,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        freqs,
        base_cent_s,
        label="Baseline",
        color="#4A4A4A",
        linestyle="--",
        linewidth=2.4,
        zorder=3,
    )
    ax.plot(
        freqs,
        task_cent_s,
        label="Task",
        color="#B22222",
        linewidth=2.4,
        zorder=3,
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.grid(True, alpha=0.3)
    _apply_global_psd_xy_axes(
        ax,
        np.concatenate(
            [
                base_cent_s,
                task_cent_s,
                base_lo_s,
                base_hi_s,
                task_lo_s,
                task_hi_s,
            ]
        ),
    )
    _hide_top_right_spines(ax)
    ax.legend()
    fig.tight_layout()
    # 不使用 bbox_inches="tight"，避免裁剪导致整体宽高比偏离 figsize 设定
    fig.savefig(save_path, dpi=300, format="svg")
    plt.close(fig)
    print(f"已保存: {save_path}")


def plot_psd_task_vs_baseline_smoothed_no_sem_svg(
    freqs,
    base_curve,
    task_curve,
    save_path,
    smooth_window=3,
    font_size=None,
):
    """画 task/baseline PSD（dB），只展示平滑均值线，不展示 SEM。"""
    import matplotlib as mpl
    if font_size is None:
        font_size = PSD_DISPLAY_FONT_SIZE
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial"]
    mpl.rcParams["font.size"] = font_size
    mpl.rcParams["axes.titlesize"] = font_size
    mpl.rcParams["axes.labelsize"] = font_size
    mpl.rcParams["xtick.labelsize"] = font_size
    mpl.rcParams["ytick.labelsize"] = font_size
    mpl.rcParams["legend.fontsize"] = font_size
    mpl.rcParams["font.weight"] = "normal"
    mpl.rcParams["axes.titleweight"] = "normal"
    mpl.rcParams["axes.labelweight"] = "normal"

    freqs = np.asarray(freqs, dtype=float)
    base_curve = np.asarray(base_curve, dtype=float)
    task_curve = np.asarray(task_curve, dtype=float)

    if not (freqs.shape[0] == base_curve.shape[0] == task_curve.shape[0]):
        n = int(min(freqs.shape[0], base_curve.shape[0], task_curve.shape[0]))
        freqs = freqs[:n]
        base_curve = base_curve[:n]
        task_curve = task_curve[:n]

    eps = np.finfo(float).eps
    base_db = 10.0 * np.log10(base_curve + eps)
    task_db = 10.0 * np.log10(task_curve + eps)

    base_db_s = _moving_average_1d(base_db, window=smooth_window)
    task_db_s = _moving_average_1d(task_db, window=smooth_window)

    packed = _cap_psd_freq_series_at_display_limit(freqs, base_db_s, task_db_s)
    freqs = packed[0]
    base_db_s = packed[1]
    task_db_s = packed[2]

    fig, ax = plt.subplots(figsize=PSD_DISPLAY_FIGSIZE_INCHES)
    ax.plot(
        freqs,
        base_db_s,
        label="Baseline",
        color="#4A4A4A",
        linestyle="--",
        linewidth=2.6,
        zorder=3,
    )
    ax.plot(
        freqs,
        task_db_s,
        label="Task",
        color="#B22222",
        linewidth=2.6,
        zorder=3,
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB)")

    ax.grid(True, alpha=0.3)
    _apply_global_psd_xy_axes(ax, np.concatenate([base_db_s, task_db_s]))
    _hide_top_right_spines(ax)
    ax.legend()
    fig.tight_layout()
    # 不使用 bbox_inches="tight"，避免裁剪导致整体宽高比偏离 figsize 设定
    fig.savefig(save_path, dpi=300, format="svg")
    plt.close(fig)
    print(f"已保存: {save_path}")


def _power_to_db(x):
    x = np.asarray(x, dtype=float)
    eps = np.finfo(float).eps
    return 10.0 * np.log10(np.maximum(x, eps))


def group_aggregate(values_2d):
    """values_2d: (n_subj, n_ch). 返回 (n_ch,)"""
    if values_2d.shape[0] == 0:
        return None
    if GROUP_AGG == "mean":
        return np.mean(values_2d, axis=0)
    return np.median(values_2d, axis=0)


def group_aggregate_psd_freq(values_2d):
    """values_2d: (n_subj, n_freq). 群体 PSD 曲线对被试维聚合，规则与 GROUP_AGG 一致。"""
    if values_2d.shape[0] == 0:
        return None
    if GROUP_AGG == "mean":
        return np.mean(values_2d, axis=0)
    return np.median(values_2d, axis=0)


def wilcoxon_channel_pvals(delta_subjects):
    """delta_subjects: (n_subj, n_ch) => pvals (n_ch,)"""
    n_subj, n_ch = delta_subjects.shape
    pvals = np.ones((n_ch,), dtype=float)
    for ch in range(n_ch):
        x = delta_subjects[:, ch]
        if np.allclose(x, 0):
            pvals[ch] = 1.0
            continue
        if n_subj < 3:
            pvals[ch] = 1.0
            continue
        try:
            stat = wilcoxon(x, zero_method="wilcox", alternative="two-sided", mode="auto")
            pvals[ch] = stat.pvalue
        except Exception:
            pvals[ch] = 1.0
    return pvals


def plot_topomap_svg(
    topo_data,
    info,
    label_text,
    save_path,
    mask=None,
    vmin=None,
    vmax=None,
    cmap=None,
    colorbar_label="percent change (%)",
    symmetric_scale=True,
):
    """画 topomap（SVG），顶部标记 label_text，显著通道用 mask 加粗。"""
    import matplotlib as mpl

    mpl.rcParams["font.family"] = PLOT_PARAMS["font_family"]
    mpl.rcParams["font.size"] = PLOT_PARAMS["font_size"]

    topo_data = np.asarray(topo_data, dtype=float)
    if vmin is None or vmax is None:
        if symmetric_scale:
            data_range = max(abs(float(np.min(topo_data))), abs(float(np.max(topo_data))))
            vmin, vmax = -data_range, data_range
        else:
            vmin, vmax = float(np.min(topo_data)), float(np.max(topo_data))

    fig, ax = plt.subplots(figsize=(8, 6))
    mask_params = dict(
        marker=PLOT_PARAMS["mask_marker"],
        markerfacecolor=PLOT_PARAMS["mask_facecolor"],
        markeredgecolor=PLOT_PARAMS["mask_edgecolor"],
        linewidth=0,
        markersize=PLOT_PARAMS["mask_markersize"],
    )

    im = mne.viz.plot_topomap(
        topo_data,
        info,
        axes=ax,
        show=False,
        contours=6,
        sensors=False,
        outlines="head",
        extrapolate="head",
        cmap=(cmap if cmap is not None else PLOT_PARAMS["cmap"]),
        sphere=0.09,
        mask=mask,
        mask_params=mask_params if mask is not None else None,
        vlim=(vmin, vmax),
    )

    cbar = plt.colorbar(im[0], ax=ax)
    cbar.set_label(colorbar_label, rotation=0, fontsize=28)
    cbar.ax.tick_params(labelsize=28)
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontfamily(PLOT_PARAMS["font_family"])
        tick_label.set_fontsize(28)

    fig.text(
        0.5,
        0.99,
        str(label_text),
        ha="center",
        va="top",
        fontsize=28,
        fontfamily=PLOT_PARAMS["font_family"],
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="svg")
    plt.close(fig)
    print(f"已保存: {save_path}")


def replot_topomaps_from_cache():
    """从统一子目录 npy_cache/median_topo_psd_unified 重绘 SVG（不写根目录 npy_cache）。"""
    npy_cache_dir = get_active_npy_cache_dir()
    info_path = os.path.join(npy_cache_dir, "topomap_info.fif")
    if not os.path.isfile(info_path):
        raise FileNotFoundError(
            f"缺少 {info_path}，请先完整运行本脚本一次以生成缓存。"
        )
    info = _read_measurement_info(info_path)
    vc_path = os.path.join(npy_cache_dir, "valid_channels.json")
    if os.path.isfile(vc_path):
        with open(vc_path, "r", encoding="utf-8") as f:
            valid_channels = json.load(f)
    else:
        valid_channels = list(info.ch_names)

    for band_name in FREQ_BANDS:
        vm_path = os.path.join(npy_cache_dir, f"vmin_vmax_band_{band_name}.npz")
        if not os.path.isfile(vm_path):
            print(f"[replot 跳过] {band_name}: 无 {vm_path}")
            continue
        vm = np.load(vm_path, allow_pickle=True)
        draw_delta = DRAW_LEFT_RIGHT_DELTA
        if "draw_left_right_delta" in vm.files:
            draw_delta = bool(np.asarray(vm["draw_left_right_delta"]).item())

        sm_path = os.path.join(npy_cache_dir, f"sig_mask_delta_{band_name}.npy")
        sig_mask_delta = np.load(sm_path) if os.path.isfile(sm_path) else None

        all_mat = None
        all_tb_mat = None
        all_base_mat = None
        all_task_mat = None
        delta_mat = None
        ga_path = os.path.join(npy_cache_dir, f"all_subjects_{band_name}.npy")
        ga_tb_path = os.path.join(npy_cache_dir, f"all_subjects_task_minus_baseline_{band_name}.npy")
        ga_base_path = os.path.join(npy_cache_dir, f"all_subjects_baseline_power_{band_name}.npy")
        ga_task_path = os.path.join(npy_cache_dir, f"all_subjects_task_power_{band_name}.npy")
        ds_path = os.path.join(npy_cache_dir, f"delta_subjects_{band_name}.npy")
        if os.path.isfile(ga_path):
            all_mat = np.load(ga_path)
        if os.path.isfile(ga_tb_path):
            all_tb_mat = np.load(ga_tb_path)
        if os.path.isfile(ga_base_path):
            all_base_mat = np.load(ga_base_path)
        if os.path.isfile(ga_task_path):
            all_task_mat = np.load(ga_task_path)
        if os.path.isfile(ds_path):
            delta_mat = np.load(ds_path)

        # 各指标独立色阶：不再跨指标共享 vmin/vmax，避免小量纲图被“压灰”

        if draw_delta:
            if delta_mat is not None and delta_mat.size > 0:
                delta_maps = {
                    "median": np.median(delta_mat, axis=0),
                    "mean": np.mean(delta_mat, axis=0),
                }
                for agg_name, group_delta in delta_maps.items():
                    delta_svg = os.path.join(
                        OUTPUT_DIR, f"{band_name.lower()}_delta_left-right_{agg_name}.svg"
                    )
                    plot_topomap_svg(
                        group_delta,
                        info,
                        label_text=f"{band_name.lower()} left-right ({agg_name})",
                        save_path=delta_svg,
                        mask=sig_mask_delta,
                        vmin=None,
                        vmax=None,
                        colorbar_label="delta percent change (left-right)",
                        symmetric_scale=True,
                    )
            cp_path = os.path.join(npy_cache_dir, f"consistency_left_gt_right_{band_name}.npy")
            if os.path.isfile(cp_path):
                consistency_pct = np.load(cp_path)
                consistency_svg = os.path.join(
                    OUTPUT_DIR, f"{band_name.lower()}_consistency_left_gt_right.svg"
                )
                plot_topomap_svg(
                    consistency_pct,
                    info,
                    label_text=f"{band_name.lower()} consistency % left>right",
                    save_path=consistency_svg,
                    mask=sig_mask_delta,
                    vmin=0.0,
                    vmax=100.0,
                    cmap="viridis",
                    colorbar_label="% left > right",
                    symmetric_scale=False,
                )

        if all_mat is not None and all_mat.size > 0:
            all_maps = {
                "median": np.median(all_mat, axis=0),
                "mean": np.mean(all_mat, axis=0),
            }
            for agg_name, group_all in all_maps.items():
                all_svg = os.path.join(OUTPUT_DIR, f"{band_name.lower()}_all_{agg_name}.svg")
                plot_topomap_svg(
                    group_all,
                    info,
                    label_text=f"{band_name.lower()} all ({agg_name})",
                    save_path=all_svg,
                    mask=None,
                    vmin=None,
                    vmax=None,
                    colorbar_label="percent change (%)",
                    symmetric_scale=True,
                )

        if all_tb_mat is not None and all_tb_mat.size > 0:
            # 与 main() 一致：先对被试维聚合功率，再取 dB 差（避免 median(log差)≠log(median)混用）
            if all_task_mat is not None and all_base_mat is not None and all_task_mat.shape == all_base_mat.shape:
                gt = group_aggregate(all_task_mat)
                gb = group_aggregate(all_base_mat)
                if gt is not None and gb is not None:
                    group_tb = _power_to_db(gt) - _power_to_db(gb)
                    tb_cbar = "task - baseline (dB)"
                else:
                    group_tb = None
            else:
                group_tb = group_aggregate(all_tb_mat)
                tb_cbar = "task - baseline"
            if group_tb is not None:
                tb_svg = os.path.join(
                    OUTPUT_DIR,
                    f"{band_name.lower()}_task-minus-baseline{OUTPUT_ALIGNED_TAG}.svg",
                )
                plot_topomap_svg(
                    group_tb,
                    info,
                    label_text=f"{band_name.lower()} task-baseline ({GROUP_AGG})",
                    save_path=tb_svg,
                    mask=None,
                    vmin=None,
                    vmax=None,
                    colorbar_label=tb_cbar,
                    symmetric_scale=True,
                )
        else:
            print(f"[replot 跳过] {band_name}: 缺少 all_subjects_task_minus_baseline_{band_name}.npy")

        if all_base_mat is not None and all_base_mat.size > 0:
            for agg_name, base_map in {
                "median": np.median(all_base_mat, axis=0),
                "mean": np.mean(all_base_mat, axis=0),
            }.items():
                base_svg = os.path.join(OUTPUT_DIR, f"{band_name.lower()}_baseline-power_{agg_name}.svg")
                plot_topomap_svg(
                    base_map,
                    info,
                    label_text=f"{band_name.lower()} baseline power ({agg_name})",
                    save_path=base_svg,
                    mask=None,
                    vmin=None,
                    vmax=None,
                    colorbar_label="baseline power",
                    symmetric_scale=True,
                )
        else:
            print(f"[replot 跳过] {band_name}: 缺少 all_subjects_baseline_power_{band_name}.npy")

        if all_task_mat is not None and all_task_mat.size > 0:
            for agg_name, task_map in {
                "median": np.median(all_task_mat, axis=0),
                "mean": np.mean(all_task_mat, axis=0),
            }.items():
                task_svg = os.path.join(OUTPUT_DIR, f"{band_name.lower()}_task-power_{agg_name}.svg")
                plot_topomap_svg(
                    task_map,
                    info,
                    label_text=f"{band_name.lower()} task power ({agg_name})",
                    save_path=task_svg,
                    mask=None,
                    vmin=None,
                    vmax=None,
                    colorbar_label="task power",
                    symmetric_scale=True,
                )
        else:
            print(f"[replot 跳过] {band_name}: 缺少 all_subjects_task_power_{band_name}.npy")

    # 重绘群体 PSD 对比图（对齐管线缓存）
    psd_base_path = os.path.join(npy_cache_dir, PSD_CURVE_CACHE_SUBJECTS_BASE)
    psd_task_path = os.path.join(npy_cache_dir, PSD_CURVE_CACHE_SUBJECTS_TASK)
    psd_freqs_path = os.path.join(npy_cache_dir, PSD_CURVE_CACHE_FREQS)
    if os.path.isfile(psd_base_path) and os.path.isfile(psd_task_path) and os.path.isfile(psd_freqs_path):
        base_mat = np.load(psd_base_path)
        task_mat = np.load(psd_task_path)
        freqs = np.load(psd_freqs_path)
        if base_mat.size > 0 and task_mat.size > 0:
            base_curve = group_aggregate_psd_freq(base_mat)
            task_curve = group_aggregate_psd_freq(task_mat)
            psd_svg = os.path.join(
                OUTPUT_DIR, f"global_psd_task_vs_baseline{OUTPUT_ALIGNED_TAG}.svg"
            )
            plot_psd_task_vs_baseline_svg(freqs, base_curve, task_curve, psd_svg)
            psd_svg_s = os.path.join(
                OUTPUT_DIR,
                f"global_psd_task_vs_baseline_smoothed_no_sem{OUTPUT_ALIGNED_TAG}.svg",
            )
            plot_psd_task_vs_baseline_smoothed_no_sem_svg(
                freqs,
                base_curve,
                task_curve,
                psd_svg_s,
                smooth_window=3,
            )
            if DRAW_GLOBAL_PSD_UNCERTAINTY:
                psd_svg_unc = os.path.join(
                    OUTPUT_DIR,
                    f"global_psd_task_vs_baseline_smoothed_uncertainty{OUTPUT_ALIGNED_TAG}.svg",
                )
                plot_psd_task_vs_baseline_with_uncertainty_svg(
                    freqs,
                    base_mat,
                    task_mat,
                    psd_svg_unc,
                    smooth_window=3,
                )
    else:
        print(
            f"[replot 跳过] 群体 PSD 对比图：缺少 {PSD_CURVE_CACHE_SUBJECTS_BASE} / "
            f"{PSD_CURVE_CACHE_SUBJECTS_TASK} / {PSD_CURVE_CACHE_FREQS}"
        )

    print("=" * 70)
    print("[replot-only] 已从统一缓存目录重绘 SVG:", npy_cache_dir)
    print(f"输出目录: {OUTPUT_DIR}")


def _has_required_replot_cache():
    """检查 median_topo_psd_unified 子目录内是否具备续画所需缓存。"""
    npy_cache_dir = get_active_npy_cache_dir()
    if not os.path.isdir(npy_cache_dir):
        return False
    info_path = os.path.join(npy_cache_dir, "topomap_info.fif")
    if not os.path.isfile(info_path):
        return False
    # 至少一个频带具备 task-baseline 新缓存即可认为可续画新图
    for band_name in FREQ_BANDS:
        tb = os.path.join(npy_cache_dir, f"all_subjects_task_minus_baseline_{band_name}.npy")
        bp = os.path.join(npy_cache_dir, f"all_subjects_baseline_power_{band_name}.npy")
        tp = os.path.join(npy_cache_dir, f"all_subjects_task_power_{band_name}.npy")
        if os.path.isfile(tb) and os.path.isfile(bp) and os.path.isfile(tp):
            return True
    return False


def main():
    # 直接运行脚本时，仅由顶部配置控制执行模式，不依赖命令行参数。
    # - REPLOT_ONLY=False（默认）: 完整重算并更新 median_topo_psd_unified 缓存
    # - REPLOT_ONLY=True: 只根据缓存重绘 SVG
    use_replot = bool(REPLOT_ONLY)
    if use_replot and AUTO_FULL_RUN_IF_CACHE_MISSING and (not _has_required_replot_cache()):
        print("[AUTO] 检测到新图所需缓存不完整，自动切换为完整重算。")
        use_replot = False
    if use_replot:
        replot_topomaps_from_cache()
        return

    entries = []
    for cfg in SUBJECTS_CONFIG:
        subject_name = cfg["subject_name"]
        entry = load_subject_entry(subject_name)
        if entry is not None:
            entries.append(entry)

    if not entries:
        raise RuntimeError("没有可用被试数据")

    common_channels = get_common_channels(entries)
    info, valid_channels = build_info_and_channel_order(entries, common_channels)

    npy_cache_dir = get_active_npy_cache_dir()
    os.makedirs(npy_cache_dir, exist_ok=True)
    print(f"[INFO] 统一缓存目录（不覆盖 npy_cache 根目录旧文件）: {npy_cache_dir}")

    info_cache_path = os.path.join(npy_cache_dir, "topomap_info.fif")
    _write_measurement_info(info_cache_path, info)
    print(f"[INFO] 已写入 topomap 用 info 缓存: {info_cache_path}")

    with open(os.path.join(npy_cache_dir, "valid_channels.json"), "w", encoding="utf-8") as f:
        json.dump(valid_channels, f, ensure_ascii=False, indent=2)
    with open(os.path.join(npy_cache_dir, "plot_params.json"), "w", encoding="utf-8") as f:
        json.dump(PLOT_PARAMS, f, ensure_ascii=False, indent=2)

    delta_subjects = {band: [] for band in FREQ_BANDS}
    all_subjects = {band: [] for band in FREQ_BANDS}
    all_tb_subjects = {band: [] for band in FREQ_BANDS}
    all_base_power_subjects = {band: [] for band in FREQ_BANDS}
    all_task_power_subjects = {band: [] for band in FREQ_BANDS}
    psd_base_subject_curves = []
    psd_task_subject_curves = []
    psd_curve_freqs = None

    print(f"[INFO] 用于计算的被试数: {len(entries)}")

    for si, entry in enumerate(entries, start=1):
        subj = entry["subject_name"]
        print(f"[{si}/{len(entries)}] 处理被试: {subj}")
        trials = entry["trials"]
        labels_df = entry["labels_df"]
        ch_labels = entry["ch_labels"]

        ch_to_idx = {ch: i for i, ch in enumerate(ch_labels)}
        channel_indices = [ch_to_idx[ch] for ch in valid_channels if ch in ch_to_idx]
        if len(channel_indices) != len(valid_channels):
            print(f"[跳过] {subj} 通道不完整")
            continue
        trials = trials[:, channel_indices, :]

        label0_idx, label1_idx = _collect_label_indices(labels_df, trials.shape[0])
        label_all_idx = sorted(set(label0_idx + label1_idx))

        if len(label_all_idx) == 0:
            continue

        # 额外缓存：群体 PSD 对比曲线（baseline vs task）
        base_curve, freqs = compute_subject_mean_psd_curve(
            trials[label_all_idx], t_window=BASE_WINDOW, fmin=PSD_CURVE_FMIN, fmax=PSD_CURVE_FMAX
        )
        task_curve, freqs_task = compute_subject_mean_psd_curve(
            trials[label_all_idx], t_window=STIM_WINDOW, fmin=PSD_CURVE_FMIN, fmax=PSD_CURVE_FMAX
        )
        if base_curve is not None and task_curve is not None:
            if psd_curve_freqs is None:
                psd_curve_freqs = np.asarray(freqs, dtype=np.float32)
                psd_base_subject_curves.append(np.asarray(base_curve, dtype=np.float32))
                # task 也统一插值到基准频率网格（首个被试 base 频率）
                task_aligned = np.interp(psd_curve_freqs, np.asarray(freqs_task, dtype=np.float32), np.asarray(task_curve, dtype=np.float32)).astype(np.float32)
                psd_task_subject_curves.append(task_aligned)
            else:
                # 不同被试窗口长度可能导致 Welch 频率点数不同；统一插值到首个被试频率网格
                freqs = np.asarray(freqs, dtype=np.float32)
                freqs_task = np.asarray(freqs_task, dtype=np.float32)
                base_curve = np.asarray(base_curve, dtype=np.float32)
                task_curve = np.asarray(task_curve, dtype=np.float32)
                base_aligned = np.interp(psd_curve_freqs, freqs, base_curve).astype(np.float32)
                task_aligned = np.interp(psd_curve_freqs, freqs_task, task_curve).astype(np.float32)
                psd_base_subject_curves.append(base_aligned)
                psd_task_subject_curves.append(task_aligned)

        for band_name, (fmin, fmax) in FREQ_BANDS.items():
            print(f"    - band={band_name} ({fmin}-{fmax}Hz) ...")
            # all
            pct_all = aggregate_subject_median_percent(trials[label_all_idx], fmin=fmin, fmax=fmax)
            if pct_all is not None:
                all_subjects[band_name].append(pct_all)
            tb_all = aggregate_subject_median_task_minus_baseline(
                trials[label_all_idx], fmin=fmin, fmax=fmax
            )
            if tb_all is not None:
                all_tb_subjects[band_name].append(tb_all)
            base_all, task_all = aggregate_subject_median_base_and_task_power(
                trials[label_all_idx], fmin=fmin, fmax=fmax
            )
            if base_all is not None and task_all is not None:
                all_base_power_subjects[band_name].append(base_all)
                all_task_power_subjects[band_name].append(task_all)

            # delta
            if len(label0_idx) > 0 and len(label1_idx) > 0:
                pct_left = aggregate_subject_median_percent(trials[label0_idx], fmin=fmin, fmax=fmax)
                pct_right = aggregate_subject_median_percent(trials[label1_idx], fmin=fmin, fmax=fmax)
                if pct_left is not None and pct_right is not None:
                    delta_subjects[band_name].append(pct_left - pct_right)

    for band_name in FREQ_BANDS:
        has_delta = len(delta_subjects[band_name]) >= 3
        has_all = len(all_subjects[band_name]) > 0
        has_all_tb = len(all_tb_subjects[band_name]) > 0
        has_base_power = len(all_base_power_subjects[band_name]) > 0
        has_task_power = len(all_task_power_subjects[band_name]) > 0

        if DRAW_LEFT_RIGHT_DELTA:
            if not has_delta:
                print(f"[跳过输出] {band_name} delta 被试太少：{len(delta_subjects[band_name])}")
                continue
        else:
            if not has_all:
                print(f"[跳过输出] {band_name} 无 all 数据")
                continue

        group_delta = None
        delta_mat = None
        sig_mask = None
        crit_p = ALPHA
        sig_channels = []
        consistency_pct = None

        if DRAW_LEFT_RIGHT_DELTA and has_delta:
            delta_mat = np.vstack(delta_subjects[band_name])  # (n_subj, n_ch)
            group_delta = group_aggregate(delta_mat)

            pvals = wilcoxon_channel_pvals(delta_mat)
            if USE_FDR:
                sig_mask, crit_p = fdr_bh(pvals, alpha=ALPHA)
            else:
                sig_mask = pvals <= ALPHA
                crit_p = ALPHA

            sig_channels = [valid_channels[i] for i in range(len(valid_channels)) if sig_mask[i]]

            np.save(os.path.join(npy_cache_dir, f"delta_subjects_{band_name}.npy"), delta_mat)
            np.save(os.path.join(npy_cache_dir, f"group_delta_{band_name}.npy"), group_delta)
            np.save(os.path.join(npy_cache_dir, f"pvals_delta_{band_name}.npy"), pvals)
            np.save(os.path.join(npy_cache_dir, f"sig_mask_delta_{band_name}.npy"), sig_mask)
            consistency_pct = np.mean(delta_mat > 0, axis=0) * 100.0
            np.save(os.path.join(npy_cache_dir, f"consistency_left_gt_right_{band_name}.npy"), consistency_pct)

        group_all = None
        group_all_tb = None
        group_base_power = None
        group_task_power = None
        if has_all:
            all_mat = np.vstack(all_subjects[band_name])
            group_all = group_aggregate(all_mat)
            np.save(os.path.join(npy_cache_dir, f"all_subjects_{band_name}.npy"), all_mat)
            np.save(os.path.join(npy_cache_dir, f"group_all_{band_name}.npy"), group_all)
        if has_all_tb:
            all_tb_mat = np.vstack(all_tb_subjects[band_name])
            group_all_tb = group_aggregate(all_tb_mat)
            np.save(
                os.path.join(npy_cache_dir, f"all_subjects_task_minus_baseline_{band_name}.npy"),
                all_tb_mat,
            )
            np.save(
                os.path.join(npy_cache_dir, f"group_task_minus_baseline_{band_name}.npy"),
                group_all_tb,
            )
        if has_base_power:
            all_base_mat = np.vstack(all_base_power_subjects[band_name])
            group_base_power = group_aggregate(all_base_mat)
            np.save(
                os.path.join(npy_cache_dir, f"all_subjects_baseline_power_{band_name}.npy"),
                all_base_mat,
            )
            np.save(
                os.path.join(npy_cache_dir, f"group_baseline_power_{band_name}.npy"),
                group_base_power,
            )
        if has_task_power:
            all_task_mat = np.vstack(all_task_power_subjects[band_name])
            group_task_power = group_aggregate(all_task_mat)
            np.save(
                os.path.join(npy_cache_dir, f"all_subjects_task_power_{band_name}.npy"),
                all_task_mat,
            )
            np.save(
                os.path.join(npy_cache_dir, f"group_task_power_{band_name}.npy"),
                group_task_power,
            )

        combined_parts = []
        if group_all is not None:
            combined_parts.append(group_all)
        if len(combined_parts) >= 1:
            combined = np.concatenate(combined_parts)
        elif group_all_tb is not None:
            combined = np.asarray(group_all_tb)
        elif group_base_power is not None:
            combined = np.asarray(group_base_power)
        elif group_task_power is not None:
            combined = np.asarray(group_task_power)
        elif group_delta is not None:
            combined = np.asarray(group_delta)
        else:
            print(f"[跳过输出] {band_name} 无可用聚合数据")
            continue

        data_range = max(abs(float(np.min(combined))), abs(float(np.max(combined))))
        vmin, vmax = -data_range, data_range
        np.savez(
            os.path.join(npy_cache_dir, f"vmin_vmax_band_{band_name}.npz"),
            vmin=vmin,
            vmax=vmax,
            alpha=ALPHA,
            use_fdr=USE_FDR,
            group_agg=GROUP_AGG,
            baseline_window=BASE_WINDOW,
            stim_window=STIM_WINDOW,
            draw_left_right_delta=DRAW_LEFT_RIGHT_DELTA,
        )

        if DRAW_LEFT_RIGHT_DELTA and consistency_pct is not None:
            with open(os.path.join(npy_cache_dir, f"sig_channels_delta_{band_name}.txt"), "w", encoding="utf-8") as f:
                f.write(f"band={band_name}, USE_FDR={USE_FDR}, alpha={ALPHA}, crit_p={crit_p}\n")
                f.write("significant_channels:\n")
                for ch in sig_channels:
                    f.write(f"- {ch}\n")

            with open(os.path.join(npy_cache_dir, f"consistency_left_gt_right_{band_name}.txt"), "w", encoding="utf-8") as f:
                f.write(f"band={band_name}\n")
                f.write("channel\tpercent_left_gt_right\n")
                for ch_name, pct in zip(valid_channels, consistency_pct):
                    f.write(f"{ch_name}\t{pct:.2f}\n")

            delta_svg = os.path.join(OUTPUT_DIR, f"{band_name.lower()}_delta_left-right.svg")
            plot_topomap_svg(
                group_delta,
                info,
                label_text=f"{band_name.lower()} left-right",
                save_path=delta_svg,
                mask=sig_mask,
                vmin=None,
                vmax=None,
                colorbar_label="delta percent change (left-right)",
                symmetric_scale=True,
            )

            consistency_svg = os.path.join(OUTPUT_DIR, f"{band_name.lower()}_consistency_left_gt_right.svg")
            plot_topomap_svg(
                consistency_pct,
                info,
                label_text=f"{band_name.lower()} consistency % left>right",
                save_path=consistency_svg,
                mask=sig_mask,
                vmin=0.0,
                vmax=100.0,
                cmap="viridis",
                colorbar_label="% left > right",
                symmetric_scale=False,
            )

        if group_all is not None:
            all_svg = os.path.join(OUTPUT_DIR, f"{band_name.lower()}_all.svg")
            plot_topomap_svg(
                group_all,
                info,
                label_text=f"{band_name.lower()} all",
                save_path=all_svg,
                mask=None,
                vmin=None,
                vmax=None,
                colorbar_label="percent change (%)",
                symmetric_scale=True,
            )
        if group_all_tb is not None:
            tb_svg = os.path.join(
                OUTPUT_DIR,
                f"{band_name.lower()}_task-minus-baseline{OUTPUT_ALIGNED_TAG}.svg",
            )
            if group_task_power is not None and group_base_power is not None:
                tb_map = _power_to_db(group_task_power) - _power_to_db(group_base_power)
                tb_cbar = "task - baseline (dB)"
            else:
                tb_map = group_all_tb
                tb_cbar = "task - baseline"
            plot_topomap_svg(
                tb_map,
                info,
                label_text=f"{band_name.lower()} task-baseline ({GROUP_AGG})",
                save_path=tb_svg,
                mask=None,
                vmin=None,
                vmax=None,
                colorbar_label=tb_cbar,
                symmetric_scale=True,
            )
        if group_base_power is not None:
            base_svg = os.path.join(OUTPUT_DIR, f"{band_name.lower()}_baseline-power.svg")
            plot_topomap_svg(
                group_base_power,
                info,
                label_text=f"{band_name.lower()} baseline power",
                save_path=base_svg,
                mask=None,
                vmin=None,
                vmax=None,
                colorbar_label="baseline power",
                symmetric_scale=True,
            )
        if group_task_power is not None:
            task_svg = os.path.join(OUTPUT_DIR, f"{band_name.lower()}_task-power.svg")
            plot_topomap_svg(
                group_task_power,
                info,
                label_text=f"{band_name.lower()} task power",
                save_path=task_svg,
                mask=None,
                vmin=None,
                vmax=None,
                colorbar_label="task power",
                symmetric_scale=True,
            )

    print("=" * 70)
    if len(psd_base_subject_curves) > 0 and len(psd_task_subject_curves) > 0 and psd_curve_freqs is not None:
        psd_base_mat = np.vstack(psd_base_subject_curves)
        psd_task_mat = np.vstack(psd_task_subject_curves)
        np.save(os.path.join(npy_cache_dir, PSD_CURVE_CACHE_SUBJECTS_BASE), psd_base_mat)
        np.save(os.path.join(npy_cache_dir, PSD_CURVE_CACHE_SUBJECTS_TASK), psd_task_mat)
        np.save(
            os.path.join(npy_cache_dir, PSD_CURVE_CACHE_FREQS),
            np.asarray(psd_curve_freqs, dtype=np.float32),
        )

        base_curve = group_aggregate_psd_freq(psd_base_mat)
        task_curve = group_aggregate_psd_freq(psd_task_mat)
        psd_svg = os.path.join(
            OUTPUT_DIR, f"global_psd_task_vs_baseline{OUTPUT_ALIGNED_TAG}.svg"
        )
        plot_psd_task_vs_baseline_svg(psd_curve_freqs, base_curve, task_curve, psd_svg)
        psd_svg_s = os.path.join(
            OUTPUT_DIR,
            f"global_psd_task_vs_baseline_smoothed_no_sem{OUTPUT_ALIGNED_TAG}.svg",
        )
        plot_psd_task_vs_baseline_smoothed_no_sem_svg(
            psd_curve_freqs,
            base_curve,
            task_curve,
            psd_svg_s,
            smooth_window=3,
        )
        if DRAW_GLOBAL_PSD_UNCERTAINTY:
            psd_svg_unc = os.path.join(
                OUTPUT_DIR,
                f"global_psd_task_vs_baseline_smoothed_uncertainty{OUTPUT_ALIGNED_TAG}.svg",
            )
            plot_psd_task_vs_baseline_with_uncertainty_svg(
                psd_curve_freqs,
                psd_base_mat,
                psd_task_mat,
                psd_svg_unc,
                smooth_window=3,
            )

    if DRAW_LEFT_RIGHT_DELTA:
        print("完成：delta(topomap+显著点) + consistency + all(topomap)")
    else:
        print("完成：仅 all(topomap)（left-right delta 已关闭）")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

