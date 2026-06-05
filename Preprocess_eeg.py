import argparse
import os
import numpy as np
import re
import mne
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from extract_channel_info import make_montage
from collections import defaultdict
import mffpy
import logging
import json
import pandas as pd
from mne.preprocessing import ICA
from pathlib import Path

try:
    from mne_icalabel import label_components
    IC_LABEL_AVAILABLE = True
except ImportError:
    IC_LABEL_AVAILABLE = False
    label_components = None
    print("错误: 未安装 mne-icalabel，无法使用 ICLabel。请先: pip install mne-icalabel")

try:
    from pyprep import PrepPipeline
    PYPREP_AVAILABLE = True
except ImportError:
    print("警告: 未安装 PyPREP，将使用手动指定的坏电极列表")
    PYPREP_AVAILABLE = False

from subject_mff_benchmark_config import SUBJECTS_CONFIG

# 配置参数
BASE_DATA_DIRS = [r"A:\\", r"A:/standard_data_noica"]
SAMPLING_RATE = 250

# 如果 PyPREP 不可用，使用手动指定的坏电极列表作为备用
MANUAL_BAD_ELECTRODES = ['E48', 'E119', 'E49', 'E113', 'E44', 'E114',
                         'E131', 'E132', 'E56', 'E107', 'E126', 'E127',
                         'E43', 'E120', 'E45', 'E108', 'E57', 'E100', 'E99', 'E63']

# 插值 + 滤波 + ICA + ICLabel 后的输出根目录
OUTPUT_ROOT_INTERP_ICA = r"A:/standard_data_interp_ica_99"

# ICA：0 < n < 1 时为保留方差比例（与 MNE 一致）；ICLabel 伪迹类且 proba>阈值则剔除
ICA_N_COMPONENTS = 0.99
ICLABEL_PROBA_THRESHOLD = 0.7

# main() 无参数直接运行（IDE 点运行）时默认只处理该被试；需全员时用命令行 --all-subjects
DEFAULT_RUN_SUBJECT_NAME = "liuzehao"


def _is_iclabel_artifact(label_name: str) -> bool:
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


def _find_existing_mff_path(mff_rel_path):
    """在 BASE_DATA_DIRS 中查找存在的 mff 目录。"""
    for base_dir in BASE_DATA_DIRS:
        candidate_path = os.path.join(base_dir, mff_rel_path)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def _save_figure_or_figlist(fig_obj, output_file_stem):
    """兼容 MNE 返回 Figure 或 Figure 列表的保存逻辑。"""
    if fig_obj is None:
        return
    if isinstance(fig_obj, list):
        for idx, fig in enumerate(fig_obj):
            fig.savefig(f"{output_file_stem}_{idx + 1:02d}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
    else:
        fig_obj.savefig(f"{output_file_stem}.png", dpi=200, bbox_inches="tight")
        plt.close(fig_obj)


def plot_subject_ica_components(
    subject_name,
    file_index=1,
    max_components=8,
    proba_threshold=ICLABEL_PROBA_THRESHOLD,
):
    """仅针对单个被试可视化 ICA 将剔除的若干分量。"""
    if not (IC_LABEL_AVAILABLE and label_components is not None):
        print("错误: 需要 mne-icalabel 才能可视化 ICLabel 分量。请先: pip install mne-icalabel")
        return

    subject_cfg = next((cfg for cfg in SUBJECTS_CONFIG if cfg.get("subject_name") == subject_name), None)
    if subject_cfg is None:
        print(f"错误: 在 SUBJECTS_CONFIG 中找不到被试 {subject_name}")
        return

    mff_files = subject_cfg.get("mff_files", [])
    if not mff_files:
        print(f"错误: 被试 {subject_name} 没有配置 mff_files")
        return

    if file_index < 1 or file_index > len(mff_files):
        print(f"错误: file_index={file_index} 超出范围，当前被试共有 {len(mff_files)} 个文件")
        return

    mff_rel_path = mff_files[file_index - 1]["mff"]
    mff_dir = _find_existing_mff_path(mff_rel_path)
    if mff_dir is None:
        print(f"错误: 未找到 MFF 文件目录 {mff_rel_path}，搜索目录: {BASE_DATA_DIRS}")
        return

    print(f"开始 ICA 可视化: subject={subject_name}, file_index={file_index}, mff={mff_rel_path}")
    raw = mne.io.read_raw_egi(mff_dir, preload=True)

    # 保持与主流程一致：只保留前 128 个 EEG 通道
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    if len(eeg_picks) > 128:
        eeg_picks = eeg_picks[:128]
    raw.pick(eeg_picks)

    # 与主流程一致：E### -> EEG###
    channel_rename_dict = {}
    for ch_name in raw.ch_names:
        if ch_name.startswith("E") and ch_name[1:].isdigit():
            channel_rename_dict[ch_name] = f"EEG{int(ch_name[1:]):03d}"
    if channel_rename_dict:
        raw.rename_channels(channel_rename_dict)

    # montage
    coordinates_path = os.path.join(mff_dir, "coordinates.xml")
    if os.path.exists(coordinates_path):
        out_subject_dir = os.path.join(OUTPUT_ROOT_INTERP_ICA, subject_name)
        os.makedirs(out_subject_dir, exist_ok=True)
        montage_path = os.path.join(out_subject_dir, "electrode_montage.fif")
        make_montage(coordinates_path, montage_path)
        montage = mne.channels.read_dig_fif(montage_path)
        raw.set_montage(montage, on_missing="warn")
        print("已设置电极 montage")
    else:
        print("警告: 未找到 coordinates.xml，继续但可能影响 topomap 显示")

    # 预处理与主流程保持一致
    raw.filter(l_freq=0.1, h_freq=45.0, method="fir", fir_window="hamming")
    raw.notch_filter(freqs=[50])
    raw.set_eeg_reference("average")

    ica = ICA(
        n_components=ICA_N_COMPONENTS,
        random_state=97,
        method="infomax",
        max_iter=500,
    )
    ica.fit(raw)

    ic_labels = label_components(raw, ica, method="iclabel")
    comp_labels = ic_labels["labels"]
    comp_proba = ic_labels["y_pred_proba"]

    exclude_idx = [
        i
        for i, (lab, p) in enumerate(zip(comp_labels, comp_proba))
        if _is_iclabel_artifact(lab) and float(p) > float(proba_threshold)
    ]

    if exclude_idx:
        print(f"ICLabel 判定可剔除分量数: {len(exclude_idx)}")
    else:
        print("ICLabel 未找到满足阈值的可剔除伪迹分量，将展示高置信度伪迹分量")

    if exclude_idx:
        plot_picks = exclude_idx[:max_components]
    else:
        artifact_idx = [i for i, lab in enumerate(comp_labels) if _is_iclabel_artifact(lab)]
        artifact_idx_sorted = sorted(artifact_idx, key=lambda i: float(comp_proba[i]), reverse=True)
        plot_picks = artifact_idx_sorted[:max_components]

    if not plot_picks:
        print("没有可绘制的伪迹分量，结束。")
        return

    plot_dir = Path(OUTPUT_ROOT_INTERP_ICA) / subject_name / "ica_component_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 保存文本摘要
    summary_path = plot_dir / f"{subject_name}_file{file_index}_ica_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"subject_name: {subject_name}\n")
        f.write(f"mff: {mff_rel_path}\n")
        f.write(f"proba_threshold: {proba_threshold}\n")
        f.write(f"exclude_idx: {exclude_idx}\n")
        f.write(f"plot_picks: {plot_picks}\n")
        f.write("component_details:\n")
        for i, (lab, p) in enumerate(zip(comp_labels, comp_proba)):
            f.write(f"  - ic={i}, label={lab}, proba={float(p):.4f}\n")

    print(f"将绘制分量: {plot_picks}")
    _save_figure_or_figlist(
        ica.plot_components(picks=plot_picks, show=False),
        str(plot_dir / f"{subject_name}_file{file_index}_ica_components"),
    )
    _save_figure_or_figlist(
        ica.plot_properties(raw, picks=plot_picks, show=False),
        str(plot_dir / f"{subject_name}_file{file_index}_ica_properties"),
    )
    _save_figure_or_figlist(
        ica.plot_sources(raw, show=False),
        str(plot_dir / f"{subject_name}_file{file_index}_ica_sources"),
    )

    print(f"ICA 可视化已保存到: {plot_dir}")
    print(f"摘要文件: {summary_path}")


def _resolve_bad_channel_names(bad_list, ch_names):
    """坏道名为 E###（或已与 ch_names 一致）；匹配重命名后的 EEG### 或原始 E###。"""
    out = []
    for name in bad_list:
        if name in ch_names:
            out.append(name)
            continue
        if name.startswith('E') and name[1:].isdigit():
            eeg = f'EEG{int(name[1:]):03d}'
            if eeg in ch_names:
                out.append(eeg)
    return out


def extract_labels(benchmark_file_path):
    """从benchmark文件提取trial的标签"""
    if not os.path.exists(benchmark_file_path):
        print(f"错误: benchmark文件不存在: {benchmark_file_path}")
        return {}

    try:
        # 尝试不同的编码方式读取文件
        encodings = ['utf-16-le', 'utf-16', 'utf-16-be', 'utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
        file_content = None

        for encoding in encodings:
            try:
                with open(benchmark_file_path, 'r', encoding=encoding, errors='strict') as f:
                    candidate = f.read()
                # 校验是否为合理内容（包含关键字段）
                if re.search(r"List1:|RESP|ImageDisplay", candidate):
                    file_content = candidate
                    print(f"成功使用 {encoding} 编码读取benchmark文件")
                    break
            except UnicodeDecodeError:
                continue

        if file_content is None:
            # 退化方案：忽略错误读取
            with open(benchmark_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            print("以 utf-8(ignore) 方式读取benchmark文件（可能存在部分字符丢失）")

        # 归一化特殊换行符并按行分割
        file_content = file_content.replace('\u2028', '\n').replace('\u2029', '\n')
        lines = re.split(r"\r\n|\r|\n", file_content)
        print(f"benchmark文件总共有 {len(lines)} 行")

        # 查找trial和标签
        labels = {}
        current_trial = None

        # 匹配格式: List1: 1 和 ImageDisplay1.RESP: 1/2
        list_pattern = r'List1:\s*(\d+)'
        resp_pattern = r'ImageDisplay1\.RESP:\s*([12])'

        for i, line in enumerate(lines):
            line = line.strip()

            # 查找List1: trial编号
            list_match = re.match(list_pattern, line)
            if list_match:
                current_trial = int(list_match.group(1))
                continue

            # 查找ImageDisplay1.RESP: 标签
            resp_match = re.match(resp_pattern, line)
            if resp_match and current_trial is not None:
                resp_value = int(resp_match.group(1))
                # 映射: 1->0, 2->1
                label = 0 if resp_value == 1 else 1
                labels[current_trial] = label
                current_trial = None  # 重置，避免重复匹配

        print(f"总共提取到 {len(labels)} 个trial的标签")
        return labels

    except Exception as e:
        print(f"读取benchmark文件错误: {e}")
        return {}




def extract_labels_and_rts(benchmark_file_path, verbose=True):
    """从benchmark文件同时提取trial标签与反应时间(RT, ms)"""
    if not os.path.exists(benchmark_file_path):
        if verbose:
            print(f"错误: benchmark文件不存在: {benchmark_file_path}")
        return {}, {}

    try:
        # 尝试不同的编码方式读取文件
        encodings = ['utf-16-le', 'utf-16', 'utf-16-be', 'utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
        file_content = None

        for encoding in encodings:
            try:
                with open(benchmark_file_path, 'r', encoding=encoding, errors='strict') as f:
                    candidate = f.read()
                # 校验是否为合理内容（包含关键字段）
                if re.search(r"List1:|RESP|RT:|ImageDisplay", candidate):
                    file_content = candidate
                    if verbose:
                        print(f"成功使用 {encoding} 编码读取benchmark文件")
                    break
            except UnicodeDecodeError:
                continue

        if file_content is None:
            # 退化方案：忽略错误读取
            with open(benchmark_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            if verbose:
                print("以 utf-8(ignore) 方式读取benchmark文件（可能存在部分字符丢失）")

        # 归一化特殊换行符并按行分割
        file_content = file_content.replace('\u2028', '\n').replace('\u2029', '\n')
        lines = re.split(r"\r\n|\r|\n", file_content)
        if verbose:
            print(f"benchmark文件总共有 {len(lines)} 行")

        labels = {}
        rts = {}
        current_trial = None

        list_pattern = r'List1:\s*(\d+)'
        resp_pattern = r'ImageDisplay\d+\.RESP:\s*([12])'
        rt_pattern = r'ImageDisplay\d+\.RT:\s*(\d+)'

        for line in lines:
            line = line.strip()

            list_match = re.match(list_pattern, line)
            if list_match:
                current_trial = int(list_match.group(1))
                continue

            if current_trial is None:
                continue

            resp_match = re.match(resp_pattern, line)
            if resp_match:
                resp_value = int(resp_match.group(1))
                label = 0 if resp_value == 1 else 1
                if current_trial not in labels:
                    labels[current_trial] = label
                continue

            rt_match = re.match(rt_pattern, line)
            if rt_match:
                rt_ms = int(rt_match.group(1))
                if current_trial not in rts:
                    rts[current_trial] = rt_ms
                continue

        if verbose:
            print(f"总共提取到 {len(labels)} 个trial标签，{len(rts)} 个trial RT")
        return labels, rts

    except Exception as e:
        if verbose:
            print(f"读取benchmark文件错误: {e}")
        return {}, {}


def find_trial_times(log_file_path):
    """从log文件找到trial的开始时间，匹配PLTECIEvents格式"""
    if not os.path.exists(log_file_path):
        print(f"错误: log文件不存在: {log_file_path}")
        return []

    try:
        # 优先采用二进制级别匹配
        try:
            with open(log_file_path, 'rb') as fb:
                raw_bytes = fb.read()
            # 仅匹配包含 PLTECIEvents 和 TBEG 的事件行
            bexpr = re.compile(rb"(\d{2}:\d{2}:\d{2}\.\d{3})\.\d{3}:[^\r\n]*?PLTECIEvents[^\r\n]*?TBEG", re.IGNORECASE)
            trial_times = []
            for m in bexpr.finditer(raw_bytes):
                time_str = m.group(1).decode('ascii', errors='ignore')
                time_parts = time_str.split(':')
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds_parts = time_parts[2].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1])
                trial_start_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
                snippet = m.group(0)[:200]
                try:
                    snippet_text = snippet.decode('utf-8', errors='ignore')
                except Exception:
                    snippet_text = repr(snippet)
                trial_times.append({
                    'trial_num': len(trial_times) + 1,
                    'time': time_str,
                    'start_seconds': trial_start_seconds,
                    'line': snippet_text,
                    'line_number': None,
                })
            if len(trial_times) > 0:
                print(f"二进制扫描找到 {len(trial_times)} 个TBEG 事件")
                # 排序并编号
                trial_times.sort(key=lambda x: x['start_seconds'])
                for i, t in enumerate(trial_times):
                    t['trial_num'] = i + 1
                return trial_times
        except Exception as e:
            print(f"二进制扫描失败，改用文本解析: {e}")

        # 尝试不同的编码方式读取文件
        encodings = ['utf-16-le', 'utf-16', 'utf-16-be', 'utf-8-sig', 'utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
        file_content = None

        for encoding in encodings:
            try:
                with open(log_file_path, 'r', encoding=encoding, errors='strict') as f:
                    candidate = f.read()
                if re.search(r"PLTECI|TBEG|Event:", candidate, re.IGNORECASE):
                    file_content = candidate
                    print(f"成功使用 {encoding} 编码读取log文件")
                    break
            except UnicodeDecodeError:
                continue

        if file_content is None:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            print("以 utf-8(ignore) 方式读取log文件（可能存在部分字符丢失）")

        # 归一化特殊换行符并按行分割
        file_content = file_content.replace('\u2028', '\n').replace('\u2029', '\n')
        lines = re.split(r"\r\n|\r|\n", file_content)
        print(f"log文件总共有 {len(lines)} 行")

        # 查找trial事件
        trial_times = []
        pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3})\.\d{3}:\s*[^\r\n]*PLTECIEvents[^\r\n]*TBEG'

        for i, line in enumerate(lines):
            raw_line = line
            line = line.strip()
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                time_str = match.group(1)
                time_parts = time_str.split(':')
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds_parts = time_parts[2].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1])
                trial_start_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0

                trial_times.append({
                    'trial_num': len(trial_times) + 1,
                    'time': time_str,
                    'start_seconds': trial_start_seconds,
                    'line': raw_line,
                    'line_number': i + 1
                })

        # 如果按行解析未找到，退化为全文正则搜索
        if len(trial_times) == 0:
            print("按行解析未找到TBEG，尝试全文扫描...")
            trial_times = []
            iter_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}\.\d{3})\.\d{3}:\s*[^\n\r]*PLTECIEvents[^\n\r]*TBEG", re.IGNORECASE)
            for m in iter_pattern.finditer(file_content):
                time_str = m.group(1)
                time_parts = time_str.split(':')
                hours = int(time_parts[0])
                minutes = int(time_parts[1])
                seconds_parts = time_parts[2].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1])
                trial_start_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
                trial_times.append({
                    'trial_num': len(trial_times) + 1,
                    'time': time_str,
                    'start_seconds': trial_start_seconds,
                    'line': m.group(0)[:200],
                    'line_number': None
                })
            print(f"全文扫描找到 {len(trial_times)} 个TBEG 事件")

        print(f"总共找到 {len(trial_times)} 个trial")

        # 按时间顺序排序
        trial_times.sort(key=lambda x: x['start_seconds'])

        # 重新编号
        for i, trial in enumerate(trial_times):
            trial['trial_num'] = i + 1

        return trial_times

    except Exception as e:
        print(f"读取log文件错误: {e}")
        return []




def process_subject(
    subject_config,
    output_root="A:/standard_data_noica",
    apply_ica=True,
    update_ic_reject_only=False,
):
    """处理单个受试者的多个MFF文件数据。
    update_ic_reject_only=True：仅重算 ICA 剔除数并写回已有 trial_info.json，不保存 trials/labels/rts。
    """
    subject_name = subject_config["subject_name"]
    mff_files = subject_config["mff_files"]
    trial_duration = subject_config["trial_duration"]

    print(f"\n{'='*60}")
    print(f"开始处理受试者: {subject_name}")
    print(f"{'='*60}\n")
    if apply_ica:
        print(
            f"ICA + ICLabel：已启用（ICA n_components={ICA_N_COMPONENTS}；"
            f"ICLabel 伪迹类且 proba>{ICLABEL_PROBA_THRESHOLD} 则剔除）"
        )
    else:
        print("ICA：已关闭")
    print(f"输出目录: {output_root}")
    print(f"MFF文件配置: {len(mff_files)} 个文件")

    # 创建输出目录
    processed_data_path = os.path.join(output_root, subject_name)
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path, exist_ok=True)
    print(f"输出目录: {processed_data_path}")

    # 收集所有文件的trial时间和标签
    all_trial_times = []
    all_labels = []
    all_rts = []
    start_datetimes = []
    raw_objects = []
    log_files = []
    selected_trials_list = []
    file_info = []

    # 逐个处理每个MFF文件
    for i, mff_config in enumerate(mff_files):
        print(f"\n{'-'*40}")
        print(f"处理文件 {i+1}/{len(mff_files)}: {mff_config['mff']}")
        print(f"{'-'*40}")
        print(f"目标: 提取 {mff_config['max_trials']} 个trials, {'倒序' if mff_config['reverse_order'] else '正序'}")

        # 在多个目录中查找MFF文件
        mff_dir = None
        for base_dir in BASE_DATA_DIRS:
            candidate_path = os.path.join(base_dir, mff_config["mff"])
            if os.path.exists(candidate_path):
                mff_dir = candidate_path
                print(f"在 {base_dir} 中找到MFF文件")
                break

        if mff_dir is None:
            print(f"错误: 在任何配置的目录中都找不到MFF文件 {mff_config['mff']}")
            print(f"搜索的目录: {BASE_DATA_DIRS}")
            continue

        benchmark_file = os.path.join(mff_dir, mff_config["benchmark"])
        print(f"MFF目录: {mff_dir}")
        print(f"Benchmark文件: {benchmark_file}")
        print(f"Benchmark文件存在: {os.path.exists(benchmark_file)}")

        # 查找log文件
        log_file = None
        mff_name = os.path.basename(mff_dir).replace('.mff', '')

        possible_log_names = [
            f"log_{mff_name}.txt",
            f"{mff_name}.txt",
            "log.txt"
        ]

        for log_name in possible_log_names:
            candidate = os.path.join(mff_dir, log_name)
            if os.path.exists(candidate):
                log_file = candidate
                break

        if log_file is None:
            print(f"警告: 未找到log文件，尝试在目录中查找...")
            if os.path.exists(mff_dir):
                txt_files = [f for f in os.listdir(mff_dir) if f.lower().endswith('.txt')]
                print(f"目录中的TXT文件: {txt_files}")
                for f in os.listdir(mff_dir):
                    if f.lower().endswith('.txt') and 'log' in f.lower():
                        log_file = os.path.join(mff_dir, f)
                        print(f"找到log文件: {log_file}")
                        break

        # 检查文件是否存在
        if not os.path.exists(mff_dir):
            print(f"错误: MFF目录不存在，跳过文件 {mff_config['mff']}")
            continue

        # 读取MFF文件
        print(f"正在读取MFF文件...")
        raw = mne.io.read_raw_egi(mff_dir, preload=True)
        fo = mffpy.Reader(mff_dir)
        start_datetime = fo.startdatetime
        start_datetimes.append(start_datetime)

        print(f"文件数据形状: {raw.get_data().shape}")
        print(f"通道数量: {len(raw.ch_names)}")
        print(f"采样率: {raw.info['sfreq']} Hz")
        print(f"开始时间: {start_datetime}")

        raw_objects.append(raw)
        log_files.append(log_file)

        # 查找trial时间
        print(f"正在查找trial时间...")
        trial_times = find_trial_times(log_file) if log_file else []
        print(f"找到 {len(trial_times)} 个trials")

        # 根据配置选择trials
        max_trials = mff_config['max_trials']
        reverse_order = mff_config['reverse_order']

        if len(trial_times) >= max_trials:
            if reverse_order:
                # 倒序选择最后max_trials个
                selected_trials = trial_times[-max_trials:]
                print(f"选择最后 {max_trials} 个trials")
            else:
                # 正序选择前max_trials个
                selected_trials = trial_times[:max_trials]
                print(f"选择前 {max_trials} 个trials")
        else:
            selected_trials = trial_times
            print(f"警告: trials数量不足({len(trial_times)} < {max_trials})，使用全部trials")

        # 标记来源文件
        for trial in selected_trials:
            trial['source_file'] = i + 1

        selected_trials_list.append(selected_trials)
        all_trial_times.append(trial_times)  # 保存所有trials，用于标签对齐

        # 提取标签与RT
        print(f"正在提取标签与RT...")
        labels, rts = extract_labels_and_rts(benchmark_file)
        print(f"提取到 {len(labels)} 个标签，{len(rts)} 个RT")
        all_labels.append(labels)
        all_rts.append(rts)

        # 保存文件信息
        file_info.append({
            'file_index': i + 1,
            'mff_file': mff_config["mff"],
            'benchmark_file': mff_config["benchmark"],
            'log_file': log_file,
            'start_datetime': start_datetime,
            'total_trials': len(trial_times),
            'selected_trials': len(selected_trials),
            'max_trials': max_trials,
            'reverse_order': reverse_order,
            'num_labels': len(labels),
            'num_rts': len(rts),
            'raw_shape': raw.get_data().shape
        })

        print(f"文件 {i+1} 处理完成！")

    if not raw_objects:
        print(f"错误: 没有成功读取任何MFF文件")
        return

    print(f"\n{'='*40}")
    print("所有文件处理完成，开始合并数据")
    print(f"{'='*40}")

    # 打印汇总信息
    print("\n文件处理汇总:")
    total_selected = 0
    for info in file_info:
        print(f"  文件{info['file_index']}: {info['selected_trials']}/{info['total_trials']} trials 选择")
        total_selected += info['selected_trials']
    print(f"总共选择: {total_selected} 个trials")

    # 使用第一个成功读取的文件的montage
    coordinates_path = None
    for base_dir in BASE_DATA_DIRS:
        candidate_path = os.path.join(base_dir, mff_files[0]["mff"], "coordinates.xml")
        if os.path.exists(candidate_path):
            coordinates_path = candidate_path
            print(f"在 {base_dir} 中找到坐标文件")
            break
    montage_path = os.path.join(processed_data_path, "electrode_montage.fif")

    if os.path.exists(coordinates_path):
        make_montage(coordinates_path, montage_path)
        montage = mne.channels.read_dig_fif(montage_path)
        print(f"电极数量: {len(montage.dig)}")
    else:
        print("警告: 坐标文件不存在，将使用默认montage")
        montage = None

    # 预处理函数：坏道标记 → 球面插值（需 montage）→ 滤波/陷波/重参考 → ICA + ICLabel（可选）
    def preprocess_raw(raw_data, montage=None, apply_ica=True, bad_channels=None):
        """对单个raw对象进行预处理。bad_channels 为检测到的坏道（E###），将插值而非删除。"""
        channels_before = len(raw_data.ch_names)
        rejected_ic_count = 0

        # 仅保留前128个EEG通道
        eeg_picks = mne.pick_types(raw_data.info, eeg=True)
        if len(eeg_picks) == 0:
            print("警告: 未检测到EEG通道，保持原通道不变")
        else:
            if len(eeg_picks) > 128:
                eeg_picks = eeg_picks[:128]
            raw_data.pick(eeg_picks)
            print(f"已选择前{len(raw_data.ch_names)}个EEG通道")

        # 设置 Montage（插值前必须设置空间信息）
        if montage is not None:
            channel_rename_dict = {}
            for ch_name in raw_data.ch_names:
                if ch_name.startswith('E') and ch_name[1:].isdigit():
                    num = int(ch_name[1:])
                    new_name = f'EEG{num:03d}'  # EEG001, EEG002, etc.
                    channel_rename_dict[ch_name] = new_name

            if channel_rename_dict:
                raw_data.rename_channels(channel_rename_dict)
                print(f"已重命名 {len(channel_rename_dict)} 个通道以匹配montage格式")

            raw_data.set_montage(montage, on_missing='warn')
            print("已设置电极 montage")
        else:
            print("警告: 未设置 montage，坏道将无法插值，将尝试丢弃坏道")

        # 坏道：插值（而非删除），保持 128 通道拓扑完整
        if bad_channels:
            to_interp = _resolve_bad_channel_names(bad_channels, raw_data.ch_names)
            if to_interp:
                raw_data.info['bads'] = to_interp
                try:
                    raw_data.interpolate_bads(reset_bads=True)
                    print(f"已对 {len(to_interp)} 个坏道进行球面插值: {to_interp}")
                except Exception as exc:
                    print(f"插值失败 ({exc})，改为移除这些通道")
                    raw_data.info['bads'] = []
                    raw_data.drop_channels(to_interp)
            else:
                print("坏道列表与当前通道名无交集，跳过插值")
        else:
            print("无坏道列表，跳过插值")

        channels_after = len(raw_data.ch_names)
        print(f"插值/裁剪后通道数: {channels_before} -> {channels_after}")

        raw_data.filter(l_freq=0.1, h_freq=45.0, method="fir", fir_window="hamming")
        raw_data.notch_filter(freqs=[50])
        raw_data.set_eeg_reference("average")

        if apply_ica:
            ica = ICA(
                n_components=ICA_N_COMPONENTS,
                random_state=97,
                method="infomax",
                max_iter=500,
            )

            try:
                ica.fit(raw_data)
                if not (IC_LABEL_AVAILABLE and label_components is not None):
                    raise ImportError(
                        "未安装 mne-icalabel，无法使用 ICLabel 自动判别 ICA 分量。"
                        "请先: pip install mne-icalabel"
                    )

                ic_labels = label_components(raw_data, ica, method="iclabel")
                comp_labels = ic_labels["labels"]
                comp_proba = ic_labels["y_pred_proba"]

                proba_threshold = ICLABEL_PROBA_THRESHOLD
                exclude_idx = [
                    i
                    for i, (lab, p) in enumerate(zip(comp_labels, comp_proba))
                    if _is_iclabel_artifact(lab) and float(p) > proba_threshold
                ]

                if exclude_idx:
                    ica.exclude = exclude_idx
                    raw_data = ica.apply(raw_data.copy())
                    rejected_ic_count = len(exclude_idx)
                    print(f"ICA/ICLabel：已剔除 {len(exclude_idx)} 个分量: {exclude_idx}")
                else:
                    print("ICA/ICLabel：无分量达到剔除阈值，保留全部 ICA 分量")
            except Exception as exc:
                print(f"ICA 处理失败，跳过 ICA: {exc}")

        return raw_data, rejected_ic_count

    # 首先对第一个文件进行坏通道检测，确定统一的坏通道列表
    print("\n开始坏通道检测...")
    bad_channels_global = []

    if PYPREP_AVAILABLE:
        print("使用 PyPREP 对第一个文件进行坏通道检测...")
        raw_for_detection = raw_objects[0].copy()

        eeg_picks = mne.pick_types(raw_for_detection.info, eeg=True)
        if len(eeg_picks) > 128:
            eeg_picks = eeg_picks[:128]
        raw_for_detection.pick(eeg_picks)

        vref_candidates = ['VREF', 'EEG129', 'EEG128']
        for vref_name in vref_candidates:
            if vref_name in raw_for_detection.ch_names:
                try:
                    vref_type = raw_for_detection.get_channel_types(picks=[vref_name])[0]
                    if vref_type == 'eeg':
                        raw_for_detection.set_channel_types({vref_name: 'misc'})
                    break
                except Exception:
                    continue

        if montage is not None:
            channel_rename_dict = {}
            for ch_name in raw_for_detection.ch_names:
                if ch_name.startswith('E') and ch_name[1:].isdigit():
                    num = int(ch_name[1:])
                    new_name = f'EEG{num:03d}'
                    channel_rename_dict[ch_name] = new_name

            if channel_rename_dict:
                raw_for_detection.rename_channels(channel_rename_dict)

            raw_for_detection.set_montage(montage, on_missing='warn')

        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": [50],
            "max_iterations": 4
        }

        try:
            prep = PrepPipeline(raw_for_detection, prep_params, montage, ransac=True)
            prep.fit()

            bad_channels = prep.noisy_channels_original['bad_all']
            bad_channels_eeg_format = [str(ch) for ch in bad_channels]

            bad_channels_global = []
            for ch in bad_channels_eeg_format:
                if ch.startswith('EEG') and ch[3:].isdigit():
                    num = int(ch[3:])
                    e_format = f'E{num}'
                    bad_channels_global.append(e_format)
                else:
                    bad_channels_global.append(ch)

            print(f"PyPREP检测到坏通道(EEG格式): {bad_channels_eeg_format}")
            print(f"转换为E格式用于插值: {bad_channels_global}")
            print(f"全局坏通道检测结果: {len(bad_channels_global)} 个坏道")

        except Exception as e:
            print(f"PyPREP 检测失败: {e}")
            print("将使用手动指定的坏电极列表作为全局坏通道...")
            bad_channels_global = MANUAL_BAD_ELECTRODES.copy()
            print(f"手动坏通道列表: {bad_channels_global}")
    else:
        print("PyPREP 不可用，使用手动指定的坏电极列表作为全局坏通道...")
        bad_channels_global = MANUAL_BAD_ELECTRODES.copy()
        print(f"手动坏通道列表: {bad_channels_global}")

    # 预处理所有raw对象，使用统一的坏通道列表
    print("\n开始预处理所有文件...")
    processed_raws = []
    rejected_ic_counts_per_file = []
    for i, raw in enumerate(raw_objects):
        print(f"预处理文件 {i+1}...")
        processed_raw, rejected_ic_count = preprocess_raw(
            raw, montage=montage, apply_ica=apply_ica, bad_channels=bad_channels_global
        )
        processed_raws.append(processed_raw)
        rejected_ic_counts_per_file.append(int(rejected_ic_count))
    print("预处理完成")
    rejected_ic_count_total = int(sum(rejected_ic_counts_per_file))
    print(f"该被试 ICA/ICLabel 共剔除分量数: {rejected_ic_count_total}")

    if update_ic_reject_only:
        trial_info_path = os.path.join(processed_data_path, f"{subject_name}_trial_info.json")
        if not os.path.isfile(trial_info_path):
            print(f"错误: 未找到已有 {subject_name}_trial_info.json，无法仅更新 IC 剔除数: {trial_info_path}")
            return
        with open(trial_info_path, "r", encoding="utf-8") as f:
            trial_info = json.load(f)
        trial_info.setdefault("preprocessing", {})
        trial_info["preprocessing"]["rejected_ic_count_total"] = rejected_ic_count_total
        with open(trial_info_path, "w", encoding="utf-8") as f:
            json.dump(trial_info, f, ensure_ascii=False, indent=2)
        print(f"已仅更新 trial_info 中的 preprocessing.rejected_ic_count_total，未改写其他数据文件。")
        print(f"  -> {trial_info_path}")
        return

    # 拼接所有选择的trials（简单拼接，不考虑时间）
    print(f"\n拼接所有文件的trials...")
    merged_trials = []
    current_global_trial_num = 1

    for file_idx, selected_trials in enumerate(selected_trials_list):
        print(f"文件{file_idx+1}: 添加 {len(selected_trials)} 个trials")
        for trial in selected_trials:
            new_trial = trial.copy()
            new_trial['global_trial_num'] = current_global_trial_num
            merged_trials.append(new_trial)
            current_global_trial_num += 1

    print(f"拼接后总共有 {len(merged_trials)} 个trials")

    # 合并标签 - 按照选择的trials顺序
    print(f"合并标签...")
    merged_labels = {}
    merged_rts = {}
    global_trial_num = 1

    for file_idx, (selected_trials, labels, rts) in enumerate(zip(selected_trials_list, all_labels, all_rts)):
        print(f"文件{file_idx+1}: 处理 {len(selected_trials)} 个trials的标签")

        for trial in selected_trials:
            original_trial_num = trial['trial_num']
            # 从对应文件的标签与RT中查找（未找到则为 None）
            merged_labels[global_trial_num] = labels.get(original_trial_num, None)
            merged_rts[global_trial_num] = rts.get(original_trial_num, None)

            global_trial_num += 1

    print(f"合并后总共有 {len(merged_labels)} 个标签")

    # 提取trial数据
    samples_per_trial = int(trial_duration * SAMPLING_RATE)
    epochs = []
    kept_trials_info = []

    print(f"\n开始提取trial数据，每个trial {trial_duration}秒...")

    for idx, trial in enumerate(merged_trials, start=1):
        source_file = trial['source_file']
        original_start_seconds = trial['start_seconds']

        # 选择对应的raw对象（文件索引从1开始，列表从0开始）
        raw_to_use = processed_raws[source_file - 1]

        start_sample = int(original_start_seconds * SAMPLING_RATE)
        stop_sample = start_sample + samples_per_trial

        if original_start_seconds < 0 or stop_sample / SAMPLING_RATE > raw_to_use.times[-1]:
            print(f"跳过 Trial {idx} (文件{source_file}): 超出数据范围 ({original_start_seconds:.3f}s -> {(stop_sample / SAMPLING_RATE):.3f}s)")
            continue

        data = raw_to_use.get_data(start=start_sample, stop=stop_sample)
        if data.shape[0] != len(raw_to_use.ch_names) or data.shape[1] != samples_per_trial:
            print(f"警告 Trial {idx} (文件{source_file}): 形状异常 {data.shape}，将尝试继续")

        epochs.append(data)
        kept_trials_info.append({
            'trial_num': idx,
            'global_trial_num': trial['global_trial_num'],
            'original_trial_num': trial['trial_num'],
            'source_file': source_file,
            'time': trial['time'],
            'start_seconds': original_start_seconds,
            'line': trial.get('line', ''),
            'line_number': trial.get('line_number', None)
        })

    if len(epochs) == 0:
        print("未能提取到有效trial数据，跳过数据保存。")
    else:
        # 合并所有epochs
        epochs_array = np.stack(epochs, axis=0)
        print(f"提取的epochs形状: {epochs_array.shape}  (trial, channel, time)")

        # 保存trial数据
        np.save(os.path.join(processed_data_path, f"{subject_name}_trials.npy"), epochs_array)

        # 保存trial信息
        trial_info = {
            'subject_name': subject_name,
            'data_name': f"{subject_name}",
            'sampling_rate': SAMPLING_RATE,
            'trial_duration': trial_duration,
            'num_trials': len(kept_trials_info),
            'channels': processed_raws[0].ch_names,  # 使用第一个文件的通道信息
            'trial_times': kept_trials_info,
            'mff_files': [config["mff"] for config in mff_files],
            'log_files': log_files,
        }

        # 对齐并保存标签
        if merged_labels:
            label_list = []
            for i in range(1, len(kept_trials_info) + 1):
                kept = kept_trials_info[i - 1]
                orig_trial_num = kept['original_trial_num']
                # CSV 的 Trial 是“保留后的序号”；merged_labels 的 key 是 global_trial_num
                label_list.append({
                    'Trial': i,
                    'OriginalTrial': orig_trial_num,
                    'Label': merged_labels.get(kept['global_trial_num'], None)
                })

            labels_df = pd.DataFrame(label_list)
            labels_df.to_csv(os.path.join(processed_data_path, f"{subject_name}_labels.csv"), index=False)
            trial_info['labels'] = {int(row.Trial): (None if pd.isna(row.Label) else int(row.Label)) for _, row in labels_df.iterrows()}
            trial_info['num_labels'] = int(labels_df['Label'].notna().sum())

            # 打印标签分布
            if labels_df['Label'].notna().any():
                label_counts = labels_df.dropna()['Label'].value_counts().sort_index()
                print("\n标签分布:")
                for label, count in label_counts.items():
                    print(f"  标签 {int(label)}: {int(count)} 个trial")

        # 对齐并保存RT（ms）
        if merged_rts:
            rt_list = []
            for i in range(1, len(kept_trials_info) + 1):
                kept = kept_trials_info[i - 1]
                rt_list.append({
                    'Trial': i,
                    'OriginalTrial': kept['original_trial_num'],
                    'RT_ms': merged_rts.get(kept['global_trial_num'], None)
                })

            rts_df = pd.DataFrame(rt_list)
            rts_df.to_csv(os.path.join(processed_data_path, f"{subject_name}_rts.csv"), index=False)
            trial_info['rts_ms'] = {int(row.Trial): (None if pd.isna(row.RT_ms) else int(row.RT_ms)) for _, row in rts_df.iterrows()}
            trial_info['num_rts'] = int(rts_df['RT_ms'].notna().sum())

        trial_info['preprocessing'] = {
            'bad_channel_policy': (
                'interpolate_bads_filter_avgref_ica_iclabel' if apply_ica
                else 'interpolate_bads_filter_avgref_no_ica'
            ),
            'ica': 'enabled' if apply_ica else 'disabled',
            'iclabel': (
                f'artifact components (eye/muscle/heart/line/channel noise) with proba > {ICLABEL_PROBA_THRESHOLD} excluded'
                if apply_ica else 'n/a'
            ),
            'ica_n_components': ICA_N_COMPONENTS if apply_ica else None,
            'bad_channels_detected': bad_channels_global,
            'rejected_ic_count_total': rejected_ic_count_total,
        }

        with open(os.path.join(processed_data_path, f"{subject_name}_trial_info.json"), 'w', encoding='utf-8') as f:
            json.dump(trial_info, f, ensure_ascii=False, indent=2)

        print(f"\n已保存trial数据到: {processed_data_path}")
        print(f"- 数据文件: {subject_name}_trials.npy (形状: {epochs_array.shape})")
        print(f"- 信息文件: {subject_name}_trial_info.json")
        if merged_labels:
            print(f"- 标签文件: {subject_name}_labels.csv")
        if merged_rts:
            print(f"- RT文件: {subject_name}_rts.csv")

        # 显示前几个trial的详细信息
        print(f"\n前5个trial的详细信息:")
        for i in range(min(5, len(kept_trials_info))):
            trial = kept_trials_info[i]
            orig_num = trial['original_trial_num']
            key = trial['global_trial_num']
            lab = merged_labels.get(key, 'N/A') if merged_labels else 'N/A'
            rt_ms = merged_rts.get(key, 'N/A') if merged_rts else 'N/A'
            print(f"  Trial {trial['trial_num']} (原始: {orig_num}, 文件: {trial['source_file']}): {trial['time']} - 标签: {lab} - RT_ms: {rt_ms}")

    print(f"\n受试者 {subject_name} 的数据处理完成！\n")


def main():
    """主函数：坏道插值 → 滤波 → ICA + ICLabel，结果写入 <output_root>/<subject_name>/。

    无额外参数时默认只处理 DEFAULT_RUN_SUBJECT_NAME（便于 IDE 直接运行）；全员请加 --all-subjects。
    默认 output_root 为 OUTPUT_ROOT_INTERP_ICA；可用 --output-here / --output-root 覆盖输出根目录。
    """
    parser = argparse.ArgumentParser(description="EEG 预处理与导出（extract_jiachen_2）")
    parser.set_defaults(update_ic_reject_only=False)
    parser.add_argument(
        "--plot-ica-subject",
        type=str,
        default=None,
        help="仅绘制单个受试者的 ICA 分量（传入 subject_name）",
    )
    parser.add_argument(
        "--plot-ica-file-index",
        type=int,
        default=1,
        help="绘制 ICA 时使用该受试者第几个 mff 文件（从 1 开始）",
    )
    parser.add_argument(
        "--plot-ica-max-components",
        type=int,
        default=8,
        help="最多绘制多少个 ICA 分量",
    )
    parser.add_argument(
        "--plot-ica-proba-threshold",
        type=float,
        default=ICLABEL_PROBA_THRESHOLD,
        help="ICLabel 判为可剔除伪迹分量的阈值（伪迹类且 proba>threshold 则剔除，默认与主流程 ICLABEL_PROBA_THRESHOLD 一致）",
    )
    parser.add_argument(
        "--update-ic-reject-only",
        action="store_true",
        dest="update_ic_reject_only",
        help="仅重算各文件 ICA/ICLabel 剔除分量总数，并写回已有 trial_info.json；不保存 trials.npy / labels.csv / rts.csv",
    )
    parser.add_argument(
        "--full-export",
        action="store_false",
        dest="update_ic_reject_only",
        help="执行完整导出（保存 trials.npy / labels.csv / rts.csv）",
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="处理 SUBJECTS_CONFIG 中的全部被试（覆盖默认仅跑一人）",
    )
    parser.add_argument(
        "--only-subject",
        action="append",
        dest="only_subjects",
        default=None,
        metavar="NAME",
        help=(
            "只处理指定 subject_name；可重复传入多人。"
            "不传本参数且未传 --all-subjects 时，默认仅处理 "
            f"{DEFAULT_RUN_SUBJECT_NAME!r}（与 DEFAULT_RUN_SUBJECT_NAME 一致）"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="覆盖默认输出根目录 OUTPUT_ROOT_INTERP_ICA；结果写入 <output-root>/<subject_name>/",
    )
    parser.add_argument(
        "--output-here",
        action="store_true",
        help="将输出根目录设为运行脚本时的当前工作目录（cwd），等价于 --output-root .",
    )
    args = parser.parse_args()

    if not IC_LABEL_AVAILABLE:
        print("错误: 需要 mne-icalabel 才能运行 ICA/ICLabel。请先: pip install mne-icalabel")
        return

    if args.plot_ica_subject:
        plot_subject_ica_components(
            subject_name=args.plot_ica_subject,
            file_index=args.plot_ica_file_index,
            max_components=args.plot_ica_max_components,
            proba_threshold=args.plot_ica_proba_threshold,
        )
        return

    if args.output_root is not None:
        output_root = os.path.abspath(os.path.expanduser(args.output_root))
    elif args.output_here:
        output_root = os.path.abspath(os.getcwd())
    else:
        output_root = OUTPUT_ROOT_INTERP_ICA

    if args.all_subjects and args.only_subjects:
        print("提示: 同时传入 --all-subjects 与 --only-subject，按 --all-subjects 处理全部被试。")

    if args.all_subjects:
        configs_to_run = list(SUBJECTS_CONFIG)
    elif args.only_subjects:
        want = set(args.only_subjects)
        configs_to_run = [c for c in SUBJECTS_CONFIG if c.get("subject_name") in want]
        found = {c.get("subject_name") for c in configs_to_run}
        missing = want - found
        if missing:
            print(f"错误: 以下被试不在 SUBJECTS_CONFIG 中: {sorted(missing)}")
            return
        if not configs_to_run:
            print("错误: --only-subject 未匹配到任何被试")
            return
    else:
        configs_to_run = [
            c for c in SUBJECTS_CONFIG if c.get("subject_name") == DEFAULT_RUN_SUBJECT_NAME
        ]
        if not configs_to_run:
            print(
                f"错误: 默认被试 {DEFAULT_RUN_SUBJECT_NAME!r} 不在 SUBJECTS_CONFIG 中。"
                "请检查 subject_mff_benchmark_config，或改用 --only-subject / --all-subjects。"
            )
            return
        print(f"默认仅处理被试: {DEFAULT_RUN_SUBJECT_NAME}（全员请使用命令行参数 --all-subjects）")

    mode = "仅更新 trial_info 中 rejected_ic_count_total" if args.update_ic_reject_only else "完整导出"
    n_run = len(configs_to_run)
    print(f"本次处理 {n_run} 名受试者，输出根目录: {output_root}（模式: {mode}）")

    for cfg in configs_to_run:
        sn = cfg["subject_name"]
        print("=" * 60)
        if args.update_ic_reject_only:
            print(f"开始受试者: {sn}（仅更新 IC 剔除数 → trial_info.json）")
        else:
            print(f"开始处理受试者: {sn}（插值坏道 → 滤波 → ICA + ICLabel）")
        print("=" * 60)

        try:
            process_subject(
                cfg,
                output_root=output_root,
                apply_ica=True,
                update_ic_reject_only=args.update_ic_reject_only,
            )
        except Exception as e:
            print(f"\n处理受试者 {sn} 时发生错误: {e}")
            import traceback
            traceback.print_exc()

        print("=" * 60)
        print(f"受试者 {sn} 处理结束。")
        print("=" * 60)


if __name__ == "__main__":
    main()
