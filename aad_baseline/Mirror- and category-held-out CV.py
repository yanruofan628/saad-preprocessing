"""
EEGNet 2s 整段交叉验证 - 默认只跑 Lawhern EEGNet。
- 数据: A:/standard_data_interp_ica/<被试>/（插值+滤波+ICA+ICLabel 预处理）
- 划分: 默认五折左右平衡 pair-consistent；--subcategory-6fold 子类别 6 折；
  --stratified-kfold 分层五折（无左右约束）。
- 结果: 默认 eegnet_2s_cv5_pairconsistent_summary；SKF / 子类别 6 折 对应各自子目录。
- 断点: 若汇总里该被试已有 status=ok 且 by_trial_mean 有效则跳过；nan 或失败则重跑。
"""
import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from shallownet3_model import ShallowNet3

from pair_consistent_splits import (
    PairBalanceError,
    SUBCATEGORY_N_FOLDS,
    SUBCATEGORY_VAL_TYPES_PER_FOLD,
    enumerate_subcategory_keys_from_benchmarks,
    load_trial_subcategory_keys_strict,
    pair_consistent_kfold_splits_hybrid,
    subcategory_pair_6fold_splits,
    trial_pair_keys_pair_consistent_strict,
    verify_pair_consistent_splits_relaxed,
)

# 数据配置
SAMPLING_RATE = 250
BASE_DATA_PATH = "A:/standard_data_interp_ica"
# 汇总表目录：与 split_suffix _pairconsistent / _skf 一致
RESULTS_ROOT_PAIRCONSISTENT = "A:/standard_data_interp_ica/eegnet_2s_cv5_pairconsistent_summary"
RESULTS_ROOT_SKF = "A:/standard_data_interp_ica/eegnet_2s_cv5_skf_summary"
RESULTS_ROOT_SUBCAT6 = "A:/standard_data_interp_ica/eegnet_2s_subcat6fold_summary"
DEFAULT_RESULTS_ROOT = RESULTS_ROOT_PAIRCONSISTENT
SUBJECT_NAMES = [
    'zhanghanglei', 'zhangyufei', 'jinxiaoyue', 'chenxianwei', 'yeziyuan', 'yanxingzhuo',
    'zhangzhiyao', 'haoxiang', 'hehaohuai', 'qiuhaiyun', 'zhouyu', 'honghaokai', 'caolulu',
    'yanyinsong',
    'zengdexin',
    'huangxiaohang', 'xufan',
    'liuzehao', 'jichengzhi', 'qiusiqi',
    'machenxiang',
    'lizhuhang', 'zhangyajie',
]
# SUBJECT_NAMES = [
#     # 'zhanghanglei', 'zhangyufei', 'jinxiaoyue', 'chenxianwei', 'yeziyuan', 'yanxingzhuo',
#     # 'zhangzhiyao', 'haoxiang', 'hehaohuai', 'qiuhaiyun', 'zhouyu', 'honghaokai', 'caolulu',
#     # 'yanyinsong',
#     #'zengdexin',
#     'huangxiaohang'
# ]


@dataclass
class DataMeta:
    sampling_rate_hz: float
    num_channels: int
    samples_per_trial: Optional[int] = None


def find_data_files(data_dir: Path) -> Tuple[Path, Path, Path]:
    trials = list(data_dir.glob("*trials.npy"))
    labels = list(data_dir.glob("*labels.csv"))
    info = list(data_dir.glob("*trial_info.json"))
    if len(trials) != 1 or len(labels) != 1 or len(info) != 1:
        raise FileNotFoundError(
            f"期望在 {data_dir} 找到各1个 trials.npy/labels.csv/trial_info.json，实际数量: "
            f"trials={len(trials)}, labels={len(labels)}, info={len(info)}"
        )
    return trials[0], labels[0], info[0]


def load_meta(info_path: Path, override_fs: Optional[float], override_channels: Optional[int]) -> DataMeta:
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    fs_candidates = [
        info.get("fs"),
        info.get("sampling_rate"),
        info.get("sampling_rate_hz"),
        info.get("sfreq"),
    ]
    ch_candidates = [
        info.get("n_channels"),
        info.get("num_channels"),
        info.get("channels"),
    ]
    spt_candidates = [
        info.get("samples_per_trial"),
        info.get("n_samples_per_trial"),
    ]

    fs = override_fs if override_fs is not None else next((x for x in fs_candidates if x is not None), None)
    num_channels = override_channels if override_channels is not None else next(
        (x for x in ch_candidates if x is not None), None)
    if isinstance(num_channels, list):
        num_channels = len(num_channels)
    samples_per_trial = next((x for x in spt_candidates if x is not None), None)

    if fs is None or num_channels is None:
        raise ValueError("trial_info.json 未提供采样率或通道数，且未通过命令行覆盖。")

    return DataMeta(float(fs), int(num_channels), int(samples_per_trial) if samples_per_trial is not None else None)


def reshape_trials(trials_2d: np.ndarray, num_channels: int) -> np.ndarray:
    if trials_2d.ndim == 3:
        return trials_2d
    if trials_2d.ndim != 2:
        raise ValueError(f"期望 2D 或 3D 数组，得到 {trials_2d.ndim}D")

    num_trials, flat_len = trials_2d.shape
    if flat_len % num_channels != 0:
        raise ValueError(
            f"无法按 {num_channels} 通道整除展平长度 {flat_len}，请检查 meta 或数据。"
        )
    samples_per_trial = flat_len // num_channels
    return trials_2d.reshape(num_trials, num_channels, samples_per_trial)


def exponential_moving_standardize(
    data: np.ndarray,
    alpha: float = 0.01,
    eps: float = 1e-8,
    init_block_size: Optional[int] = None
) -> np.ndarray:
    original_ndim = data.ndim
    if original_ndim == 2:
        data = data[np.newaxis, :, :]
        was_2d = True
    elif original_ndim == 3:
        was_2d = False
    else:
        raise ValueError(f"数据必须是2D或3D，得到 {original_ndim}D")

    n_samples, n_channels, n_timepoints = data.shape
    standardized = np.zeros_like(data)

    for sample_idx in range(n_samples):
        sample_data = data[sample_idx]
        ema = np.zeros_like(sample_data)
        emsd = np.zeros_like(sample_data)

        if init_block_size is not None and n_timepoints > init_block_size:
            init_mean = np.mean(sample_data[:, :init_block_size], axis=1, keepdims=True)
            init_std = np.std(sample_data[:, :init_block_size], axis=1, keepdims=True) + eps
            ema[:, 0] = init_mean[:, 0]
            emsd[:, 0] = init_std[:, 0] ** 2
        else:
            ema[:, 0] = sample_data[:, 0]
            emsd[:, 0] = eps ** 2

        for t in range(1, n_timepoints):
            ema[:, t] = alpha * sample_data[:, t] + (1 - alpha) * ema[:, t - 1]
            squared_diff = (sample_data[:, t] - ema[:, t]) ** 2
            emsd[:, t] = alpha * squared_diff + (1 - alpha) * emsd[:, t - 1]

        emsd = np.sqrt(emsd)
        standardized[sample_idx] = (sample_data - ema) / (emsd + eps)

    if was_2d:
        standardized = standardized[0]

    return standardized


def select_last_seconds(data: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    last_n = int(round(seconds * fs))
    if last_n <= 0:
        raise ValueError("截取秒数需为正。")
    if data.shape[-1] < last_n:
        raise ValueError(
            f"每trial样本数 {data.shape[-1]} 小于所需最后 {last_n} 样本，请检查采样率或trial长度。"
        )
    return data[..., -last_n:]


def apply_baseline_correction(
    data: np.ndarray,
    fs: float,
    baseline_start: float = 0.0,
    baseline_end: float = 4.0
) -> np.ndarray:
    n_trials, n_channels, n_timepoints = data.shape
    corrected_data = np.zeros_like(data)
    baseline_start_idx = int(round(baseline_start * fs))
    baseline_end_idx = int(round(baseline_end * fs))

    if baseline_start_idx < 0 or baseline_end_idx > n_timepoints:
        raise ValueError(
            f"基线窗口 [{baseline_start}, {baseline_end}] 超出数据范围 [0, {n_timepoints/fs:.2f}]"
        )
    if baseline_start_idx >= baseline_end_idx:
        raise ValueError(f"基线窗口开始时间必须小于结束时间")

    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            baseline_window = data[trial_idx, ch_idx, baseline_start_idx:baseline_end_idx]
            baseline_mean = np.mean(baseline_window)
            corrected_data[trial_idx, ch_idx, :] = data[trial_idx, ch_idx, :] - baseline_mean

    return corrected_data


def split_trials_to_1s_samples(
    data: np.ndarray,
    labels: np.ndarray,
    fs: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将每个trial的2秒数据拆成2个1秒样本。
    data: (n_trials, n_channels, n_timepoints), n_timepoints=500(2秒)
    返回: data_1s (n_trials*2, n_channels, 250), labels_1s, trial_ids
    """
    n_trials = data.shape[0]
    seg_samples = int(1.0 * fs)  # 250
    if data.shape[2] < seg_samples * 2:
        raise ValueError(f"数据时长不足2秒，无法拆成2个1秒样本")

    samples_1 = data[:, :, :seg_samples]   # 3-4秒
    samples_2 = data[:, :, seg_samples:]     # 4-5秒
    data_1s = np.concatenate([samples_1, samples_2], axis=0)
    labels_1s = np.concatenate([labels, labels], axis=0)
    trial_ids = np.concatenate([np.arange(n_trials), np.arange(n_trials)])

    return data_1s, labels_1s, trial_ids


# --------------- 网络选择 ---------------
# 本脚本默认仅 eegnet；仍可通过 --network 指定 eegnet_simple / shallownet3
NETWORK_CHOICE = 'eegnet'

# --------------- 划分方式 ---------------
# 只跑 by_trial 五折；要跑两种可改为 ['random_5fold', 'by_trial_5fold']
SPLIT_METHODS = ['by_trial_5fold']


def _get_fc_size(n_samples: int, f2: int, pool1: int, pool2: int) -> int:
    """Lawhern EEGNet Block3 池化后的时间维长度（Block1/3 用 same 填充），用于 FC 输入维度。"""
    t = (n_samples // pool1) // pool2
    return f2 * 1 * max(1, t)


class EEGNet(nn.Module):
    """Lawhern et al. EEGNet：Block1 时间卷积 + Block2 深度卷积 + Block3 深度可分离卷积。
    默认 f1=8, f2=8, kernel_t=41, kernel_3=8，参数量与 EEGNetSimple 同量级，减轻过拟合。"""

    def __init__(self, n_channels: int, n_samples: int, n_classes: int, f1: int = 8, f2: int = 8,
                 kernel_t: int = 41, kernel_3: int = 8, pool1: int = 4, pool2: int = 8, dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.f1, self.f2 = f1, f2
        self.kernel_t = kernel_t
        self.kernel_3 = kernel_3
        self.pool1, self.pool2 = pool1, pool2

        # Block 1: 时间卷积 (1, C, T) -> (F1, C, T')
        self.conv1 = nn.Conv2d(1, f1, (1, kernel_t), padding=(0, kernel_t // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)

        # Block 2: 深度卷积，每个 F1 通道独立 (C, 1) 卷积
        self.depthwise_conv2 = nn.Conv2d(f1, f1, (n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1)
        self.pool2_layer = nn.AvgPool2d((1, pool1))

        # Block 3: 深度可分离 = 深度 (1, kernel_3) + 逐点 1x1
        self.depthwise_conv3 = nn.Conv2d(f1, f1, (1, kernel_3), groups=f1, padding=(0, kernel_3 // 2), bias=False)
        self.pointwise_conv3 = nn.Conv2d(f1, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool3_layer = nn.AvgPool2d((1, pool2))

        fc_size = _get_fc_size(n_samples, f2, pool1, pool2)
        self.fc = nn.Linear(fc_size, n_classes)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_p, training=self.training)

        x = self.depthwise_conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_p, training=self.training)
        x = self.pool2_layer(x)

        x = self.depthwise_conv3(x)
        x = self.pointwise_conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = F.dropout(x, self.dropout_p, training=self.training)
        x = self.pool3_layer(x)

        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


class EEGNetSimple(nn.Module):
    """原脚本的简化版：conv_ica + conv_time + adaptive pool + conv_class。"""

    def __init__(self, n_channels: int, n_samples: int, n_classes: int):
        super(EEGNetSimple, self).__init__()
        self.conv_time = nn.Conv2d(1, 20, (1, 41), stride=(1, 1), bias=False)
        self.conv_ica = nn.Conv2d(1, 8, (n_channels, 1), stride=(1, 1), bias=False)
        self.batch1 = nn.BatchNorm2d(20, momentum=0.1, affine=True, eps=1e-5)
        self.poolmean = nn.AdaptiveAvgPool2d((8, 1))
        self.conv_class = nn.Conv2d(1, n_classes, kernel_size=(20, 8), stride=(1, 1), bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_ica(x)
        x = F.dropout(x, 0.15, training=self.training)
        x = torch.permute(x, (0, 2, 1, 3))
        x = self.conv_time(x)
        x = self.batch1(x)
        x = F.dropout(x, 0.15, training=self.training)
        x = torch.mul(x, x)
        x = self.poolmean(x)
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.conv_class(x)
        x = x.squeeze()
        return x


def get_network_class_and_tag(network_name: str) -> Tuple[type, str]:
    """返回 (模型类, 用于保存路径的短标签)。shallownet3 构造方式不同，在 run_subject_cv 里单独分支。"""
    name = (network_name or NETWORK_CHOICE).strip().lower()
    if name == 'eegnet_simple':
        return EEGNetSimple, 'eegnet_simple'
    if name == 'shallownet3':
        return ShallowNet3, 'shallownet3'
    return EEGNet, 'eegnet'


def train_eegnet(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu',
                 class_weights: torch.Tensor = None, patience=30):
    """不保存模型，只记录 val 上最高准确率及对应的 y_true, y_pred。与 shallownet3 一致。"""
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=7e-4)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.8)

    best_val_balanced_acc = 0
    patience_counter = 0
    best_y_true, best_y_pred = None, None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_predictions, val_true_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(batch_y.cpu().numpy())

        val_balanced_acc = balanced_accuracy_score(val_true_labels, val_predictions)

        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_y_true, best_y_pred = np.array(val_true_labels), np.array(val_predictions)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'  Epoch {epoch}, Train Loss: {train_loss / len(train_loader):.4f}, Val: {val_balanced_acc:.4f} (best: {best_val_balanced_acc:.4f})')

        if patience_counter >= patience:
            break

        scheduler.step()

    if best_y_true is None:
        model.eval()
        val_predictions, val_true_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(batch_y.cpu().numpy())
        best_y_true, best_y_pred = np.array(val_true_labels), np.array(val_predictions)
    return best_val_balanced_acc, best_y_true, best_y_pred


def train_shallownet3(
    model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu',
    class_weights: Optional[torch.Tensor] = None, patience=30
):
    """ShallowNet3。不保存模型，只记录 val 上最高准确率及对应的 y_true, y_pred。与 train_eegnet 一致。"""
    criterion = nn.NLLLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=7e-4)
    scheduler = StepLR(optimizer, step_size=40, gamma=0.8)
    best_val_balanced_acc = 0
    patience_counter = 0
    best_y_true, best_y_pred = None, None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_predictions, val_true_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(batch_y.cpu().numpy())
        val_balanced_acc = balanced_accuracy_score(val_true_labels, val_predictions)

        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_y_true, best_y_pred = np.array(val_true_labels), np.array(val_predictions)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'  Epoch {epoch}, Train Loss: {train_loss / len(train_loader):.4f}, Val: {val_balanced_acc:.4f} (best: {best_val_balanced_acc:.4f})')

        if patience_counter >= patience:
            break
        scheduler.step()

    if best_y_true is None:
        model.eval()
        val_predictions, val_true_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(batch_y.cpu().numpy())
        best_y_true, best_y_pred = np.array(val_true_labels), np.array(val_predictions)
    return best_val_balanced_acc, best_y_true, best_y_pred


def run_subject_cv(
    data_dir: Path,
    device: torch.device,
    fs: float = 250,
    random_state: int = 42,
    epochs: int = 300,
    batch_size: int = 16,
    lr: float = 0.0008,
    n_folds: int = 5,
    verbose: bool = True,
    network_name: Optional[str] = None,
    results_dir: Optional[Path] = None,
    use_pair_consistent: bool = True,
    use_subcategory_6fold: bool = False,
    write_per_subject_files: bool = True,
) -> Dict[str, Any]:
    """
    对单个被试跑 CV（默认五折 by_trial）；--subcategory-6fold 时为 6 折。
    network_name: 'eegnet' | 'eegnet_simple'，不传则使用全局 NETWORK_CHOICE。
    返回: {'random_5fold': {...}, 'by_trial_5fold': {...}}
    若失败返回 {'error': str}
    """
    model_cls, path_tag = get_network_class_and_tag(network_name or NETWORK_CHOICE)
    trials_path, labels_path, info_path = find_data_files(data_dir)
    with info_path.open("r", encoding="utf-8") as f:
        info_json = json.load(f)
    trial_duration = info_json.get("trial_duration")

    try:
        meta = load_meta(info_path, fs, None)
        if verbose:
            print(f"  元信息: fs={meta.sampling_rate_hz}, 通道数={meta.num_channels}")
    except (FileNotFoundError, ValueError) as e:
        trials = np.load(trials_path)
        inferred_channels = trials.shape[1] if trials.ndim == 3 else 128
        meta = DataMeta(sampling_rate_hz=fs, num_channels=inferred_channels, samples_per_trial=None)

    if trial_duration is None:
        trials = np.load(trials_path)
        trials_3d = reshape_trials(trials, meta.num_channels)
        trial_duration = trials_3d.shape[2] / meta.sampling_rate_hz

    if verbose:
        print(f"  试次时长: {trial_duration:.2f} 秒 (需要 >= 5 秒才进行 3–5s 刺激段分类)")

    trials = np.load(trials_path)
    labels_df = pd.read_csv(labels_path)
    if 'Label' in labels_df.columns:
        labels = labels_df['Label'].to_numpy()
    else:
        labels = labels_df.iloc[:, 1].to_numpy() if labels_df.shape[1] >= 2 else labels_df.iloc[:, 0].to_numpy()

    n_keys = len(labels)
    full_pair_keys: list = []
    if use_pair_consistent and not use_subcategory_6fold:
        full_pair_keys = trial_pair_keys_pair_consistent_strict(
            info_json, data_dir, n_keys, verbose=verbose
        )

    trials_3d = reshape_trials(trials, meta.num_channels)
    data_full = select_last_seconds(trials_3d, meta.sampling_rate_hz, trial_duration)
    data_baseline_corrected = apply_baseline_correction(
        data_full, fs=meta.sampling_rate_hz, baseline_start=0.0, baseline_end=3.0
    )

    if trial_duration < 5.0:
        if verbose:
            print(f"  跳过: 数据时长 {trial_duration:.2f} 秒不足（需要 >= 5 秒）")
        return {'error': f'数据时长{trial_duration}秒不足'}

    stimulus_start_idx = int(round(3.0 * meta.sampling_rate_hz))
    stimulus_end_idx = int(round(5.0 * meta.sampling_rate_hz))
    data = data_baseline_corrected[:, :, stimulus_start_idx:stimulus_end_idx]

    if data.shape[0] != labels.shape[0]:
        min_n = min(data.shape[0], labels.shape[0])
        data = data[:min_n]
        labels = labels[:min_n]
        full_pair_keys = full_pair_keys[:min_n]

    valid_mask = ~pd.isna(labels)
    if not valid_mask.all():
        bad_pos = np.where(~valid_mask)[0]
        if verbose:
            detail_parts = [
                f"剔除标签为 NaN 的试次，共 {bad_pos.size} 条",
                f"顺序位置(1-based): {(bad_pos + 1).tolist()}",
            ]
            if "Trial" in labels_df.columns:
                n0 = len(labels)
                tcol = labels_df["Trial"].to_numpy()[:n0]
                detail_parts.append(f"labels.csv 的 Trial 列: {tcol[bad_pos].tolist()}")
            print(f"  {' | '.join(detail_parts)}")
        data = data[valid_mask]
        labels = labels[valid_mask]
        # 仅左右平衡路径会填充 full_pair_keys；默认 SKF 下为空列表，勿按下标筛
        if len(full_pair_keys) == len(valid_mask):
            full_pair_keys = [full_pair_keys[i] for i, m in enumerate(valid_mask) if m]

    trial_pair_keys = full_pair_keys
    n_trials = data.shape[0]
    if n_trials == 0:
        return {'error': '无可用trial'}

    if use_pair_consistent and not use_subcategory_6fold and verbose:
        ctr = Counter(trial_pair_keys)
        n_pair2 = sum(1 for c in ctr.values() if c == 2)
        n_other = len(ctr) - n_pair2
        if n_other > 0 or any(k and k[0] == "__singleton__" for k in ctr):
            print(
                f"  折分: 混合模式 — 恰 2 试次/刺激对的成对保持同折，"
                f"其余试次在五折间轮询均分（当前 {len(ctr)} 个不同 key，其中恰 2 次重复: {n_pair2}）"
            )

    # 使用完整 2s 刺激段，每 trial 一个样本
    data_2s = data  # (n_trials, n_channels, 500)
    trial_ids = np.arange(n_trials)
    unique_labels = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[label] for label in labels])
    n_classes = len(unique_labels)
    if n_classes < 2:
        return {'error': f'有效标签类别数仅 {n_classes}，无法训练分类器'}
    n_channels = data_2s.shape[1]
    n_timepoints = data_2s.shape[2]
    standardization_method = 'exponential_moving'

    fold_splits_subcat: Optional[list] = None
    n_folds_effective = n_folds
    if use_subcategory_6fold:
        try:
            keys_sorted = enumerate_subcategory_keys_from_benchmarks(
                info_json, data_dir, verbose=verbose
            )
            trial_subcat_keys = load_trial_subcategory_keys_strict(
                info_json, data_dir, n_trials, verbose=verbose
            )
            fold_splits_subcat = subcategory_pair_6fold_splits(
                trial_subcat_keys, keys_sorted
            )
        except PairBalanceError as e:
            return {'error': str(e)}
        n_folds_effective = SUBCATEGORY_N_FOLDS
        if verbose:
            print("\n  ---------- 子类别 6 折划分（固定、每人相同）----------")
            print(
                "  规则: 全部 session benchmark 枚举出 24 种「无序子类别对」，"
                "按 (左类,右类) 字典序排序后，连续每 4 种作为一折的验证集类型；"
                "该类型下所有 trial（含左右平衡）进 val，其余进 train。"
            )
            for f in range(SUBCATEGORY_N_FOLDS):
                lo = SUBCATEGORY_VAL_TYPES_PER_FOLD * f
                block = keys_sorted[lo : lo + SUBCATEGORY_VAL_TYPES_PER_FOLD]
                n_val = int(fold_splits_subcat[f][1].size)
                parts = [f"({a!r}, {b!r})" for a, b in block]
                print(f"    第 {f + 1}/6 折 — val 的 4 种子类别对: " + " | ".join(parts))
                print(f"             该折 val 试次数: {n_val}")
            print("  ------------------------------------------------\n")
        for fi, (tr_idx, va_idx) in enumerate(fold_splits_subcat):
            if tr_idx.size == 0 or va_idx.size == 0:
                return {
                    'error': (
                        f'子类别6折: 第 {fi + 1} 折 train 或 val 为空 '
                        f'(train={tr_idx.size}, val={va_idx.size})'
                    )
                }
            if len(np.unique(y_encoded[tr_idx])) < 2:
                return {
                    'error': (
                        f'子类别6折: 第 {fi + 1} 折 train 仅含单一类别，无法训练'
                    )
                }
            if len(np.unique(y_encoded[va_idx])) < 2:
                return {
                    'error': (
                        f'子类别6折: 第 {fi + 1} 折 val 仅含单一类别，无法计算有意义的 balanced accuracy'
                    )
                }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = {}
    out_dir: Optional[Path]
    if write_per_subject_files:
        out_dir = results_dir if results_dir is not None else data_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None
    if use_subcategory_6fold:
        split_tag = "subcat6fold"
    elif use_pair_consistent:
        split_tag = "pairconsistent"
    else:
        split_tag = "skf"

    for split_method in SPLIT_METHODS:
        if verbose:
            cv_name = f"{n_folds_effective}折" if use_subcategory_6fold else "五折"
            print(f"  开始{cv_name}交叉验证: {split_method} ...")
        if use_subcategory_6fold:
            fold_splits = fold_splits_subcat
        elif use_pair_consistent:
            fold_splits = pair_consistent_kfold_splits_hybrid(
                trial_pair_keys, n_folds=n_folds, random_state=random_state
            )
            if not verify_pair_consistent_splits_relaxed(trial_pair_keys, fold_splits):
                raise PairBalanceError("内部校验失败：成对试次未完全同折。")
        else:
            fold_splits = list(skf.split(np.arange(n_trials), y_encoded))

        fold_accs = []
        cm_agg = np.zeros((n_classes, n_classes), dtype=np.int64)
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            if verbose:
                print(f"    Fold {fold + 1}/{n_folds_effective} ...")
            train_mask = np.isin(trial_ids, train_idx)
            val_mask = np.isin(trial_ids, val_idx)
            X_train = data_2s[train_mask]
            y_train = y_encoded[train_mask]
            X_val = data_2s[val_mask]
            y_val = y_encoded[val_mask]

            if standardization_method == 'exponential_moving':
                X_train_scaled = exponential_moving_standardize(X_train, alpha=0.01, eps=1e-8, init_block_size=None)
                X_val_scaled = exponential_moving_standardize(X_val, alpha=0.01, eps=1e-8, init_block_size=None)
            else:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
                X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(X_val.shape)

            X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val_scaled).unsqueeze(1)
            y_val_tensor = torch.LongTensor(y_val)

            train_loader = DataLoader(
                TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size, shuffle=False
            )

            model = model_cls(n_channels=n_channels, n_samples=n_timepoints, n_classes=n_classes).to(device)
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            total_train = y_train.shape[0]
            weights_np = np.array([
                total_train / (n_classes * counts_train[np.where(unique_train == c)[0][0]])
                for c in unique_train
            ], dtype=np.float32)
            class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)

            best_acc, y_true, y_pred = train_eegnet(
                model, train_loader, val_loader,
                epochs=epochs, lr=lr, device=device, class_weights=class_weights, patience=30
            )
            fold_accs.append(best_acc)
            cm_fold = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
            cm_agg += cm_fold

        mean_acc = np.mean(fold_accs)
        std_acc = np.std(fold_accs)
        results[split_method] = {'mean': mean_acc, 'std': std_acc, 'fold_accs': fold_accs, 'confusion_matrix': cm_agg}

        if verbose:
            print(f"  {split_method}: {mean_acc:.4f} ± {std_acc:.4f}")

        if write_per_subject_files and out_dir is not None:
            out_path = out_dir / f"{path_tag}_results_3-5s_2s_cv5_{split_method}_{split_tag}.txt"
            with out_path.open("w", encoding="utf-8") as f:
                if use_subcategory_6fold:
                    cv_line = "cv_split: subcategory 6-fold (4 unordered subcategory pair types per val fold, fixed)\n"
                elif use_pair_consistent:
                    cv_line = "cv_split: pairconsistent (AB/BA same fold)\n"
                else:
                    cv_line = "cv_split: skf (StratifiedKFold)\n"
                nfold_label = "6-Fold" if use_subcategory_6fold else "5-Fold"
                f.write(
                    f"network: {path_tag}\nsegment: 2s (full)\nsplit_method: {split_method}\n"
                    f"{cv_line}"
                )
                f.write(f"各Fold Balanced Accuracy: {fold_accs}\n")
                f.write(f"{nfold_label} CV Balanced Accuracy: {mean_acc:.6f} ± {std_acc:.6f}\n")

            np.savez(
                out_dir / f"{path_tag}_cm_{split_method}_2s_{split_tag}.npz",
                confusion_matrix=cm_agg,
                class_labels=unique_labels
            )

    return results


def _summary_row_is_valid(row: Dict[str, Any]) -> bool:
    """汇总表里该行是否视为已成功、可跳过重跑。"""
    if str(row.get("status", "")).strip().lower() != "ok":
        return False
    m = row.get("by_trial_mean")
    if m is None or pd.isna(m):
        return False
    try:
        mf = float(m)
    except (TypeError, ValueError):
        return False
    return np.isfinite(mf)


def _load_existing_summary_by_subject(summary_path: Path, network_tag: str) -> Dict[str, Dict[str, Any]]:
    if not summary_path.exists():
        return {}
    try:
        old = pd.read_csv(summary_path, encoding="utf-8-sig")
    except Exception:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for _, r in old.iterrows():
        if str(r.get("network", "")) != network_tag:
            continue
        out[str(r["subject"])] = r.to_dict()
    return out


def main():
    parser = argparse.ArgumentParser(
        description="EEGNet 2s CV：默认五折左右平衡 pair-consistent；"
        "可选 --subcategory-6fold 或 --stratified-kfold"
    )
    parser.add_argument(
        "--network", "-n",
        type=str,
        default="eegnet",
        choices=["eegnet", "eegnet_simple", "both"],
        help="默认仅 eegnet；both=依次 eegnet 与 eegnet_simple",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help=f"默认 {BASE_DATA_PATH}",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="汇总 CSV 目录；未指定时：默认 pairconsistent→eegnet_2s_cv5_pairconsistent_summary；"
        "--stratified-kfold→skf；--subcategory-6fold→subcat6；传 inline 则写脚本目录",
    )
    parser.add_argument(
        "--stratified-kfold",
        action="store_true",
        help="使用 StratifiedKFold 五折（无左右约束）；默认使用左右平衡",
    )
    parser.add_argument(
        "--write-per-subject-files",
        action="store_true",
        help="每人额外写入 txt/npz（默认关闭，仅一张汇总表）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="忽略已有汇总中的有效行，全员重跑",
    )
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument(
        "--subcategory-6fold",
        action="store_true",
        help="启用子类别无序对 6 折（benchmark 须 24 种；结果写入 eegnet_2s_subcat6fold_summary）",
    )
    parser.add_argument(
        "--pairconsistent-5fold",
        action="store_true",
        help="兼容旧参数（默认已启用左右平衡，可省略）",
    )
    args = parser.parse_args()

    if args.network == "both":
        networks_to_run = ["eegnet", "eegnet_simple"]
    else:
        networks_to_run = [args.network]

    if args.subcategory_6fold and args.pairconsistent_5fold:
        parser.error("不能同时指定 --subcategory-6fold 与 --pairconsistent-5fold")

    # 默认：五折左右平衡。--stratified-kfold → 分层五折；--subcategory-6fold → 子类别 6 折（与 SKF/左右平衡互斥）
    if args.stratified_kfold:
        use_subcategory_6fold = False
        use_pair_consistent = False
    else:
        use_subcategory_6fold = args.subcategory_6fold
        use_pair_consistent = not use_subcategory_6fold
    write_per_subject = args.write_per_subject_files

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = Path(args.data_root or BASE_DATA_PATH)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else list(SUBJECT_NAMES)

    if args.results_root == "inline":
        results_base: Optional[Path] = None
    elif args.results_root is not None:
        results_base = Path(args.results_root)
    else:
        if use_subcategory_6fold:
            results_base = Path(RESULTS_ROOT_SUBCAT6)
        else:
            results_base = Path(
                RESULTS_ROOT_PAIRCONSISTENT if use_pair_consistent else RESULTS_ROOT_SKF
            )

    print(f"使用设备: {device}")
    print(f"网络顺序: {networks_to_run}")
    print(f"样本: 2s (每 trial 一个)")
    print(f"数据根目录: {base_path}")
    if use_subcategory_6fold:
        split_desc = "子类别无序对 6 折（每折 val 4 种类型，固定可复现）"
    elif use_pair_consistent:
        split_desc = "左右平衡 pair-consistent"
    else:
        split_desc = "StratifiedKFold（无左右约束）"
    print(f"划分: {split_desc}")
    print(f"每人 txt/npz: {'开' if write_per_subject else '关（仅汇总 CSV）'}")
    print(f"断点续跑: {'关' if args.no_resume else '开（汇总里 ok 且 by_trial_mean 有效则跳过）'}")
    print(f"汇总目录: {results_base or 'inline：汇总 CSV 在脚本目录'}")
    print(f"被试列表: {len(subjects)} 人")
    print("=" * 60)

    if use_subcategory_6fold:
        split_suffix = "_subcat6fold"
    elif use_pair_consistent:
        split_suffix = "_pairconsistent"
    else:
        split_suffix = "_skf"

    for network_choice in networks_to_run:
        _, path_tag = get_network_class_and_tag(network_choice)
        print("\n" + "#" * 60)
        print(f"# 当前网络: {path_tag}")
        print("#" * 60)

        if use_subcategory_6fold:
            summary_name = f"{path_tag}_all_subjects_2s_subcat6fold_summary.csv"
        else:
            summary_name = f"{path_tag}_all_subjects_2s_cv5_summary{split_suffix}.csv"
        if results_base is not None:
            results_base.mkdir(parents=True, exist_ok=True)
            summary_path = results_base / summary_name
        else:
            summary_path = Path(__file__).parent / summary_name

        existing_by_subject = (
            {} if args.no_resume else _load_existing_summary_by_subject(summary_path, path_tag)
        )

        all_results = []
        for i, subject in enumerate(subjects):
            data_dir = base_path / subject

            if not args.no_resume:
                prev = existing_by_subject.get(subject)
                if prev is not None and _summary_row_is_valid(prev):
                    print(f"\n[{i + 1}/{len(subjects)}] {subject} | {path_tag} — 跳过（汇总已有有效结果）")
                    all_results.append(prev)
                    continue

            if write_per_subject and results_base is not None:
                res_sub = results_base / subject
                res_sub.mkdir(parents=True, exist_ok=True)
            elif write_per_subject and results_base is None:
                res_sub = data_dir
            else:
                res_sub = None

            print(f"\n[{i + 1}/{len(subjects)}] {subject} | {path_tag}")

            if not data_dir.exists():
                print(f"  跳过: 目录不存在")
                all_results.append({
                    'subject': subject,
                    'network': path_tag,
                    'random_mean': np.nan, 'random_std': np.nan,
                    'by_trial_mean': np.nan, 'by_trial_std': np.nan,
                    'status': '目录不存在'
                })
                continue

            try:
                res = run_subject_cv(
                    data_dir,
                    device,
                    verbose=True,
                    network_name=network_choice,
                    results_dir=res_sub,
                    use_pair_consistent=use_pair_consistent,
                    use_subcategory_6fold=use_subcategory_6fold,
                    write_per_subject_files=write_per_subject,
                )
            except Exception as e:
                print(f"  错误: {e}")
                all_results.append({
                    'subject': subject,
                    'network': path_tag,
                    'random_mean': np.nan, 'random_std': np.nan,
                    'by_trial_mean': np.nan, 'by_trial_std': np.nan,
                    'status': str(e)
                })
                continue

            if 'error' in res:
                print(f"  跳过: {res['error']}")
                all_results.append({
                    'subject': subject,
                    'network': path_tag,
                    'random_mean': np.nan, 'random_std': np.nan,
                    'by_trial_mean': np.nan, 'by_trial_std': np.nan,
                    'status': res['error']
                })
                continue

            all_results.append({
                'subject': subject,
                'network': path_tag,
                'random_mean': res.get('random_5fold', {}).get('mean', np.nan),
                'random_std': res.get('random_5fold', {}).get('std', np.nan),
                'by_trial_mean': res.get('by_trial_5fold', {}).get('mean', np.nan),
                'by_trial_std': res.get('by_trial_5fold', {}).get('std', np.nan),
                'status': 'ok'
            })

        df = pd.DataFrame(all_results)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print("\n" + "=" * 60)
        print(f"网络: {path_tag}，汇总已保存: {summary_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
