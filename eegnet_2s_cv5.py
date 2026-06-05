"""
EEGNet (Lawhern) 2秒样本五折交叉验证 - 跑所有被试
- 使用完整 2s 刺激段（3-5s），不切 1s 片段
- 数据与划分与 shallownet3 脚本一致，仅模型为 EEGNet要按论文再对一下层序和是否有 LSTM 等细节。
- 结果存：eegnet_*_2s_cv5_*
"""
import argparse
import json
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
from sklearn.model_selection import StratifiedKFold

from pair_consistent_splits import (
    PairBalanceError,
    pair_consistent_kfold_splits_hybrid,
    trial_pair_keys_pair_consistent_strict,
    verify_pair_consistent_splits_relaxed,
)

# 数据配置（与 eegnet 脚本一致）
SAMPLING_RATE = 250
BASE_DATA_PATH = 'A:/standard_data_interp_ica'
SUBJECT_NAMES = [
    'zhanghanglei', 'zhangyufei', 'jinxiaoyue', 'chenxianwei', 'yeziyuan', 'yanxingzhuo',
    'zhangzhiyao', 'haoxiang', 'hehaohuai', 'qiuhaiyun', 'zhouyu', 'honghaokai', 'caolulu',
    'yanyinsong', 'zengdexin', 'huanghaoxiang', 'xufan', 'liuzehao', 'jichengzhi', 'qiusiqi',
    'machenxiang', 'lizhuhang', 'zhangyajie',
]

# 结果文件名前缀（与 shallownet3 同目录、不同名）
PATH_TAG = 'eegnet'
SPLIT_METHODS = ['by_trial_5fold']
SEGMENT_SUFFIX = '_2s'


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
    fs_candidates = [info.get("fs"), info.get("sampling_rate"), info.get("sampling_rate_hz"), info.get("sfreq")]
    ch_candidates = [info.get("n_channels"), info.get("num_channels"), info.get("channels")]
    spt_candidates = [info.get("samples_per_trial"), info.get("n_samples_per_trial")]
    fs = override_fs if override_fs is not None else next((x for x in fs_candidates if x is not None), None)
    num_channels = override_channels if override_channels is not None else next((x for x in ch_candidates if x is not None), None)
    if isinstance(num_channels, list):
        num_channels = len(num_channels)
    samples_per_trial = next((x for x in spt_candidates if x is not None), None)
    if fs is None or num_channels is None:
        raise ValueError("trial_info.json 未提供采样率或通道数。")
    return DataMeta(float(fs), int(num_channels), int(samples_per_trial) if samples_per_trial is not None else None)


def reshape_trials(trials_2d: np.ndarray, num_channels: int) -> np.ndarray:
    if trials_2d.ndim == 3:
        return trials_2d
    if trials_2d.ndim != 2:
        raise ValueError(f"期望 2D 或 3D 数组，得到 {trials_2d.ndim}D")
    num_trials, flat_len = trials_2d.shape
    if flat_len % num_channels != 0:
        raise ValueError(f"无法按 {num_channels} 通道整除展平长度 {flat_len}")
    return trials_2d.reshape(num_trials, num_channels, flat_len // num_channels)


def exponential_moving_standardize(
    data: np.ndarray, alpha: float = 0.01, eps: float = 1e-8, init_block_size: Optional[int] = None
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
    if last_n <= 0 or data.shape[-1] < last_n:
        raise ValueError("截取秒数或数据长度异常")
    return data[..., -last_n:]


def apply_baseline_correction(
    data: np.ndarray, fs: float, baseline_start: float = 0.0, baseline_end: float = 4.0
) -> np.ndarray:
    n_trials, n_channels, n_timepoints = data.shape
    corrected_data = np.zeros_like(data)
    baseline_start_idx = int(round(baseline_start * fs))
    baseline_end_idx = int(round(baseline_end * fs))
    if baseline_start_idx >= baseline_end_idx:
        raise ValueError("基线窗口开始时间必须小于结束时间")
    for trial_idx in range(n_trials):
        for ch_idx in range(n_channels):
            baseline_window = data[trial_idx, ch_idx, baseline_start_idx:baseline_end_idx]
            corrected_data[trial_idx, ch_idx, :] = data[trial_idx, ch_idx, :] - np.mean(baseline_window)
    return corrected_data


def _get_fc_size(n_samples: int, f2: int, pool1: int, pool2: int) -> int:
    """Lawhern EEGNet Block3 池化后的时间维长度，用于 FC 输入维度。"""
    t = (n_samples // pool1) // pool2
    return f2 * 1 * max(1, t)


class EEGNet(nn.Module):
    """Lawhern et al. EEGNet：Block1 时间卷积 + Block2 深度卷积 + Block3 深度可分离卷积。"""

    def __init__(self, n_channels: int, n_samples: int, n_classes: int, f1: int = 8, f2: int = 8,
                 kernel_t: int = 41, kernel_3: int = 8, pool1: int = 4, pool2: int = 8, dropout: float = 0.25):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, f1, (1, kernel_t), padding=(0, kernel_t // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise_conv2 = nn.Conv2d(f1, f1, (n_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1)
        self.pool2_layer = nn.AvgPool2d((1, pool1))
        self.depthwise_conv3 = nn.Conv2d(f1, f1, (1, kernel_3), groups=f1, padding=(0, kernel_3 // 2), bias=False)
        self.pointwise_conv3 = nn.Conv2d(f1, f2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool3_layer = nn.AvgPool2d((1, pool2))
        fc_size = _get_fc_size(n_samples, f2, pool1, pool2)
        self.fc = nn.Linear(fc_size, n_classes)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def train_eegnet(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu',
                 class_weights: Optional[torch.Tensor] = None, patience=30
) -> Tuple[float, np.ndarray, np.ndarray]:
    """不保存模型，只记录 val 上最高准确率及对应的 y_true, y_pred。返回 (best_acc, y_true, y_pred)。"""
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
    pair_consistent: bool = False,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    trials_path, labels_path, info_path = find_data_files(data_dir)
    with info_path.open("r", encoding="utf-8") as f:
        info_json = json.load(f)
    trial_duration = info_json.get("trial_duration")
    try:
        meta = load_meta(info_path, fs, None)
        if verbose:
            print(f"  元信息: fs={meta.sampling_rate_hz}, 通道数={meta.num_channels}")
    except (FileNotFoundError, ValueError):
        trials = np.load(trials_path)
        inferred_channels = trials.shape[1] if trials.ndim == 3 else 128
        meta = DataMeta(sampling_rate_hz=fs, num_channels=inferred_channels, samples_per_trial=None)

    if trial_duration is None:
        trials = np.load(trials_path)
        trials_3d = reshape_trials(trials, meta.num_channels)
        trial_duration = trials_3d.shape[2] / meta.sampling_rate_hz

    trials = np.load(trials_path)
    labels_df = pd.read_csv(labels_path)
    labels = labels_df['Label'].to_numpy() if 'Label' in labels_df.columns else labels_df.iloc[:, 1].to_numpy() if labels_df.shape[1] >= 2 else labels_df.iloc[:, 0].to_numpy()
    n_before = len(labels)
    full_pair_keys: list = []
    if pair_consistent:
        full_pair_keys = trial_pair_keys_pair_consistent_strict(
            info_json, data_dir, n_before, verbose=verbose
        )
    trials_3d = reshape_trials(trials, meta.num_channels)
    data_full = select_last_seconds(trials_3d, meta.sampling_rate_hz, trial_duration)
    data_baseline_corrected = apply_baseline_correction(data_full, fs=meta.sampling_rate_hz, baseline_start=0.0, baseline_end=3.0)
    if trial_duration < 5.0:
        return {'error': f'数据时长{trial_duration}秒不足'}

    stimulus_start_idx = int(round(3.0 * meta.sampling_rate_hz))
    stimulus_end_idx = int(round(5.0 * meta.sampling_rate_hz))
    data = data_baseline_corrected[:, :, stimulus_start_idx:stimulus_end_idx]
    if data.shape[0] != labels.shape[0]:
        min_n = min(data.shape[0], labels.shape[0])
        data, labels = data[:min_n], labels[:min_n]
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
        data, labels = data[valid_mask], labels[valid_mask]
        full_pair_keys = [full_pair_keys[i] for i, m in enumerate(valid_mask) if m]
    trial_pair_keys = full_pair_keys
    n_trials = data.shape[0]
    if n_trials == 0:
        return {'error': '无可用trial'}

    if pair_consistent and verbose:
        ctr = Counter(trial_pair_keys)
        n_pair2 = sum(1 for k, c in ctr.items() if c == 2)
        n_other = len(ctr) - n_pair2
        if n_other > 0 or any(k and k[0] == "__singleton__" for k in ctr):
            print(
                f"  折分: 混合模式 — 恰 2 试次/刺激对的成对保持同折，"
                f"其余试次在五折间轮询均分（当前 {len(ctr)} 个不同 key，其中恰 2 次重复: {n_pair2}）"
            )

    # 使用完整 2s 刺激段，不切 1s
    data_2s = data  # (n_trials, n_channels, 500)
    labels_2s = labels
    trial_ids = np.arange(n_trials)
    unique_labels = np.unique(labels_2s)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y_encoded = np.array([label_to_idx[label] for label in labels_2s])
    n_classes = len(unique_labels)
    n_channels = data_2s.shape[1]
    n_timepoints = data_2s.shape[2]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = {}
    out_dir = results_dir if results_dir is not None else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_tag = "pairconsistent" if pair_consistent else "stratified"
    for split_method in SPLIT_METHODS:
        if pair_consistent:
            fold_splits = pair_consistent_kfold_splits_hybrid(
                trial_pair_keys, n_folds=n_folds, random_state=random_state
            )
            if not verify_pair_consistent_splits_relaxed(trial_pair_keys, fold_splits):
                raise PairBalanceError("内部校验失败：成对试次未完全同折。")
        elif split_method == 'random_5fold':
            fold_splits = list(skf.split(np.arange(len(data_2s)), y_encoded))
        else:
            fold_splits = list(skf.split(np.arange(n_trials), y_encoded))

        fold_accs_final = []   # 早停 best 模型在 val 折上的准确率
        fold_accs_max_epoch = []  # 训练过程中 val 折上的最高单代准确率
        cm_agg = np.zeros((n_classes, n_classes), dtype=np.int64)
        for fold, (train_idx, val_idx) in enumerate(fold_splits):
            if split_method == 'random_5fold':
                X_train = data_2s[train_idx]
                y_train = y_encoded[train_idx]
                X_val = data_2s[val_idx]
                y_val = y_encoded[val_idx]
            else:
                train_mask = np.isin(trial_ids, train_idx)
                val_mask = np.isin(trial_ids, val_idx)
                X_train = data_2s[train_mask]
                y_train = y_encoded[train_mask]
                X_val = data_2s[val_mask]
                y_val = y_encoded[val_mask]

            X_train_scaled = exponential_moving_standardize(X_train, alpha=0.01, eps=1e-8)
            X_val_scaled = exponential_moving_standardize(X_val, alpha=0.01, eps=1e-8)
            train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_train_scaled).unsqueeze(1), torch.LongTensor(y_train)),
                batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_val_scaled).unsqueeze(1), torch.LongTensor(y_val)),
                batch_size=batch_size, shuffle=False
            )

            model = EEGNet(
                n_channels=n_channels,
                n_samples=n_timepoints,
                n_classes=n_classes,
            ).to(device)
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
            acc_final = acc_max_epoch = best_acc
            fold_accs_final.append(acc_final)
            fold_accs_max_epoch.append(acc_max_epoch)
            cm_agg += confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))

        mean_final = np.mean(fold_accs_final)
        std_final = np.std(fold_accs_final)
        mean_max = np.mean(fold_accs_max_epoch)
        std_max = np.std(fold_accs_max_epoch)
        results[split_method] = {
            'mean_final': mean_final, 'std_final': std_final, 'fold_accs_final': fold_accs_final,
            'mean_max_epoch': mean_max, 'std_max_epoch': std_max, 'fold_accs_max_epoch': fold_accs_max_epoch,
            'confusion_matrix': cm_agg
        }
        if verbose:
            print(f"  {split_method} [final]:  {mean_final:.4f} ± {std_final:.4f}")
            print(f"  {split_method} [max]:   {mean_max:.4f} ± {std_max:.4f}")

        fname = f"{PATH_TAG}_results_3-5s_2s_cv5_{split_method}{SEGMENT_SUFFIX}_{split_tag}.txt"
        out_path = out_dir / fname
        with out_path.open("w", encoding="utf-8") as f:
            f.write(
                f"network: {PATH_TAG}\nsegment: 2s (full)\nsplit_method: {split_method}\n"
                f"cv_split: {split_tag} (左右平衡对同 fold)\n"
            )
            f.write(f"各Fold [final] (早停best模型在val折上的准确率): {fold_accs_final}\n")
            f.write(f"5-Fold CV [final]: {mean_final:.6f} ± {std_final:.6f}\n")
            f.write(f"各Fold [max] (val折上最高单代准确率): {fold_accs_max_epoch}\n")
            f.write(f"5-Fold CV [max]: {mean_max:.6f} ± {std_max:.6f}\n")
        np.savez(
            out_dir / f"{PATH_TAG}_cm_{split_method}{SEGMENT_SUFFIX}_{split_tag}.npz",
            confusion_matrix=cm_agg,
            class_labels=unique_labels
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="EEGNet (Lawhern) 2s 五折 CV（完整刺激段）")
    parser.add_argument("--data-root", type=str, default=None, help="被试数据父目录（默认 A:/standard_data_noica）")
    parser.add_argument("--results-root", type=str, default=None,
                        help="结果根目录；若设则写入 <results-root>/<subject>/，否则写入各被试数据目录")
    parser.add_argument("--pair-consistent", action="store_true",
                        help="左右平衡音频对（AB/BA）划分到同一 fold（非分层 StratifiedKFold）")
    parser.add_argument("--subjects", type=str, default=None,
                        help="逗号分隔被试 id；默认使用脚本内 SUBJECT_NAMES")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = Path(args.data_root or BASE_DATA_PATH)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else list(SUBJECT_NAMES)
    print(f"使用设备: {device}")
    print(f"网络: {PATH_TAG}")
    print(f"样本: 完整 2s 刺激段 (3-5s)")
    print(f"数据根目录: {base_path}")
    print(f"划分: {'pair-consistent (对同 fold)' if args.pair_consistent else 'StratifiedKFold'}")
    print("=" * 60)
    all_results = []
    for i, subject in enumerate(subjects):
        data_dir = base_path / subject
        res_sub = Path(args.results_root) / subject if args.results_root else None
        if res_sub is not None:
            res_sub.mkdir(parents=True, exist_ok=True)
        print(f"\n[{i + 1}/{len(subjects)}] {subject}")
        if not data_dir.exists():
            all_results.append({
                'subject': subject, 'by_trial_final_mean': np.nan, 'by_trial_final_std': np.nan,
                'by_trial_max_mean': np.nan, 'by_trial_max_std': np.nan, 'status': '目录不存在'
            })
            continue
        try:
            res = run_subject_cv(
                data_dir, device, verbose=True,
                pair_consistent=args.pair_consistent,
                results_dir=res_sub,
            )
        except Exception as e:
            all_results.append({
                'subject': subject, 'by_trial_final_mean': np.nan, 'by_trial_final_std': np.nan,
                'by_trial_max_mean': np.nan, 'by_trial_max_std': np.nan, 'status': str(e)
            })
            continue
        if 'error' in res:
            all_results.append({
                'subject': subject, 'by_trial_final_mean': np.nan, 'by_trial_final_std': np.nan,
                'by_trial_max_mean': np.nan, 'by_trial_max_std': np.nan, 'status': res['error']
            })
            continue
        by_trial = res.get('by_trial_5fold', {})
        all_results.append({
            'subject': subject,
            'by_trial_final_mean': by_trial.get('mean_final', np.nan),
            'by_trial_final_std': by_trial.get('std_final', np.nan),
            'by_trial_max_mean': by_trial.get('mean_max_epoch', np.nan),
            'by_trial_max_std': by_trial.get('std_max_epoch', np.nan),
            'status': 'ok'
        })
    df = pd.DataFrame(all_results)
    split_suffix = "_pairconsistent" if args.pair_consistent else "_stratified"
    summary_path = Path(__file__).parent / f"{PATH_TAG}_all_subjects_2s_cv5_summary{SEGMENT_SUFFIX}{split_suffix}.csv"
    if args.results_root:
        summary_path = Path(args.results_root) / summary_path.name
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 60)
    print(f"网络: {PATH_TAG}，2s 样本，所有人结果已保存至: {summary_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
