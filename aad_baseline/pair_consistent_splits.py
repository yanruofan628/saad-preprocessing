# -*- coding: utf-8 -*-
"""
左右平衡音频对（AB/BA）在五折中保持同一 fold，避免一对的两个呈现分属训练/验证。
与 eeg_audio_residual.py 中 build_pair_consistent_folds 一致：parse_pair_name → 无序键 tuple(sorted([audio_a, audio_b]))，
再打乱键后轮询分配到各 fold；输出 sklearn 式 (train_idx, val_idx) 列表。

「左右平衡」严格模式（trial_pair_keys_pair_consistent_strict）：
- 从 SUBJECTS_CONFIG 指定的 benchmark 读 wavfile；与 eeg_audio_residual / load_all_subjects_data 一致，
  **优先用 folder1/2/3 的 CSV 将短文件名映射为 original_name**（parse_pair_name 才能识别），
  无映射时再对 wavfile 直解析。
- benchmark 与 trial_times 对齐：**每个 session 内按试次顺序**对应 benchmark 第 1…N 条，不依赖 log 的 original_trial_num。
- 映射路径：环境变量 ATTENTION_SWITCH_AUDIO_MAPPING_ROOT（默认与 eeg_audio_residual 相同），
  或 trial_info['audio_mapping_files'] 为 3 个 CSV 路径列表。
- 校验：无 singleton；若不同无序刺激对数量 **大于 200** 则通过；否则须为偶数试次、n/2 个对、每对恰好 2 次。

非 pair-consistent 流程仍可使用 trial_pair_keys_from_trial_info（宽松，仅调试等场景）。

子类别 6 折（subcategory_unordered_key_from_pair_name、enumerate_subcategory_keys_from_benchmarks、
subcategory_pair_6fold_splits）：无序「左右子类别名」对与 benchmark 枚举 24 种一致时，按排序后固定 6 组×4 种做 val。
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from linear_regression_loudness_models import parse_experiment_data_txt, parse_pair_name
from subject_mff_benchmark_config import SUBJECTS_CONFIG

# 不同无序刺激对数大于该值时，不再要求「偶数试次 + 每键恰好 2 次」的强约束
PAIR_BALANCE_RELAX_THRESHOLD_DISTINCT_PAIRS = 200

# 子类别无序对（左右子类别名 sorted tuple）固定 6 折划分：24 种类型 → 每折 val 恰好 4 种
SUBCATEGORY_PAIR_TYPE_COUNT = 24
SUBCATEGORY_N_FOLDS = 6
SUBCATEGORY_VAL_TYPES_PER_FOLD = 4

# 与 extract_jiachen_2.BASE_DATA_DIRS 一致，用于定位 .mff 目录
_DEFAULT_MFF_SEARCH_ROOTS: Tuple[Path, ...] = (
    Path("A:/"),
    Path(r"A:\\"),
    Path("A:/standard_data_noica"),
)

# 与 eeg_audio_residual.MAPPING_BASE_DIR / MAPPING_FILES 一致；可 export ATTENTION_SWITCH_AUDIO_MAPPING_ROOT 覆盖
_DEFAULT_AUDIO_MAPPING_ROOT = os.environ.get(
    "ATTENTION_SWITCH_AUDIO_MAPPING_ROOT",
    r"D:\D\research\audioset下载\audio_pairs_2s\分组音频",
)


class PairBalanceError(ValueError):
    """左右平衡约束未满足，或无法仅从 benchmark 建立完整刺激对索引时抛出。"""


def assert_even_trial_count_for_pair_balance(n: int) -> None:
    """左右平衡要求试次数为偶数，以便分成 n/2 个无序对（每对 AB+BA 各 1 试次）。"""
    if n < 2:
        raise PairBalanceError("左右平衡至少需要 2 个试次。")
    if n % 2 != 0:
        raise PairBalanceError(
            f"左右平衡要求试次数为偶数（每对刺激 2 试次），当前 n={n}。"
        )


def validate_balanced_pair_keys(trial_pair_keys: List[Tuple]) -> None:
    """
    校验：不允许 singleton。
    若不同无序刺激对数量 > PAIR_BALANCE_RELAX_THRESHOLD_DISTINCT_PAIRS，则不再做强校验（不报错）；
    否则要求偶数试次、n/2 个对、每对恰好 2 次。
    """
    n = len(trial_pair_keys)
    for i, k in enumerate(trial_pair_keys):
        if len(k) > 0 and k[0] == "__singleton__":
            raise PairBalanceError(
                f"试次 {i + 1} 未归入有效刺激对（singleton），请检查 benchmark wavfile 与 trial_times 对齐。"
            )
    ctr = Counter(trial_pair_keys)
    n_distinct = len(ctr)
    if n_distinct > PAIR_BALANCE_RELAX_THRESHOLD_DISTINCT_PAIRS:
        return

    assert_even_trial_count_for_pair_balance(n)
    expected_pairs = n // 2
    if n_distinct != expected_pairs:
        raise PairBalanceError(
            f"左右平衡需要 {n} 个试次对应 {expected_pairs} 个不同无序刺激对（每对 2 试次），"
            f"当前不同键数={n_distinct}。"
        )
    bad = [(k, c) for k, c in ctr.items() if c != 2]
    if bad:
        k0, c0 = bad[0]
        raise PairBalanceError(
            f"每个无序刺激对应出现 2 次（AB/BA）；示例键 {k0!r} 出现了 {c0} 次（共 {len(bad)} 个键计数异常）。"
        )


def extract_pair_name_from_log_line(line: str) -> Optional[str]:
    """从 MFF log 行中提取音频对名字符串（含 main_/sub_/nn_* 前缀与 + 连接）。"""
    if not line or "+" not in line:
        return None
    # 按空白与常见分隔符切分，找恰好含一个 '+' 的 token
    for token in re.split(r'[\s,;|"<>\[\]]+', line):
        if token.count("+") != 1:
            continue
        t = token.strip().strip("'\"")
        if not t:
            continue
        if t.lower().endswith(".wav"):
            t = t[:-4]
        parsed = parse_pair_name(t)
        if parsed[0] is not None:
            return t
    # 回退：在行内搜索类似 xxx+yyy 的片段
    m = re.search(
        r"((?:nn_main_|nn_sub_|main_|sub_)[^\s+\\/]+\+(?:nn_main_|nn_sub_|main_|sub_)?[^\s+\\/]+)",
        line,
    )
    if m:
        t = m.group(1)
        if t.lower().endswith(".wav"):
            t = t[:-4]
        if parse_pair_name(t)[0] is not None:
            return t
    return None


def normalized_stimulus_pair_key(pair_name: str) -> Optional[Tuple[str, ...]]:
    """与 fusion 脚本相同：无序对 (audio_a, audio_b) 作为 key。"""
    pr = parse_pair_name(pair_name)
    if pr[0] is None:
        return None
    left_cat, left_id, right_cat, right_id, _ = pr
    audio_a = f"{left_cat}_{left_id}"
    audio_b = f"{right_cat}_{right_id}"
    return tuple(sorted([audio_a, audio_b]))


def canonical_pair_id_str(pair_name: str) -> Optional[str]:
    """
    无序刺激对唯一 ID（用于 240 平衡对合并与 CSV 列）。
    使用 ASCII 分隔符 \\x1e，避免与类别名中的下划线等冲突。
    """
    k = normalized_stimulus_pair_key(pair_name)
    if k is None:
        return None
    return f"{k[0]}\x1e{k[1]}"


def flip_playback_lr_to_canonical_ab(pair_name: str) -> Optional[int]:
    """
    将「播放左减右」声学差转为「词典序 canonical 先侧 A 减后侧 B」：A,B 为 sorted(左耳 clip, 右耳 clip)。

    若播放时左耳 clip 即为 canonical A → +1（Δ_playback 已是 A−B）。
    若播放时左耳 clip 为 canonical B → −1（需 Δ = −Δ_playback 得 A−B）。
    """
    pr = parse_pair_name(pair_name)
    if pr[0] is None:
        return None
    left_cat, left_id, right_cat, right_id, _ = pr
    left_audio = f"{left_cat}_{left_id}"
    right_audio = f"{right_cat}_{right_id}"
    a0, a1 = tuple(sorted([left_audio, right_audio]))
    if left_audio == a0:
        return 1
    if left_audio == a1:
        return -1
    return None


def subcategory_unordered_key_from_pair_name(pair_name: str) -> Optional[Tuple[str, str]]:
    """
    无序「子类别对」键：parse_pair_name 得到的左右类别名，排序后二元组。
    与左右平衡用的 normalized_stimulus_pair_key（含 clip id）不同，仅用于子类别 6 折划分。
    """
    pr = parse_pair_name(pair_name)
    if pr[0] is None:
        return None
    left_cat, _left_id, right_cat, _right_id, _exp = pr
    return tuple(sorted((left_cat, right_cat)))


def wavfile_to_pair_name(wavfile: Optional[str]) -> Optional[str]:
    """从 benchmark 的 wavfile 字段得到与 build_pair_consistent_folds 一致的 pair_name（无路径、无 .wav）。"""
    if not wavfile or not str(wavfile).strip():
        return None
    base = os.path.basename(str(wavfile).strip())
    if not base:
        return None
    stem = base[:-4] if base.lower().endswith(".wav") else base
    pr = parse_pair_name(stem)
    if pr[0] is None:
        pr = parse_pair_name(base)
    if pr[0] is None:
        return None
    return stem


def _parse_audio_mapping_custom(file_path: Path) -> Dict[str, str]:
    """
    与 eeg_audio_residual.parse_audio_mapping_custom 一致：
    短文件名（new_name）→ 完整立体声对名 original_name（main_...+...）。
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        mapping: Dict[str, str] = {}
        if "new_name" in df.columns and "original_name" in df.columns:
            for _, row in df.iterrows():
                if pd.isna(row["new_name"]) or pd.isna(row["original_name"]):
                    continue
                new_name = str(row["new_name"]).strip()
                original_name = str(row["original_name"]).strip()
                if not new_name or not original_name:
                    continue
                mapping[new_name] = original_name
                if new_name.lower().endswith(".wav"):
                    mapping[new_name[:-4]] = original_name
        elif len(df.columns) >= 2:
            col1, col2 = df.columns[0], df.columns[1]
            for _, row in df.iterrows():
                if pd.isna(row[col1]) or pd.isna(row[col2]):
                    continue
                original_name = str(row[col1]).strip()
                new_name = str(row[col2]).strip()
                if not original_name or not new_name:
                    continue
                mapping[new_name] = original_name
                if new_name.lower().endswith(".wav"):
                    mapping[new_name[:-4]] = original_name
        return mapping
    except Exception:
        return {}


def _default_audio_mapping_csv_paths() -> List[Path]:
    root = Path(_DEFAULT_AUDIO_MAPPING_ROOT)
    return [
        root / "folder1" / "file_mapping_folder1.csv",
        root / "folder2" / "file_mapping_folder2.csv",
        root / "folder3" / "file_mapping_folder3.csv",
    ]


def _load_session_mappings_for_pair_balance(
    info: Dict[str, Any], n_sessions: int, verbose: bool
) -> List[Dict[str, str]]:
    """每个 session（对应 extract 的一个 MFF/run）一个映射表，与 eeg_audio_residual.load_all_mappings 对应。"""
    custom = info.get("audio_mapping_files")
    if isinstance(custom, list) and len(custom) > 0:
        paths = [Path(str(p)) for p in custom]
    else:
        paths = _default_audio_mapping_csv_paths()
    out: List[Dict[str, str]] = []
    for i in range(n_sessions):
        p = paths[i] if i < len(paths) else None
        if p is not None and p.is_file():
            m = _parse_audio_mapping_custom(p)
            out.append(m)
            if verbose:
                print(f"  [左右平衡] 音频映射 session {i + 1}: {p}（{len(m)} 键）")
        else:
            out.append({})
            if verbose and p is not None:
                print(f"  [左右平衡] 映射文件缺失，wavfile 将尝试直解析: {p}")
    return out


def wavfile_to_pair_name_with_mapping(
    wavfile: Optional[str], mapping: Optional[Dict[str, str]]
) -> Optional[str]:
    """
    与 eeg_audio_residual.load_all_subjects_data 中逻辑一致：先映射到 original_name，再作为 pair_name。
    无映射或映射失败时回退到 wavfile_to_pair_name（wavfile 本身已是完整对名时可用）。
    """
    if not wavfile or not str(wavfile).strip():
        return None
    base = os.path.basename(str(wavfile).strip())
    stem = base[:-4] if base.lower().endswith(".wav") else base
    if mapping:
        orig = mapping.get(base) or mapping.get(stem)
        if not orig and base.lower().endswith(".wav"):
            orig = mapping.get(base[:-4])
        if orig:
            s = str(orig).strip()
            if s.lower().endswith(".wav"):
                s = s[:-4]
            pr = parse_pair_name(s)
            if pr[0] is not None:
                return s
    return wavfile_to_pair_name(wavfile)


def _find_mff_directory(mff_name: str, extra_roots: Optional[List[Path]] = None) -> Optional[Path]:
    roots: List[Path] = []
    if extra_roots:
        roots.extend(extra_roots)
    roots.extend(list(_DEFAULT_MFF_SEARCH_ROOTS))
    for root in roots:
        cand = root / mff_name
        if cand.is_dir():
            return cand
    return None


def _find_benchmark_in_mff_dir(mff_dir: Path) -> Optional[Path]:
    bms = sorted(mff_dir.glob("benchmark*.txt"))
    if not bms:
        return None
    return bms[0]


def _benchmark_filename_from_extract_config(
    subject_name: Optional[str], mff_name: str
) -> Optional[str]:
    """
    使用 subject_mff_benchmark_config.SUBJECTS_CONFIG 中列出的 (mff, benchmark) 文件名，
    与 extract 里 os.path.join(mff_dir, mff_config[\"benchmark\"]) 一致。
    """
    mff_name = str(mff_name).strip()
    if subject_name:
        sn = str(subject_name).strip()
        for cfg in SUBJECTS_CONFIG:
            if cfg.get("subject_name") != sn:
                continue
            for mff_cfg in cfg.get("mff_files") or []:
                if mff_cfg.get("mff") == mff_name:
                    return mff_cfg.get("benchmark")
    for cfg in SUBJECTS_CONFIG:
        for mff_cfg in cfg.get("mff_files") or []:
            if mff_cfg.get("mff") == mff_name:
                return mff_cfg.get("benchmark")
    return None


def _resolve_benchmark_path(
    mff_dir: Path,
    mff_name: str,
    subject_name: Optional[str],
    verbose: bool,
) -> Optional[Path]:
    """宽松模式：优先 SUBJECTS_CONFIG 中的文件名，否则 benchmark*.txt 字典序第一个。"""
    bname = _benchmark_filename_from_extract_config(subject_name, mff_name)
    if bname:
        cand = mff_dir / bname
        if cand.is_file():
            return cand
        if verbose:
            print(f"  [pair-consistent] 配置中的 benchmark 不存在，将尝试 glob: {cand}")
    return _find_benchmark_in_mff_dir(mff_dir)


def _resolve_benchmark_path_strict(
    mff_dir: Path,
    mff_name: str,
    subject_name: Optional[str],
) -> Path:
    """严格模式：仅使用 SUBJECTS_CONFIG 中的文件名，不做 glob。"""
    bname = _benchmark_filename_from_extract_config(subject_name, mff_name)
    if not bname:
        raise PairBalanceError(
            f"MFF「{mff_name}」未在 subject_mff_benchmark_config.SUBJECTS_CONFIG 中找到 benchmark 文件名，"
            f"请核对 subject_name={subject_name!r} 与 extract 配置是否一致。"
        )
    cand = mff_dir / bname
    if not cand.is_file():
        raise PairBalanceError(
            f"benchmark 文件不存在（配置要求路径）: {cand}"
        )
    return cand


def _format_mff_search_tried(mff_name: str, extra_roots: Optional[List[Path]]) -> str:
    roots: List[Path] = []
    if extra_roots:
        roots.extend(extra_roots)
    roots.extend(list(_DEFAULT_MFF_SEARCH_ROOTS))
    tried = [str(r / mff_name) for r in roots]
    return "; ".join(tried)


def _within_session_sequential_index(tt: List[Any], i: int) -> int:
    """
    trial_times 全局第 i 条 trial 在其 source_file 对应 session 内、按 extract 合并顺序的 0-based 下标。
    与同 session 内第几条 trial 对应 benchmark 的第几条记录（数量一致时顺序对齐），不依赖 log 的 original_trial_num。
    """
    if i < 0 or i >= len(tt) or not isinstance(tt[i], dict):
        return 0
    sf = int(tt[i].get("source_file") or 1)
    k = 0
    for j in range(i):
        rj = tt[j]
        if isinstance(rj, dict) and int(rj.get("source_file") or 1) == sf:
            k += 1
    return k


def load_trial_pair_names_strict(
    info: Dict[str, Any],
    data_dir: Path,
    n_trials: int,
    verbose: bool = False,
) -> List[str]:
    """
    仅从 SUBJECTS_CONFIG 指定的 benchmark 读取 wavfile，与 trial_times 对齐。
    对齐方式：每个 session（source_file）内，按 trial_times 中该 session 试次的先后顺序，
    依次对应 benchmark 解析结果的第 1、2、… 条（0-based 下标与 session 内顺序一致），
    不使用 original_trial_num。任一环节失败抛出 PairBalanceError（不回退）。
    """
    mff_files = info.get("mff_files") or []
    if not mff_files:
        raise PairBalanceError("trial_info 缺少 mff_files，无法定位 benchmark。")

    subject_name = info.get("subject_name") or info.get("data_name")
    extra_roots: List[Path] = [Path(data_dir), Path(data_dir).parent]

    session_mappings = _load_session_mappings_for_pair_balance(
        info, len(mff_files), verbose=verbose
    )

    session_pair_lists: List[List[str]] = []

    for session_idx, mff_name in enumerate(mff_files):
        mff_dir = _find_mff_directory(str(mff_name), extra_roots=extra_roots)
        if mff_dir is None:
            raise PairBalanceError(
                f"未找到 MFF 目录「{mff_name}」。已尝试: {_format_mff_search_tried(str(mff_name), extra_roots)}"
            )
        bench = _resolve_benchmark_path_strict(mff_dir, str(mff_name), subject_name)
        try:
            parsed = parse_experiment_data_txt(str(bench))
        except Exception as e:
            raise PairBalanceError(f"解析 benchmark 失败 {bench}: {e}") from e
        mapping = session_mappings[session_idx] if session_idx < len(session_mappings) else {}
        row_names: List[str] = []
        for ti, t in enumerate(parsed):
            w = t.get("wavfile")
            pn = wavfile_to_pair_name_with_mapping(w, mapping)
            if not pn:
                raise PairBalanceError(
                    f"{bench.name} 第 {ti + 1} 条 trial 的 wavfile 无法解析为刺激对（无映射或映射后仍非 main_/sub_+ 格式）: {w!r}"
                )
            row_names.append(pn)
        session_pair_lists.append(row_names)
        if verbose:
            print(
                f"  [左右平衡] session {session_idx + 1}: {bench.name} → {len(row_names)} 条 wavfile→pair_name"
            )

    tt = info.get("trial_times") or []
    if len(tt) < n_trials:
        raise PairBalanceError(
            f"trial_times 条目数 {len(tt)} 小于所需试次数 {n_trials}。"
        )

    out: List[str] = []
    for i in range(n_trials):
        if not isinstance(tt[i], dict):
            raise PairBalanceError(f"trial_times[{i}] 不是字典，无法对齐 benchmark。")
        row = tt[i]
        sf = int(row.get("source_file") or 1)
        if sf < 1 or sf > len(session_pair_lists):
            raise PairBalanceError(
                f"trial {i + 1}: source_file={sf} 超出 session 数 {len(session_pair_lists)}。"
            )
        plist = session_pair_lists[sf - 1]
        j = _within_session_sequential_index(tt, i)
        if j < 0 or j >= len(plist):
            raise PairBalanceError(
                f"trial {i + 1}: session {sf} 内顺序第 {j + 1} 条在 benchmark 中无对应条目（该 session 共 {len(plist)} 条）。"
            )
        out.append(plist[j])

    return out


def enumerate_subcategory_keys_from_benchmarks(
    info: Dict[str, Any],
    data_dir: Path,
    verbose: bool = False,
) -> List[Tuple[str, str]]:
    """
    遍历所有 session 的 benchmark，从每条 wavfile 解析无序子类别对，去重后排序返回。
    用于校验种类数是否为 24 及生成固定 6 折的 val 类型分组。
    """
    mff_files = info.get("mff_files") or []
    if not mff_files:
        raise PairBalanceError("trial_info 缺少 mff_files，无法枚举子类别。")

    subject_name = info.get("subject_name") or info.get("data_name")
    extra_roots: List[Path] = [Path(data_dir), Path(data_dir).parent]

    session_mappings = _load_session_mappings_for_pair_balance(
        info, len(mff_files), verbose=verbose
    )

    keys_set: set = set()
    for session_idx, mff_name in enumerate(mff_files):
        mff_dir = _find_mff_directory(str(mff_name), extra_roots=extra_roots)
        if mff_dir is None:
            raise PairBalanceError(
                f"[子类别枚举] 未找到 MFF 目录「{mff_name}」。已尝试: {_format_mff_search_tried(str(mff_name), extra_roots)}"
            )
        bench = _resolve_benchmark_path_strict(mff_dir, str(mff_name), subject_name)
        try:
            parsed = parse_experiment_data_txt(str(bench))
        except Exception as e:
            raise PairBalanceError(f"[子类别枚举] 解析 benchmark 失败 {bench}: {e}") from e
        mapping = session_mappings[session_idx] if session_idx < len(session_mappings) else {}
        for ti, t in enumerate(parsed):
            w = t.get("wavfile")
            pn = wavfile_to_pair_name_with_mapping(w, mapping)
            if not pn:
                raise PairBalanceError(
                    f"[子类别枚举] {bench.name} 第 {ti + 1} 条 wavfile 无法解析为 pair_name: {w!r}"
                )
            sk = subcategory_unordered_key_from_pair_name(pn)
            if sk is None:
                raise PairBalanceError(
                    f"[子类别枚举] {bench.name} 第 {ti + 1} 条无法解析子类别: pair_name={pn!r}"
                )
            keys_set.add(sk)
        if verbose:
            print(
                f"  [子类别枚举] session {session_idx + 1}: {bench.name} → "
                f"累计 {len(keys_set)} 种无序子类别对"
            )

    keys_sorted: List[Tuple[str, str]] = sorted(keys_set)
    return keys_sorted


def load_trial_subcategory_keys_strict(
    info: Dict[str, Any],
    data_dir: Path,
    n_trials: int,
    verbose: bool = False,
) -> List[Tuple[str, str]]:
    """与 trial_times 对齐的每条试次的无序子类别对键（与 load_trial_pair_names_strict 同序）。"""
    names = load_trial_pair_names_strict(info, data_dir, n_trials, verbose=verbose)
    out: List[Tuple[str, str]] = []
    for i, pn in enumerate(names):
        sk = subcategory_unordered_key_from_pair_name(pn)
        if sk is None:
            raise PairBalanceError(
                f"试次 {i + 1}: 无法从 pair_name 解析子类别无序对: {pn!r}"
            )
        out.append(sk)
    return out


def subcategory_pair_6fold_splits(
    trial_subcat_keys: List[Tuple[str, str]],
    keys_sorted: List[Tuple[str, str]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    确定性 6 折：keys_sorted 须为 benchmark 枚举并排序后的 24 个无序子类别对；
    第 f 折（0-based）验证集为第 4f…4f+3 种类型对应的全部试次，其余试次为训练集。
    """
    if len(keys_sorted) != SUBCATEGORY_PAIR_TYPE_COUNT:
        sample = keys_sorted[: min(8, len(keys_sorted))]
        raise PairBalanceError(
            f"子类别无序对种类数应为 {SUBCATEGORY_PAIR_TYPE_COUNT}，当前 benchmark 枚举得到 {len(keys_sorted)}。"
            f" 示例键: {sample}"
        )
    if len(set(keys_sorted)) != len(keys_sorted):
        raise PairBalanceError("子类别键列表存在重复，无法唯一分折。")

    n = len(trial_subcat_keys)
    universe = set(keys_sorted)
    for i, k in enumerate(trial_subcat_keys):
        if k not in universe:
            raise PairBalanceError(
                f"试次 {i + 1} 的子类别键 {k!r} 不在 benchmark 枚举的 {SUBCATEGORY_PAIR_TYPE_COUNT} 种内。"
            )

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n, dtype=np.int64)
    for f in range(SUBCATEGORY_N_FOLDS):
        block = keys_sorted[
            SUBCATEGORY_VAL_TYPES_PER_FOLD * f : SUBCATEGORY_VAL_TYPES_PER_FOLD * (f + 1)
        ]
        val_types = frozenset(block)
        val_idx = np.array(
            [i for i in range(n) if trial_subcat_keys[i] in val_types],
            dtype=np.int64,
        )
        train_idx = np.setdiff1d(all_idx, val_idx)
        splits.append((train_idx, val_idx))
    return splits


def trial_pair_keys_pair_consistent_strict(
    info: Dict[str, Any],
    data_dir: Path,
    n_trials: int,
    verbose: bool = False,
) -> List[Tuple]:
    """
    左右平衡专用：benchmark 严格加载 + 规范化键；折分建议用 pair_consistent_kfold_splits_hybrid + verify_pair_consistent_splits_relaxed。
    """
    names = load_trial_pair_names_strict(info, data_dir, n_trials, verbose=verbose)
    keys: List[Tuple] = []
    for i, pn in enumerate(names):
        nk = normalized_stimulus_pair_key(pn)
        if nk is None:
            raise PairBalanceError(
                f"试次 {i + 1}: pair_name 无法规范化为无序刺激键: {pn!r}"
            )
        keys.append(nk)
    return keys


def load_trial_pair_names_from_benchmarks(
    info: Dict[str, Any],
    data_dir: Optional[Path],
    n_trials: int,
    verbose: bool = False,
) -> List[Optional[str]]:
    """
    按 extract 合并顺序：各 session 的 parse_experiment_data_txt 与 trial_times 对齐，
    每个 session 内按试次先后顺序依次对应 benchmark 的第 1、2、… 条（与 original_trial_num 无关）。
    与 load_hanglei_trials / build_pair_consistent_folds 使用同一套 wavfile→parse_pair_name 语义。
    """
    out: List[Optional[str]] = [None] * n_trials
    mff_files = info.get("mff_files") or []
    if not mff_files:
        return out

    subject_name = info.get("subject_name") or info.get("data_name")

    extra_roots: List[Path] = []
    if data_dir is not None:
        dd = Path(data_dir)
        extra_roots.append(dd)
        extra_roots.append(dd.parent)

    session_mappings = _load_session_mappings_for_pair_balance(
        info, len(mff_files), verbose=verbose
    )

    # session_idx -> plist；第 i 条全局 trial 用 session 内顺序下标 j 取 plist[j]
    session_pair_lists: List[List[Optional[str]]] = []

    for session_idx, mff_name in enumerate(mff_files):
        mff_dir = _find_mff_directory(str(mff_name), extra_roots=extra_roots)
        if mff_dir is None:
            if verbose:
                print(f"  [pair-consistent] 未找到 MFF 目录: {mff_name}")
            session_pair_lists.append([])
            continue
        bench = _resolve_benchmark_path(mff_dir, str(mff_name), subject_name, verbose)
        if bench is None or not bench.is_file():
            if verbose:
                print(f"  [pair-consistent] 未找到 benchmark 文件: {mff_dir}")
            session_pair_lists.append([])
            continue
        try:
            parsed = parse_experiment_data_txt(str(bench))
        except Exception as e:
            if verbose:
                print(f"  [pair-consistent] 解析 benchmark 失败 {bench}: {e}")
            session_pair_lists.append([])
            continue
        mapping = session_mappings[session_idx] if session_idx < len(session_mappings) else {}
        plist: List[Optional[str]] = [
            wavfile_to_pair_name_with_mapping(t.get("wavfile"), mapping) for t in parsed
        ]
        session_pair_lists.append(plist)
        if verbose:
            n_ok = sum(1 for x in plist if x is not None)
            print(f"  [pair-consistent] session {session_idx + 1}: {bench.name} → {len(plist)} trials, {n_ok} 个可解析 pair_name")

    tt = info.get("trial_times") or []
    for i in range(n_trials):
        if i >= len(tt) or not isinstance(tt[i], dict):
            continue
        row = tt[i]
        sf = int(row.get("source_file") or 1)
        if sf < 1 or sf > len(session_pair_lists):
            continue
        plist = session_pair_lists[sf - 1]
        j = _within_session_sequential_index(tt, i)
        if 0 <= j < len(plist):
            out[i] = plist[j]

    return out


def trial_pair_keys_from_trial_info(
    info: Dict[str, Any],
    n_trials: int,
    data_dir: Optional[Path] = None,
    prefer_benchmark: bool = True,
    verbose: bool = False,
) -> List[Tuple]:
    """
    按 trial_info['trial_times'] 与 trials 行顺序对齐，返回每个 trial 的分组 key。
    优先使用 benchmark 中 wavfile 推导的 pair_name（与 build_pair_consistent_folds 一致），
    否则 trial_times[i]['pair_name']，再否则从 log 行提取。
    无法解析的 trial 使用独立 key (__singleton__, i)。
    """
    tt = info.get("trial_times") or []
    bench_names: List[Optional[str]] = [None] * n_trials
    if prefer_benchmark and data_dir is not None:
        bench_names = load_trial_pair_names_from_benchmarks(info, data_dir, n_trials, verbose=verbose)

    keys: List[Tuple] = []
    for i in range(n_trials):
        pn: Optional[str] = None
        if i < len(bench_names) and bench_names[i]:
            pn = bench_names[i]
        if not pn and i < len(tt) and isinstance(tt[i], dict):
            row = tt[i]
            raw_pn = row.get("pair_name")
            if raw_pn:
                pn = str(raw_pn).strip()
        if not pn and i < len(tt) and isinstance(tt[i], dict):
            line = str(tt[i].get("line") or "")
            pn = extract_pair_name_from_log_line(line)
        nk = normalized_stimulus_pair_key(pn) if pn else None
        if nk is None:
            keys.append(("__singleton__", i))
        else:
            keys.append(nk)
    return keys


def pair_consistent_kfold_splits(
    trial_pair_keys: List[Tuple],
    n_folds: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    与 AB balence.build_pair_consistent_folds 一致：
    先按无序刺激对 key 合并 trial，打乱 key 后轮询分配到各 fold，再生成 K 组 (train_idx, val_idx)。
    """
    n_trials = len(trial_pair_keys)
    key_to_indices: Dict[Tuple, List[int]] = {}
    for i, k in enumerate(trial_pair_keys):
        key_to_indices.setdefault(k, []).append(i)

    keys = list(key_to_indices.keys())
    rng = np.random.RandomState(random_state)
    rng.shuffle(keys)

    folds_dict = {i: [] for i in range(n_folds)}
    for j, k in enumerate(keys):
        folds_dict[j % n_folds].extend(key_to_indices[k])

    folds = [np.array(folds_dict[i], dtype=np.int64) for i in range(n_folds)]
    all_idx = np.arange(n_trials, dtype=np.int64)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        val_idx = folds[k]
        train_idx = np.setdiff1d(all_idx, val_idx)
        splits.append((train_idx, val_idx))
    return splits


def pair_consistent_kfold_splits_hybrid(
    trial_pair_keys: List[Tuple],
    n_folds: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    混合折分：无序刺激对 key 恰好出现 2 次的试次作为一整块放入同一折（保持 AB/BA 同折）；
    singleton、出现次数≠2 的 key 的各试次视为独立，打乱后在各 fold 间轮询分配，使尽量均匀。
    """
    n_trials = len(trial_pair_keys)
    key_to_indices: Dict[Tuple, List[int]] = defaultdict(list)
    for i, k in enumerate(trial_pair_keys):
        key_to_indices[k].append(i)

    balanced_units: List[List[int]] = []
    loose: List[int] = []
    for k, idxs in key_to_indices.items():
        if len(k) > 0 and k[0] == "__singleton__":
            loose.extend(idxs)
        elif len(idxs) == 2:
            balanced_units.append(idxs)
        else:
            loose.extend(idxs)

    rng = np.random.RandomState(random_state)
    rng.shuffle(balanced_units)
    folds_dict: Dict[int, List[int]] = {i: [] for i in range(n_folds)}
    for j, unit in enumerate(balanced_units):
        folds_dict[j % n_folds].extend(unit)

    loose_arr = np.array(loose, dtype=np.int64)
    rng.shuffle(loose_arr)
    for j, tid in enumerate(loose_arr):
        folds_dict[j % n_folds].append(int(tid))

    folds = [np.array(folds_dict[i], dtype=np.int64) for i in range(n_folds)]
    all_idx = np.arange(n_trials, dtype=np.int64)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        val_idx = folds[k]
        train_idx = np.setdiff1d(all_idx, val_idx)
        splits.append((train_idx, val_idx))
    return splits


def verify_pair_consistent_splits(
    trial_pair_keys: List[Tuple], splits: List[Tuple[np.ndarray, np.ndarray]]
) -> bool:
    """同一 pair key（非 singleton）的 trial 在每一折上应整组在 train 或整组在 val。"""
    key_to_idxs: Dict[Tuple, List[int]] = defaultdict(list)
    for i, k in enumerate(trial_pair_keys):
        if k[0] != "__singleton__":
            key_to_idxs[k].append(i)
    for _train_idx, val_idx in splits:
        val_set = set(int(x) for x in val_idx.tolist())
        for _k, idxs in key_to_idxs.items():
            n_in = sum(1 for i in idxs if i in val_set)
            if n_in not in (0, len(idxs)):
                return False
    return True


def verify_pair_consistent_splits_relaxed(
    trial_pair_keys: List[Tuple], splits: List[Tuple[np.ndarray, np.ndarray]]
) -> bool:
    """
    仅要求：同一无序刺激对 key 且恰好 2 个试次时，每折上 validation 要么包含这 2 个要么都不包含。
    singleton 或计数≠2 的 key 不校验（可跨折）。
    """
    key_to_idxs: Dict[Tuple, List[int]] = defaultdict(list)
    for i, k in enumerate(trial_pair_keys):
        if len(k) > 0 and k[0] == "__singleton__":
            continue
        key_to_idxs[k].append(i)
    for _train_idx, val_idx in splits:
        val_set = set(int(x) for x in val_idx.tolist())
        for _k, idxs in key_to_idxs.items():
            if len(idxs) != 2:
                continue
            n_in = sum(1 for i in idxs if i in val_set)
            if n_in not in (0, 2):
                return False
    return True
