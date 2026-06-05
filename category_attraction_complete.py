#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from collections import defaultdict
from scipy import stats
import statsmodels.api as sm
from scipy.stats import binomtest
import math
from subject_mff_benchmark_config import SUBJECTS_CONFIG

print("=== 大类吸引力分析工具 ===")
print("开始执行...")

# 统一字体样式：Arial, 18, normal
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.unicode_minus'] = False

CATEGORY_COLOR_MAP = {
    'music': '#9594C0',
    'speech': '#D16552',
    'High Ecology': '#92D5D0',
    'Low Ecology': '#BCE1A4'
}
DEFAULT_CATEGORY_COLOR = '#8D99AE'

# 大类柱状图横轴顺序（与文稿一致：Music, Speech, HE, LE）
MAIN_CATEGORY_BAR_ORDER = ["music", "speech", "High Ecology", "Low Ecology"]


def _order_attraction_by_main_bar_order(items):
    """按 MAIN_CATEGORY_BAR_ORDER 排列；未在表中的类别排在后面。"""
    if not items:
        return items
    by_cat = {it["category"]: it for it in items}
    out = []
    for c in MAIN_CATEGORY_BAR_ORDER:
        if c in by_cat:
            out.append(by_cat[c])
    for c, row in by_cat.items():
        if c not in MAIN_CATEGORY_BAR_ORDER:
            out.append(row)
    return out


# matplotlib 中 offset points：1 inch = 72 pt，1 inch = 2.54 cm
_PT_PER_CM = 72.0 / 2.54


def apply_unified_font_style(fig):
    """强制将整张图字体统一为 Arial 18 normal。"""
    for ax in fig.axes:
        text_items = [ax.title, ax.xaxis.label, ax.yaxis.label]
        text_items.extend(ax.get_xticklabels())
        text_items.extend(ax.get_yticklabels())
        for txt in text_items:
            txt.set_fontname('Arial')
            txt.set_fontsize(18)
            txt.set_fontweight('normal')

        legend = ax.get_legend()
        if legend is not None:
            for txt in legend.get_texts():
                txt.set_fontname('Arial')
                txt.set_fontsize(18)
                txt.set_fontweight('normal')
            legend_title = legend.get_title()
            if legend_title is not None:
                legend_title.set_fontname('Arial')
                legend_title.set_fontsize(18)
                legend_title.set_fontweight('normal')

# 新版23人数据路径配置（与预处理脚本保持一致）
BASE_DATA_DIRS = [r"A:\\", r"A:/standard_data_noica"]
# 每人 3 个 session：benchmark 文件名最后一位为 1/2/3 时，分别对应 folder1 / folder2 / folder3
MAPPING_ROOT = r"D:\D\research\audioset下载\audio_pairs_2s\分组音频"

# 小类名 -> 大类（与文稿表一致；键须与 original_name 里子类字符串完全一致）
SUBCATEGORY_TO_MAIN_CATEGORY = {
    "Baby cry, infant cry": "High Ecology",
    "Telephone bell ringing": "High Ecology",
    "Computer keyboard": "Low Ecology",
    "Helicopter": "Low Ecology",
    "Male speech, man speaking": "speech",
    "Female speech, woman speaking": "speech",
    "Bass drum": "music",
    "Sad music": "music",
}

# 解析 sub_ 左段时用「最长前缀匹配」；顺序按长度降序
_SUBCATEGORY_PARSE_ORDER = sorted(
    SUBCATEGORY_TO_MAIN_CATEGORY.keys(), key=len, reverse=True
)

def parse_experiment_data_txt(file_path):
    """解析TXT格式的实验数据文件"""
    print(f"正在解析TXT实验数据: {os.path.basename(file_path)}")
    
    trials = []
    # 尝试不同的编码方式
    encodings = ['utf-8', 'utf-16', 'gbk', 'gb2312', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"成功使用编码: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"无法使用任何编码读取文件: {file_path}")
    
    # 找到trial数据开始的位置
    for i, line in enumerate(lines):
        if line.strip().startswith('Level: 3') and i+1 < len(lines) and '*** LogFrame Start ***' in lines[i+1]:
            # 这是一个trial的开始
            trial_data = {}
            
            # 解析trial信息
            for j in range(i+1, min(i+20, len(lines))):
                current_line = lines[j].strip()
                
                if 'wavfile:' in current_line:
                    trial_data['wavfile'] = current_line.split('wavfile:')[1].strip()
                elif 'ImageDisplay1.RESP:' in current_line:
                    trial_data['response'] = int(current_line.split('ImageDisplay1.RESP:')[1].strip())
                elif '*** LogFrame End ***' in current_line:
                    break
            
            if 'wavfile' in trial_data and 'response' in trial_data:
                trials.append(trial_data)
    
    print(f"成功解析 {len(trials)} 个trial")
    return trials

def parse_experiment_data_csv(file_path):
    """解析CSV格式的实验数据文件"""
    print(f"正在解析CSV实验数据: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"成功读取 {len(df)} 行数据")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return []
    
    trials = []
    for _, row in df.iterrows():
        # 从CSV中提取信息
        audio_file = row['音频文件']
        choice = row['选择']
        
        # 将选择转换为数字：left=1, right=2
        if choice.lower() == 'left':
            response = 1
        elif choice.lower() == 'right':
            response = 2
        else:
            continue  # 跳过无效选择
        
        trials.append({
            'wavfile': audio_file,
            'response': response
        })
    
    print(f"成功解析 {len(trials)} 个trial")
    return trials

def parse_audio_mapping(file_path):
    """解析音频映射"""
    print(f"解析映射文件: {os.path.basename(file_path)}")
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        mapping = {}
        for _, row in df.iterrows():
            new_name = row['new_name']
            original_name = row['original_name']
            mapping[new_name] = original_name
        
        print(f"成功解析 {len(mapping)} 个文件映射")
        return mapping
    except Exception as e:
        print(f"读取映射文件失败: {e}")
        return {}

def parse_audio_filename(filename):
    """解析音频文件名。当前实验均为 sub_ 开头时，用小类最长前缀匹配 + 表驱动大类映射。"""
    try:
        name_without_ext = filename.replace(".wav", "").strip()

        if name_without_ext.startswith("sub_") or name_without_ext.startswith("SUB_"):
            ls, rs = parse_sub_pair_stimulus_name(name_without_ext)
            if ls and rs:
                return "sub", ls, rs
            return None, None, None

        if name_without_ext.startswith("nn_main_"):
            experiment_type = "nn_main"
            name_part = name_without_ext[8:]
        elif name_without_ext.startswith("nn_sub_"):
            experiment_type = "nn_sub"
            name_part = name_without_ext[7:]
        elif name_without_ext.startswith("main_"):
            experiment_type = "main"
            name_part = name_without_ext[5:]
        else:
            return None, None, None

        if "+" not in name_part:
            return None, None, None

        left_part, right_part = name_part.split("+", 1)

        left_underscore_pos = left_part.find("_")
        if left_underscore_pos == -1:
            return None, None, None
        left_category = left_part[:left_underscore_pos]

        right_underscore_pos = right_part.find("_")
        if right_underscore_pos == -1:
            return None, None, None
        right_category = right_part[:right_underscore_pos]

        return experiment_type, left_category, right_category

    except Exception as e:
        print(f"解析文件名失败 {filename}: {e}")
        return None, None, None

def get_main_category(sub_category):
    """根据子类别获取主类别（与 SUBCATEGORY_TO_MAIN_CATEGORY 一致）。"""
    if sub_category is None:
        return None
    return SUBCATEGORY_TO_MAIN_CATEGORY.get(sub_category.strip())


def parse_sub_pair_stimulus_name(name_without_ext):
    """
    解析 sub_ 开头的 original 文件名（无 .wav 后缀）。
    规则：+ 右侧，第一个 '_' 之前为右小类；左侧去掉 sub_ 后，用已知小类名做最长前缀匹配
    （兼容子类名中的逗号；不依赖「第几个下划线」以免与 YouTube id 段混淆）。
    返回 (left_sub, right_sub) 或 (None, None)。
    """
    if not name_without_ext or "+" not in name_without_ext:
        return None, None
    ne = name_without_ext.strip()
    if not (ne.startswith("sub_") or ne.startswith("SUB_")):
        return None, None
    plus = ne.index("+")
    left_rest = ne[4:plus]
    right_chunk = ne[plus + 1 :]
    right_us = right_chunk.split(".")[0] if "." in right_chunk else right_chunk
    u = right_us.find("_")
    right_sub = right_us[:u].strip() if u >= 0 else right_us.strip()

    left_sub = None
    for cand in _SUBCATEGORY_PARSE_ORDER:
        if left_rest == cand or left_rest.startswith(cand + "_"):
            left_sub = cand
            break
    if left_sub is None or not right_sub:
        return None, None
    if right_sub not in SUBCATEGORY_TO_MAIN_CATEGORY:
        return None, None
    return left_sub, right_sub

def analyze_session(data_file, mapping_file, file_type='txt'):
    """分析单个session的数据"""
    print(f"\n=== 分析Session: {os.path.basename(data_file)} ===")
    
    # 提取被试ID（规范化：去除数字后缀、空格、下划线）
    def extract_subject_id(path, file_type):
        try:
            if file_type == 'csv':
                base = os.path.splitext(os.path.basename(path))[0]
                # 先去除下划线后的数字部分（如 _1, _2），然后取第一部分
                raw_id = base.split('_')[0]
                # 如果文件名是 "Liu Yaorui2" 这样的格式（没有下划线），需要处理空格和数字
                if '_' not in base:
                    # 直接使用整个文件名作为 raw_id，后面会处理空格和数字
                    raw_id = base
            else:
                # txt 的被试名在父目录名
                parent = os.path.basename(os.path.dirname(path))
                raw_id = parent.split('_')[0]
            
            # 规范化ID：去除空格、去除数字后缀
            import re
            # 先去除空格（处理 "Liu Yaorui2" -> "LiuYaorui2"）
            cleaned_id = raw_id.replace(' ', '')
            # 去除所有数字后缀（如 jiachen1017 -> jiachen, aiwenkai2 -> aiwenkai, LiuYaorui2 -> LiuYaorui）
            cleaned_id = re.sub(r'\d+$', '', cleaned_id)
            return cleaned_id
        except Exception:
            return 'unknown'

    subject_id = extract_subject_id(data_file, file_type)

    # 解析数据
    if file_type == 'txt':
        trials = parse_experiment_data_txt(data_file)
    else:
        trials = parse_experiment_data_csv(data_file)
    
    mapping = parse_audio_mapping(mapping_file)
    
    # 分析每个trial
    session_results = []
    
    for trial in trials:
        wavfile = trial['wavfile']
        response = trial['response']
        
        # 从wavfile中提取文件名
        filename = os.path.basename(wavfile)
        
        # 获取原始文件名
        if filename not in mapping:
            continue
        
        original_filename = mapping[filename]
        
        # 解析文件名
        experiment_type, left_category, right_category = parse_audio_filename(original_filename)
        
        if experiment_type is None:
            continue
        
        # 获取主类别（小类名 -> 大类，见 SUBCATEGORY_TO_MAIN_CATEGORY）
        left_main = get_main_category(left_category)
        right_main = get_main_category(right_category)

        if left_main is None or right_main is None:
            continue

        result_data = {
            'experiment_type': experiment_type,
            'left_category': left_category,
            'right_category': right_category,
            'response': response,
            'original_filename': original_filename,
            'subject_id': subject_id,
            'left_main': left_main,
            'right_main': right_main,
        }

        session_results.append(result_data)
    
    print(f"成功分析 {len(session_results)} 个trial")
    return session_results

def calculate_attraction_with_counts(all_results, experiment_type):
    """计算吸引力并返回原始计数数据"""
    print(f"\n=== 计算{experiment_type}类型吸引力和原始计数 ===")
    
    # 只处理指定实验类型的数据
    filtered_results = [r for r in all_results if r['experiment_type'] == experiment_type]
    
    print(f"找到 {len(filtered_results)} 个{experiment_type}类型的trial")
    
    if len(filtered_results) == 0:
        return [], {}
    
    # 收集每个类别的原始选择数据
    category_counts = defaultdict(lambda: {'total': 0, 'selected': 0})
    
    for result in filtered_results:
        if experiment_type in ["main", "nn_main"]:
            left_category = result["left_main"]
            right_category = result["right_main"]
            if left_category == right_category:
                continue
        elif experiment_type == "sub":
            # 全部 sub 试次：按大类累计；允许跨大类、也允许同一大类内两小类
            left_category = result["left_main"]
            right_category = result["right_main"]
        else:
            left_category = result["left_category"]
            right_category = result["right_category"]
        
        response = result['response']
        
        # 对于每个类别，记录它被选择的次数和总出现次数
        if response == 1:  # 选择左声道
            category_counts[left_category]['total'] += 1
            category_counts[left_category]['selected'] += 1
            category_counts[right_category]['total'] += 1
            category_counts[right_category]['selected'] += 0
        else:  # 选择右声道
            category_counts[left_category]['total'] += 1
            category_counts[left_category]['selected'] += 0
            category_counts[right_category]['total'] += 1
            category_counts[right_category]['selected'] += 1
    
    # 计算吸引力和统计
    attraction_data = []
    
    for category, counts in category_counts.items():
        total = counts['total']
        selected = counts['selected']
        attraction = selected / total if total > 0 else 0.5
        
        attraction_data.append({
            'category': category,
            'attraction': attraction,
            'total_opportunities': total,
            'selected_times': selected
        })
    
    print(f"计算了 {len(attraction_data)} 个类别的吸引力")
    return attraction_data, dict(category_counts)


def calculate_attraction_high_low_ecology_pairs_only(all_results, experiment_type="main"):
    """
    仅统计「一大类为 High Ecology、另一大类为 Low Ecology」的 trial（无序），
    再按与 calculate_attraction_with_counts 相同规则累计选中次数。
    输出顺序固定为 High Ecology, Low Ecology，便于作图配色。
    """
    he, le = "High Ecology", "Low Ecology"
    if experiment_type not in ("main", "nn_main", "sub"):
        print(f"high_low_ecology_pairs_only 仅支持 main / nn_main / sub，收到: {experiment_type}")
        return [], {}

    filtered_results = [r for r in all_results if r.get("experiment_type") == experiment_type]
    pair_trials = []
    for r in filtered_results:
        lm, rm = r.get("left_main"), r.get("right_main")
        if lm is None or rm is None:
            continue
        if {lm, rm} == {he, le}:
            pair_trials.append(r)

    print(
        f"\n=== High–Low Ecology 配对 trial only（{experiment_type}）: "
        f"{len(pair_trials)} trials ==="
    )

    if not pair_trials:
        return [], {}

    category_counts = defaultdict(lambda: {"total": 0, "selected": 0})
    for result in pair_trials:
        left_category = result["left_main"]
        right_category = result["right_main"]
        response = result["response"]
        if response == 1:
            category_counts[left_category]["total"] += 1
            category_counts[left_category]["selected"] += 1
            category_counts[right_category]["total"] += 1
        else:
            category_counts[left_category]["total"] += 1
            category_counts[right_category]["total"] += 1
            category_counts[right_category]["selected"] += 1

    attraction_data = []
    for category in (he, le):
        if category not in category_counts:
            continue
        counts = category_counts[category]
        total = counts["total"]
        selected = counts["selected"]
        attraction = selected / total if total > 0 else 0.5
        attraction_data.append(
            {
                "category": category,
                "attraction": attraction,
                "total_opportunities": total,
                "selected_times": selected,
            }
        )

    print(f"计算了 {len(attraction_data)} 个类别（仅 HE–LE 试次）")
    return attraction_data, dict(category_counts)


def calculate_attraction(all_results, experiment_type):
    """计算吸引力（用于小提琴图）；每个点包含其配对大类，且排除自配对
    额外返回被试层面的平均得分 subject_points: [{category, subject_id, subject_score}]"""
    print(f"\n=== 计算{experiment_type}类型吸引力 ===")
    
    # 只处理指定实验类型的数据
    filtered_results = [r for r in all_results if r['experiment_type'] == experiment_type]
    
    print(f"找到 {len(filtered_results)} 个{experiment_type}类型的trial")
    
    if len(filtered_results) == 0:
        return []
    
    # 按刺激对分组
    stimulus_pairs = defaultdict(list)
    
    for result in filtered_results:
        if experiment_type in ['main', 'nn_main']:
            # 大类对比，使用主类别
            left_category = result['left_main']
            right_category = result['right_main']
        else:
            # 小类对比，使用子类别
            left_category = result['left_category']
            right_category = result['right_category']
        
        response = result['response']
        
        # 创建刺激对标识符（排序确保一致性）
        pair_key = tuple(sorted([left_category, right_category]))
        
        # 记录选择结果
        stimulus_pairs[pair_key].append({
            'left_category': left_category,
            'right_category': right_category,
            'response': response,
            'subject_id': result.get('subject_id', 'unknown'),
        })
    
    # 对于小提琴图，我们需要收集每个类别在所有刺激对中的选择数据
    # 同时记录该数据点对应的“配对类别”，并排除自配对
    attraction_points = []
    # 被试-大类聚合：统计每位被试在每个大类的选择比例（排除自配对）
    subject_category_counts = defaultdict(lambda: {'selected': 0, 'total': 0})
    
    for pair_key, trials in stimulus_pairs.items():
        cat1, cat2 = pair_key
        
        # 计算cat1的吸引力
        cat1_left_trials = [t for t in trials if t['left_category'] == cat1]
        cat1_right_trials = [t for t in trials if t['right_category'] == cat1]
        
        cat1_attraction_left = sum(1 for t in cat1_left_trials if t['response'] == 1) / len(cat1_left_trials) if cat1_left_trials else 0.5
        cat1_attraction_right = sum(1 for t in cat1_right_trials if t['response'] == 2) / len(cat1_right_trials) if cat1_right_trials else 0.5
        
        cat1_attraction = (cat1_attraction_left + cat1_attraction_right) / 2
        
        # 计算cat2的吸引力
        cat2_left_trials = [t for t in trials if t['left_category'] == cat2]
        cat2_right_trials = [t for t in trials if t['right_category'] == cat2]
        
        cat2_attraction_left = sum(1 for t in cat2_left_trials if t['response'] == 1) / len(cat2_left_trials) if cat2_left_trials else 0.5
        cat2_attraction_right = sum(1 for t in cat2_right_trials if t['response'] == 2) / len(cat2_right_trials) if cat2_right_trials else 0.5
        
        cat2_attraction = (cat2_attraction_left + cat2_attraction_right) / 2
        
        # 为每个类别添加一个数据点，并记录其配对的大类；排除自配对
        if cat1 != cat2:
            attraction_points.append({
                'category': cat1,
                'attraction': cat1_attraction,
                'pair_category': cat2
            })
            attraction_points.append({
                'category': cat2,
                'attraction': cat2_attraction,
                'pair_category': cat1
            })

            # 统计被试层面的数据
            for t in trials:
                subj = t.get('subject_id', 'unknown')
                # 对 cat1 作为目标类的计数
                if t['left_category'] == cat1:
                    subject_category_counts[(subj, cat1)]['total'] += 1
                    if t['response'] == 1:
                        subject_category_counts[(subj, cat1)]['selected'] += 1
                if t['right_category'] == cat1:
                    subject_category_counts[(subj, cat1)]['total'] += 1
                    if t['response'] == 2:
                        subject_category_counts[(subj, cat1)]['selected'] += 1
                # 对 cat2 作为目标类的计数
                if t['left_category'] == cat2:
                    subject_category_counts[(subj, cat2)]['total'] += 1
                    if t['response'] == 1:
                        subject_category_counts[(subj, cat2)]['selected'] += 1
                if t['right_category'] == cat2:
                    subject_category_counts[(subj, cat2)]['total'] += 1
                    if t['response'] == 2:
                        subject_category_counts[(subj, cat2)]['selected'] += 1
    
    # 直接返回包含配对信息的数据点
    attraction_data = attraction_points
    # 汇总被试层面的平均分
    subject_points = []
    for (subj, cat), cnt in subject_category_counts.items():
        if cnt['total'] > 0:
            subject_points.append({
                'subject_id': subj,
                'category': cat,
                'subject_score': cnt['selected'] / cnt['total']
            })
    
    print(f"生成了 {len(attraction_data)} 个吸引力数据点（用于小提琴图），{len(subject_points)} 个被试散点")
    return attraction_data, subject_points


ALPHA_FAMILY = 0.05


def _bonferroni_adjusted_p(p, k_tests):
    """Bonferroni 调整 p 值：min(1, p * k)。"""
    if k_tests is None or k_tests <= 0:
        return min(1.0, float(p))
    return min(1.0, float(p) * int(k_tests))


def _bonferroni_significant(p, k_tests, alpha=ALPHA_FAMILY):
    """Bonferroni：原始 p < alpha/k 等价于 adjusted_p < alpha。"""
    if k_tests is None or k_tests <= 0:
        return float(p) < alpha
    return float(p) < (alpha / float(k_tests))


def perform_statistical_tests(attraction_data_with_counts):
    """执行统计检验。

    多重比较：Bonferroni，两个独立家族——
    (1) 各类 vs 0.5 的二项检验，k = 类别数；
    (2) 类间两两 z 检验，k = n*(n-1)/2。
    """
    results = {
        "binomial_tests": [],
        "pairwise_comparisons": [],
    }

    n_cat = len(attraction_data_with_counts)
    k_binom = max(1, n_cat)
    k_pair = max(1, n_cat * (n_cat - 1) // 2) if n_cat >= 2 else 1

    # 1. 二项检验：每个类别 vs 0.5（家族大小 k_binom）
    for item in attraction_data_with_counts:
        category = item["category"]
        selected = item["selected_times"]
        total = item["total_opportunities"]
        attraction = item["attraction"]

        if attraction > 0.5:
            result = binomtest(selected, total, p=0.5, alternative="greater")
        else:
            result = binomtest(selected, total, p=0.5, alternative="less")

        p_value = float(result.pvalue)
        p_adj = _bonferroni_adjusted_p(p_value, k_binom)
        sig = _bonferroni_significant(p_value, k_binom)

        results["binomial_tests"].append(
            {
                "category": category,
                "p_value": p_value,
                "p_value_bonferroni": p_adj,
                "k_bonferroni_family": k_binom,
                "significant": sig,
            }
        )

    # 2. 成对比较：两个独立比例的 z 检验（家族大小 k_pair）
    for i in range(n_cat):
        for j in range(i + 1, n_cat):
            cat1 = attraction_data_with_counts[i]
            cat2 = attraction_data_with_counts[j]

            n1 = cat1["total_opportunities"]
            k1 = cat1["selected_times"]
            n2 = cat2["total_opportunities"]
            k2 = cat2["selected_times"]

            p1 = k1 / n1
            p2 = k2 / n2

            p_combined = (k1 + k2) / (n1 + n2)
            se = math.sqrt(p_combined * (1 - p_combined) * (1 / n1 + 1 / n2))

            if se > 0:
                z = (p1 - p2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            else:
                p_value = 1.0

            p_value = float(p_value)
            p_adj = _bonferroni_adjusted_p(p_value, k_pair)
            sig = _bonferroni_significant(p_value, k_pair)

            results["pairwise_comparisons"].append(
                {
                    "category1": cat1["category"],
                    "category2": cat2["category"],
                    "p_value": p_value,
                    "p_value_bonferroni": p_adj,
                    "k_bonferroni_family": k_pair,
                    "significant": sig,
                }
            )

    print(
        f"[Bonferroni] 二项检验家族 k={k_binom}（阈值 p < {ALPHA_FAMILY/k_binom:.4f}）；"
        f"两两比较家族 k={k_pair}（阈值 p < {ALPHA_FAMILY/k_pair:.4f}）。"
    )
    return results

def plot_results_with_stats(
    attraction_data_with_counts, output_dir, experiment_type, show_pairwise_comparisons=True
):
    """绘制带统计检验的柱状图。

    show_pairwise_comparisons: 同一批 forced-choice 试次下两类比例互补（和为 1）时，
    类间「独立双比例 z 检验」不适用，应设为 False，仅保留相对 chance 的二项检验。
    显著性：二项与两两比较各自采用 Bonferroni（家族内 alpha=0.05）。
    """
    print(f"\n=== 绘制{experiment_type}类型结果（带统计检验） ===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if not attraction_data_with_counts:
        print("没有数据可以绘制")
        return

    attraction_data_with_counts = _order_attraction_by_main_bar_order(
        list(attraction_data_with_counts)
    )

    # 执行统计检验（顺序与柱顺序一致）
    stats_results = perform_statistical_tests(attraction_data_with_counts)
    
    if show_pairwise_comparisons:
        print(f"\n成对比较结果：")
        print(f"总比较数: {len(stats_results['pairwise_comparisons'])}")
        significant_count = sum(
            1 for c in stats_results["pairwise_comparisons"] if c["significant"]
        )
        print(f"显著比较数: {significant_count}")
    
    # 转换为DataFrame
    df = pd.DataFrame(attraction_data_with_counts)
    
    # 设置图形风格
    sns.set_theme(style="whitegrid")
    
    # 创建柱状图（高度与 plot_rt_excluded_violin.py 一致）
    fig_width_cm = 12.42
    fig_height_cm = 7.84
    plt.figure(figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54))
    
    categories = df['category'].values
    attractions = df['attraction'].values
    
    ax = plt.gca()
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")

    bar_colors = [
        CATEGORY_COLOR_MAP.get(str(c), DEFAULT_CATEGORY_COLOR) for c in categories
    ]
    x_positions = np.arange(len(categories)) * 0.32
    bars = ax.bar(x_positions, attractions, 
                  color=bar_colors,
                  alpha=0.85, edgecolor='black', linewidth=1.2, width=0.14)
    
    # 添加数值标签和统计显著性
    y_max = 0
    for i, (bar, category, attraction) in enumerate(zip(bars, categories, attractions)):
        height = bar.get_height()
        y_max = max(y_max, height)
        label_y = height + 0.012
        label_x = bar.get_x() + bar.get_width() / 2.
        
        # 数值标签
        ax.text(label_x, label_y,
                f'{height:.3f}', ha='center', va='bottom', fontsize=13, fontweight='normal')
        
        # 二项检验显著性
        binom_test = stats_results['binomial_tests'][i]
        if binom_test['significant']:
            # 灰色星号：右 0.3 cm、下 0.2 cm（相对锚点再上移 0.1 cm）
            ax.annotate(
                '*',
                xy=(label_x + bar.get_width() * 0.3, label_y - 0.01),
                xytext=(0.3 * _PT_PER_CM, -0.2 * _PT_PER_CM),
                textcoords='offset points',
                ha='left',
                va='bottom',
                fontsize=20,
                color='gray',
                fontweight='normal',
            )
    
    # 两两比较：不再逐对连线；若全部 Bonferroni 显著，则横跨所有柱画一条大括号 + 一句说明
    n = len(categories)
    max_annotation_height = y_max
    tick_len = 0.015
    if show_pairwise_comparisons and n >= 2:
        pair_list = stats_results.get("pairwise_comparisons", [])
        all_pairwise_sig = bool(pair_list) and all(
            c.get("significant", False) for c in pair_list
        )
        if all_pairwise_sig:
            x_left = bars[0].get_x() + bars[0].get_width() / 2.0
            x_right = bars[-1].get_x() + bars[-1].get_width() / 2.0
            if x_left > x_right:
                x_left, x_right = x_right, x_left
            # 黑色线段：按纵轴 [0, 1.0] 跨度换算 (0.3+0.2) cm 为数据坐标
            ylim_span_data = 1.0 - 0.0
            bracket_line_lift = ((0.3 + 0.2) / fig_height_cm) * ylim_span_data
            bracket_y = y_max + 0.10 + bracket_line_lift
            ax.plot(
                [x_left, x_right],
                [bracket_y, bracket_y],
                color="black",
                linewidth=1.5,
            )
            ax.plot(
                [x_left, x_left],
                [bracket_y, bracket_y - tick_len],
                color="black",
                linewidth=1.5,
            )
            ax.plot(
                [x_right, x_right],
                [bracket_y, bracket_y - tick_len],
                color="black",
                linewidth=1.5,
            )
            mid_x = (x_left + x_right) / 2.0
            # 文案在锚点基础上再上移 0.3 cm（含此前 0.2 cm + 本次 0.1 cm）
            ax.annotate(
                "All pairwise p < 0.05 (Bonf.)",
                xy=(mid_x, bracket_y + 0.018),
                xytext=(0, 0.3 * _PT_PER_CM),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="normal",
            )
            max_annotation_height = max(max_annotation_height, bracket_y + 0.12)
    
    # Chance 线于网格之后绘制（虚线、更粗），见下方 ax.axhline

    # 美化图表（不显示横轴标题 Category；保留 Music/Speech 等刻度标签）
    plt.ylabel('Average Attraction', fontsize=14, fontweight='normal')

    upper_limit = 1.0
    ax.set_ylim(0.0, upper_limit)
    y_ticks = np.arange(0.0, upper_limit + 1e-9, 0.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks])
    # 内框网格：横纵均为虚线
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="--",
        linewidth=0.9,
        alpha=0.45,
        color="#A8A8A8",
    )
    # 0.5 chance：虚线、更粗
    ax.axhline(
        y=0.5,
        color="#9E9E9E",
        linestyle="--",
        alpha=1.0,
        linewidth=2.4,
        zorder=10,
    )
    display_labels = [
        "Music"
        if c == "music"
        else "Speech"
        if c == "speech"
        else "HE"
        if c == "High Ecology"
        else "LE"
        if c == "Low Ecology"
        else c
        for c in categories
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels, rotation=0, ha='center')
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_fontweight('normal')
    for label in ax.get_yticklabels():
        label.set_fontweight('normal')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{experiment_type}_category_attraction_bar.svg')
    apply_unified_font_style(plt.gcf())
    # 仅缩小纵轴刻度字号（比全局小2号）
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.show()
    
    print(f"柱状图已保存: {output_path}")

def plot_results(attraction_data_with_pair, output_dir, experiment_type):
    
    # 将输入数据转换为DataFrame
    # attraction_data_with_pair 仅含配对点；同时尝试从同名函数返回中获取被试散点
    if isinstance(attraction_data_with_pair, tuple):
        attraction_data, subject_points = attraction_data_with_pair
    else:
        attraction_data = attraction_data_with_pair
        subject_points = []
    df = pd.DataFrame(attraction_data)
    # 图B：刺激对吸引力分布小提琴图
    # 根据类别数量调整图形大小
    unique_categories = df['category'].nunique()
    if unique_categories > 8:
        fig_width = max(18, unique_categories * 1.2)
        fig_height = 10
    else:
        fig_width = 14
        fig_height = 9

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=150)
    
    # 创建小提琴图
    category_palette = {cat: CATEGORY_COLOR_MAP.get(cat, DEFAULT_CATEGORY_COLOR) for cat in df['category'].unique()}
    vp = sns.violinplot(data=df, x='category', y='attraction', 
                       inner='box', palette=category_palette, saturation=0.9, ax=ax)
    
    # 定义配对大类的颜色映射（与柱状图一致）
    pair_category_colors = {cat: CATEGORY_COLOR_MAP.get(cat, DEFAULT_CATEGORY_COLOR) for cat in CATEGORY_COLOR_MAP}
    
    # 为每个点设置颜色
    if 'pair_category' in df.columns:
        df['point_color'] = df['pair_category'].map(pair_category_colors).fillna(DEFAULT_CATEGORY_COLOR)
    else:
        df['point_color'] = DEFAULT_CATEGORY_COLOR  # 默认颜色
    
    # 在小提琴图上叠加原始数据点，按配对大类着色
    sns.stripplot(data=df, x='category', y='attraction', hue='pair_category',
                  palette=pair_category_colors, alpha=0.85, jitter=True, dodge=False, 
                  size=6, linewidth=1.5, edgecolor='black', legend=False, ax=ax)

    # 注意：被试散点已移至独立图，此处不再叠加
    
    # 添加Chance Level参考线
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Chance Level (0.5)')
    
    # 美化图表
    ax.set_title('Stimulus Pair Attraction Distribution', fontsize=16, fontweight='bold', pad=24)
    ax.set_xlabel('Category', fontsize=14, fontweight='bold')
    ax.set_ylabel('Attraction', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12, loc='upper right', frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    fig.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.25)
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{experiment_type}_category_attraction_violin.png')
    apply_unified_font_style(fig)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"小提琴图已保存: {output_path}")
    
    # 返回subject_points供独立图使用
    return subject_points

def plot_subject_level_results(subject_points, output_dir, experiment_type):
    """绘制被试水平分析图"""
    if len(subject_points) == 0:
        print("没有被试水平数据")
        return
    
    df_subjects = pd.DataFrame(subject_points)
    df_subjects['subject_id'] = df_subjects['subject_id'].astype(str)
    
    # 定义9位被试的固定配色
    subject_colors = {
        'liyanchen': '#1f77b4',
        'shimin': '#ff7f0e',
        'jiachen': '#2ca02c',
        'aiwenkai': '#d62728',
        'lironghua': '#9467bd',
        'mayunmiao': '#8c564b',
        'ShangZiyang': '#e377c2',
        'wjy': '#7f7f7f',
        'LiuYaorui': '#bcbd22',
        'Liu Yaorui': '#bcbd22'
    }
    
    plt.figure(figsize=(12, 8))
    
    # 1. 背景：被试水平的总体分布（箱线图）
    sns.boxplot(data=df_subjects, x='category', y='subject_score', 
               width=0.7, palette="pastel", saturation=0.4, fliersize=0,
               showcaps=True, whis=[0, 100],
               boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 2},
               medianprops={'color': 'black', 'linewidth': 2},
               whiskerprops={'color': 'black', 'linewidth': 2},
               capprops={'color': 'black', 'linewidth': 2})
    
    # 2. 个体轨迹线
    sns.lineplot(data=df_subjects, x='category', y='subject_score',
                units='subject_id', estimator=None,
                hue='subject_id', palette=subject_colors,
                alpha=0.4, linewidth=1)
    
    # 3. 个体数据点
    sns.stripplot(data=df_subjects, x='category', y='subject_score',
                 hue='subject_id', palette=subject_colors,
                 jitter=True, size=5, alpha=0.7, edgecolor='black', linewidth=0.8)
    
    # 4. 总体趋势线（基于被试均值）
    sns.lineplot(data=df_subjects, x='category', y='subject_score',
                estimator='mean', errorbar='sd',
                color='blue', linewidth=3, marker='o', markersize=8)
    
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.title('Individual Patterns and Group-Level Effects\n(All analysis at subject level)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Category', fontsize=14, fontweight='bold')
    plt.ylabel('Subject-Level Attraction Score', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.legend(title='Subject', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=0, ha='center')
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{experiment_type}_category_attraction_subjects.png')
    apply_unified_font_style(plt.gcf())
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"被试水平图已保存: {output_path}")

def calculate_subcategory_attraction(all_results, experiment_type):
    """计算小类吸引力（sub或nn_sub），返回包含主类别信息的数据"""
    print(f"\n=== 计算{experiment_type}类型小类吸引力 ===")
    
    # 只处理指定实验类型的数据
    filtered_results = [r for r in all_results if r['experiment_type'] == experiment_type]
    
    print(f"找到 {len(filtered_results)} 个{experiment_type}类型的trial")
    
    if len(filtered_results) == 0:
        return [], []
    
    # 收集每个小类的原始选择数据（包含主类别信息）
    subcategory_counts = defaultdict(lambda: {'total': 0, 'selected': 0})
    # 被试-小类聚合
    subject_subcategory_counts = defaultdict(lambda: {'selected': 0, 'total': 0})
    
    for result in filtered_results:
        # 小类对比，使用子类别
        left_subcategory = result['left_category']
        right_subcategory = result['right_category']
        left_main = result.get('left_main')
        right_main = result.get('right_main')
        subject_id = result.get('subject_id', 'unknown')
        response = result['response']
        
        # 对于每个小类，记录它被选择的次数和总出现次数
        if response == 1:  # 选择左声道
            subcategory_counts[left_subcategory]['total'] += 1
            subcategory_counts[left_subcategory]['selected'] += 1
            subcategory_counts[right_subcategory]['total'] += 1
            subcategory_counts[right_subcategory]['selected'] += 0
            
            # 被试水平统计
            subject_subcategory_counts[(subject_id, left_subcategory)]['total'] += 1
            subject_subcategory_counts[(subject_id, left_subcategory)]['selected'] += 1
            subject_subcategory_counts[(subject_id, right_subcategory)]['total'] += 1
        else:  # 选择右声道
            subcategory_counts[left_subcategory]['total'] += 1
            subcategory_counts[left_subcategory]['selected'] += 0
            subcategory_counts[right_subcategory]['total'] += 1
            subcategory_counts[right_subcategory]['selected'] += 1
            
            # 被试水平统计
            subject_subcategory_counts[(subject_id, left_subcategory)]['total'] += 1
            subject_subcategory_counts[(subject_id, right_subcategory)]['total'] += 1
            subject_subcategory_counts[(subject_id, right_subcategory)]['selected'] += 1
    
    # 计算吸引力和统计
    attraction_data = []
    for subcategory, counts in subcategory_counts.items():
        total = counts['total']
        selected = counts['selected']
        attraction = selected / total if total > 0 else 0.5
        
        # 获取该小类所属的主类别
        main_category = get_main_category(subcategory)
        
        attraction_data.append({
            'sub_category': subcategory,
            'main_category': main_category,
            'attraction_score': attraction,
            'total_opportunities': total,
            'selected_times': selected
        })
    
    # 汇总被试层面的平均分
    subject_points = []
    for (subj, subcat), cnt in subject_subcategory_counts.items():
        if cnt['total'] > 0:
            main_cat = get_main_category(subcat)
            subject_points.append({
                'subject_id': subj,
                'sub_category': subcat,
                'main_category': main_cat,
                'subject_score': cnt['selected'] / cnt['total']
            })
    
    print(f"计算了 {len(attraction_data)} 个小类的吸引力")
    return attraction_data, subject_points

def perform_subcategory_statistical_tests(attraction_data):
    """对小类吸引力数据执行统计检验（按主类别分组）。

    多重比较：在每个主类别内，Bonferroni 两个独立家族——
    (1) 各小类 vs 0.5，k = 该主类下小类个数；
    (2) 小类两两 z 检验，k = m*(m-1)/2。
    """
    results = {}

    df = pd.DataFrame(attraction_data)
    df = df[df["main_category"].notna()].copy()

    for main_cat in df["main_category"].unique():
        df_main = df[df["main_category"] == main_cat].copy()
        main_results = {"binomial_tests": [], "pairwise_comparisons": []}

        m = len(df_main)
        k_binom = max(1, m)
        k_pair = max(1, m * (m - 1) // 2) if m >= 2 else 1

        for _, row in df_main.iterrows():
            subcategory = row["sub_category"]
            selected = row["selected_times"]
            total = row["total_opportunities"]
            attraction = row["attraction_score"]

            if attraction > 0.5:
                result = binomtest(selected, total, p=0.5, alternative="greater")
            else:
                result = binomtest(selected, total, p=0.5, alternative="less")

            p_value = float(result.pvalue)
            p_adj = _bonferroni_adjusted_p(p_value, k_binom)
            sig = _bonferroni_significant(p_value, k_binom)

            main_results["binomial_tests"].append(
                {
                    "sub_category": subcategory,
                    "p_value": p_value,
                    "p_value_bonferroni": p_adj,
                    "k_bonferroni_family": k_binom,
                    "significant": sig,
                }
            )

        for i in range(m):
            for j in range(i + 1, m):
                subcat1 = df_main.iloc[i]
                subcat2 = df_main.iloc[j]

                n1 = subcat1["total_opportunities"]
                k1 = subcat1["selected_times"]
                n2 = subcat2["total_opportunities"]
                k2 = subcat2["selected_times"]

                p1 = k1 / n1
                p2 = k2 / n2

                p_combined = (k1 + k2) / (n1 + n2)
                se = math.sqrt(p_combined * (1 - p_combined) * (1 / n1 + 1 / n2))

                if se > 0:
                    z = (p1 - p2) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                else:
                    p_value = 1.0

                p_value = float(p_value)
                p_adj = _bonferroni_adjusted_p(p_value, k_pair)
                sig = _bonferroni_significant(p_value, k_pair)

                main_results["pairwise_comparisons"].append(
                    {
                        "sub_category1": subcat1["sub_category"],
                        "sub_category2": subcat2["sub_category"],
                        "p_value": p_value,
                        "p_value_bonferroni": p_adj,
                        "k_bonferroni_family": k_pair,
                        "significant": sig,
                    }
                )

        print(
            f"[Bonferroni] {main_cat}: 二项 k={k_binom}（p < {ALPHA_FAMILY/k_binom:.4f}）；"
            f"两两 k={k_pair}（p < {ALPHA_FAMILY/k_pair:.4f}）。"
        )
        results[main_cat] = main_results

    return results

def plot_subcategory_barplot(attraction_data, output_dir, experiment_type):
    """绘制小类吸引分数分面柱状图（带统计检验）"""
    print(f"\n=== 绘制{experiment_type}类型小类吸引分数分面柱状图（带统计检验） ===")
    
    if len(attraction_data) == 0:
        print("没有数据可以绘制")
        return
    
    df = pd.DataFrame(attraction_data)
    
    # 确保主类别存在
    df = df[df['main_category'].notna()].copy()
    
    if len(df) == 0:
        print("没有包含主类别信息的数据")
        return
    
    # 执行统计检验
    stats_results = perform_subcategory_statistical_tests(attraction_data)
    
    # 按主类别创建分面图
    main_categories = df['main_category'].unique()
    n_main_cats = len(main_categories)
    
    # 设置图形大小（根据主类别数量调整，为统计标注留出更多空间）
    fig_width = max(12, n_main_cats * 3)
    fig, axes = plt.subplots(1, n_main_cats, figsize=(fig_width, 7), sharey=True)
    
    if n_main_cats == 1:
        axes = [axes]
    
    for idx, main_cat in enumerate(main_categories):
        ax = axes[idx]
        df_main = df[df['main_category'] == main_cat].copy()
        df_main = df_main.sort_values('attraction_score', ascending=False)
        bar_color = CATEGORY_COLOR_MAP.get(main_cat, DEFAULT_CATEGORY_COLOR)
        
        # 绘制柱状图
        bars = ax.bar(df_main['sub_category'], df_main['attraction_score'],
                      color=bar_color, alpha=0.85, edgecolor='black', linewidth=1.2)
        
        # 获取该主类别的统计结果
        main_stats = stats_results.get(main_cat, {'binomial_tests': [], 'pairwise_comparisons': []})
        binom_map = {item['sub_category']: item for item in main_stats.get('binomial_tests', [])}
        
        # 添加数值标签
        y_max = 0
        max_annotation_height = 0
        short_labels = []

        for bar, (_, row) in zip(bars, df_main.iterrows()):
            height = bar.get_height()
            y_max = max(y_max, height)
            label_x = bar.get_x() + bar.get_width() / 2
            label_y = height + 0.012
            short_label = row['sub_category'].split(' ')[0] if isinstance(row['sub_category'], str) else row['sub_category']
            short_labels.append(short_label)
            
            # 数值标签
            ax.text(label_x, label_y,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            max_annotation_height = max(max_annotation_height, label_y)
        
        # 添加Chance Level参考线
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Chance Level (0.5)')
        
        ax.set_title(main_cat, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Subcategory', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Attraction Score', fontsize=12, fontweight='bold')
        upper_limit = max(max_annotation_height + 0.08, y_max + 0.12)
        ax.set_ylim(0.2, upper_limit)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.tick_params(axis='x', rotation=0)
        ax.set_xticklabels(short_labels)
        for label in ax.get_xticklabels():
            label.set_fontsize(16)
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight('bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{experiment_type.upper()} Subcategory Attraction by Main Category', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{experiment_type}_subcategory_attraction_facet.png')
    apply_unified_font_style(plt.gcf())
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"分面柱状图已保存: {output_path}")

def plot_subcategory_subjects(subject_points, output_dir, experiment_type):
    """绘制小类被试轨迹分面图"""
    print(f"\n=== 绘制{experiment_type}类型小类被试轨迹分面图 ===")
    
    if len(subject_points) == 0:
        print("没有被试水平数据")
        return
    
    df_subjects = pd.DataFrame(subject_points)
    df_subjects = df_subjects[df_subjects['main_category'].notna()].copy()
    df_subjects['subject_id'] = df_subjects['subject_id'].astype(str)
    
    if len(df_subjects) == 0:
        print("没有包含主类别信息的被试数据")
        return
    
    # 定义被试的固定配色
    subject_colors = {
        'liyanchen': '#1f77b4',
        'shimin': '#ff7f0e',
        'jiachen': '#2ca02c',
        'aiwenkai': '#d62728',
        'lironghua': '#9467bd',
        'mayunmiao': '#8c564b',
        'ShangZiyang': '#e377c2',
        'wjy': '#7f7f7f',
        'LiuYaorui': '#bcbd22',
        'Liu Yaorui': '#bcbd22'
    }
    
    # 按主类别创建分面图
    main_categories = sorted(df_subjects['main_category'].unique())
    n_main_cats = len(main_categories)
    
    # 设置图形大小
    fig_width = max(12, n_main_cats * 4)
    fig, axes = plt.subplots(1, n_main_cats, figsize=(fig_width, 7), sharey=True)
    
    if n_main_cats == 1:
        axes = [axes]
    
    for idx, main_cat in enumerate(main_categories):
        ax = axes[idx]
        df_main = df_subjects[df_subjects['main_category'] == main_cat].copy()
        
        # 1. 背景：被试水平的总体分布（箱线图）
        sns.boxplot(data=df_main, x='sub_category', y='subject_score', 
                   width=0.7, palette="pastel", saturation=0.4, fliersize=0, ax=ax,
                   showcaps=True, whis=[0, 100],
                   boxprops={'facecolor': 'white', 'edgecolor': 'black', 'linewidth': 2},
                   medianprops={'color': 'black', 'linewidth': 2},
                   whiskerprops={'color': 'black', 'linewidth': 2},
                   capprops={'color': 'black', 'linewidth': 2})
        
        # 2. 个体轨迹线
        sns.lineplot(data=df_main, x='sub_category', y='subject_score',
                    units='subject_id', estimator=None,
                    hue='subject_id', palette=subject_colors,
                    alpha=0.4, linewidth=1, ax=ax, legend=False)
        
        # 3. 个体数据点（去掉图例）
        sns.stripplot(data=df_main, x='sub_category', y='subject_score',
                     hue='subject_id', palette=subject_colors,
                     jitter=True, size=4, alpha=0.7, edgecolor='black', linewidth=0.8, ax=ax, legend=False)
        
        # 4. 总体趋势线
        sns.lineplot(data=df_main, x='sub_category', y='subject_score',
                    estimator='mean', errorbar='sd',
                    color='blue', linewidth=2.5, marker='o', markersize=6, ax=ax, legend=False)
        
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.set_title(main_cat, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Subcategory', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Subject-Level Attraction Score', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=0)
    short_xticks = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        short_text = text.split(' ')[0] if isinstance(text, str) else text
        short_xticks.append(short_text)
    ax.set_xticklabels(short_xticks)
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{experiment_type.upper()} Individual Patterns by Main Category\n(All analysis at subject level)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{experiment_type}_subcategory_subjects_facet.png')
    apply_unified_font_style(plt.gcf())
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"分面被试轨迹图已保存: {output_path}")

def main():
    """主函数"""
    print("=== 大类吸引力分析工具 ===")
    print("开始执行...")

    def resolve_mff_dir(mff_name):
        for base in BASE_DATA_DIRS:
            candidate = os.path.join(base, mff_name)
            if os.path.isdir(candidate):
                return candidate
        return None

    def get_mapping_file_from_benchmark(benchmark_name):
        # 例如 benchmark_1_10-2-3.txt -> session 3 -> folder3/file_mapping_folder3.csv
        match = re.search(r"benchmark_1_10-\d+-(\d+)\.txt$", benchmark_name, flags=re.IGNORECASE)
        if not match:
            return None
        folder_idx = match.group(1)
        return os.path.join(
            MAPPING_ROOT, f"folder{folder_idx}", f"file_mapping_folder{folder_idx}.csv"
        )

    output_dir = r"D:\D\research\audioset下载\category_attraction_results"

    print(f"受试者数量: {len(SUBJECTS_CONFIG)}")
    
    # 分析所有数据
    all_results = []

    total_runs = 0
    missing_runs = 0
    for subject_cfg in SUBJECTS_CONFIG:
        for run_cfg in subject_cfg.get("mff_files", []):
            total_runs += 1
            mff_name = run_cfg.get("mff")
            benchmark_name = run_cfg.get("benchmark")
            if not mff_name or not benchmark_name:
                missing_runs += 1
                continue

            mff_dir = resolve_mff_dir(mff_name)
            if mff_dir is None:
                print(f"警告: 未找到MFF目录，跳过: {mff_name}")
                missing_runs += 1
                continue

            data_file = os.path.join(mff_dir, benchmark_name)
            mapping_file = get_mapping_file_from_benchmark(benchmark_name)

            if not os.path.exists(data_file):
                print(f"警告: 未找到benchmark文件，跳过: {data_file}")
                missing_runs += 1
                continue
            if mapping_file is None or not os.path.exists(mapping_file):
                print(f"警告: 未找到映射文件，跳过: {benchmark_name} -> {mapping_file}")
                missing_runs += 1
                continue

            results = analyze_session(data_file, mapping_file, 'txt')
            all_results.extend(results)

    print(f"总run数: {total_runs}，成功读取: {total_runs - missing_runs}，跳过: {missing_runs}")
    
    print(f"总共处理了 {len(all_results)} 个trial")
    
    if not all_results:
        print("没有找到有效数据")
        return
    
    # 全部 sub_ 试次：按小类解析、按大类累计吸引力（柱状图 + HE–LE 子集）
    exp_type = "sub"
    attraction_data_with_counts, _ = calculate_attraction_with_counts(all_results, exp_type)
    if len(attraction_data_with_counts) > 0:
        attraction_ordered = _order_attraction_by_main_bar_order(
            list(attraction_data_with_counts)
        )
        plot_results_with_stats(attraction_ordered, output_dir, exp_type)
        df = pd.DataFrame(attraction_ordered)
        csv_path = os.path.join(output_dir, f'{exp_type}_category_attraction_data.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"数据已保存: {csv_path}")
    else:
        print(f"没有{exp_type}类型的数据")

    # 仅 High Ecology vs Low Ecology 配对的 trial（大类层面）
    he_le_tag = f"{exp_type}_high_low_ecology_only"
    attraction_he_le, _ = calculate_attraction_high_low_ecology_pairs_only(all_results, exp_type)
    if len(attraction_he_le) > 0:
        he_le_ordered = _order_attraction_by_main_bar_order(list(attraction_he_le))
        plot_results_with_stats(
            he_le_ordered, output_dir, he_le_tag, show_pairwise_comparisons=False
        )
        df_hl = pd.DataFrame(he_le_ordered)
        csv_hl = os.path.join(output_dir, f"{he_le_tag}_category_attraction_data.csv")
        df_hl.to_csv(csv_hl, index=False, encoding="utf-8-sig")
        print(f"仅 HE–LE 配对数据已保存: {csv_hl}")
    else:
        print(f"没有 {exp_type} 类型的 High–Low Ecology 配对 trial，跳过专图")
    
    print(f"\n分析完成！结果保存到: {output_dir}")

if __name__ == "__main__":
    main()

