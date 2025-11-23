# -*- coding: utf-8 -*-
"""手动验证 ONNX 模型的小工具。

使用方法：
1. 在“配置区”设置 PROGRAM、CPU、METHOD、ONNX_ROOT 等参数。
2. 确保 onnxruntime/onnx 等依赖已安装 (requirements.txt 已包含)。
3. 运行 `python manual_verify_onnx.py`，按照提示输入“规模”，脚本会自动从
    exclusive_runtime*.csv 中查找对应真实值并对比预测结果。
"""
from __future__ import annotations

import sys
import io
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence, Union, Optional, Dict

import numpy as np
import pandas as pd
import onnxruntime as ort

# 解决 Windows 控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ========== 配置区 ==========
# KF | FFT | AES | MD5 | SHA256 | MPC 等
PROGRAM = 'AES'.upper()
# R5 | A72 | M7 等
CPU = 'A72'.upper()
# rf | svr | mlp | curve | xgboost | hybrid
METHOD = 'HYBRID'.lower()
# ONNX 根目录（run_ensemble_top4.py 默认导出到 onnx/Program-Method-CPU/）
ONNX_ROOT = Path('onnx')
# lower_bound 配置文件
CONFIG_FILE = Path('config_lower_bounds.csv')
# 如果想手动指定模型文件，可在此列出（相对于目标目录）
# e.g. MODEL_FILES = ['model_seed1_hybrid.onnx', ['model_seed2_base.onnx','model_seed2_residual.onnx']]
MODEL_FILES: List[Union[str, Sequence[str]]] = []
# 自动发现模型时最多保留多少个（默认 4 个 Top 模型）
MAX_MODELS = 4
# 输入提示信息中是否显示当前配置
PRINT_CONFIG = True

# ========== 工具函数 ==========
@dataclass
class OnnxPredictor:
    """封装单个 ONNX 模型或一对 base/residual 模型。"""
    paths: List[Path]

    def __post_init__(self) -> None:
        self.sessions: List[ort.InferenceSession] = []
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(f"ONNX 模型不存在: {path}")
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            self.sessions.append(session)
            self.input_names.append(session.get_inputs()[0].name)
            self.output_names.append(session.get_outputs()[0].name)

    @property
    def label(self) -> str:
        """根据文件名生成友好的标签。"""
        if len(self.paths) == 1:
            return self.paths[0].stem
        common = self.paths[0].stem
        return f"{common}+res"

    def predict(self, x_value: float) -> float:
        """对单个输入值进行预测。"""
        arr = np.array([[float(x_value)]], dtype=np.float32)
        total = 0.0
        for session, input_name in zip(self.sessions, self.input_names):
            outputs = session.run(None, {input_name: arr})
            total += float(np.squeeze(outputs[0]))
        return total


def load_host_data() -> pd.DataFrame:
    """读取真实测量数据并按配置进行过滤。"""
    encrypt_programs = {'AES', 'MD5', 'SHA256'}
    csv_file = 'exclusive_runtime_encrypt.csv' if PROGRAM in encrypt_programs else 'exclusive_runtime.csv'
    df = pd.read_csv(csv_file)

    # 统一大小写，便于匹配
    df['cpu'] = df['cpu'].astype(str).str.upper()
    df['program'] = df['program'].astype(str).str.upper()

    host_df = df[(df['cpu'] == CPU) & (df['program'] == PROGRAM)].copy()
    if host_df.empty:
        raise ValueError(f"数据集中不存在 PROGRAM={PROGRAM}, CPU={CPU} 的记录")
    host_df['input'] = host_df['input'].astype(float)
    host_df['time'] = host_df['time'].astype(float)
    return host_df


def load_config_lower_bound() -> float:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"未找到配置文件: {CONFIG_FILE}")

    df = pd.read_csv(CONFIG_FILE)
    if df.empty:
        raise ValueError(f"配置文件 {CONFIG_FILE} 为空")

    df['algorithm'] = df['algorithm'].astype(str).str.upper()
    df['platform'] = df['platform'].astype(str).str.upper()
    df['method'] = df['method'].astype(str).str.lower()

    mask = (
        (df['algorithm'] == PROGRAM)
        & (df['platform'] == CPU)
        & (df['method'] == METHOD)
    )

    subset = df[mask]
    if subset.empty:
        raise ValueError(
            f"配置文件中未找到 PROGRAM={PROGRAM}, CPU={CPU}, METHOD={METHOD} 的 lower_bound"
        )

    value = subset.iloc[0]['lower_bound']
    try:
        lower_bound = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"配置中的 lower_bound 无法转换为数字: {value}")

    return lower_bound


def lookup_truth_stats(host_df: Optional[pd.DataFrame], input_value: float, atol: float = 1e-9) -> Optional[Dict[str, float]]:
    if host_df is None or host_df.empty:
        return None
    mask = np.isclose(host_df['input'], float(input_value), atol=atol)
    matches = host_df[mask]
    if matches.empty:
        return None
    return {
        'mean': float(matches['time'].mean()),
        'median': float(matches['time'].median()),
        'min': float(matches['time'].min()),
        'max': float(matches['time'].max()),
        'count': int(matches.shape[0])
    }


def _to_path_list(entry: Union[str, Sequence[str]], folder: Path) -> List[Path]:
    if isinstance(entry, (list, tuple)):
        return [folder / item for item in entry]
    return [folder / entry]


def discover_model_groups(folder: Path) -> List[List[Path]]:
    """自动扫描文件夹，返回 ONNX 模型路径列表。

    如果发现 `*_base.onnx` 与 `*_residual.onnx` 配对，则组合成一组。
    其他文件按单个模型处理。
    """
    if not folder.exists():
        raise FileNotFoundError(f"未找到 ONNX 目录: {folder}")

    files = sorted(folder.glob('*.onnx'))
    if not files:
        raise FileNotFoundError(f"目录 {folder} 中不存在 ONNX 文件")

    base_map = {}
    residual_map = {}
    singles: List[Path] = []

    for path in files:
        name = path.name
        if '_base' in name:
            key = name.replace('_base', '')
            base_map[key] = path
        elif '_residual' in name:
            key = name.replace('_residual', '')
            residual_map[key] = path
        else:
            singles.append(path)

    groups: List[List[Path]] = []
    used_pairs = set()
    for key, base_path in base_map.items():
        if key in residual_map:
            groups.append([base_path, residual_map[key]])
            used_pairs.add(key)

    # 未配对的 base/residual 作为单独模型处理
    for key, base_path in base_map.items():
        if key not in used_pairs:
            groups.append([base_path])
    for key, res_path in residual_map.items():
        if key not in used_pairs:
            groups.append([res_path])

    groups.extend([[p] for p in singles])
    return groups


def load_predictors() -> List[OnnxPredictor]:
    target_dir = ONNX_ROOT / f"{PROGRAM}-{METHOD}-{CPU}"
    if MODEL_FILES:
        group_paths = []
        for entry in MODEL_FILES:
            group_paths.append(_to_path_list(entry, target_dir))
    else:
        group_paths = discover_model_groups(target_dir)

    predictors: List[OnnxPredictor] = []
    for paths in group_paths[:MAX_MODELS]:
        predictors.append(OnnxPredictor(paths))
    if not predictors:
        raise RuntimeError("未加载任何 ONNX 模型，请检查配置")
    return predictors


def format_error(pred: float, truth: float) -> str:
    abs_err = pred - truth
    rel_err = abs_err / truth * 100 if truth != 0 else float('inf')
    return f"预测={pred:.4f} ms, 误差={abs_err:+.4f} ms ({rel_err:+.2f}%)"


def interactive_loop(predictors: List[OnnxPredictor], host_df: Optional[pd.DataFrame], lower_bound: float) -> None:
    if PRINT_CONFIG:
        print("=" * 80)
        print(f"手动验证配置 -> PROGRAM={PROGRAM}, CPU={CPU}, METHOD={METHOD}, LOWER_BOUND={lower_bound}")
        print(f"模型目录: {ONNX_ROOT / f'{PROGRAM}-{METHOD}-{CPU}'}")
        print(f"已加载 {len(predictors)} 个 ONNX 模型: {[p.label for p in predictors]}")
        if host_df is not None and not host_df.empty:
            available_inputs = sorted(host_df['input'].unique())[:10]
            sample_hint = f"样例输入: {available_inputs} ..." if available_inputs else ""
            print("真实值将自动从 CSV 中查找 (按 exact input 匹配)。")
            if sample_hint:
                print(sample_hint)
        else:
            print("警告: 未找到真实数据，结果中将不显示误差。")
        print("请输入: <输入规模> ，或输入 q 退出。")
        print("=" * 80)

    while True:
        try:
            raw = input("规模 >>> ").strip()
        except EOFError:
            print("\n收到 EOF，退出。")
            break

        if not raw:
            continue
        if raw.lower() in {'q', 'quit', 'exit'}:
            print("退出手动验证。")
            break

        parts = raw.split()
        if len(parts) == 0:
            continue

        if len(parts) > 1:
            print("多余参数已忽略。")

        try:
            input_size = float(parts[0])
        except ValueError:
            print("请输入合法的数值规模，例如: 1024")
            continue

        truth_info = lookup_truth_stats(host_df, input_size)
        truth_val = truth_info['mean'] if truth_info else None
        if truth_info:
            print(
                f"真实数据: N={truth_info['count']}, 平均={truth_info['mean']:.4f} ms, "
                f"中位数={truth_info['median']:.4f} ms, 范围=[{truth_info['min']:.4f}, {truth_info['max']:.4f}]"
            )
            if truth_info['min'] < lower_bound:
                print("预测值可能处于不可靠区间")
        else:
            if host_df is not None:
                print("提示: 数据集中没有该输入规模的真实值，无法计算误差。")

        preds = []
        for idx, predictor in enumerate(predictors, start=1):
            try:
                pred = predictor.predict(input_size)
            except Exception as exc:
                print(f"模型#{idx} ({predictor.label}) 推理失败: {exc}")
                continue
            preds.append(pred)
            if truth_val is not None:
                print(f"模型#{idx} [{predictor.label}]: {format_error(pred, truth_val)}")
            else:
                print(f"模型#{idx} [{predictor.label}]: 预测={pred:.4f} ms")

        if not preds:
            print("没有成功的预测结果。")
            continue

        ensemble_mean = float(np.mean(preds))
        ensemble_median = float(np.median(preds))
        if truth_val is not None:
            print(f"--> Ensemble(平均):  {format_error(ensemble_mean, truth_val)}")
            print(f"--> Ensemble(中位数):{format_error(ensemble_median, truth_val)}")
        else:
            print(f"--> Ensemble 预测(平均): {ensemble_mean:.4f} ms")
            print(f"--> Ensemble 预测(中位数): {ensemble_median:.4f} ms")
        print("-" * 80)


def main() -> None:
    try:
        predictors = load_predictors()
    except Exception as exc:
        print(f"加载 ONNX 模型失败: {exc}")
        return

    try:
        lower_bound = load_config_lower_bound()
    except Exception as exc:
        print(f"加载 lower_bound 失败: {exc}")
        return

    try:
        host_df = load_host_data()
    except Exception as exc:
        print(f"加载真实数据失败: {exc}")
        host_df = None

    interactive_loop(predictors, host_df, lower_bound)


if __name__ == '__main__':
    main()
