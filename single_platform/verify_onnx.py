# -*- coding: utf-8 -*-
"""
评估 ONNX 模型的预测性能
直接比较 ONNX 模型预测值与真实值的差异,计算 MAPE、R2、RMSE 等指标
"""

import sys
import io
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ========== 配置区 ==========
# 设置要验证的模型
PROGRAM = 'KF'  # KF | FFT | AES | MD5 | SHA256 | MPC
CPU = 'A72'       # R5 | A72 | M7
METHOD = 'rf'   # rf | svr | mlp | curve | xgboost | hybrid

# ONNX 模型文件夹 (格式: onnx/Program-Method-CPU)
ONNX_FOLDER = os.path.join("onnx", f"{PROGRAM}-{METHOD}-{CPU}")

# 测试配置
TEST_SIZE = 1  # 从所有数据中随机抽取的比例（0.3 = 30%）
REFERENCE_SEED = 1  # 随机种子，用于可重复的随机抽样
# 忽略低于该阈值的运行时间数据（ms）
LOWER_BOUND = 0

# ========== 数据准备 ==========
print("="*80)
print("ONNX Model Performance Evaluation")
print("="*80)
print(f"Program: {PROGRAM}, CPU: {CPU}, Method: {METHOD}")

if PROGRAM not in ('AES','MD5','SHA256'):
    data = pd.read_csv("exclusive_runtime.csv")
else:
    data = pd.read_csv("exclusive_runtime_encrypt.csv")

host_data = data[(data['cpu'] == CPU) & (data['program'] == PROGRAM)]
host_data = host_data[host_data['time'] > LOWER_BOUND]

features = ['input']
output = 'time'

# 从所有数据中随机抽取测试样本
print(f"数据集总样本数: {len(host_data)}")
print(f"随机抽取比例: {TEST_SIZE*100:.0f}%")

# 随机抽样（不划分训练集，只抽取测试集）
test_data = host_data.sample(frac=TEST_SIZE, random_state=REFERENCE_SEED)
X_test = test_data[features]
y_true = test_data[output].values

print(f"随机抽取的测试集 (种子={REFERENCE_SEED}): {len(X_test)} 个样本")
print(f"将对所有导出的模型在此测试集上进行评估")
print()

# ========== 导入 ONNX Runtime ==========
try:
    import onnxruntime as ort
    print(f"✓ ONNXRuntime version: {ort.__version__}")
except ImportError:
    print("✗ ONNXRuntime not found. Please install: pip install onnxruntime")
    sys.exit(1)

# ========== 辅助函数 ==========
def load_onnx_model(onnx_path):
    """加载 ONNX 模型并返回 session"""
    if not os.path.exists(onnx_path):
        return None
    return ort.InferenceSession(onnx_path)

def predict_onnx(session, X_input):
    """使用 ONNX 模型预测"""
    if session is None:
        return None
    
    # 转换为 float32
    X_array = X_input.values if hasattr(X_input, 'values') else X_input
    if len(X_array.shape) == 1:
        X_array = X_array.reshape(-1, 1)
    X_float32 = X_array.astype(np.float32)
    
    # 推理
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: X_float32})[0]
    
    return output.flatten()

# ========== 主评估流程 ==========
# 1. 查找 ONNX 模型文件
if not os.path.exists(ONNX_FOLDER):
    print(f"\n✗ 错误: 文件夹不存在: {ONNX_FOLDER}/")
    print(f"  请先运行 run_ensemble_top4.py 导出 ONNX 模型")
    sys.exit(1)

onnx_files = [f for f in os.listdir(ONNX_FOLDER) if f.endswith('.onnx')]
if len(onnx_files) == 0:
    print(f"\n✗ 错误: 在 {ONNX_FOLDER}/ 中未找到 ONNX 文件")
    sys.exit(1)

# 2. 识别所有种子的模型
model_seeds = set()
for f in onnx_files:
    import re
    match = re.search(r'seed(\d+)', f)
    if match:
        seed = int(match.group(1))
        model_seeds.add(seed)

model_seeds = sorted(list(model_seeds))

if len(model_seeds) == 0:
    print(f"\n✗ 错误: 无法识别任何模型种子")
    sys.exit(1)

print(f"\n✓ 找到 {len(model_seeds)} 个模型，开始评估...")

# 3. 对每个种子的模型进行评估

all_results = []

for seed in model_seeds:
    # 简化输出：不显示过程信息
    
    # 查找该种子的 ONNX 文件
    seed_files = [f for f in onnx_files if f'seed{seed}' in f]
    
    if len(seed_files) == 0:
        print(f"✗ 警告: 未找到种子 {seed} 的模型文件")
        continue
    
    # 检查是否是 hybrid 模型（需要 base 和 residual 两个文件）
    base_file = None
    res_file = None
    single_file = None
    
    for f in seed_files:
        if 'base' in f:
            base_file = f
        elif 'residual' in f:
            res_file = f
        else:
            single_file = f
    
    # 加载并预测
    y_pred_onnx = None
    
    if base_file and res_file:
        # Hybrid 模型
        base_path = os.path.join(ONNX_FOLDER, base_file)
        res_path = os.path.join(ONNX_FOLDER, res_file)
        
        base_session = load_onnx_model(base_path)
        res_session = load_onnx_model(res_path)
        
        if base_session and res_session:
            base_pred = predict_onnx(base_session, X_test)
            res_pred = predict_onnx(res_session, X_test)
            y_pred_onnx = base_pred + res_pred
        else:
            print(f"✗ 错误: 种子 {seed} 的混合模型加载失败")
            continue
    
    elif single_file:
        # 单个模型
        model_path = os.path.join(ONNX_FOLDER, single_file)
        session = load_onnx_model(model_path)
        
        if session:
            y_pred_onnx = predict_onnx(session, X_test)
        else:
            print(f"✗ 错误: 种子 {seed} 的模型加载失败")
            continue
    else:
        print(f"✗ 错误: 种子 {seed} 无法识别模型类型")
        continue
    
    # 计算评估指标
    mape = mean_absolute_percentage_error(y_true, y_pred_onnx)
    r2 = r2_score(y_true, y_pred_onnx)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_onnx))
    
    eps = 1e-12
    mean_y = np.mean(y_true)
    rmse_pct_mean = 100.0 * rmse / (mean_y + eps)
    
    # 保存结果
    all_results.append({
        'seed': seed,
        'mape': mape,
        'r2': r2,
        'rmse': rmse,
        'rmse_pct_mean': rmse_pct_mean
    })
    
    mape_pct = mape * 100

# 4. 显示汇总结果
print("\n" + "="*80)
print("ONNX模型评估结果")
print("="*80)

if len(all_results) == 0:
    print("\n✗ 错误: 没有成功评估的模型")
    sys.exit(1)

print(f"\n{'种子':<10} {'MAPE%':<10} {'R2':<12} {'RMSE':<12} {'RMSE%_mean':<12} {'等级':<15}")
print("-" * 75)

for result in all_results:
    m = result
    mape_pct = m['mape'] * 100
    rmse_pct = m['rmse_pct_mean']
    
    # 评级
    if mape_pct < 10.0 and rmse_pct < 10.0:
        grade = "优秀"
    elif mape_pct < 20.0 and rmse_pct < 20.0:
        grade = "良好"
    else:
        grade = "一般"
    
    print(f"{m['seed']:<10} {mape_pct:<10.2f} {m['r2']:<12.6f} {m['rmse']:<12.4f} {rmse_pct:<12.2f} {grade:<15}")

print("="*80)
