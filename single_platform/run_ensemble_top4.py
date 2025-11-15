# -*- coding: utf-8 -*-
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from scipy.optimize import curve_fit
from xgboost import XGBRegressor

# ========== 配置区 ==========
# KF | FFT | AES | MD5 | SHA256 | MPC
predicted_app = 'KF'.upper()
# R5 | A72 | M7
host_cpu = 'A72'.upper()
# rf | svr | mlp | curve | xgboost | hybrid
PREDICT_METHOD = 'hybrid'.lower()
# 每个拟合方法将在这些随机种子下运行
SEEDS = [1, 2, 6, 42, 123, 2025, 33550336]
# 测试集占总数据比重
TEST_SIZE = 0.3
# 忽略低于该阈值的运行时间数据（ms）
LOWER_BOUND = 10
# 打印详细信息
PRINT_DETAILS = False
# 导出ONNX后自动验证模型
AUTO_VERIFY_ONNX = True

# ========== 数据准备 ==========
if predicted_app not in ('AES','MD5','SHA256'):
    data = pd.read_csv("exclusive_runtime.csv")
else:
    data = pd.read_csv("exclusive_runtime_encrypt.csv")
host_data = data[(data['cpu'] == host_cpu) & (data['program'] == predicted_app)]
host_data = host_data[(host_data['time'] > LOWER_BOUND)]

features = ['input']
output = 'time'

# ========= 全局拟合函数（用于 pickle 序列化）==========
def cubic_func(N, a, b, c, d):
    return a * N**3 + b * N**2 + c * N + d

def fft_complexity(N, a, b):
    return a * N * np.log2(N) + b

def linear_func(N, a, b):
    return a * N + b

def quad_func(N, a, b, c):
    return a * N**2 + b * N + c

# ========= 评估函数 ==========
def mape_grade(mape_value: float) -> str:
    perc = mape_value * 100
    if perc < 10:
        return "优秀"
    if perc < 20:
        return "中等"
    return "差"

def rmse_pct_grade(pct_value: float) -> str:
    if pct_value < 10:
        return "优秀"
    if pct_value < 20:
        return "中等"
    return "差"

# ========= 训练单个模型并返回模型对象 ==========
def train_one_model(seed: int, predict_method: str = PREDICT_METHOD):
    """训练单个模型并返回 (模型对象, scaler, 测试数据, 预测结果, 指标字典)"""
    train_data, test_data = train_test_split(host_data, test_size=TEST_SIZE, random_state=seed)
    if train_data.empty or test_data.empty:
        raise ValueError("数据划分失败：训练集或测试集为空。")

    X_train = train_data[features]
    y_train = train_data[output]
    X_test = test_data[features]
    y_true = test_data[output].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_obj = None  # 用于保存模型对象或参数
    y_pred = None

    # 根据预测方法训练模型
    if predict_method == 'rf':
        if predicted_app == 'FFT':
            X_train_fe = train_data[['input']].copy()
            X_test_fe = test_data[['input']].copy()
            X_train_fe['n_log_n'] = X_train_fe['input'] * np.log2(X_train_fe['input'])
            X_test_fe['n_log_n'] = X_test_fe['input'] * np.log2(X_test_fe['input'])
            model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
            model.fit(X_train_fe[['n_log_n']], y_train)
            y_pred = model.predict(X_test_fe[['n_log_n']])
            model_obj = {'model': model, 'feature_type': 'n_log_n'}
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_obj = {'model': model, 'feature_type': 'raw'}

    elif predict_method == 'svr':
        model = SVR(kernel='rbf', C=1000, epsilon=0.5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        model_obj = {'model': model, 'scaler': scaler}

    elif predict_method == 'mlp':
        if predicted_app == 'FFT':
            X_train_fe = train_data[['input']].copy()
            X_test_fe = test_data[['input']].copy()
            X_train_fe['n_log_n'] = X_train_fe['input'] * np.log2(X_train_fe['input'])
            X_test_fe['n_log_n'] = X_test_fe['input'] * np.log2(X_test_fe['input'])
            scaler_mlp = StandardScaler()
            X_train_m = scaler_mlp.fit_transform(X_train_fe[['n_log_n']])
            X_test_m = scaler_mlp.transform(X_test_fe[['n_log_n']])
            mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', max_iter=8000, random_state=seed)
            mlp.fit(X_train_m, y_train)
            y_pred = mlp.predict(X_test_m)
            model_obj = {'model': mlp, 'scaler': scaler_mlp, 'feature_type': 'n_log_n'}
        else:
            mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', max_iter=8000, random_state=seed)
            mlp.fit(X_train_scaled, y_train)
            y_pred = mlp.predict(X_test_scaled)
            model_obj = {'model': mlp, 'scaler': scaler}

    elif predict_method == 'curve':
        X_train_cf = X_train[features[0]].values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_cf = y_train.values if hasattr(y_train, 'values') else y_train
        X_test_cf = X_test[features[0]].values if isinstance(X_test, pd.DataFrame) else X_test

        if predicted_app == 'KF':
            params, _ = curve_fit(cubic_func, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = cubic_func(X_test_cf, *params)
            model_obj = {'func': cubic_func, 'params': params, 'app': 'KF'}

        elif predicted_app == 'FFT':
            params, _ = curve_fit(fft_complexity, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = fft_complexity(X_test_cf, *params)
            model_obj = {'func': fft_complexity, 'params': params, 'app': 'FFT'}

        elif predicted_app in ('AES', 'MD5', 'SHA256'):
            params, _ = curve_fit(linear_func, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = linear_func(X_test_cf, *params)
            model_obj = {'func': linear_func, 'params': params, 'app': predicted_app}

        elif predicted_app == 'MPC' and host_cpu == 'A72':
            params, _ = curve_fit(quad_func, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = quad_func(X_test_cf, *params)
            model_obj = {'func': quad_func, 'params': params, 'app': 'MPC-A72'}

        else:
            params, _ = curve_fit(cubic_func, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = cubic_func(X_test_cf, *params)
            model_obj = {'func': cubic_func, 'params': params, 'app': 'default'}

    elif predict_method == 'xgboost':
        model = XGBRegressor(n_estimators=500, max_depth=15, learning_rate=0.05, 
                            random_state=seed, n_jobs=-1, verbosity=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_obj = {'model': model}

    elif predict_method == 'hybrid':
        X_train_arr = X_train[features[0]].values if isinstance(X_train, pd.DataFrame) else X_train
        X_test_arr = X_test[features[0]].values if isinstance(X_test, pd.DataFrame) else X_test
        y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train

        if predicted_app in ('KF'):
            params, _ = curve_fit(cubic_func, X_train_arr, y_train_arr, maxfev=10000)
            base_train = cubic_func(X_train_arr, *params)
            base_test = cubic_func(X_test_arr, *params)
            base_func = cubic_func
            base_params = params

        elif predicted_app == 'FFT':
            X_train_fe = pd.DataFrame({'input': X_train_arr})
            X_test_fe = pd.DataFrame({'input': X_test_arr})
            X_train_fe['n_log_n'] = X_train_fe['input'] * np.log2(X_train_fe['input'])
            X_test_fe['n_log_n'] = X_test_fe['input'] * np.log2(X_test_fe['input'])
            from sklearn.linear_model import LinearRegression
            base_lr = LinearRegression().fit(X_train_fe[['n_log_n']], y_train_arr)
            base_train = base_lr.predict(X_train_fe[['n_log_n']])
            base_test = base_lr.predict(X_test_fe[['n_log_n']])
            base_func = base_lr
            base_params = None

        elif predicted_app in ('AES', 'MD5', 'SHA256') or (predicted_app == 'MPC' and host_cpu == 'M7'):
            from sklearn.linear_model import LinearRegression
            X_train_lin = X_train_arr.reshape(-1, 1)
            X_test_lin = X_test_arr.reshape(-1, 1)
            base_lr = LinearRegression().fit(X_train_lin, y_train_arr)
            base_train = base_lr.predict(X_train_lin)
            base_test = base_lr.predict(X_test_lin)
            base_func = base_lr
            base_params = None

        elif predicted_app == 'MPC' and host_cpu in ('A72', 'R5'):
            coefs = np.polyfit(X_train_arr, y_train_arr, 2)
            a, b, c = coefs
            base_train = quad_func(X_train_arr, a, b, c)
            base_test = quad_func(X_test_arr, a, b, c)
            base_func = quad_func
            base_params = coefs
        else:
            raise ValueError(f"不支持的程序类型：{predicted_app}")

        residuals_train = y_train_arr - base_train
        res_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=seed, n_jobs=-1)
        # Use DataFrame to preserve feature names
        X_train_res_df = pd.DataFrame(X_train_arr.reshape(-1, 1), columns=['input'])
        X_test_res_df = pd.DataFrame(X_test_arr.reshape(-1, 1), columns=['input'])
        res_model.fit(X_train_res_df, residuals_train)
        res_pred_test = res_model.predict(X_test_res_df)
        y_pred = base_test + res_pred_test
        model_obj = {'base_func': base_func, 'base_params': base_params, 'res_model': res_model, 'app': predicted_app}

    else:
        raise ValueError(f"不支持的预测方式: {predict_method}")

    # 计算指标
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    eps = 1e-12
    mean_y = np.mean(y_true)
    range_y = np.max(y_true) - np.min(y_true)
    rmse_pct_mean = 100.0 * rmse / (mean_y + eps)
    rmse_pct_range = 100.0 * rmse / (range_y + eps)

    metrics = {
        'seed': seed,
        'mape': mape,
        'r2': r2,
        'rmse': rmse,
        'rmse_pct_mean': rmse_pct_mean,
        'rmse_pct_range': rmse_pct_range,
        'n_test': len(y_true)
    }

    return model_obj, scaler, test_data, y_pred, y_true, metrics


# ========= 使用单个模型进行预测 ==========
def predict_with_model(model_obj, X_input, predict_method: str):
    """根据模型对象和方法对输入数据进行预测"""
    if predict_method == 'rf':
        if model_obj['feature_type'] == 'n_log_n':
            X_fe = X_input.copy()
            X_fe['n_log_n'] = X_fe['input'] * np.log2(X_fe['input'])
            return model_obj['model'].predict(X_fe[['n_log_n']])
        else:
            return model_obj['model'].predict(X_input)

    elif predict_method == 'svr':
        X_scaled = model_obj['scaler'].transform(X_input)
        return model_obj['model'].predict(X_scaled)

    elif predict_method == 'mlp':
        if 'feature_type' in model_obj and model_obj['feature_type'] == 'n_log_n':
            X_fe = X_input.copy()
            X_fe['n_log_n'] = X_fe['input'] * np.log2(X_fe['input'])
            X_scaled = model_obj['scaler'].transform(X_fe[['n_log_n']])
        else:
            X_scaled = model_obj['scaler'].transform(X_input)
        return model_obj['model'].predict(X_scaled)

    elif predict_method == 'curve':
        X_arr = X_input[features[0]].values if isinstance(X_input, pd.DataFrame) else X_input
        return model_obj['func'](X_arr, *model_obj['params'])

    elif predict_method == 'xgboost':
        return model_obj['model'].predict(X_input)

    elif predict_method == 'hybrid':
        X_arr = X_input[features[0]].values if isinstance(X_input, pd.DataFrame) else X_input
        
        if model_obj['base_params'] is not None:
            # curve_fit based
            base_pred = model_obj['base_func'](X_arr, *model_obj['base_params'])
        else:
            # sklearn model based
            if model_obj['app'] == 'FFT':
                X_fe = pd.DataFrame({'input': X_arr})
                X_fe['n_log_n'] = X_fe['input'] * np.log2(X_fe['input'])
                base_pred = model_obj['base_func'].predict(X_fe[['n_log_n']])
            else:
                base_pred = model_obj['base_func'].predict(X_arr.reshape(-1, 1))
        
        # Use DataFrame for residual prediction to preserve feature names
        X_res_df = pd.DataFrame(X_arr.reshape(-1, 1), columns=['input'])
        res_pred = model_obj['res_model'].predict(X_res_df)
        return base_pred + res_pred

    else:
        raise ValueError(f"不支持的预测方式: {predict_method}")


# ========= 主程序：训练 Top-4 集成模型 ==========
if __name__ == '__main__':
    print("="*80)
    print(f"Top-4 集成模型训练与评估")
    print(f"程序: {predicted_app}, CPU: {host_cpu}, 方法: {PREDICT_METHOD}")
    print(f"测试集比例: {TEST_SIZE}, 随机种子: {SEEDS}")
    print("="*80)

    # Step 1: Train models with all seeds and collect metrics
    print("\n步骤 1: 使用所有随机种子训练模型...")
    all_models = []
    
    for seed in SEEDS:
        print(f"  训练种子 {seed}...", end=" ")
        try:
            model_obj, scaler, test_data, y_pred, y_true, metrics = train_one_model(seed, PREDICT_METHOD)
            all_models.append({
                'seed': seed,
                'model': model_obj,
                'scaler': scaler,
                'test_data': test_data,
                'y_pred': y_pred,
                'y_true': y_true,
                'metrics': metrics
            })
            print(f"MAPE={metrics['mape']:.4f}, R2={metrics['r2']:.4f}")
        except Exception as e:
            print(f"失败: {e}")
            continue

    if len(all_models) == 0:
        print("错误: 没有模型训练成功！")
        exit(1)

    # Step 2: Filter and select Top-4 models based on strict criteria
    print("\n步骤 2: 、筛选模型...")
    print("  标准: MAPE < 10% (优秀) 且 RMSE%_mean < 10% (优秀)")
    
    # Filter models that meet "Excellent" criteria
    excellent_models = []
    for model_info in all_models:
        m = model_info['metrics']
        mape_pct = m['mape'] * 100
        rmse_pct = m['rmse_pct_mean']
        
        # Both MAPE and RMSE% must be < 10% (Excellent grade)
        if mape_pct < 10.0 and rmse_pct < 10.0:
            excellent_models.append(model_info)
    
    print(f"\n  在 {len(all_models)} 个模型中找到 {len(excellent_models)} 个符合'优秀'标准的模型")
    
    if len(excellent_models) == 0:
        print("\n  警告: 没有模型符合'优秀'标准！")
        print("  回退到仅按 MAPE 选择 Top-4...")
        all_models.sort(key=lambda x: x['metrics']['mape'])
        top4_models = all_models[:4]
    elif len(excellent_models) < 4:
        print(f"\n  信息: 找到 {len(excellent_models)} 个优秀模型")
        print(f"  仅使用这 {len(excellent_models)} 个优秀模型进行集成")
        # Use only excellent models, don't fill with non-excellent ones
        # Sort by combined score
        for model_info in excellent_models:
            m = model_info['metrics']
            mape_score = m['mape'] * 100
            rmse_score = m['rmse_pct_mean']
            model_info['combined_score'] = mape_score + rmse_score
        
        excellent_models.sort(key=lambda x: x['combined_score'])
        top4_models = excellent_models
    else:
        # Sort excellent models by combined score: normalized MAPE + normalized RMSE%
        # Lower is better for both
        for model_info in excellent_models:
            m = model_info['metrics']
            # Normalize to 0-1 range within excellent models
            mape_score = m['mape'] * 100  # Already percentage
            rmse_score = m['rmse_pct_mean']
            # Combined score (equal weight)
            model_info['combined_score'] = mape_score + rmse_score
        
        excellent_models.sort(key=lambda x: x['combined_score'])
        top4_models = excellent_models[:4]
        print("  基于 MAPE + RMSE%_mean 综合得分从优秀模型中选择 Top-4")

    print(f"\n已选模型 ({len(top4_models)} 个模型):")
    print(f"{'排名':<6} {'种子':<10} {'MAPE%':<10} {'R2':<12} {'RMSE':<12} {'RMSE%_mean':<10} {'评估':<15}")
    print("-" * 80)
    for rank, model_info in enumerate(top4_models, 1):
        m = model_info['metrics']
        mape_pct = m['mape'] * 100
        rmse_pct = m['rmse_pct_mean']
        
        # Determine overall grade (both must be excellent)
        if mape_pct < 10.0 and rmse_pct < 10.0:
            grade = "优秀"
        elif mape_pct < 20.0 and rmse_pct < 20.0:
            grade = "良好"
        else:
            grade = "一般"
        
        print(f"{rank:<6} {m['seed']:<10} {mape_pct:<10.2f} {m['r2']:<12.6f} {m['rmse']:<12.4f} {rmse_pct:<10.2f} {grade:<15}")

    # Step 3: Use unified test set (from first seed)
    print(f"\n步骤 3: 在统一测试集上评估已选模型...")
    # Use a unified test set for fair comparison
    reference_seed = SEEDS[0]
    _, unified_test_data = train_test_split(host_data, test_size=TEST_SIZE, random_state=reference_seed)
    X_unified_test = unified_test_data[features]
    y_unified_true = unified_test_data[output].values
    
    print(f"  统一测试集 (种子={reference_seed}, 样本数={len(y_unified_true)})")

    # Collect predictions from selected models on unified test set
    top4_predictions = []
    unified_metrics = []
    
    for model_info in top4_models:
        try:
            y_pred_unified = predict_with_model(model_info['model'], X_unified_test, PREDICT_METHOD)
            top4_predictions.append(y_pred_unified)
            
            # Calculate metrics on unified test set
            mape_u = mean_absolute_percentage_error(y_unified_true, y_pred_unified)
            r2_u = r2_score(y_unified_true, y_pred_unified)
            rmse_u = np.sqrt(mean_squared_error(y_unified_true, y_pred_unified))
            
            eps = 1e-12
            mean_y = np.mean(y_unified_true)
            rmse_pct_u = 100.0 * rmse_u / (mean_y + eps)
            
            unified_metrics.append({
                'seed': model_info['seed'],
                'mape': mape_u,
                'r2': r2_u,
                'rmse': rmse_u,
                'rmse_pct': rmse_pct_u
            })
        except Exception as e:
            print(f"  错误: 种子 {model_info['seed']} 预测失败 - {e}")
            continue

    if len(top4_predictions) < len(top4_models):
        print(f"  警告: 仅 {len(top4_predictions)} 个模型预测成功！")
    
    # Display unified test results in table format
    print(f"\n统一测试集评估结果 ({len(unified_metrics)} 个模型):")
    print(f"{'排名':<6} {'种子':<10} {'MAPE%':<10} {'R2':<12} {'RMSE':<12} {'RMSE%_mean':<10} {'评估':<15}")
    print("-" * 80)
    for rank, m in enumerate(unified_metrics, 1):
        mape_pct = m['mape'] * 100
        rmse_pct = m['rmse_pct']
        
        # Determine overall grade
        if mape_pct < 10.0 and rmse_pct < 10.0:
            grade = "优秀"
        elif mape_pct < 20.0 and rmse_pct < 20.0:
            grade = "良好"
        else:
            grade = "一般"
        
        print(f"{rank:<6} {m['seed']:<10} {mape_pct:<10.2f} {m['r2']:<12.6f} {m['rmse']:<12.4f} {rmse_pct:<10.2f} {grade:<15}")

    # 步骤 4: 可视化对比 (如果需要)
    if PRINT_DETAILS:
        print("\n步骤 4: 生成可视化图表...")
        
        # 图: 输入规模 vs 预测/真实值曲线 (显示所有 Top-4 模型)
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(X_unified_test[features[0]].values)
        X_sorted = X_unified_test[features[0]].values[sorted_idx]
        y_true_sorted = y_unified_true[sorted_idx]
        
        plt.plot(X_sorted, y_true_sorted, 'o', markersize=4, alpha=0.8, label='真实值', color='black')
        
        # Plot all top4 models
        colors = ['blue', 'green', 'orange', 'purple']
        for i, pred in enumerate(top4_predictions):
            y_pred_sorted = pred[sorted_idx]
            plt.plot(X_sorted, y_pred_sorted, '-', linewidth=2, alpha=0.7, 
                    label=f'模型排名-{i+1} (种子={top4_models[i]["seed"]})',
                    color=colors[i % len(colors)])
        
        plt.xlabel('输入规模', fontsize=12)
        plt.ylabel('运行时间 (ms)', fontsize=12)
        plt.title(f'{predicted_app} 在 {host_cpu}: Top-{len(top4_models)} 模型', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        print("  显示图表...")
        plt.show()

    # 步骤 5: 导出模型为 ONNX 格式
    print(f"\n步骤 5: 导出模型为 ONNX 格式...")
    try:
        import onnx
        import onnxruntime as ort
        import os
        
        # Create onnx root folder if not exists
        onnx_root = "onnx"
        if not os.path.exists(onnx_root):
            os.makedirs(onnx_root)
        
        # Create output folder: onnx/Program-Method-CPU (e.g., onnx/MPC-curve-M7)
        output_folder = os.path.join(onnx_root, f"{predicted_app}-{PREDICT_METHOD}-{host_cpu}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Import ONNX creation functions
        from export_to_onnx import (
            create_cubic_onnx, create_quadratic_onnx, 
            create_linear_onnx, create_fft_onnx,
            export_sklearn_model_to_onnx,
            export_xgboost_to_onnx,
            export_hybrid_model_to_onnx
        )
        
        # Export all top4 models
        onnx_files = []
        failed_exports = []
        
        for idx, model_info in enumerate(top4_models):
            seed = model_info['seed']
            model_obj = model_info['model']
            scaler = model_info.get('scaler', None)
            
            try:
                # Handle curve fitting models
                if 'func' in model_obj and 'params' in model_obj:
                    func = model_obj['func']
                    params = model_obj['params']
                    
                    if func == cubic_func:
                        onnx_model = create_cubic_onnx(params, f"model_seed{seed}_cubic")
                        model_type = "cubic"
                    elif func == fft_complexity:
                        onnx_model = create_fft_onnx(params, f"model_seed{seed}_fft")
                        model_type = "fft"
                    elif func == linear_func:
                        onnx_model = create_linear_onnx(params, f"model_seed{seed}_linear")
                        model_type = "linear"
                    elif func == quad_func:
                        onnx_model = create_quadratic_onnx(params, f"model_seed{seed}_quad")
                        model_type = "quad"
                    else:
                        failed_exports.append((seed, "Unknown function type"))
                        continue
                    
                    filename = os.path.join(output_folder, f"model_seed{seed}_{model_type}.onnx")
                    onnx.save(onnx_model, filename)
                    
                    # Verify
                    session = ort.InferenceSession(filename)
                    test_input = np.array([[100.0]], dtype=np.float32)
                    onnx_pred = session.run(None, {session.get_inputs()[0].name: test_input})[0]
                    python_pred = func(100.0, *params)
                    
                    onnx_files.append(filename)
                
                # Handle XGBoost models (must check BEFORE sklearn models!)
                elif 'model' in model_obj and hasattr(model_obj.get('model'), 'get_booster'):
                    xgb_model = model_obj['model']
                    
                    onnx_model = export_xgboost_to_onnx(
                        xgb_model,
                        input_dim=1,
                        model_name=f"model_seed{seed}_xgboost"
                    )
                    
                    filename = os.path.join(output_folder, f"model_seed{seed}_xgboost.onnx")
                    onnx.save(onnx_model, filename)
                    
                    # Verify
                    session = ort.InferenceSession(filename)
                    test_input = np.array([[100.0]], dtype=np.float32)
                    onnx_pred = session.run(None, {session.get_inputs()[0].name: test_input})[0]
                    python_pred = xgb_model.predict(test_input)[0]
                    
                    onnx_files.append(filename)
                
                # Handle sklearn models (RandomForest, SVR, MLP)
                elif 'model' in model_obj and hasattr(model_obj['model'], 'predict'):
                    sklearn_model = model_obj['model']
                    model_scaler = model_obj.get('scaler', None)
                    feature_type = model_obj.get('feature_type', 'raw')
                    
                    # Determine model type
                    model_class = type(sklearn_model).__name__
                    
                    # Determine input dimension
                    if feature_type == 'n_log_n':
                        input_dim = 1  # n_log_n is single feature
                    else:
                        input_dim = 1  # raw input is single feature
                    
                    onnx_model = export_sklearn_model_to_onnx(
                        sklearn_model, 
                        model_scaler, 
                        input_dim=input_dim,
                        model_name=f"model_seed{seed}_{PREDICT_METHOD}"
                    )
                    
                    filename = os.path.join(output_folder, f"model_seed{seed}_{PREDICT_METHOD}_{model_class}.onnx")
                    onnx.save(onnx_model, filename)
                    
                    # Verify
                    session = ort.InferenceSession(filename)
                    test_input_val = 100.0
                    
                    if feature_type == 'n_log_n':
                        test_feature = np.array([[test_input_val * np.log2(test_input_val)]], dtype=np.float32)
                    else:
                        test_feature = np.array([[test_input_val]], dtype=np.float32)
                    
                    if model_scaler is not None:
                        test_feature_scaled = model_scaler.transform(test_feature)
                        # Convert to DataFrame if RandomForest to avoid warning
                        if model_class == 'RandomForestRegressor':
                            import pandas as pd
                            test_df = pd.DataFrame(test_feature_scaled, columns=['input'])
                            python_pred = sklearn_model.predict(test_df)[0]
                        else:
                            python_pred = sklearn_model.predict(test_feature_scaled)[0]
                        # For ONNX, use original input (pipeline handles scaling)
                        onnx_pred = session.run(None, {session.get_inputs()[0].name: test_feature})[0]
                    else:
                        # Convert to DataFrame if RandomForest to avoid warning
                        if model_class == 'RandomForestRegressor':
                            import pandas as pd
                            if feature_type == 'n_log_n':
                                test_df = pd.DataFrame(test_feature, columns=['n_log_n'])
                            else:
                                test_df = pd.DataFrame(test_feature, columns=['input'])
                            python_pred = sklearn_model.predict(test_df)[0]
                        else:
                            python_pred = sklearn_model.predict(test_feature)[0]
                        onnx_pred = session.run(None, {session.get_inputs()[0].name: test_feature})[0]
                    
                    onnx_files.append(filename)
                
                # Handle hybrid models
                elif 'base_func' in model_obj and 'res_model' in model_obj:
                    base_func = model_obj['base_func']
                    base_params = model_obj['base_params']
                    res_model = model_obj['res_model']
                    app_name = model_obj.get('app', predicted_app)
                    
                    onnx_result = export_hybrid_model_to_onnx(
                        base_func,
                        base_params,
                        res_model,
                        input_dim=1,
                        app_name=app_name,
                        model_name=f"model_seed{seed}_hybrid"
                    )
                    
                    # Check if result is dict (separate models) or single model
                    if isinstance(onnx_result, dict) and onnx_result.get('type') == 'separate_hybrid':
                        # Save base and residual separately
                        base_filename = os.path.join(output_folder, f"model_seed{seed}_base.onnx")
                        res_filename = os.path.join(output_folder, f"model_seed{seed}_residual.onnx")
                        
                        onnx.save(onnx_result['base'], base_filename)
                        onnx.save(onnx_result['residual'], res_filename)
                        
                        # Verify both models
                        test_input = np.array([[100.0]], dtype=np.float32)
                        
                        base_session = ort.InferenceSession(base_filename)
                        res_session = ort.InferenceSession(res_filename)
                        
                        base_onnx = base_session.run(None, {base_session.get_inputs()[0].name: test_input})[0]
                        res_onnx = res_session.run(None, {res_session.get_inputs()[0].name: test_input})[0]
                        onnx_combined = base_onnx[0][0] + res_onnx[0][0]
                        
                        # Python prediction
                        test_arr = np.array([100.0])
                        if base_params is not None:
                            # curve-based
                            base_pred = base_func(test_arr, *base_params)[0]
                        else:
                            # sklearn-based
                            base_pred = base_func.predict(test_arr.reshape(-1, 1))[0]
                        
                        # Convert to DataFrame for RandomForest to avoid warning
                        test_res_df = pd.DataFrame(test_arr.reshape(-1, 1), columns=['input'])
                        res_pred = res_model.predict(test_res_df)[0]
                        python_pred = base_pred + res_pred
                        
                        onnx_files.append(base_filename)
                        onnx_files.append(res_filename)
                    else:
                        # Single combined model
                        filename = os.path.join(output_folder, f"model_seed{seed}_hybrid.onnx")
                        onnx.save(onnx_result, filename)
                        
                        # Verify
                        session = ort.InferenceSession(filename)
                        test_input = np.array([[100.0]], dtype=np.float32)
                        onnx_pred = session.run(None, {session.get_inputs()[0].name: test_input})[0]
                        
                        # Python prediction
                        test_arr = np.array([100.0])
                        if base_params is not None:
                            # curve-based
                            base_pred = base_func(test_arr, *base_params)[0]
                        else:
                            # sklearn-based
                            base_pred = base_func.predict(test_arr.reshape(-1, 1))[0]
                        
                        # Convert to DataFrame for RandomForest to avoid warning
                        test_res_df = pd.DataFrame(test_arr.reshape(-1, 1), columns=['input'])
                        res_pred = res_model.predict(test_res_df)[0]
                        python_pred = base_pred + res_pred
                        
                        onnx_files.append(filename)
                
                else:
                    failed_exports.append((seed, "Unknown model type"))
                    continue
            
            except Exception as e:
                failed_exports.append((seed, str(e)))
                continue
        
        # Summary output
        if len(failed_exports) == 0:
            print(f"  ✓ 成功导出 {len(onnx_files)} 个ONNX模型到: {output_folder}/")
        else:
            print(f"  部分导出失败:")
            print(f"    成功: {len(onnx_files)} 个模型")
            print(f"    失败: {len(failed_exports)} 个模型")
            print(f"  失败详情:")
            for seed, error in failed_exports:
                print(f"    - Seed {seed}: {error}")
            print(f"  输出目录: {output_folder}/")
        
    except ImportError as e:
        print(f"  ✗ ONNX导出失败: 缺少必需的包")
        print(f"    请安装: pip install -r requirements.txt")
        print(f"    错误: {e}")
    except Exception as e:
        print(f"  ✗ ONNX导出失败: {e}")
        import traceback
        traceback.print_exc()

    # 步骤 6: 自动验证导出的ONNX模型（可选）
    if AUTO_VERIFY_ONNX and len(onnx_files) > 0:
        print(f"\n步骤 6: 验证导出的ONNX模型...")
        print(f"  调用 verify_onnx.py 进行模型验证...")
        
        try:
            # 动态修改 verify_onnx.py 的配置并执行
            import subprocess
            import tempfile
            import shutil
            
            # 读取 verify_onnx.py 内容
            verify_script_path = "verify_onnx.py"
            if not os.path.exists(verify_script_path):
                print(f"  ✗ 错误: 未找到 {verify_script_path}")
            else:
                with open(verify_script_path, 'r', encoding='utf-8') as f:
                    verify_content = f.read()
                
                # 替换配置参数以匹配当前训练配置
                import re
                verify_content = re.sub(
                    r"PROGRAM = '[^']*'",
                    f"PROGRAM = '{predicted_app}'",
                    verify_content
                )
                verify_content = re.sub(
                    r"CPU = '[^']*'",
                    f"CPU = '{host_cpu}'",
                    verify_content
                )
                verify_content = re.sub(
                    r"METHOD = '[^']*'",
                    f"METHOD = '{PREDICT_METHOD}'",
                    verify_content
                )
                verify_content = re.sub(
                    r"LOWER_BOUND = \d+",
                    f"LOWER_BOUND = {LOWER_BOUND}",
                    verify_content
                )
                
                # 创建临时文件
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', 
                                                delete=False, encoding='utf-8') as tmp:
                    tmp.write(verify_content)
                    tmp_path = tmp.name
                
                # 执行验证脚本
                result = subprocess.run(
                    [sys.executable, tmp_path],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                # 清理临时文件
                os.unlink(tmp_path)
                
                # 显示验证结果
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print(f"  ✗ 验证失败 (退出码: {result.returncode})")
                    if result.stderr:
                        print(f"  错误信息:\n{result.stderr}")
                
        except Exception as e:
            print(f"  ✗ 自动验证失败: {e}")
            print(f"  您可以手动运行: python verify_onnx.py")
    elif AUTO_VERIFY_ONNX:
        print(f"\n步骤 6: 跳过验证（无成功导出的模型）")

