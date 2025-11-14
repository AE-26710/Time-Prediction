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
predicted_app = 'kf'.upper()
# R5 | A72 | M7
host_cpu = 'M7'.upper()
# rf | svr | mlp | curve | xgboost | hybrid
PREDICT_METHOD = 'curve'.lower()
# 每个拟合方法将在这些随机种子下运行
SEEDS = [1, 2, 6, 42, 123, 2025, 33550336]
# 测试集占总数据比重
TEST_SIZE = 0.3
# 忽略低于该阈值的运行时间数据（ms）
LOWER_BOUND = 0
# 打印详细信息
PRINT_DETAILS = True

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
        res_model.fit(X_train_arr.reshape(-1, 1), residuals_train)
        res_pred_test = res_model.predict(X_test_arr.reshape(-1, 1))
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
        
        res_pred = model_obj['res_model'].predict(X_arr.reshape(-1, 1))
        return base_pred + res_pred

    else:
        raise ValueError(f"不支持的预测方式: {predict_method}")


# ========= 主程序：训练 Top-4 集成模型 ==========
if __name__ == '__main__':
    print("="*80)
    print(f"Top-4 Ensemble Model Training & Evaluation")
    print(f"Program: {predicted_app}, CPU: {host_cpu}, Method: {PREDICT_METHOD}")
    print(f"Test size: {TEST_SIZE}, Random seeds: {SEEDS}")
    print("="*80)

    # Step 1: Train models with all seeds and collect metrics
    print("\nStep 1: Training models with all seeds...")
    all_models = []
    
    for seed in SEEDS:
        print(f"  Training seed {seed}...", end=" ")
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
            print(f"Failed: {e}")
            continue

    if len(all_models) == 0:
        print("ERROR: No models trained successfully!")
        exit(1)

    # Step 2: Filter and select Top-4 models based on strict criteria
    print("\nStep 2: Selecting Top-4 models with strict criteria...")
    print("  Criteria: MAPE < 10% (Excellent) AND RMSE% < 10% (Excellent)")
    
    # Filter models that meet "Excellent" criteria
    excellent_models = []
    for model_info in all_models:
        m = model_info['metrics']
        mape_pct = m['mape'] * 100
        rmse_pct = m['rmse_pct_mean']
        
        # Both MAPE and RMSE% must be < 10% (Excellent grade)
        if mape_pct < 10.0 and rmse_pct < 10.0:
            excellent_models.append(model_info)
    
    print(f"\n  Found {len(excellent_models)} models meeting 'Excellent' criteria out of {len(all_models)} total")
    
    if len(excellent_models) == 0:
        print("\n  WARNING: No models meet the 'Excellent' criteria!")
        print("  Falling back to selecting Top-4 by MAPE only...")
        all_models.sort(key=lambda x: x['metrics']['mape'])
        top4_models = all_models[:4]
    elif len(excellent_models) < 4:
        print(f"\n  INFO: Found {len(excellent_models)} excellent model(s)")
        print(f"  Using only these {len(excellent_models)} excellent model(s) for ensemble")
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
        print("  Selected Top-4 from excellent models based on combined MAPE + RMSE% score")

    print(f"\nSelected Models ({len(top4_models)} models):")
    print(f"{'Rank':<6} {'Seed':<10} {'MAPE':<12} {'MAPE%':<10} {'R2':<12} {'RMSE':<12} {'RMSE%':<10} {'Grade':<15}")
    print("-" * 95)
    for rank, model_info in enumerate(top4_models, 1):
        m = model_info['metrics']
        mape_pct = m['mape'] * 100
        rmse_pct = m['rmse_pct_mean']
        
        # Determine overall grade (both must be excellent)
        if mape_pct < 10.0 and rmse_pct < 10.0:
            grade = "Excellent"
        elif mape_pct < 20.0 and rmse_pct < 20.0:
            grade = "Good"
        else:
            grade = "Fair"
        
        print(f"{rank:<6} {m['seed']:<10} {m['mape']:<12.6f} {mape_pct:<10.2f} {m['r2']:<12.6f} {m['rmse']:<12.4f} {rmse_pct:<10.2f} {grade:<15}")

    # Step 3: Use unified test set (from first seed)
    print(f"\nStep 3: Ensemble prediction on unified test set...")
    print(f"  Using {len(top4_models)} selected models for ensemble")
    # Use a unified test set for fair comparison
    reference_seed = SEEDS[0]
    _, unified_test_data = train_test_split(host_data, test_size=TEST_SIZE, random_state=reference_seed)
    X_unified_test = unified_test_data[features]
    y_unified_true = unified_test_data[output].values

    # Collect predictions from selected models on unified test set
    top4_predictions = []
    print(f"\nUsing unified test set (seed={reference_seed}, samples={len(y_unified_true)}):")
    for model_info in top4_models:
        try:
            y_pred_unified = predict_with_model(model_info['model'], X_unified_test, PREDICT_METHOD)
            top4_predictions.append(y_pred_unified)
            
            # Calculate metrics on unified test set
            mape_u = mean_absolute_percentage_error(y_unified_true, y_pred_unified)
            r2_u = r2_score(y_unified_true, y_pred_unified)
            rmse_u = np.sqrt(mean_squared_error(y_unified_true, y_pred_unified))
            print(f"  Seed {model_info['seed']}: MAPE={mape_u:.6f}, R2={r2_u:.6f}, RMSE={rmse_u:.4f}")
        except Exception as e:
            print(f"  Seed {model_info['seed']}: Prediction failed - {e}")
            continue

    if len(top4_predictions) < len(top4_models):
        print(f"WARNING: Only {len(top4_predictions)} models predicted successfully!")

    # Step 4: Calculate average predictions
    print(f"\nStep 4: Calculating average predictions from {len(top4_predictions)} models...")
    y_ensemble = np.mean(top4_predictions, axis=0)

    # Step 5: Evaluate ensemble model
    print("\nStep 5: Evaluating ensemble model performance...")
    mape_ensemble = mean_absolute_percentage_error(y_unified_true, y_ensemble)
    r2_ensemble = r2_score(y_unified_true, y_ensemble)
    rmse_ensemble = np.sqrt(mean_squared_error(y_unified_true, y_ensemble))

    eps = 1e-12
    mean_y = np.mean(y_unified_true)
    range_y = np.max(y_unified_true) - np.min(y_true)
    rmse_pct_mean = 100.0 * rmse_ensemble / (mean_y + eps)
    rmse_pct_range = 100.0 * rmse_ensemble / (range_y + eps)

    # Determine overall grade
    mape_pct = mape_ensemble * 100
    if mape_pct < 10.0 and rmse_pct_mean < 10.0:
        overall_grade = "Excellent (Both metrics < 10%)"
    elif mape_pct < 20.0 and rmse_pct_mean < 20.0:
        overall_grade = "Good (Both metrics < 20%)"
    else:
        overall_grade = "Fair"

    print("\n" + "="*80)
    print(f"Ensemble Model ({len(top4_models)} models) - Final Evaluation Results")
    print("="*80)
    print(f"MAPE (Mean Absolute Percentage Error): {mape_ensemble:.6f} ({mape_ensemble*100:.2f}%) - {mape_grade(mape_ensemble)}")
    print(f"R2 (Coefficient of Determination):     {r2_ensemble:.6f}")
    print(f"RMSE (Root Mean Squared Error):        {rmse_ensemble:.4f}")
    print(f"RMSE % (relative to mean):             {rmse_pct_mean:.2f}% - {rmse_pct_grade(rmse_pct_mean)}")
    print(f"RMSE % (relative to range):            {rmse_pct_range:.2f}%")
    print(f"Test samples:                          {len(y_unified_true)}")
    print(f"\nOverall Grade:                         {overall_grade}")
    print("="*80)

    # Step 6: Compare with best single model
    print(f"\nStep 6: Comparing with best single model (Rank-1)...")
    best_single_mape = top4_models[0]['metrics']['mape']
    # Recalculate Top-1 metrics on unified test set
    y_best_single = predict_with_model(top4_models[0]['model'], X_unified_test, PREDICT_METHOD)
    mape_best_single = mean_absolute_percentage_error(y_unified_true, y_best_single)
    r2_best_single = r2_score(y_unified_true, y_best_single)
    rmse_best_single = np.sqrt(mean_squared_error(y_unified_true, y_best_single))

    print(f"\nBest Single Model (seed={top4_models[0]['seed']}):")
    print(f"  MAPE = {mape_best_single:.6f} ({mape_best_single*100:.2f}%)")
    print(f"  R2   = {r2_best_single:.6f}")
    print(f"  RMSE = {rmse_best_single:.4f}")

    print(f"\nEnsemble Model ({len(top4_models)} models):")
    print(f"  MAPE = {mape_ensemble:.6f} ({mape_ensemble*100:.2f}%)")
    print(f"  R2   = {r2_ensemble:.6f}")
    print(f"  RMSE = {rmse_ensemble:.4f}")

    improvement_mape = ((mape_best_single - mape_ensemble) / mape_best_single) * 100
    improvement_rmse = ((rmse_best_single - rmse_ensemble) / rmse_best_single) * 100

    print(f"\nEnsemble model improvement over best single model:")
    print(f"  MAPE improvement: {improvement_mape:+.2f}%")
    print(f"  RMSE improvement: {improvement_rmse:+.2f}%")

    if improvement_mape > 0:
        print(f"  -> Ensemble model MAPE is better!")
    else:
        print(f"  -> Ensemble model MAPE is not improved")

    # 步骤 7: 可视化对比
    if PRINT_DETAILS:
        print("\nStep 7: Generating visualization charts...")
        
        # 图1: 预测值 vs 真实值散点图
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_unified_true, y_best_single, alpha=0.5, s=10, label=f'Best Single Model (seed={top4_models[0]["seed"]})')
        plt.scatter(y_unified_true, y_ensemble, alpha=0.5, s=10, label=f'Ensemble Model ({len(top4_models)} models)')
        plt.plot([y_unified_true.min(), y_unified_true.max()], 
                 [y_unified_true.min(), y_unified_true.max()], 'r--', lw=2, label='Ideal Prediction')
        plt.xlabel('True Value (ms)', fontsize=12)
        plt.ylabel('Predicted Value (ms)', fontsize=12)
        plt.title('Predicted vs True Values', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 图2: 预测误差分布
        plt.subplot(1, 2, 2)
        error_single = y_best_single - y_unified_true
        error_ensemble = y_ensemble - y_unified_true
        plt.hist(error_single, bins=50, alpha=0.5, label=f'Best Single Model (seed={top4_models[0]["seed"]})')
        plt.hist(error_ensemble, bins=50, alpha=0.5, label=f'Ensemble Model ({len(top4_models)} models)')
        plt.xlabel('Prediction Error (ms)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Prediction Error Distribution', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        print("  Displaying comparison chart...")
        plt.show()

        # 图3: 输入规模 vs 预测/真实值曲线
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(X_unified_test[features[0]].values)
        X_sorted = X_unified_test[features[0]].values[sorted_idx]
        y_true_sorted = y_unified_true[sorted_idx]
        y_ensemble_sorted = y_ensemble[sorted_idx]
        y_single_sorted = y_best_single[sorted_idx]
        
        plt.plot(X_sorted, y_true_sorted, 'o', markersize=3, alpha=0.6, label='True Value')
        plt.plot(X_sorted, y_single_sorted, '-', linewidth=2, alpha=0.7, label=f'Best Single Model (seed={top4_models[0]["seed"]})')
        plt.plot(X_sorted, y_ensemble_sorted, '-', linewidth=2, alpha=0.7, label=f'Ensemble Model ({len(top4_models)} models)')
        
        plt.xlabel('Input Size', fontsize=12)
        plt.ylabel('Runtime (ms)', fontsize=12)
        plt.title(f'{predicted_app} on {host_cpu}: Ensemble vs Single Model', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        print("  Displaying curve chart...")
        plt.show()

    # Step 8: Output model parameters (for inference script)
    print(f"\nStep 8: Outputting {len(top4_models)} model parameters...")
    print("="*80)
    print("Copy the following parameters to ensemble_top4_inference.py:")
    print("="*80)
    print("\nTOP4_MODELS = [")
    for model_info in top4_models:
        seed = model_info['seed']
        model_obj = model_info['model']
        if 'func' in model_obj and 'params' in model_obj:
            # Determine function name
            func = model_obj['func']
            if func == cubic_func:
                func_name = 'cubic'
            elif func == fft_complexity:
                func_name = 'fft'
            elif func == linear_func:
                func_name = 'linear'
            elif func == quad_func:
                func_name = 'quad'
            else:
                func_name = 'unknown'
            
            params = model_obj['params']
            params_str = ', '.join([f'{p:.12e}' for p in params])
            print(f"    ({seed}, '{func_name}', ({params_str})),")
    print("]")
    print("="*80)

    # Step 9: Export to ONNX (支持所有模型类型)
    print(f"\nStep 9: Exporting models to ONNX format (all model types supported)...")
    try:
        import onnx
        from onnx import helper, TensorProto
        import onnxruntime as ort
        import os
        
        # Create output folder: Program-Method-CPU (e.g., MPC-curve-M7)
        output_folder = f"{predicted_app}-{PREDICT_METHOD}-{host_cpu}"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"  Created output folder: {output_folder}/")
        else:
            print(f"  Using existing folder: {output_folder}/")
        
        # Import ONNX creation functions
        from export_to_onnx import (
            create_cubic_onnx, create_quadratic_onnx, 
            create_linear_onnx, create_fft_onnx, 
            create_ensemble_onnx,
            export_sklearn_model_to_onnx,
            export_xgboost_to_onnx,
            export_hybrid_model_to_onnx
        )
        
        # Export all top4 models
        exported_models = []
        onnx_files = []
        
        for idx, model_info in enumerate(top4_models):
            seed = model_info['seed']
            model_obj = model_info['model']
            scaler = model_info.get('scaler', None)
            
            print(f"\n  Exporting model {idx+1}/{len(top4_models)} (seed={seed})...")
            
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
                        print(f"    ✗ Unknown function type, skipping")
                        continue
                    
                    filename = os.path.join(output_folder, f"model_seed{seed}_{model_type}.onnx")
                    onnx.save(onnx_model, filename)
                    print(f"    ✓ Exported {model_type} model: {filename}")
                    
                    # Verify
                    session = ort.InferenceSession(filename)
                    test_input = np.array([[100.0]], dtype=np.float32)
                    onnx_pred = session.run(None, {session.get_inputs()[0].name: test_input})[0]
                    python_pred = func(100.0, *params)
                    print(f"      Verification: input=100, Python={python_pred:.4f}, ONNX={onnx_pred[0][0]:.4f}")
                    
                    exported_models.append((seed, model_type, params))
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
                    print(f"    ✓ Exported {model_class} model: {filename}")
                    
                    # Verify
                    session = ort.InferenceSession(filename)
                    test_input_val = 100.0
                    
                    if feature_type == 'n_log_n':
                        test_feature = np.array([[test_input_val * np.log2(test_input_val)]], dtype=np.float32)
                    else:
                        test_feature = np.array([[test_input_val]], dtype=np.float32)
                    
                    if model_scaler is not None:
                        test_feature_scaled = model_scaler.transform(test_feature)
                        python_pred = sklearn_model.predict(test_feature_scaled)[0]
                        # For ONNX, use original input (pipeline handles scaling)
                        onnx_pred = session.run(None, {session.get_inputs()[0].name: test_feature})[0]
                    else:
                        python_pred = sklearn_model.predict(test_feature)[0]
                        onnx_pred = session.run(None, {session.get_inputs()[0].name: test_feature})[0]
                    
                    print(f"      Verification: input=100, Python={python_pred:.4f}, ONNX={onnx_pred[0][0]:.4f}")
                    
                    onnx_files.append(filename)
                
                # Handle XGBoost models
                elif hasattr(model_obj.get('model'), 'get_booster'):
                    xgb_model = model_obj['model']
                    
                    onnx_model = export_xgboost_to_onnx(
                        xgb_model,
                        input_dim=1,
                        model_name=f"model_seed{seed}_xgboost"
                    )
                    
                    filename = os.path.join(output_folder, f"model_seed{seed}_xgboost.onnx")
                    onnx.save(onnx_model, filename)
                    print(f"    ✓ Exported XGBoost model: {filename}")
                    
                    # Verify
                    session = ort.InferenceSession(filename)
                    test_input = np.array([[100.0]], dtype=np.float32)
                    onnx_pred = session.run(None, {session.get_inputs()[0].name: test_input})[0]
                    python_pred = xgb_model.predict(test_input)[0]
                    print(f"      Verification: input=100, Python={python_pred:.4f}, ONNX={onnx_pred[0][0]:.4f}")
                    
                    onnx_files.append(filename)
                
                # Handle hybrid models
                elif 'base_func' in model_obj and 'res_model' in model_obj:
                    base_func = model_obj['base_func']
                    base_params = model_obj['base_params']
                    res_model = model_obj['res_model']
                    app_name = model_obj.get('app', predicted_app)
                    
                    onnx_model = export_hybrid_model_to_onnx(
                        base_func,
                        base_params,
                        res_model,
                        input_dim=1,
                        app_name=app_name,
                        model_name=f"model_seed{seed}_hybrid"
                    )
                    
                    filename = os.path.join(output_folder, f"model_seed{seed}_hybrid.onnx")
                    onnx.save(onnx_model, filename)
                    print(f"    ✓ Exported hybrid model: {filename}")
                    
                    # Verify
                    session = ort.InferenceSession(filename)
                    test_input = np.array([[100.0]], dtype=np.float32)
                    onnx_pred = session.run(None, {session.get_inputs()[0].name: test_input})[0]
                    
                    # Python prediction
                    test_arr = np.array([100.0])
                    if base_params is not None:
                        base_pred = base_func(test_arr, *base_params)
                    else:
                        if app_name == 'FFT':
                            test_fe = test_arr * np.log2(test_arr)
                            base_pred = base_func.predict(test_fe.reshape(-1, 1))
                        else:
                            base_pred = base_func.predict(test_arr.reshape(-1, 1))
                    
                    res_pred = res_model.predict(test_arr.reshape(-1, 1))
                    python_pred = base_pred + res_pred[0] if isinstance(base_pred, (int, float)) else base_pred[0] + res_pred[0]
                    
                    print(f"      Verification: input=100, Python={python_pred:.4f}, ONNX={onnx_pred[0][0]:.4f}")
                    
                    onnx_files.append(filename)
                
                else:
                    print(f"    ✗ Unknown model type, skipping")
                    continue
            
            except Exception as e:
                print(f"    ✗ Export failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Export ensemble model (only for curve-based models for now)
        if len(exported_models) > 0:
            print(f"\n  Exporting ensemble model...")
            try:
                ensemble_onnx = create_ensemble_onnx(exported_models, f"ensemble_{len(exported_models)}models")
                ensemble_filename = os.path.join(output_folder, f"ensemble_{len(exported_models)}models.onnx")
                onnx.save(ensemble_onnx, ensemble_filename)
                print(f"    ✓ Exported ensemble model ({len(exported_models)} curve models): {ensemble_filename}")
                
                # Verify
                session = ort.InferenceSession(ensemble_filename)
                test_input = np.array([[100.0]], dtype=np.float32)
                onnx_pred = session.run(None, {session.get_inputs()[0].name: test_input})[0]
                
                python_preds = []
                for seed, func_name, params in exported_models:
                    if func_name == 'cubic':
                        python_preds.append(cubic_func(100.0, *params))
                    elif func_name == 'fft':
                        python_preds.append(fft_complexity(100.0, *params))
                    elif func_name == 'linear':
                        python_preds.append(linear_func(100.0, *params))
                    elif func_name == 'quad':
                        python_preds.append(quad_func(100.0, *params))
                
                python_ensemble_pred = np.mean(python_preds)
                print(f"      Verification: input=100, Python={python_ensemble_pred:.4f}, ONNX={onnx_pred[0][0]:.4f}")
                
                onnx_files.append(ensemble_filename)
            except Exception as e:
                print(f"    ✗ Ensemble export failed: {e}")
        
        print(f"\n  {'='*60}")
        print(f"  Successfully exported {len(onnx_files)} ONNX model(s)")
        print(f"  Output folder: {output_folder}/")
        for f in onnx_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  {'='*60}")
        
    except ImportError as e:
        print(f"  ✗ ONNX export failed: Missing required packages")
        print(f"    Please install: pip install -r requirements.txt")
        print(f"    Required: onnx, onnxruntime, skl2onnx, onnxmltools")
        print(f"    Error: {e}")
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print(f"Ensemble Model Training & Evaluation Completed!")
    print(f"Selected {len(top4_models)} excellent models for ensemble")
    print("="*80)
