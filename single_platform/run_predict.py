import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from scipy.optimize import curve_fit
from xgboost import XGBRegressor

# ========== 配置区 ==========
# KF | FFT | AES | MD5 | SHA256 | MPC
predicted_app = 'MPC'.upper()
# R5 | A72 | M7
host_cpu = 'A72'.upper()
# rf | svr | mlp | curve | xgboost | hybrid
PREDICT_METHOD = 'curve'.lower()
# 每个拟合方法将在这些随机种子下运行
SEEDS = [1, 2, 6, 42, 123, 2025, 33550336]
# 测试集占总数据比重
TEST_SIZE = 0.3
# 忽略低于该阈值的运行时间数据（ms）
LOWER_BOUND = 0
# 打印详细信息（每个随机种子的结果、图形化评估）
PRINT_DETAILS = True

# ========== 数据准备 ==========
if predicted_app not in ('AES','MD5','SHA256'):
    data = pd.read_csv("exclusive_runtime.csv")
else:
    data = pd.read_csv("exclusive_runtime_encrypt.csv")
host_data = data[(data['cpu'] == host_cpu) & (data['program'] == predicted_app)]
host_data = host_data[(host_data['time'] > LOWER_BOUND)]

#host_data = host_data.sample(frac=0.01, random_state=42)

features = ['input']
output = 'time'

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

# ========= 运行单次预测 ==========
def run_one(seed: int, predict_method: str = PREDICT_METHOD):
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

    #随机森林回归RF,对非线性、异常值鲁棒，外推弱
    if predict_method == 'rf':
        """
        n_estimators：森林中树的数量，默认100。增大可提升效果但计算更慢。
        max_depth：每棵树的最大深度，默认None（直到叶节点纯或样本数小于min_samples_split）。可防止过拟合。
        min_samples_split：内部节点再划分所需最小样本数，默认2。增大可防止过拟合。
        min_samples_leaf：叶节点最少样本数，默认1。增大可使模型更平滑。
        max_features：每次分裂考虑的最大特征数，默认“auto”（即sqrt(n_features)）。
        random_state：随机种子，保证结果可复现。
        n_jobs：并行运行的CPU核数，-1表示用所有核。
        bootstrap：是否有放回抽样，默认True。
        oob_score：是否使用袋外样本估算泛化精度，默认False。
        criterion：分裂节点的指标，回归中常用“squared_error”（均方误差）。
        """
        if predicted_app == 'FFT':
            X_train_fe = train_data[['input']].copy()
            X_test_fe = test_data[['input']].copy()
            X_train_fe['n_log_n'] = X_train_fe['input'] * np.log2(X_train_fe['input'])
            X_test_fe['n_log_n'] = X_test_fe['input'] * np.log2(X_test_fe['input'])
            model = RandomForestRegressor(
                n_estimators=200, 
                random_state=seed, 
                n_jobs=-1
                )
            model.fit(X_train_fe[['n_log_n']], y_train)
            y_pred = model.predict(X_test_fe[['n_log_n']])
        else:
            model = RandomForestRegressor(
                n_estimators=100, 
                random_state=seed, 
                n_jobs=-1
                )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

    #支持向量回归SVR,对单变量平滑非线性拟合好；对尺度敏感需标准化；外推弱
    elif predict_method == 'svr':
        """
        kernel：核函数类型。常用有 'rbf'（高斯径向基，默认）、'linear'（线性）、'poly'（多项式）、'sigmoid'。
        C：惩罚系数，默认1.0。C越大，对误差的惩罚越强，模型更容易过拟合；C越小，容忍误差，模型更平滑。
        epsilon：ε-不敏感损失函数的宽度，默认0.1。决定了预测值与真实值之间多大误差会被忽略。
        degree：多项式核函数的次数，仅对 kernel='poly' 有效，默认3。
        gamma：核函数系数，'scale'（默认）或 'auto'，也可设为具体数值。影响非线性拟合能力。
        coef0：核函数中的常数项，对 'poly' 和 'sigmoid' 有效，默认0.0。
        shrinking：是否使用启发式收缩法，默认True。
        tol：停止训练的容忍度，默认1e-3。
        max_iter：最大迭代次数，默认-1（不限制）。
        """
        model = SVR(
            kernel='rbf', 
            C=1000, 
            epsilon=0.5
            )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    #多层感知机回归MLP,表达力强，可拟合复杂曲线；需要较多数据与正则/早停来防过拟合；对尺度敏感；外推一般
    elif predict_method == 'mlp':
        if predicted_app == 'FFT':
            X_train_fe = train_data[['input']].copy()
            X_test_fe = test_data[['input']].copy()
            X_train_fe['n_log_n'] = X_train_fe['input'] * np.log2(X_train_fe['input'])
            X_test_fe['n_log_n'] = X_test_fe['input'] * np.log2(X_test_fe['input'])
            scaler_mlp = StandardScaler()
            X_train_m = scaler_mlp.fit_transform(X_train_fe[['n_log_n']])
            X_test_m = scaler_mlp.transform(X_test_fe[['n_log_n']])
            mlp = MLPRegressor(
                hidden_layer_sizes=(100, 100), 
                activation='relu', 
                max_iter=8000, 
                random_state=seed
                )
            mlp.fit(X_train_m, y_train)
            y_pred = mlp.predict(X_test_m)
        else:
            mlp = MLPRegressor(
                hidden_layer_sizes=(100, 100), 
                activation='relu', 
                max_iter=8000, 
                random_state=seed
                )
            mlp.fit(X_train_scaled, y_train)
            y_pred = mlp.predict(X_test_scaled)

    # 曲线拟合curve，对已知复杂度的程序效果好，外推能力强
    elif predict_method == 'curve':
        X_train_cf = X_train[features[0]].values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_cf = y_train.values if hasattr(y_train, 'values') else y_train
        X_test_cf = X_test[features[0]].values if isinstance(X_test, pd.DataFrame) else X_test

        if predicted_app == 'KF':
            def cubic_func(N, a, b, c, d):
                return a * N**3 + b * N**2 + c * N + d

            params, _ = curve_fit(cubic_func, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = cubic_func(X_test_cf, *params)
            a_fit, b_fit, c_fit, d_fit = params
            print(f"拟合公式 (cubic): T(N) = {a_fit} * N^3 + {b_fit} * N^2 + {c_fit} * N + {d_fit:.6f}")

        elif predicted_app == 'FFT':
            def fft_complexity(N, a, b):
                return a * N * np.log2(N) + b

            params, _ = curve_fit(fft_complexity, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = fft_complexity(X_test_cf, *params)
            a_fit, b_fit = params
            print(f"拟合公式 (fft): T(N) = {a_fit} * N log2(N) + {b_fit:.6f}")

        elif predicted_app in ('AES', 'MD5', 'SHA256'):
            # Linear complexity: T(N) = a * N + b
            def linear_func(N, a, b):
                return a * N + b

            params, _ = curve_fit(linear_func, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = linear_func(X_test_cf, *params)
            a_fit, b_fit = params
            print(f"拟合公式 (linear): T(N) = {a_fit} * N + {b_fit:.6f}")

        elif predicted_app == 'MPC' and host_cpu == 'A72':
            # Quadratic complexity hypothesis for MPC on A72: T(N) = a*N^2 + b*N + c
            def quad_func(N, a, b, c):
                return a * N**2 + b * N + c

            params, _ = curve_fit(quad_func, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = quad_func(X_test_cf, *params)
            a_fit, b_fit, c_fit = params
            print(f"拟合公式 (quadratic): T(N) = {a_fit} * N^2 + {b_fit} * N + {c_fit:.6f}")

        else:
            def cubic_func(N, a, b, c, d):
                return a * N**3 + b * N**2 + c * N + d

            params, _ = curve_fit(cubic_func, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = cubic_func(X_test_cf, *params)

    # 极端梯度提升XGBoost,表达力强，处理大数据和高维特征好；对异常值敏感；外推能力一般
    elif predict_method == 'xgboost':
        model = XGBRegressor(
            n_estimators=500,
            max_depth=15,
            learning_rate=0.05,
            random_state=seed,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # 解析先验 + 残差学习HYBRID，兼顾可解释性与灵活性，比单一模型更稳健
    elif predict_method == 'hybrid':
        X_train_arr = X_train[features[0]].values if isinstance(X_train, pd.DataFrame) else X_train
        X_test_arr = X_test[features[0]].values if isinstance(X_test, pd.DataFrame) else X_test
        y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train

        if predicted_app in ('KF'):
            def cubic_func(N, a, b, c, d):
                return a * N**3 + b * N**2 + c * N + d

            params, _ = curve_fit(cubic_func, X_train_arr, y_train_arr, maxfev=10000)
            base_train = cubic_func(X_train_arr, *params)
            base_test = cubic_func(X_test_arr, *params)

        elif predicted_app == 'FFT':
            X_train_fe = pd.DataFrame({'input': X_train_arr})
            X_test_fe = pd.DataFrame({'input': X_test_arr})
            X_train_fe['n_log_n'] = X_train_fe['input'] * np.log2(X_train_fe['input'])
            X_test_fe['n_log_n'] = X_test_fe['input'] * np.log2(X_test_fe['input'])
            from sklearn.linear_model import LinearRegression
            base_lr = LinearRegression().fit(X_train_fe[['n_log_n']], y_train_arr)
            base_train = base_lr.predict(X_train_fe[['n_log_n']])
            base_test = base_lr.predict(X_test_fe[['n_log_n']])
        elif predicted_app in ('AES', 'MD5', 'SHA256') or (predicted_app == 'MPC' and host_cpu == 'M7'):
            # Linear base model for cryptographic/hash functions
            from sklearn.linear_model import LinearRegression
            X_train_lin = X_train_arr.reshape(-1, 1)
            X_test_lin = X_test_arr.reshape(-1, 1)
            base_lr = LinearRegression().fit(X_train_lin, y_train_arr)
            base_train = base_lr.predict(X_train_lin)
            base_test = base_lr.predict(X_test_lin)
        elif predicted_app == 'MPC' and host_cpu in ('A72', 'R5'):
            # Quadratic base model hypothesis for MPC on A72: T(N) = a*N^2 + b*N + c
            def quad_base(N, a, b, c):
                return a * N**2 + b * N + c

            # Fit via numpy polyfit for stability on 2nd-degree polynomial
            coefs = np.polyfit(X_train_arr, y_train_arr, 2)
            # np.polyfit returns [a, b, c]
            a, b, c = coefs
            base_train = quad_base(X_train_arr, a, b, c)
            base_test = quad_base(X_test_arr, a, b, c)
        else:
            raise ValueError(f"不支持的程序类型：{predicted_app}")

        residuals_train = y_train_arr - base_train
        res_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=seed, n_jobs=-1)
        res_model.fit(X_train_arr.reshape(-1, 1), residuals_train)
        res_pred_test = res_model.predict(X_test_arr.reshape(-1, 1))
        y_pred = base_test + res_pred_test

    else:
        raise ValueError(f"不支持的预测方式: {predict_method}")

    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    eps = 1e-12
    mean_y = np.mean(y_true)
    range_y = np.max(y_true) - np.min(y_true)
    max_y = np.max(y_true)

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
    return metrics, y_true, y_pred


if __name__ == '__main__':
    print(f"Predicting runtime for {predicted_app} on {host_cpu}")
    print(f"Test size: {TEST_SIZE}, Lower bound: {LOWER_BOUND}")
    print(f"Method: {PREDICT_METHOD}")
    all_metrics = []
    last_true = None
    last_pred = None
    all_results = []  # 存储所有运行结果

    for seed in SEEDS:
        try:
            metrics, y_true_seed, y_pred_seed = run_one(seed)
            all_metrics.append(metrics)
            last_true = np.array(y_true_seed)
            last_pred = np.array(y_pred_seed)
            all_results.append((np.array(y_true_seed), np.array(y_pred_seed), seed))  # 保存所有结果
            grade = mape_grade(metrics['mape'])
            rmse_mean_grade = rmse_pct_grade(metrics['rmse_pct_mean'])
            rmse_range_grade = rmse_pct_grade(metrics['rmse_pct_range'])
            if PRINT_DETAILS:
                print(
                    f"Seed {seed}: MAPE={metrics['mape']:.4f} ({grade}), R2={metrics['r2']:.4f}, "
                    f"RMSE={metrics['rmse']:.4f} (ms), RMSE%_mean={metrics['rmse_pct_mean']:.2f}% ({rmse_mean_grade}), "
                    f"RMSE%_range={metrics['rmse_pct_range']:.2f}% ({rmse_range_grade}), N={metrics['n_test']}"
                )
        except Exception as exc:
            print(f"Seed {seed} failed: {exc}")

    if all_metrics:
        df = pd.DataFrame(all_metrics)
        agg = {
            'mape_mean': df['mape'].mean(),
            'mape_std': df['mape'].std(),
            'r2_mean': df['r2'].mean(),
            'r2_std': df['r2'].std(),
            'rmse_mean': df['rmse'].mean(),
            'rmse_std': df['rmse'].std(),
            'rmse_pct_mean_mean': df['rmse_pct_mean'].mean(),
            'rmse_pct_mean_std': df['rmse_pct_mean'].std(),
            'rmse_pct_range_mean': df['rmse_pct_range'].mean(),
            'rmse_pct_range_std': df['rmse_pct_range'].std(),
            
            'total_tests': df['n_test'].sum()
        }
        # 输出汇总结果与总体评估
        print(f"\nTotal tests: {agg['total_tests']}")
        print(f"MAPE: {agg['mape_mean']:.4f} ± {agg['mape_std']:.4f} ({mape_grade(agg['mape_mean'])})")
        print(f"R2:   {agg['r2_mean']:.4f} ± {agg['r2_std']:.4f}")
        print(f"RMSE: {agg['rmse_mean']:.4f} ± {agg['rmse_std']:.4f} (ms)")

        metrics_pct = [
            ("RMSE%_mean", agg['rmse_pct_mean_mean'], agg['rmse_pct_mean_std'], rmse_pct_grade),
            ("RMSE%_range", agg['rmse_pct_range_mean'], agg['rmse_pct_range_std'], rmse_pct_grade),
        ]
        for name, mean_val, std_val, grade_fn in metrics_pct:
            print(f"{name}: {mean_val:.2f}% ± {std_val:.2f}% ({grade_fn(mean_val)})")

        # 计算总体评估（取最差等级）
        grades = {
            'MAPE': mape_grade(agg['mape_mean']),
            'RMSE%_mean': rmse_pct_grade(agg['rmse_pct_mean_mean']),
            'RMSE%_range': rmse_pct_grade(agg['rmse_pct_range_mean'])
        }
        grade_order = {'优秀': 0, '中等': 1, '差': 2}
        worst_metric, worst_grade = max(grades.items(), key=lambda kv: grade_order.get(kv[1], 2))
        print(f"\nOverall assessment (worst of MAPE / RMSE%_mean / RMSE%_range): {worst_grade}")
        
        # 绘制所有运行的预测结果散点图
        if all_results and PRINT_DETAILS:
            plt.figure(figsize=(10, 8))
            
            # 定义不同的颜色用于区分不同的随机种子
            colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
            
            # 计算全局最小最大值用于绘制对角线
            all_true_vals = np.concatenate([result[0] for result in all_results])
            all_pred_vals = np.concatenate([result[1] for result in all_results])
            min_val = min(all_true_vals.min(), all_pred_vals.min())
            max_val = max(all_true_vals.max(), all_pred_vals.max())
            
            # 绘制每个随机种子的结果
            for idx, (y_true_i, y_pred_i, seed_i) in enumerate(all_results):
                plt.scatter(y_true_i, y_pred_i, color=colors[idx], alpha=1, s=15, 
                           edgecolor='k', linewidth=0.2, label=f'Seed {seed_i}')
            
            # FFT 使用对数轴以更好地展示比例关系
            if predicted_app == 'FFT':
                plt.xscale('log')
                plt.yscale('log')
            
            # 绘制完美预测线
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', 
                    linewidth=2, label='Perfect Prediction', zorder=100)
            
            plt.title(f"Predicted vs True runtime for {predicted_app} on {host_cpu}", fontsize=16)
            plt.xlabel("True runtime (ms)", fontsize=14)
            plt.ylabel("Predicted runtime (ms)", fontsize=14)
            plt.legend(fontsize=10, loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

# 显示输入规模 vs 运行时间的散点图
if PRINT_DETAILS:
    plt.figure(figsize=(8, 6))
    plt.scatter(host_data['input'], host_data['time'], color="#1500FF", alpha=1, s=1, label='Measured Data')
    # 对于 FFT，输入规模通常为 2 的幂次，使用 log2 横轴更直观；否则使用线性横轴
    if predicted_app == 'FFT':
        plt.xscale('log', base=2)
        plt.yscale('log')
        inputs = host_data['input'].values
        # 仅保留正值，避免 log2 出错
        inputs_pos = inputs[inputs > 0]
        if inputs_pos.size > 0:
            powers = np.unique(np.floor(np.log2(inputs_pos)).astype(int))
            xticks = (2 ** powers).astype(int)
            # show numeric tick labels (no special annotation)
            plt.xticks(xticks)
    # generic labels (no extra '(log)' annotations)
    plt.title(f"Input vs Time for {predicted_app} on {host_cpu}", fontsize=16)
    plt.xlabel("Input Size", fontsize=14)
    plt.ylabel("Execution Time (ms)", fontsize=14)
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()
    plt.show()
