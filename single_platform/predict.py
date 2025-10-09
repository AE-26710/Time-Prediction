import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
from sklearn.neural_network import MLPRegressor
from scipy.optimize import curve_fit
from xgboost import XGBRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor

# ========== 配置区 ==========
# Matrix_Multiply | FFT | KF
predicted_app = 'Matrix_Multiply'
host_cpu = 'Cortex-R5F'
# random_forest | random_forest_tuned | svr | svr_tuned | mlp_tuned | curve_fit | xgboost | xgboost_tuned | hybrid
PREDICT_METHOD = 'random_forest'
SEEDS = [1, 2, 6, 42, 123, 2025, 33550336]
TEST_SIZE = 0.1

# ========== 数据准备 ==========
data = pd.read_csv("exclusive_runtime.csv")
host_data = data[(data['cpu'] == host_cpu) & (data['program'] == predicted_app)]
features = ['input']
output = 'time'

# ========= 评估函数 ==========
def mape_grade(mape_value: float) -> str:
    perc = mape_value * 100
    if perc < 10:
        return "优秀"
    if perc < 20:
        return "良好"
    return "差"


def rmse_pct_grade(pct_value: float) -> str:
    if pct_value < 10:
        return "优秀"
    if pct_value < 20:
        return "良好"
    if pct_value < 50:
        return "需改进"
    return "差"


def nrmse_std_grade(nrmse_value: float) -> str:
    if nrmse_value < 0.5:
        return "优秀"
    if nrmse_value < 1.0:
        return "需改进"
    return "差"


def run_one(seed: int):
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

    if PREDICT_METHOD == 'random_forest':
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
            model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
            model.fit(X_train_fe[['n_log_n']], y_train)
            y_pred = model.predict(X_test_fe[['n_log_n']])
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

    elif PREDICT_METHOD == 'random_forest_tuned':
        rf = RandomForestRegressor(random_state=seed)
        param_dist_rf = {
            'n_estimators': stats.randint(50, 400),
            'max_depth': [None] + list(range(3, 31)),
            'min_samples_split': stats.randint(2, 11),
            'min_samples_leaf': stats.randint(1, 11),
            'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7, None],
            'bootstrap': [True, False]
        }
        rnd_rf = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist_rf,
            n_iter=30,
            scoring='neg_mean_absolute_percentage_error',
            cv=3,
            random_state=seed,
            n_jobs=-1,
            verbose=1
        )
        if predicted_app == 'FFT':
            X_train_fe = train_data[['input']].copy()
            X_test_fe = test_data[['input']].copy()
            X_train_fe['n_log_n'] = X_train_fe['input'] * np.log2(X_train_fe['input'])
            X_test_fe['n_log_n'] = X_test_fe['input'] * np.log2(X_test_fe['input'])
            rnd_rf.fit(X_train_fe[['n_log_n']], y_train)
            print('Best params (random_forest_tuned, FFT):', rnd_rf.best_params_)
            y_pred = rnd_rf.predict(X_test_fe[['n_log_n']])
        else:
            rnd_rf.fit(X_train, y_train)
            print('Best params (random_forest_tuned):', rnd_rf.best_params_)
            y_pred = rnd_rf.predict(X_test)

    elif PREDICT_METHOD == 'svr':
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
        model = SVR(kernel='rbf', C=1000, epsilon=0.5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    elif PREDICT_METHOD == 'svr_tuned':
        base_svr = SVR()
        param_dist_svr = {
            'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
            'C': stats.loguniform(1e-2, 1e4),
            'epsilon': stats.uniform(1e-4, 1.0),
            'gamma': stats.loguniform(1e-6, 1e1),
            'degree': stats.randint(2, 6)
        }
        rnd_svr = RandomizedSearchCV(
            estimator=base_svr,
            param_distributions=param_dist_svr,
            n_iter=30,
            scoring='neg_mean_absolute_percentage_error',
            cv=3,
            random_state=seed,
            n_jobs=-1,
            verbose=1
        )
        rnd_svr.fit(X_train_scaled, y_train)
        print('Best params (svr_tuned):', rnd_svr.best_params_)
        y_pred = rnd_svr.predict(X_test_scaled)

    elif PREDICT_METHOD == 'mlp_tuned':
        base_mlp = MLPRegressor(max_iter=2000, random_state=seed)
        param_dist = {
            'regressor__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100), (200, 100, 50)],
            'regressor__activation': ['relu', 'tanh', 'logistic'],
            'regressor__solver': ['adam', 'lbfgs'],
            'regressor__alpha': stats.loguniform(1e-6, 1e-1),
            'regressor__learning_rate_init': stats.loguniform(1e-4, 1e-1),
        }
        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        ttr = TransformedTargetRegressor(regressor=base_mlp, transformer=transformer)
        rnd = RandomizedSearchCV(
            estimator=ttr,
            param_distributions=param_dist,
            n_iter=20,
            scoring='neg_mean_absolute_percentage_error',
            cv=3,
            random_state=seed,
            n_jobs=-1,
            verbose=1
        )
        rnd.fit(X_train_scaled, y_train)
        print('Best params (mlp_tuned):', rnd.best_params_)
        y_pred = rnd.predict(X_test_scaled)

    elif PREDICT_METHOD == 'curve_fit':
        X_train_cf = X_train[features[0]].values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_cf = y_train.values if hasattr(y_train, 'values') else y_train
        X_test_cf = X_test[features[0]].values if isinstance(X_test, pd.DataFrame) else X_test

        if predicted_app in ('KF', 'Matrix_Multiply'):
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

        else:
            def cubic_func(N, a, b, c, d):
                return a * N**3 + b * N**2 + c * N + d

            params, _ = curve_fit(cubic_func, X_train_cf, y_train_cf, maxfev=10000)
            y_pred = cubic_func(X_test_cf, *params)

    elif PREDICT_METHOD == 'xgboost':
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

    elif PREDICT_METHOD == 'xgboost_tuned':
        xgb = XGBRegressor(objective='reg:squarederror', random_state=seed, verbosity=0)
        param_dist_xgb = {
            'n_estimators': stats.randint(50, 1000),
            'max_depth': stats.randint(3, 16),
            'learning_rate': stats.loguniform(1e-3, 0.5),
            'subsample': stats.uniform(0.5, 0.5),
            'colsample_bytree': stats.uniform(0.5, 0.5),
            'gamma': stats.loguniform(1e-8, 10.0),
            'reg_alpha': stats.loguniform(1e-8, 1.0),
            'reg_lambda': stats.loguniform(1e-8, 10.0)
        }
        rnd_xgb = RandomizedSearchCV(
            estimator=xgb,
            param_distributions=param_dist_xgb,
            n_iter=40,
            scoring='neg_mean_absolute_percentage_error',
            cv=3,
            random_state=seed,
            n_jobs=-1,
            verbose=1
        )
        rnd_xgb.fit(X_train, y_train)
        print('Best params (xgboost_tuned):', rnd_xgb.best_params_)
        y_pred = rnd_xgb.predict(X_test)

    elif PREDICT_METHOD == 'hybrid':
        X_train_arr = X_train[features[0]].values if isinstance(X_train, pd.DataFrame) else X_train
        X_test_arr = X_test[features[0]].values if isinstance(X_test, pd.DataFrame) else X_test
        y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train

        if predicted_app in ('KF', 'Matrix_Multiply'):
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
        else:
            raise ValueError(f"未知的程序类型：{predicted_app}")

        residuals_train = y_train_arr - base_train
        res_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=seed, n_jobs=-1)
        res_model.fit(X_train_arr.reshape(-1, 1), residuals_train)
        res_pred_test = res_model.predict(X_test_arr.reshape(-1, 1))
        y_pred = base_test + res_pred_test

    else:
        raise ValueError(f"未知的预测方式: {PREDICT_METHOD}")

    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Normalized / percent RMSE variants
    eps = 1e-12
    mean_y = np.mean(y_true)
    range_y = np.max(y_true) - np.min(y_true)
    max_y = np.max(y_true)
    std_y = np.std(y_true, ddof=0)

    rmse_pct_mean = 100.0 * rmse / (mean_y + eps)
    rmse_pct_range = 100.0 * rmse / (range_y + eps)
    rmse_pct_max = 100.0 * rmse / (max_y + eps)
    nrmse_std = rmse / (std_y + eps)

    metrics = {
        'seed': seed,
        'mape': mape,
        'r2': r2,
        'rmse': rmse,
        'rmse_pct_mean': rmse_pct_mean,
        'rmse_pct_range': rmse_pct_range,
        'rmse_pct_max': rmse_pct_max,
        'nrmse_std': nrmse_std,
        'n_test': len(y_true)
    }
    return metrics, y_true, y_pred


if __name__ == '__main__':
    print(f"Predicting runtime for {predicted_app} on {host_cpu}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Method: {PREDICT_METHOD}")
    all_metrics = []
    last_true = None
    last_pred = None

    for seed in SEEDS:
        try:
            metrics, y_true_seed, y_pred_seed = run_one(seed)
            all_metrics.append(metrics)
            last_true = np.array(y_true_seed)
            last_pred = np.array(y_pred_seed)
            grade = mape_grade(metrics['mape'])
            rmse_mean_grade = rmse_pct_grade(metrics['rmse_pct_mean'])
            rmse_range_grade = rmse_pct_grade(metrics['rmse_pct_range'])
            nrmse_grade = nrmse_std_grade(metrics['nrmse_std'])
            print(
                f"Seed {seed}: MAPE={metrics['mape']:.4f} ({grade}), R2={metrics['r2']:.4f}, "
                f"RMSE={metrics['rmse']:.4f} (s), RMSE%_mean={metrics['rmse_pct_mean']:.2f}% ({rmse_mean_grade}), "
                f"RMSE%_range={metrics['rmse_pct_range']:.2f}% ({rmse_range_grade}), NRMSE_std={metrics['nrmse_std']:.2f} ({nrmse_grade}), N={metrics['n_test']}"
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
            'nrmse_std_mean': df['nrmse_std'].mean(),
            'nrmse_std_std': df['nrmse_std'].std(),
            'total_tests': df['n_test'].sum()
        }

        print("\nAggregated metrics (mean ± std):")
        print(f"MAPE: {agg['mape_mean']:.4f} ± {agg['mape_std']:.4f} ({mape_grade(agg['mape_mean'])})")
        print(f"R2:   {agg['r2_mean']:.4f} ± {agg['r2_std']:.4f}")
        print(f"RMSE: {agg['rmse_mean']:.4f} ± {agg['rmse_std']:.4f} (s)")
        rmse_mean_grade_agg = rmse_pct_grade(agg['rmse_pct_mean_mean'])
        rmse_range_grade_agg = rmse_pct_grade(agg['rmse_pct_range_mean'])
        nrmse_grade_agg = nrmse_std_grade(agg['nrmse_std_mean'])
        print(f"RMSE%_mean: {agg['rmse_pct_mean_mean']:.2f}% ± {agg['rmse_pct_mean_std']:.2f}% ({rmse_mean_grade_agg})")
        print(f"RMSE%_range: {agg['rmse_pct_range_mean']:.2f}% ± {agg['rmse_pct_range_std']:.2f}% ({rmse_range_grade_agg})")
        print(f"NRMSE_std: {agg['nrmse_std_mean']:.2f} ± {agg['nrmse_std_std']:.2f} ({nrmse_grade_agg})")
        print(f"Total test samples across seeds: {agg['total_tests']}")

        if last_true is not None and last_pred is not None and len(last_true) and len(last_pred):
            plt.figure(figsize=(8, 6))
            plt.scatter(last_true, last_pred, color='blue', alpha=0.7, label='Predicted vs True')
            min_val = min(last_true.min(), last_pred.min())
            max_val = max(last_true.max(), last_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
            plt.title(f"Predicted vs True runtime for {predicted_app}", fontsize=16)
            plt.xlabel("True runtime(s)", fontsize=14)
            plt.ylabel("Predicted runtime(s)", fontsize=14)
            plt.legend()
            plt.grid(True)
            plt.show()

"""
# 显示输入规模 vs 运行时间的散点图
plt.figure(figsize=(8, 6))
plt.scatter(host_data['input'], host_data['time'], color='blue', alpha=0.7, label='Measured Data')
plt.title(f"Input vs Time for {predicted_app} on {host_cpu}", fontsize=16)
plt.xlabel("Input Size", fontsize=14)
plt.ylabel("Execution Time (s)", fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
"""
