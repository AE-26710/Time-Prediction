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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import TransformedTargetRegressor
import scipy.stats as stats

# ========== 配置区 ==========
# Matrix_Multiply | FFT | KF
predicted_app = 'KF'
host_cpu = 'Cortex-R5F'
# random_forest | random_forest_tuned | svr | svr_tuned | mlp_tuned | curve_fit | xgboost | xgboost_tuned | hybrid
PREDICT_METHOD = 'hybrid'

# ========== 数据准备 ==========
data = pd.read_csv("exclusive_runtime.csv")
host_data = data[(data['cpu'] == host_cpu) & (data['program'] == predicted_app)]
features = ['input']
output = 'time'
train_data, test_data = train_test_split(host_data, test_size=0.1, random_state=6)
X_train = train_data[features]
y_train = train_data[output]
X_test = test_data[features]
y_true = test_data[output]

# ========== 特征标准化 ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 预测模型选择 ==========
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
        # Feature engineering: add N*log2(N) feature for FFT
        X_fe = host_data[['input']].copy()
        X_fe['n_log_n'] = X_fe['input'] * np.log2(X_fe['input'])
        X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_fe, host_data['time'], test_size=0.1, random_state=6)
        model = RandomForestRegressor(n_estimators=200, random_state=2)
        model.fit(X_train_fe[['n_log_n']], y_train_fe)
        y_pred = model.predict(X_test_fe[['n_log_n']])
        # ensure later evaluation uses the corresponding y_true
        X_test = X_test_fe[['input']]
        y_true = y_test_fe
    else:
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=2,
            )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

elif PREDICT_METHOD == 'random_forest_tuned':
    # 随机森林自动调参（RandomizedSearchCV）
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(random_state=2)
    # 参数分布：n_estimators 用整数范围，max_depth 用整数或 None，max_features 可选 'sqrt'/'log2' 或小数
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
        random_state=2,
        n_jobs=-1,
        verbose=1
    )
    # For FFT use feature engineering; otherwise use original features
    if predicted_app == 'FFT':
        X_fe = host_data[['input']].copy()
        X_fe['n_log_n'] = X_fe['input'] * np.log2(X_fe['input'])
        X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(X_fe, host_data['time'], test_size=0.1, random_state=6)
        rnd_rf.fit(X_train_fe[['n_log_n']], y_train_fe)
        print('Best params (random_forest_tuned, FFT):', rnd_rf.best_params_)
        y_pred = rnd_rf.predict(X_test_fe[['n_log_n']])
        X_test = X_test_fe[['input']]
        y_true = y_test_fe
    else:
        # 注意这里我们没有对 X 做额外变换（已在顶部对 input 做了 StandardScaler）；随机森林本身不敏感于缩放
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
    model = SVR(
        kernel='rbf', 
        C=1000, 
        epsilon=0.5
        )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

elif PREDICT_METHOD == 'svr_tuned':
    # 对 SVR 进行随机搜索调参（使用缩放后的特征）
    base_svr = SVR()
    # 参数分布：C 和 gamma 用对数均匀分布，epsilon 用均匀分布
    param_dist_svr = {
        'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
        'C': stats.loguniform(1e-2, 1e4),
        'epsilon': stats.uniform(1e-4, 1.0),
        'gamma': stats.loguniform(1e-6, 1e1),
        # degree 只在 poly kernel 有效，RandomizedSearch 会在不使用时忽略
        'degree': stats.randint(2, 6)
    }
    rnd_svr = RandomizedSearchCV(
        estimator=base_svr,
        param_distributions=param_dist_svr,
        n_iter=30,
        scoring='neg_mean_absolute_percentage_error',
        cv=3,
        random_state=2,
        n_jobs=-1,
        verbose=1
    )

    rnd_svr.fit(X_train_scaled, y_train)
    print('Best params (svr_tuned):', rnd_svr.best_params_)
    y_pred = rnd_svr.predict(X_test_scaled)

elif PREDICT_METHOD == 'mlp_tuned':
    # 更稳健的 MLP 调参：对输入特征缩放、对目标值做可选变换，并用 RandomizedSearchCV 搜索超参
    # 我们使用 TransformedTargetRegressor 给目标做 Y 变换（如对数或 Yeo-Johnson），减少异方差
    base_mlp = MLPRegressor(max_iter=2000, random_state=2)
    param_dist = {
        'regressor__hidden_layer_sizes': [(50,), (100,), (100,50), (200,100), (200,100,50)],
        'regressor__activation': ['relu', 'tanh', 'logistic'],
        'regressor__solver': ['adam', 'lbfgs'],
        'regressor__alpha': stats.loguniform(1e-6, 1e-1),
        'regressor__learning_rate_init': stats.loguniform(1e-4, 1e-1),
    }
    # 对输入先用先前定义的 scaler
    X_train_s = X_train_scaled
    X_test_s = X_test_scaled
    # 对 y 做 Yeo-Johnson 变换（适用于正负值）
    transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    ttr = TransformedTargetRegressor(regressor=base_mlp, transformer=transformer)
    # 包装到 RandomizedSearchCV（n_iter 设为 20 避免过慢）
    rnd = RandomizedSearchCV(
        estimator=ttr,
        param_distributions=param_dist,
        n_iter=20,
        scoring='neg_mean_absolute_percentage_error',
        cv=3,
        random_state=2,
        n_jobs=-1,
        verbose=1
    )
    rnd.fit(X_train_s, y_train)
    print('Best params (mlp_tuned):', rnd.best_params_)
    # 预测并还原目标变换
    y_pred = rnd.predict(X_test_s)

elif PREDICT_METHOD == 'curve_fit':
    # 三次多项式 / 复杂度模型曲线拟合
    X_train_cf = X_train[features[0]].values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_cf = y_train.values if hasattr(y_train, 'values') else y_train
    X_test_cf = X_test[features[0]].values if isinstance(X_test, pd.DataFrame) else X_test

    if predicted_app == 'KF' or predicted_app == 'Matrix_Multiply':
        # cubic complexity T(N) = a*N^3 + b*N^2 + c*N + d
        def cubic_func(N, a, b, c, d):
            return a * N**3 + b * N**2 + c * N + d
        params, _ = curve_fit(cubic_func, X_train_cf, y_train_cf)
        y_pred = cubic_func(X_test_cf, *params)
        a_fit, b_fit, c_fit, d_fit = params
        print(f"拟合公式 (cubic): T(N) = {a_fit} * N^3 + {b_fit} * N^2 + {c_fit} * N + {d_fit:.6f}")

    elif predicted_app == 'FFT':
        # FFT complexity T(N) = a * N * log2(N) + b
        def fft_complexity(N, a, b):
            return a * N * np.log2(N) + b
        params, _ = curve_fit(fft_complexity, X_train_cf, y_train_cf)
        y_pred = fft_complexity(X_test_cf, *params)
        a_fit, b_fit = params
        print(f"拟合公式 (fft): T(N) = {a_fit} * N log2(N) + {b_fit:.6f}")

    else:
        # fallback: try cubic
        def cubic_func(N, a, b, c, d):
            return a * N**3 + b * N**2 + c * N + d
        params, _ = curve_fit(cubic_func, X_train_cf, y_train_cf)
        y_pred = cubic_func(X_test_cf, *params)

elif PREDICT_METHOD == 'xgboost':
    """
    XGBRegressor（XGBoost回归）：
    n_estimators：树的数量，默认100。
    max_depth：树的最大深度，默认3。
    learning_rate：每棵树的权重缩减，默认0.1。
    subsample：训练每棵树时的子样本比例，默认1。
    colsample_bytree：每棵树随机采样的特征比例，默认1。
    gamma：分裂节点所需的最小损失减少，默认0。
    reg_alpha, reg_lambda：L1/L2正则化。
    random_state：随机种子。
    n_jobs：并行线程数。
    """
    model = XGBRegressor(
        n_estimators=500,
        max_depth=15,
        learning_rate=0.05,
        random_state=2,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

elif PREDICT_METHOD == 'xgboost_tuned':
    xgb = XGBRegressor(objective='reg:squarederror', random_state=2, verbosity=0)
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
        random_state=2,
        n_jobs=-1,
        verbose=1
    )
    rnd_xgb.fit(X_train, y_train)
    print('Best params (xgboost_tuned):', rnd_xgb.best_params_)
    y_pred = rnd_xgb.predict(X_test)

elif PREDICT_METHOD == 'hybrid':
    # Hybrid: fit base theoretical model (cubic or N*logN) then learn residuals with a small RandomForest
    from scipy.optimize import curve_fit
    # Use the existing train/test split defined earlier
    # Prepare arrays
    X_train_arr = X_train[features[0]].values if isinstance(X_train, pd.DataFrame) else X_train
    X_test_arr = X_test[features[0]].values if isinstance(X_test, pd.DataFrame) else X_test
    y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_arr = y_true.values if hasattr(y_true, 'values') else y_true

    if predicted_app == 'KF' or predicted_app == 'Matrix_Multiply':
        # cubic base
        def cubic_func(N, a, b, c, d):
            return a * N**3 + b * N**2 + c * N + d
        params, _ = curve_fit(cubic_func, X_train_arr, y_train_arr)
        base_train = cubic_func(X_train_arr, *params)
        base_test = cubic_func(X_test_arr, *params)

    elif predicted_app == 'FFT':
        # N*logN base using linear regression on engineered feature
        X_train_fe = pd.DataFrame({ 'input': X_train_arr })
        X_test_fe = pd.DataFrame({ 'input': X_test_arr })
        X_train_fe['n_log_n'] = X_train_fe['input'] * np.log2(X_train_fe['input'])
        X_test_fe['n_log_n'] = X_test_fe['input'] * np.log2(X_test_fe['input'])
        from sklearn.linear_model import LinearRegression
        base_lr = LinearRegression().fit(X_train_fe[['n_log_n']], y_train_arr)
        base_train = base_lr.predict(X_train_fe[['n_log_n']])
        base_test = base_lr.predict(X_test_fe[['n_log_n']])
    else:
        raise ValueError(f"未知的预测程序: {PREDICT_METHOD}")

    # residuals on training set
    residuals_train = y_train_arr - base_train

    # small RF to model residuals
    res_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=2)
    res_model.fit(X_train_arr.reshape(-1,1), residuals_train)

    res_pred_test = res_model.predict(X_test_arr.reshape(-1,1))
    y_pred = base_test + res_pred_test

else:
    raise ValueError(f"未知的预测方式: {PREDICT_METHOD}")

# ========== 评估与可视化 ==========
print(f"预测的 {predicted_app} 在 {host_cpu} 上的运行时间: {y_pred}")

mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"Method: {PREDICT_METHOD}")
print(f"MAPE: {mape:.4f}")
# MAPE（Mean Absolute Percentage Error，平均绝对百分比误差），衡量预测误差占真实值的比例（相对误差的平均），越小越好。
# 经验判定（仅作经验参考）：MAPE < 10%（优秀），10–20%（良好），20–50%（可接受/需改进），>50%（差）。具体阈值依应用场景而异。
print(f"R²: {r2:.4f}")
# R²（决定系数, coefficient of determination），衡量模型对数据变异的解释能力，范围通常在0到1之间，越接近1表示模型拟合越好。
# 负值表示模型表现比简单的均值预测还差。
print(f"RMSE: {rmse:.4f}")
# RMSE（Root Mean Squared Error，均方根误差），衡量预测值与真实值之间的平均偏差，单位与目标变量相同，越小越好。

plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Predicted vs True')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Perfect Prediction')
plt.title(f"Predicted vs True runtime for {predicted_app}", fontsize=16)
plt.xlabel("True runtime(s)", fontsize=14)
plt.ylabel("Predicted runtime(s)", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

"""
plt.figure(figsize=(8, 6))
plt.scatter(host_data['input'], host_data['time'], color='blue', alpha=0.7, label='Measured Data')
plt.title(f"Input vs Time for {predicted_app} on {host_cpu}", fontsize=16)
plt.xlabel("Input Size", fontsize=14)
plt.ylabel("Execution Time (s)", fontsize=14)
plt.grid(True)
plt.legend()
plt.show()
"""
