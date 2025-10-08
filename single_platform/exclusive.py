import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# 1. 读取数据
data = pd.read_csv("exclusive_runtime.csv")

# 2. 筛选所需数据
host_cpu = 'Cortex-R5F'
predicted_app = 'Matrix_Multiply'
#predicted_app = 'KF'
host_data = data[(data['cpu'] == host_cpu) & (data['program'] == predicted_app)]

# host_data['scale'] = list(range(100, 1001, 9))[:len(host_data)]

# 3. 训练模型
# 使用input作为训练数据，time作为目标
# features = ['scale']
features = ['input']
output = 'time'

#train_data = host_data
#test_data = host_data
train_data, test_data = train_test_split(host_data, test_size=0.1, random_state=6)

X_train = train_data[features]
y_train = train_data[output]

X_test = test_data[features]

# # 初始化标准化器
scaler = StandardScaler()

# # 对训练特征进行标准化
X_train_scaled = scaler.fit_transform(X_train)

# # 对测试特征进行标准化
X_test_scaled = scaler.transform(X_test)

# 使用岭回归拟合标准化后的特征
# model = Lasso(alpha=1.0)
# model = RandomForestRegressor(n_estimators=100, random_state=2)
# model = SVR(kernel='rbf', C=1000, epsilon=0.5)
model = RandomForestRegressor(n_estimators=100, random_state=2)
model.fit(X_train, y_train)

'''
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50), 
    activation='relu', 
    solver='lbfgs', 
    max_iter=10000, 
    random_state=2
    )
model = mlp.fit(X_train_scaled, y_train)
'''

# 预测运行时间
y_pred = model.predict(X_test)

print(f"预测的 {predicted_app} 在 {host_cpu} 上的运行时间: {y_pred}")

# 5. 可视化预测结果与真实值对比
y_true = test_data[output]

plt.figure(figsize=(8, 6))
plt.scatter(y_true, y_pred, color='blue', alpha=0.7, label='Predicted vs True')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Perfect Prediction')
plt.title(f"Predicted vs True runtime for {predicted_app}", fontsize=16)
plt.xlabel("True runtime(s)", fontsize=14)
plt.ylabel("Predicted runtime(s)", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# 6. 输出性能指标 (MAPE 和 R²)
mape = mean_absolute_percentage_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAPE: {mape:.4f}")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 画 input-time 散点图
plt.figure(figsize=(8, 6))
plt.scatter(host_data['input'], host_data['time'], color='blue', alpha=0.7, label='Measured Data')

# 设置标题和坐标轴标签
plt.title(f"Input vs Time for {predicted_app} on {host_cpu}", fontsize=16)
plt.xlabel("Input Size", fontsize=14)
plt.ylabel("Execution Time (s)", fontsize=14)

# 添加网格和图例
plt.grid(True)
plt.legend()
plt.show()


# 拟合Matrix Multiplication的时间复杂度模型
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# 1. 读取数据
data = pd.read_csv("exclusive_runtime.csv")

# 2. 筛选所需数据
host_cpu = 'Cortex-R5F'
predicted_app = 'Matrix_Multiply'
host_data = data[(data['cpu'] == host_cpu) & (data['program'] == predicted_app)]

# 3. 训练数据集划分
train_data, test_data = train_test_split(host_data, test_size=0.1, random_state=6)
X_train = train_data['input'].values
y_train = train_data['time'].values
X_test = test_data['input'].values
y_test = test_data['time'].values

# 4. 定义 FFT 时间复杂度模型
def fft_complexity(N, a, b):
    return a * N * np.log2(N) + b

# 4. 定义 Matrix Multiplication 时间复杂度模型
def matrix_multiplication_complexity(N, a, b, c, d):
    return a * N**3 + b * N**2 + c * N + d

# 5. 拟合参数
params, _ = curve_fit(matrix_multiplication_complexity, X_train, y_train)
a_fit, b_fit, c_fit, d_fit = params

# 6. 预测运行时间
y_pred = matrix_multiplication_complexity(X_test, a_fit, b_fit, c_fit, d_fit)

# 7. 计算误差指标
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"拟合公式: T(N) = {a_fit} * N^3 + {b_fit} * N^2 + {c_fit} * N + {d_fit:.6f}")
print(f"MAPE: {mape:.4f}")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
"""
# 8. 可视化拟合结果
N_range = np.linspace(min(X_train), max(X_train), 1000)
T_pred = matrix_multiplication_complexity(N_range, a_fit, b_fit, c_fit, d_fit)

plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.7, label='Measured Data')
plt.plot(N_range, T_pred, color='red', label='Fitted Curve')
plt.xlabel("Input Size", fontsize=14)
plt.ylabel("Execution Time (s)", fontsize=14)
plt.title(f"Matrix Multiplication Execution Time Fitting on {host_cpu}", fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# # 1. 读取数据
# data = pd.read_csv("exclusive_runtime.csv")

# # 2. 筛选所需数据
# host_cpu = 'Cortex-R5F'
# predicted_app = 'Matrix_Multiply'
# host_data = data[(data['cpu'] == host_cpu) & (data['program'] == predicted_app)]

# # 3. 训练数据集划分
# train_data, test_data = train_test_split(host_data, test_size=0.1, random_state=2)
# X_train = train_data['input'].values
# y_train = train_data['time'].values
# X_test = test_data['input'].values
# y_test = test_data['time'].values

# # 4. 定义 Matrix Multiplication 基础复杂度模型
# def matrix_multiplication_complexity(N, a, b, c, d):
#     return a * N**3 + b * N**2 + c * N + d

# # 5. 拟合基函数参数
# params, _ = curve_fit(matrix_multiplication_complexity, X_train, y_train, maxfev=10000)
# a_fit, b_fit, c_fit, d_fit = params

# # 6. 计算基函数预测 & 残差
# y_train_base = matrix_multiplication_complexity(X_train, a_fit, b_fit, c_fit, d_fit)
# residuals = y_train - y_train_base

# # 7. 用轻量模型拟合残差（这里用小随机森林）
# residual_model = RandomForestRegressor(
#     n_estimators=50,   # 不需要太大
#     max_depth=5,
#     random_state=2
# )
# residual_model.fit(X_train.reshape(-1, 1), residuals)

# # 8. 最终预测函数
# def hybrid_predict(N):
#     base = matrix_multiplication_complexity(N, a_fit, b_fit, c_fit, d_fit)
#     res = residual_model.predict(np.array(N).reshape(-1, 1))
#     return base + res

# # 9. 预测 & 评估
# y_pred = hybrid_predict(X_test)

# mape = mean_absolute_percentage_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# print(f"基拟合公式: T0(N) = {a_fit}*N^3 + {b_fit}*N^2 + {c_fit}*N + {d_fit:.6f}")
# print(f"Hybrid -> MAPE: {mape:.4f}, R²: {r2:.4f}, RMSE: {rmse:.4f}")

# # 10. 可视化
# N_range = np.linspace(min(X_train), max(X_train), 1000)
# T_base = matrix_multiplication_complexity(N_range, a_fit, b_fit, c_fit, d_fit)
# T_hybrid = hybrid_predict(N_range)

# plt.figure(figsize=(8, 6))
# plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Measured Data')
# plt.plot(N_range, T_base, color='red', linestyle='--', label='Base Cubic Fit')
# plt.plot(N_range, T_hybrid, color='green', label='Hybrid (Base + Residual)')
# plt.xlabel("Input Size", fontsize=14)
# plt.ylabel("Execution Time (s)", fontsize=14)
# plt.title(f"Matrix Multiplication Execution Time Hybrid Fit on {host_cpu}", fontsize=16)
# plt.legend()
# plt.grid(True)
# plt.show()

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

# # 1. 读取数据
# data = pd.read_csv("fft.csv")

# # 2. 筛选所需数据
# host_cpu = 'Cortex-R5F'
# predicted_app = 'FFT'
# host_data = data[(data['cpu'] == host_cpu) & (data['program'] == predicted_app)]

# # 3. 分割数据
# train_data_small = host_data[host_data['time'] < 1]
# train_data_large = host_data[host_data['time'] >= 1]

# # 4. 定义 FFT 时间复杂度模型
# def fft_complexity(N, a, b):
#     return a * N * np.log2(N) + b

# # 5. 拟合小于 1s 的数据
# X_train_small = train_data_small['input'].values
# y_train_small = train_data_small['time'].values
# params_small, _ = curve_fit(fft_complexity, X_train_small, y_train_small)
# a1, b1 = params_small

# # 6. 拟合大于等于 1s 的数据
# X_train_large = train_data_large['input'].values
# y_train_large = train_data_large['time'].values
# params_large, _ = curve_fit(fft_complexity, X_train_large, y_train_large)
# a2, b2 = params_large

# # 7. 计算预测值
# N_range = np.linspace(min(host_data['input']), max(host_data['input']), 1000)
# T_pred_small = fft_complexity(N_range, a1, b1)
# T_pred_large = fft_complexity(N_range, a2, b2)

# # 8. 计算误差
# y_pred_small = fft_complexity(X_train_small, a1, b1)
# mape_small = mean_absolute_percentage_error(y_train_small, y_pred_small)
# r2_small = r2_score(y_train_small, y_pred_small)

# y_pred_large = fft_complexity(X_train_large, a2, b2)
# mape_large = mean_absolute_percentage_error(y_train_large, y_pred_large)
# r2_large = r2_score(y_train_large, y_pred_large)

# print(f"小规模数据拟合: T(N) = {a1} * N log2(N) + {b1:.6f}")
# print(f"小规模数据 MAPE: {mape_small:.4f}, R²: {r2_small:.4f}")

# print(f"大规模数据拟合: T(N) = {a2} * N log2(N) + {b2:.6f}")
# print(f"大规模数据 MAPE: {mape_large:.4f}, R²: {r2_large:.4f}")

# # 9. 可视化
# plt.figure(figsize=(8, 6))
# plt.scatter(X_train_small, y_train_small, color='blue', alpha=0.7, label='Measured Data (Small)')
# plt.scatter(X_train_large, y_train_large, color='green', alpha=0.7, label='Measured Data (Large)')
# plt.plot(N_range, T_pred_small, color='red', linestyle='--', label='Fitted Curve (Small)')
# plt.plot(N_range, T_pred_large, color='orange', linestyle='--', label='Fitted Curve (Large)')

# plt.xlabel("Input Size", fontsize=14)
# plt.ylabel("Execution Time (s)", fontsize=14)
# plt.title(f"FFT Execution Time Segmented Fitting on {host_cpu}", fontsize=16)
# plt.legend()
# plt.grid(True)
# plt.show()
"""