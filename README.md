# 同平台程序运行时间预测（静态特征）
通过输入规模n预测运行时间T。  

预测算法：
- 卡尔曼滤波KF
- 快速傅里叶变换FFT
- 模型预测控制MPC
- 安全算法AES，MD5，SHA256

平台：
- J721的A72
- J721的R5
- OK8MP的M7

预测方法：
- 随机森林回归RF：对非线性、异常值鲁棒，外推弱  
- 支持向量回归SVR：对单变量平滑非线性拟合好；对尺度敏感需标准化；外推弱  
- 多层感知机回归MLP：表达力强，可拟合复杂曲线；需要较多数据与正则/早停来防过拟合；对尺度敏感；外推一般  
- 解析模型的非线性最小二乘CURVE：可解释、外推性好（当先验正确时）  
- 梯度提升树XGB：对非线性、异常值鲁棒，外推弱  
- 解析先验 + 残差学习HYBRID：兼顾可解释性与灵活性，常比单一模型更稳健  

## 仓库结构

### 根目录文件

- `kalman_op.c` — 原始卡尔曼滤波示例程序
- `kalman_op_A72.c` — 适配A72平台的卡尔曼滤波测试代码
- `kalman_op_M7.c` — 适配M7平台的卡尔曼滤波测试代码
- `.gitignore` — Git忽略配置文件

### 子目录
- `freertos_exclusive_runtime/` — 用于在R5平台上测试的FreeRTOS代码
  - `CPUMonitor.c`, `CPUMonitor.h` — CPU监控模块
  - `main_hello.c` — 主程序入口
  - `makefile` — 编译配置文件
- `single_platform/` — 单平台收集的运行时数据与预测脚本。***该目录是主要工作目录。***
  - **数据文件：**
    - `exclusive_runtime.csv` — 整合的运行时数据（不含加密算法）
    - `exclusive_runtime_encrypt.csv` — 整合的运行时数据（含加密算法）
    - `exclusive_runtime_o.csv` — 原始R5核时间数据（留档备用）
    - `fft-A72.csv`, `fft-M7.csv`, `fft-R5.csv` — FFT算法在各平台的数据
    - `kf-M7.csv` — 卡尔曼滤波在M7平台的数据
    - `matrix-R5.csv` — 矩阵运算在R5平台的数据
    - `mpc-A72.csv`, `mpc-R5.csv` — MPC算法在各平台的数据
  - **预测脚本：**
    - `run_predict.py` — 主预测脚本，用于对单平台数据进行时间预测
    - `run_ensemble_top4.py` — 集成学习脚本，使用Top4模型
    - `ensemble_top4_inference.py` — Top4集成模型推理脚本
    - `export_to_onnx.py` — 模型导出为ONNX格式的脚本
    - `onnx_inference_demo.py` — ONNX模型推理示例
    - `exclusive.py` — 数据处理脚本（留档备用）
  - **依赖配置：**
    - `requirements.txt` — Python依赖包列表

### 数据格式说明

CSV数据文件包含以下字段：
- `input`：输入规模n
- `time`：运行时间（毫秒/ms）***注意：KF在R5上的数据单位是秒！***
- `cpu`：平台名称（A72/R5/M7）
- `program`：运行的程序名称（kf/fft/mpc等）
