"""
自动运行所有预测方法在所有算子和平台上的脚本
读取配置文件 config_lower_bounds.csv 来确定哪些组合需要运行以及对应的 LOWER_BOUND 值
当 lower_bound 设置为 -1 时，跳过该配置
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# 导入 run_predict.py 中的核心函数
import importlib.util
spec = importlib.util.spec_from_file_location("run_predict", "run_predict.py")
run_predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_predict_module)

def run_single_configuration(algorithm, platform, method, lower_bound, test_size=0.3, print_details=False):
    """
    运行单个配置的预测
    返回汇总结果字典
    """
    print(f"\nRunning: {algorithm} on {platform} using {method.upper()} (lower_bound={lower_bound})")
    
    # 准备数据
    if algorithm not in ('AES', 'MD5', 'SHA256'):
        data = pd.read_csv("exclusive_runtime.csv")
    else:
        data = pd.read_csv("exclusive_runtime_encrypt.csv")
    
    host_data = data[(data['cpu'] == platform) & (data['program'] == algorithm)]
    host_data = host_data[(host_data['time'] > lower_bound)]
    
    if host_data.empty:
        print(f"WARNING: No data available for {algorithm} on {platform} with lower_bound={lower_bound}")
        return None
    
    print(f"Data points: {len(host_data)}")
    
    # 运行多个随机种子
    SEEDS = [1, 2, 6, 42, 123, 2025, 33550336]
    all_metrics = []
    
    for seed in SEEDS:
        try:
            # 调用 run_predict_module 中的 run_one 函数
            # 需要临时修改全局变量
            original_host_data = run_predict_module.host_data
            original_predicted_app = run_predict_module.predicted_app
            original_host_cpu = run_predict_module.host_cpu
            original_predict_method = run_predict_module.PREDICT_METHOD
            original_test_size = run_predict_module.TEST_SIZE
            original_lower_bound = run_predict_module.LOWER_BOUND
            
            run_predict_module.host_data = host_data
            run_predict_module.predicted_app = algorithm
            run_predict_module.host_cpu = platform
            run_predict_module.PREDICT_METHOD = method
            run_predict_module.TEST_SIZE = test_size
            run_predict_module.LOWER_BOUND = lower_bound
            
            metrics, y_true_seed, y_pred_seed = run_predict_module.run_one(seed, predict_method=method)
            all_metrics.append(metrics)
            
            # 恢复原始值
            run_predict_module.host_data = original_host_data
            run_predict_module.predicted_app = original_predicted_app
            run_predict_module.host_cpu = original_host_cpu
            run_predict_module.PREDICT_METHOD = original_predict_method
            run_predict_module.TEST_SIZE = original_test_size
            run_predict_module.LOWER_BOUND = original_lower_bound
            
            if print_details:
                grade = run_predict_module.mape_grade(metrics['mape'])
                rmse_mean_grade = run_predict_module.rmse_pct_grade(metrics['rmse_pct_mean'])
                rmse_range_grade = run_predict_module.rmse_pct_grade(metrics['rmse_pct_range'])
                print(
                    f"Seed {seed}: MAPE={metrics['mape']:.4f} ({grade}), R2={metrics['r2']:.4f}, "
                    f"RMSE={metrics['rmse']:.4f} (ms), RMSE%_mean={metrics['rmse_pct_mean']:.2f}% ({rmse_mean_grade}), "
                    f"RMSE%_range={metrics['rmse_pct_range']:.2f}% ({rmse_range_grade}), N={metrics['n_test']}"
                )
                
        except Exception as exc:
            print(f"  Seed {seed} failed: {exc}")
            continue
    
    if not all_metrics:
        print(f"ERROR: All seeds failed for {algorithm} on {platform} using {method}")
        return None
    
    # 计算汇总统计
    df = pd.DataFrame(all_metrics)
    agg = {
        'algorithm': algorithm,
        'platform': platform,
        'method': method,
        'lower_bound': lower_bound,
        'n_data_points': len(host_data),
        'n_seeds': len(all_metrics),
        'mape_mean': df['mape'].mean(),
        'mape_std': df['mape'].std(),
        'mape_min': df['mape'].min(),
        'mape_max': df['mape'].max(),
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
    
    # 计算总体评估
    grades = {
        'MAPE': run_predict_module.mape_grade(agg['mape_mean']),
        'RMSE%_mean': run_predict_module.rmse_pct_grade(agg['rmse_pct_mean_mean']),
        'RMSE%_range': run_predict_module.rmse_pct_grade(agg['rmse_pct_range_mean'])
    }
    grade_order = {'优秀': 0, '中等': 1, '差': 2}
    worst_metric, worst_grade = max(grades.items(), key=lambda kv: grade_order.get(kv[1], 2))
    agg['overall_grade'] = worst_grade
    
    # 打印汇总结果
    print(f"MAPE: {agg['mape_mean']:.4f} ± {agg['mape_std']:.4f} ({run_predict_module.mape_grade(agg['mape_mean'])})")
    print(f"R2:   {agg['r2_mean']:.4f} ± {agg['r2_std']:.4f}")
    print(f"RMSE: {agg['rmse_mean']:.4f} ± {agg['rmse_std']:.4f} (ms)")
    print(f"RMSE%_mean: {agg['rmse_pct_mean_mean']:.2f}% ± {agg['rmse_pct_mean_std']:.2f}% ({run_predict_module.rmse_pct_grade(agg['rmse_pct_mean_mean'])})")
    print(f"RMSE%_range: {agg['rmse_pct_range_mean']:.2f}% ± {agg['rmse_pct_range_std']:.2f}% ({run_predict_module.rmse_pct_grade(agg['rmse_pct_range_mean'])})")
    print(f"Overall Grade: {worst_grade}")
    print(f"MAPE Range: {agg['mape_min']*100:.2f}%~{agg['mape_max']*100:.2f}%")
    
    return agg

def main():
    """
    主函数：读取配置文件并运行所有配置
    """
    # 读取配置文件
    config_file = "config_lower_bounds.csv"
    if not os.path.exists(config_file):
        print(f"ERROR: Configuration file '{config_file}' not found!")
        sys.exit(1)
    
    config_df = pd.read_csv(config_file)
    print(f"Loaded {len(config_df)} configurations from {config_file}")
    
    # 过滤掉 lower_bound == -1 的配置
    config_df = config_df[config_df['lower_bound'] != -1]
    print(f"After filtering (lower_bound != -1): {len(config_df)} configurations to run")
    
    if config_df.empty:
        print("No configurations to run!")
        return
    
    # 运行所有配置
    all_results = []
    total_configs = len(config_df)
    
    for idx, row in config_df.iterrows():
        algorithm = row['algorithm'].upper()
        platform = row['platform'].upper()
        method = row['method'].lower()
        lower_bound = float(row['lower_bound'])
        
        result = run_single_configuration(
            algorithm=algorithm,
            platform=platform,
            method=method,
            lower_bound=lower_bound,
            test_size=0.3,
            print_details=True
        )
        
        if result is not None:
            all_results.append(result)
    
    print(f"\nAll predictions completed! Total successful runs: {len(all_results)}/{total_configs}")

if __name__ == '__main__':
    main()
