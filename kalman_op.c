#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For memset
#include <math.h>   // For sqrt, rand etc.
#include <time.h>   // For srand (optional, for consistent random numbers)

void mat_add(double *A, double *B, double *Result, size_t dim_n, size_t dim_m);
void mat_sub(double *A, double *B, double *Result, size_t dim_n, size_t dim_m);
void mat_mul(double *A, double *B, double *Result, size_t n, size_t k, size_t m); // A_nk, B_km, C_nm
void mat_transpose(double *A, double *Result, size_t rows, size_t cols); // Not used in provided code, but common
void mat_identity(double *I, size_t dim);
void mat_inv_diag(double *A, double *Result, size_t dim); // Inversion for diagonal matrix

// 核心卡尔曼滤波器模拟函数
void kalman_filter_core_loop(size_t dim, int steps_as_iterations) {
    if (dim == 0 || steps_as_iterations == 0) return;

    // 分配内存
    // 使用 calloc 初始化为0可以简化某些矩阵的初始化
    double *x = (double *)calloc(dim, sizeof(double));          // 状态估计向量
    double *P = (double *)calloc(dim * dim, sizeof(double));    // 协方差矩阵
    double *Q = (double *)calloc(dim * dim, sizeof(double));    // 过程噪声协方差
    double *R = (double *)calloc(dim * dim, sizeof(double));    // 测量噪声协方差
    double *K = (double *)calloc(dim * dim, sizeof(double));    // 卡尔曼增益 (dim x dim for simplified state-space model)
    double *I_mat = (double *)calloc(dim * dim, sizeof(double));// 单位矩阵
    double *z = (double *)calloc(dim, sizeof(double));          // 测量值 (假设测量维度与状态维度相同)
    
    // 临时计算矩阵和向量
    double *temp_P_plus_R = (double *)calloc(dim * dim, sizeof(double));
    double *inv_temp_P_plus_R = (double *)calloc(dim * dim, sizeof(double));
    double *temp_I_minus_K = (double *)calloc(dim * dim, sizeof(double));
    double *P_old = (double *)calloc(dim * dim, sizeof(double)); // 用于 P = (I - K) * P_old
    double *z_minus_x = (double *)calloc(dim, sizeof(double));
    double *K_mult_z_minus_x = (double *)calloc(dim, sizeof(double));


    if (!x || !P || !Q || !R || !K || !I_mat || !z || 
        !temp_P_plus_R || !inv_temp_P_plus_R || !temp_I_minus_K || !P_old || !z_minus_x || !K_mult_z_minus_x) {
        fprintf(stderr, "Kalman filter: Memory allocation failed for dim = %zu\n", dim);
        // 释放已分配的内存
        free(x); free(P); free(Q); free(R); free(K); free(I_mat); free(z);
        free(temp_P_plus_R); free(inv_temp_P_plus_R); free(temp_I_minus_K); free(P_old);
        free(z_minus_x); free(K_mult_z_minus_x);
        return;
    }

    // 初始化 (简化版)
    // P: 初始协方差 (通常是单位阵或较大的对角阵)
    // Q: 过程噪声 (对角阵)
    // R: 测量噪声 (对角阵)
    // x: 初始状态估计 (通常是0或基于先验知识)
    for (size_t i = 0; i < dim; i++) {
        P[i * dim + i] = 1.0;  // P = I
        Q[i * dim + i] = 0.01; // 较小的过程噪声
        R[i * dim + i] = 0.1;  // 较大的测量噪声
        x[i] = 0.0;            // 初始状态为0
    }
    mat_identity(I_mat, dim);
    srand(12345); // 固定种子以获得可重复的随机数

    // 模拟真实状态（仅用于生成测量值，不参与核心计时循环的性能）
    double *true_state_sim = (double *)calloc(dim, sizeof(double));
    if(!true_state_sim) { /* ... handle error ... */ free(x); /* ... free others ... */ return; }


    // 主循环 (由 iterations_in_c 控制)
    for (int step = 0; step < steps_as_iterations; step++) {
        // 1. 模拟一个简单的真实状态演化和测量 (这部分开销不应过大)
        for (size_t i = 0; i < dim; i++) {
            true_state_sim[i] = (double) (step + 1 + i); // 简单线性增加
            // 测量值 = 真实状态 + 测量噪声 (简化，假设H=I)
            z[i] = true_state_sim[i] + ((rand() / (RAND_MAX / 2.0)) - 1.0) * sqrt(R[i*dim+i]); // 噪声幅度与R相关
        }

        // --- 卡尔曼滤波器核心步骤 ---
        // 预测步骤 (假设状态转移矩阵 F=I, 控制输入 B*u=0)
        // x_pred = x; (状态预测不变)
        // P_pred = P + Q;
        mat_add(P, Q, P, dim, dim); // P 是方阵

        // 更新步骤
        // K = P_pred * H^T * inv(H * P_pred * H^T + R)
        // 简化: 假设 H = I (测量直接反映状态)
        // K = P * inv(P + R)
        mat_add(P, R, temp_P_plus_R, dim, dim);
        mat_inv_diag(temp_P_plus_R, inv_temp_P_plus_R, dim); // 假设 (P+R) 是对角可逆的 (非常强的简化)
                                                            // 真实EKF需要更通用的矩阵求逆或Cholesky分解等
        mat_mul(P, inv_temp_P_plus_R, K, dim, dim, dim);    // K = P * inv(temp_P_plus_R)

        // x_est = x_pred + K * (z - H * x_pred)
        // 简化: x = x + K * (z - x) (因为 x_pred = x, H=I)
        for (size_t i = 0; i < dim; i++) {
            z_minus_x[i] = z[i] - x[i];
        }
        // K_mult_z_minus_x = K * z_minus_x (矩阵 * 向量)
        memset(K_mult_z_minus_x, 0, dim * sizeof(double));
        for (size_t i = 0; i < dim; i++) { // 结果向量的行
            for (size_t j = 0; j < dim; j++) { // K的列 / z_minus_x的行
                K_mult_z_minus_x[i] += K[i * dim + j] * z_minus_x[j];
            }
        }
        for (size_t i = 0; i < dim; i++) {
            x[i] += K_mult_z_minus_x[i];
        }
        

        // P_est = (I - K * H) * P_pred
        // 简化: P = (I - K) * P_pred (因为 H=I, P_pred就是当前的P)
        memcpy(P_old, P, dim * dim * sizeof(double)); // 保存 P_pred
        mat_sub(I_mat, K, temp_I_minus_K, dim, dim);
        mat_mul(temp_I_minus_K, P_old, P, dim, dim, dim);
    }

    // 防止优化：对最终状态x做一个简单的操作
    double final_sum = 0;
    for(size_t i=0; i<dim; ++i) final_sum += x[i];
    if (final_sum > 1e10 && final_sum < 1e11) printf("Kalman final sum check.\n");


    // 释放内存
    free(x); free(P); free(Q); free(R); free(K); free(I_mat); free(z); 
    free(true_state_sim);
    free(temp_P_plus_R); free(inv_temp_P_plus_R); free(temp_I_minus_K); free(P_old);
    free(z_minus_x); free(K_mult_z_minus_x);
}

// --- 简单的矩阵运算实现 (假设都是方阵) ---
// 单位矩阵
void mat_identity(double *I, size_t dim) {
    memset(I, 0, sizeof(double) * dim * dim);
    for (size_t i = 0; i < dim; i++) I[i * dim + i] = 1.0;
}

// 加法 C = A + B (A, B, C 都是 dim x m_cols 的矩阵)
void mat_add(double *A, double *B, double *Result, size_t dim_n, size_t dim_m) {
    for (size_t i = 0; i < dim_n * dim_m; i++) Result[i] = A[i] + B[i];
}

// 减法 C = A - B
void mat_sub(double *A, double *B, double *Result, size_t dim_n, size_t dim_m) {
    for (size_t i = 0; i < dim_n * dim_m; i++) Result[i] = A[i] - B[i];
}

// 乘法 C_nm = A_nk * B_km
void mat_mul(double *A, double *B, double *Result, size_t n, size_t k_common, size_t m) {
    memset(Result, 0, sizeof(double) * n * m);
    for (size_t i = 0; i < n; i++) {        // C的行
        for (size_t j = 0; j < m; j++) {    // C的列
            double sum_val = 0.0; 
            for (size_t l = 0; l < k_common; l++) { // 内积
                sum_val += A[i * k_common + l] * B[l * m + j];
            }
            Result[i * m + j] = sum_val;
        }
    }
}

// 对角阵求逆 Result = inv(A)
void mat_inv_diag(double *A, double *Result, size_t dim) {
    memset(Result, 0, sizeof(double) * dim * dim); // 先清零
    for (size_t i = 0; i < dim; i++) {
        if (A[i * dim + i] != 0.0)
            Result[i * dim + i] = 1.0 / A[i * dim + i];
        // else Result[i*dim+i] 保持为0，或者可以报错/置为极大值
    }
}


// Python脚本将按参数名排序传递命令行参数。
// 如果config.json中参数是 "N_dim", 则它会是第一个参数。
// ITERATIONS_IN_C 会是最后一个参数。
int main(int argc, char *argv[]) {
    if (argc < 3) { // 需要 N_dim 和 iterations_in_c
        fprintf(stderr, "Usage: %s <N_dimension> <iterations_in_c>\n", argv[0]);
        fprintf(stderr, "N_dimension specifies the state dimension for Kalman filter.\n");
        return 1;
    }

    size_t dim_val = (size_t)atoi(argv[1]);      // 状态维度 N
    int iterations = atoi(argv[2]);          // C代码内部的迭代/模拟步数

    if (dim_val <= 0 || iterations <= 0) {
        fprintf(stderr, "Error: Dimension N_dim and iterations_in_c must be positive.\n");
        fprintf(stderr, "Received: N_dim=%zu, iterations=%d\n", dim_val, iterations);
        return 1;
    }

    // printf("Kalman Filter: Dimension = %zu, Simulation Steps (Iterations in C) = %d\n", dim_val, iterations);
    kalman_filter_core_loop(dim_val, iterations);

    return 0;
}
