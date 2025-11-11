#include <stdio.h>
#include <stdlib.h>
#include <string.h> // For memset
#include <math.h>   // For sqrt, rand etc.
#include <time.h>   // For srand (optional, for consistent random numbers)
#include <stdint.h>

#define MAX_DIM 40

void mat_add(double* A, double* B, double* Result, size_t dim_n, size_t dim_m);
void mat_sub(double* A, double* B, double* Result, size_t dim_n, size_t dim_m);
void mat_mul(double* A, double* B, double* Result, size_t n, size_t k, size_t m); // A_nk, B_km, C_nm
void mat_transpose(double* A, double* Result, size_t rows, size_t cols); // Not used in provided code, but common
void mat_identity(double* I, size_t dim);
void mat_inv_diag(double* A, double* Result, size_t dim); // Inversion for diagonal matrix

// 核心卡尔曼滤波器模拟函数
int kalman_filter_core_loop(size_t dim, int steps_as_iterations) {
    if (dim == 0 || steps_as_iterations == 0 || dim > MAX_DIM) return 1;

    // 使用静态内存在数据段上分配，避免栈溢出
    static double x[MAX_DIM];
    static double P[MAX_DIM * MAX_DIM];
    static double Q[MAX_DIM * MAX_DIM];
    static double R[MAX_DIM * MAX_DIM];
    static double K[MAX_DIM * MAX_DIM];
    static double I_mat[MAX_DIM * MAX_DIM];
    static double z[MAX_DIM];

    // 临时计算矩阵和向量
    static double temp_P_plus_R[MAX_DIM * MAX_DIM];
    static double inv_temp_P_plus_R[MAX_DIM * MAX_DIM];
    static double temp_I_minus_K[MAX_DIM * MAX_DIM];
    static double P_old[MAX_DIM * MAX_DIM];
    static double z_minus_x[MAX_DIM];
    static double K_mult_z_minus_x[MAX_DIM];
    static double true_state_sim[MAX_DIM];

    // 由于是静态数组，每次调用函数时需要手动清零
    memset(x, 0, sizeof(double) * dim);
    memset(P, 0, sizeof(double) * dim * dim);
    memset(Q, 0, sizeof(double) * dim * dim);
    memset(R, 0, sizeof(double) * dim * dim);
    memset(K, 0, sizeof(double) * dim * dim);
    memset(I_mat, 0, sizeof(double) * dim * dim);
    memset(z, 0, sizeof(double) * dim);
    memset(temp_P_plus_R, 0, sizeof(double) * dim * dim);
    memset(inv_temp_P_plus_R, 0, sizeof(double) * dim * dim);
    memset(temp_I_minus_K, 0, sizeof(double) * dim * dim);
    memset(P_old, 0, sizeof(double) * dim * dim);
    memset(z_minus_x, 0, sizeof(double) * dim);
    memset(K_mult_z_minus_x, 0, sizeof(double) * dim);
    memset(true_state_sim, 0, sizeof(double) * dim);

    // 初始化 (简化版)
    // P: 初始协方差 (通常是单位阵或较大的对角阵)
    // Q: 过程噪声 (对角阵)
    // R: 测量噪声 (对角阵)
    // x: 初始状态估计 (通常是0或基于先验知识)
    for (size_t i = 0; i < dim; i++) {
        P[i * dim + i] = 1.0;  // P = I
        Q[i * dim + i] = 0.5; // 较小的过程噪声
        R[i * dim + i] = 0.5;  // 较大的测量噪声
        x[i] = 0.0;            // 初始状态为0
    }
    mat_identity(I_mat, dim);
    srand(12345); // 固定种子以获得可重复的随机数

    // 主循环 (由 iterations_in_c 控制)
    for (int step = 0; step < steps_as_iterations; step++) {
        // 1. 模拟一个简单的真实状态演化和测量 (这部分开销不应过大)
        for (size_t i = 0; i < dim; i++) {
            true_state_sim[i] = (double)(step + 1 + i); // 简单线性增加
            // 测量值 = 真实状态 + 测量噪声 (简化，假设H=I)
            z[i] = true_state_sim[i] + ((rand() / (RAND_MAX / 2.0)) - 1.0) * sqrt(R[i * dim + i]); // 噪声幅度与R相关
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
    for (size_t i = 0; i < dim; ++i) final_sum += x[i];
    if (final_sum > 1e10 && final_sum < 1e11) printf("Kalman final sum check.\n");


    // 静态分配不需要释放内存

    return 0;
}

// --- 简单的矩阵运算实现 (假设都是方阵) ---
// 单位矩阵
void mat_identity(double* I, size_t dim) {
    memset(I, 0, sizeof(double) * dim * dim);
    for (size_t i = 0; i < dim; i++) I[i * dim + i] = 1.0;
}

// 加法 C = A + B (A, B, C 都是 dim x m_cols 的矩阵)
void mat_add(double* A, double* B, double* Result, size_t dim_n, size_t dim_m) {
    for (size_t i = 0; i < dim_n * dim_m; i++) Result[i] = A[i] + B[i];
}

// 减法 C = A - B
void mat_sub(double* A, double* B, double* Result, size_t dim_n, size_t dim_m) {
    for (size_t i = 0; i < dim_n * dim_m; i++) Result[i] = A[i] - B[i];
}

// 乘法 C_nm = A_nk * B_km
void mat_mul(double* A, double* B, double* Result, size_t n, size_t k_common, size_t m) {
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
void mat_inv_diag(double* A, double* Result, size_t dim) {
    memset(Result, 0, sizeof(double) * dim * dim); // 先清零
    for (size_t i = 0; i < dim; i++) {
        if (A[i * dim + i] != 0.0)
            Result[i * dim + i] = 1.0 / A[i * dim + i];
        // else Result[i*dim+i] 保持为0，或者可以报错/置为极大值
    }
}

int main() {
    for (int i = 10; i <= MAX_DIM; i += 9) {
        struct timespec start, end;
        // 计时函数来自飞书上邱高的fft_a.h,202-214
        clock_gettime(CLOCK_MONOTONIC, &start);
        int ret = kalman_filter_core_loop(i, 1);
        if (ret != 0) {
            fprintf(stderr, "Kalman filter core loop failed for dim = %d with error code %d\n", i, ret);
            continue;
        }
        clock_gettime(CLOCK_MONOTONIC, &end);
        double seconds = (end.tv_sec - start.tv_sec);
        double nanoseconds = (end.tv_nsec - start.tv_nsec) / 1e6;
        if (end.tv_nsec < start.tv_nsec) {
            seconds -= 1;
            nanoseconds += 1000;
        }
        double elapsed = seconds + nanoseconds;
        printf("%d,%f,A72,KF\n", i, elapsed);
    }
    return 0;
}
