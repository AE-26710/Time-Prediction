#include "mpc.h"
#include <time.h>

int main()
{
    #pragma omp parallel 
    {
        // 打印线程信息
        printf("Thread %d / %d\n", omp_get_thread_num(), omp_get_num_threads());
    }
    State_t s = {
        .x = 0.0f,
        .y = 0.0f,
        .psi = atanf(ref_path_derivative(0.0f)),  // 用参考路径的切线方向初始化
        .v = 5.0f
    };

    // 初始化误差项
    s.cte = ref_path(s.x) - s.y;
    s.epsi = s.psi - atanf(ref_path_derivative(s.x));

    // Controls_t plan[MAX_HORIZON];
    Controls_t plan_out;

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // mpc_control_with_plan(s, plan);  // 优化控制
    mpc_control_with_plan(s, &plan_out);  // SoA控制输出
    print_predicted_trajectory(s, &plan_out);

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long elapsed_us = (end_time.tv_sec - start_time.tv_sec) * 1000000L +
                      (end_time.tv_nsec - start_time.tv_nsec) / 1000;

    printf(">>> MPC took %ld us\n", elapsed_us);
    return 0;
}