#ifndef _MINIMAL_MPC_OPT_H
#define _MINIMAL_MPC_OPT_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <arm_neon.h>
#include <omp.h>

#define DT 0.05f
#define LF 2.67f
#define REF_V 10.0f
#define MAX_HORIZON 156    
#define MAX_ITER 50       
#define LEARNING_RATE 0.002f 
#define GRAD_EPS 1e-2f 

#define WEIGHT_CTE 10.0f
#define WEIGHT_EPSI 10.0f
#define WEIGHT_V 5.0f
#define WEIGHT_DELTA 0.1f
#define WEIGHT_A 0.1f
#define WEIGHT_DELTA_DIFF 10.0f
#define WEIGHT_A_DIFF 5.0f
#define MAX_ROLLOUT_LEN 50

#ifndef M_PI
#define M_PI 3.14159265358979323846f   
#endif

#define MAX_DELTA 0.5f
#define MAX_A 1.0f
#define MAX_DELTA_RATE 0.5f
#define MAX_A_RATE 2.0f

#define SIGN(x) ((x) > 0 ? 1.0f : ((x) < 0 ? -1.0f : 0.0f))

typedef struct {
    float x;
    float y;
    float psi;
    float v;
    float cte;
    float epsi;
} State_t;

// SoA
typedef struct {
    float delta[MAX_HORIZON] __attribute__((aligned(16)));
    float a[MAX_HORIZON] __attribute__((aligned(16)));
} Controls_t;

Controls_t u_seq, grad;
// 参考路径函数
// static inline float ref_path(float x) { return 0.5f * x; }
// static inline float ref_path_derivative(float x) { return 0.5f; }
static inline float ref_path(float x) {
    return sinf(0.2f * x) + 0.1f * x;
}
static inline float ref_path_derivative(float x) {
    return 0.2f * cosf(0.2f * x) + 0.1f;
}

// 角度归一化函数
static inline float normalize_angle(float angle) {
    while (angle > M_PI)
        angle -= 2.0f * M_PI;
    while (angle < -M_PI)
        angle += 2.0f * M_PI;
    return angle;
}

// // NEON优化的三角函数近似
static inline float32x4_t neon_sinf(float32x4_t x) {
    // 泰勒展开: sin(x) ≈ x - x^3/6 + x^5/120
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t x5 = vmulq_f32(x3, x2);

    float32x4_t result = vmlsq_f32(x, vdupq_n_f32(1.0f / 6.0f), x3);
    result = vmlaq_f32(result, vdupq_n_f32(1.0f / 120.0f), x5);
    return result;
}

static inline float32x4_t neon_cosf(float32x4_t x) {
    // 泰勒展开: cos(x) ≈ 1 - x^2/2 + x^4/24
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);

    float32x4_t result = vmlsq_f32(vdupq_n_f32(1.0f), vdupq_n_f32(0.5f), x2);
    result = vmlaq_f32(result, vdupq_n_f32(1.0f / 24.0f), x4);
    return result;
}

static inline float32x4_t neon_tanf(float32x4_t x) {
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t x5 = vmulq_f32(x3, x2);

    float32x4_t term1 = vmlaq_f32(x, x3, vdupq_n_f32(1.0f / 3.0f));
    float32x4_t term2 = vmlaq_f32(term1, x5, vdupq_n_f32(2.0f / 15.0f));
    return term2;
}

static inline void simulate(State_t *s, float *cos_psi, float *sin_psi, float delta, float a) {
    float vx = *cos_psi;
    float vy = *sin_psi;

    float delta_psi = s->v * tanf(delta) / LF * DT;
    float cos_dpsi = cosf(delta_psi);
    float sin_dpsi = sinf(delta_psi);

    float new_vx = vx * cos_dpsi - vy * sin_dpsi;
    float new_vy = vy * cos_dpsi + vx * sin_dpsi;

    s->x += s->v * new_vx * DT;
    s->y += s->v * new_vy * DT;

    // 推进速度
    s->v += a * DT;
    if (s->v < 0.01f) s->v = 0.01f;
    else if (s->v > 20.0f) s->v = 20.0f;

    // 更新方向角与 cos/sin
    *cos_psi = new_vx;
    *sin_psi = new_vy;
    s->psi = atan2f(new_vy, new_vx);
}

// 估算局部代价项（用于梯度计算）
static inline float local_cost(const State_t *s,
                                const Controls_t *u,
                                const Controls_t *prev_u,
                                int t) {
    float ref_x = s->x;
    float f_x = ref_path(ref_x);
    float f_prime = ref_path_derivative(ref_x);
    float desired_psi = atanf(f_prime);

    float cte = s->y - f_x;
    float epsi = normalize_angle(s->psi - desired_psi);
    float v_err = REF_V - s->v;

    float cost = WEIGHT_CTE * cte * cte +
                 WEIGHT_EPSI * epsi * epsi +
                 WEIGHT_V * v_err * v_err +
                 WEIGHT_DELTA * u->delta[t] * u->delta[t] +
                 WEIGHT_A * u->a[t] * u->a[t];
        
    // printf("WEIGHT_CTE * cte * cte = %.3f, "
    //        "WEIGHT_EPSI * epsi * epsi = %.3f, "
    //        "WEIGHT_V * v_err * v_err = %.3f, "
    //        "WEIGHT_DELTA * u->delta[t] * u->delta[t] = %.3f, "
    //        "WEIGHT_A * u->a[t] * u->a[t] = %.3f\n",
    //        WEIGHT_CTE * cte * cte,
    //        WEIGHT_EPSI * epsi * epsi,
    //        WEIGHT_V * v_err * v_err,
    //        WEIGHT_DELTA * u->delta[t] * u->delta[t],
    //        WEIGHT_A * u->a[t] * u->a[t]);
    // printf("[debug] local_cost: t=%d, x=%.3f, y=%.3f, ref_x=%.3f, f_x=%.3f, cte=%.3f, epsi=%.3f, v=%.3f\n, cost = %.3f\n",
    //     t, s->x, s->y, ref_x, f_x, cte, fabsf(epsi), s->v, cost);

    if (prev_u != NULL && t > 0) {
        float delta_diff = u->delta[t] - prev_u->delta[t - 1];
        float a_diff = u->a[t] - prev_u->a[t - 1];

        cost += WEIGHT_DELTA_DIFF * delta_diff * delta_diff;
        cost += WEIGHT_A_DIFF * a_diff * a_diff;

        float delta_rate = fabsf(delta_diff / DT);
        if (delta_rate > MAX_DELTA_RATE){
            float penalty = delta_rate - MAX_DELTA_RATE;
            cost += 1000.0f * penalty * penalty;
        }
    }

    if (s->v < 0.0f) {
        cost += 5000.0f * s->v * s->v;
    }

    if (fabsf(epsi) > M_PI / 2.0f) {
        float penalty = fabsf(epsi) - (M_PI / 2.0f);
        cost += 1000.0f * penalty * penalty;
    }

    return cost;
}

static float rollout_tail_cost(State_t start,
                               float cos_psi,
                               float sin_psi,
                               const Controls_t *u_seq,
                               int start_t, int end_t) {
    float cost = 0.0f;
    State_t s = start;
    int horizon_len = MAX_HORIZON;
    int max_t = (start_t + horizon_len - 1 < end_t) ? (start_t + horizon_len - 1) : end_t;

    for (int t = start_t; t <= max_t; ++t) {
        simulate(&s, &cos_psi, &sin_psi, u_seq->delta[t], u_seq->a[t]);
        cost += local_cost(&s, u_seq, u_seq, t);
    }
    // printf("[debug] rollout_tail_cost: t=%d ~ %d, start_x=%.3f, start_y=%.3f, v=%.3f\n, cost = %.3f\n", start_t, end_t, start.x, start.y, start.v, cost);
    return cost;
}

static void compute_gradient(State_t current,
                            Controls_t *u_seq,
                            Controls_t *grad) 
{
    State_t traj[MAX_HORIZON+1];
    float cos_arr[MAX_HORIZON+1];
    float sin_arr[MAX_HORIZON+1];
    
    traj[0] = current;
    cos_arr[0] = cosf(current.psi);
    sin_arr[0] = sinf(current.psi);
    
    for (int t = 0; t < MAX_HORIZON; t++) {
        traj[t+1] = traj[t];
        cos_arr[t+1] = cos_arr[t];
        sin_arr[t+1] = sin_arr[t];
        simulate(&traj[t+1], &cos_arr[t+1], &sin_arr[t+1], 
                 u_seq->delta[t], u_seq->a[t]);
    }

    float cost_base_tail[MAX_HORIZON];
    #pragma omp parallel for
    for (int t = 0; t < MAX_HORIZON; t++) {
        cost_base_tail[t] = rollout_tail_cost(
            traj[t], cos_arr[t], sin_arr[t], 
            u_seq, t, MAX_HORIZON-1
        );
    }

    #pragma omp parallel for
    for (int t = 0; t < MAX_HORIZON; t++) {

        float orig_delta = u_seq->delta[t];
        float orig_a = u_seq->a[t];
        
        // delta梯度
        u_seq->delta[t] = orig_delta + GRAD_EPS;
        float cost_plus_delta = rollout_tail_cost(
            traj[t], cos_arr[t], sin_arr[t], 
            u_seq, t, MAX_HORIZON-1
        );
        u_seq->delta[t] = orig_delta;
        
        // a梯度
        u_seq->a[t] = orig_a + GRAD_EPS;
        float cost_plus_a = rollout_tail_cost(
            traj[t], cos_arr[t], sin_arr[t], 
            u_seq, t, MAX_HORIZON-1
        );
        u_seq->a[t] = orig_a;
        
        // 有限差分
        grad->delta[t] = (cost_plus_delta - cost_base_tail[t]) / GRAD_EPS;
        grad->a[t] = (cost_plus_a - cost_base_tail[t]) / GRAD_EPS;
    }
}

// 主控制器：使用梯度下降优化控制序列
static inline void neon_update_controls(Controls_t* u_seq, const Controls_t* grad) {
    // 根据梯度来调整控制序列 梯度正就减小控制变量，梯度负就增大控制变量
    const float32x4_t v_lr = vdupq_n_f32(LEARNING_RATE);
    const float32x4_t v_max_delta = vdupq_n_f32(MAX_DELTA);
    const float32x4_t v_min_delta = vdupq_n_f32(-MAX_DELTA);
    const float32x4_t v_max_a = vdupq_n_f32(MAX_A);
    const float32x4_t v_min_a = vdupq_n_f32(-MAX_A);

    #pragma omp parallel for simd
    for (int i = 0; i < MAX_HORIZON; i += 4) {
        // Delta处理
        float32x4_t v_delta = vld1q_f32(&u_seq->delta[i]);
        float32x4_t v_grad_d = vld1q_f32(&grad->delta[i]);
        // 除以10000是为了平衡梯度影响，防止由于梯度很大导致车辆偏转的变化幅度大
        v_delta = vmlsq_f32(v_delta, v_lr, v_grad_d / 10000); 
        v_delta = vmaxq_f32(vminq_f32(v_delta, v_max_delta), v_min_delta);
        // printf("[debug] neon_update_controls: i=%d, delta=%.3f, grad_delta=%.3f\n", i, vgetq_lane_f32(v_delta, 0), vgetq_lane_f32(v_grad_d, 0));
        vst1q_f32(&u_seq->delta[i], v_delta);

        // A处理
        float32x4_t v_a = vld1q_f32(&u_seq->a[i]);
        float32x4_t v_grad_a = vld1q_f32(&grad->a[i]);
        // 除以1000是为了平衡梯度影响,防止由于梯度很大导致车辆加速的变化幅度大
        v_a = vmlsq_f32(v_a, v_lr, v_grad_a / 1000); 
        v_a = vmaxq_f32(vminq_f32(v_a, v_max_a), v_min_a);
        // printf("[debug] neon_update_controls: i=%d, a=%.3f, grad_a=%.3f\n", i, vgetq_lane_f32(v_a, 0), vgetq_lane_f32(v_grad_a, 0));
        vst1q_f32(&u_seq->a[i], v_a);
    }
}

// 打印最终预测出的轨迹
static inline void print_predicted_trajectory(State_t s, const Controls_t* plan_out) {
    State_t state = s;
    float cos_psi = cosf(state.psi);
    float sin_psi = sinf(state.psi);

    for (int t = 0; t < MAX_HORIZON; ++t) {
        // 逐步推进一个时间步
        simulate(&state, &cos_psi, &sin_psi, plan_out->delta[t], plan_out->a[t]);

        float ref_y = ref_path(state.x);
        float cte = state.y - ref_y;

        printf("[predict] Step %3d: x=%.3f, y=%.3f, ref_y=%.3f, cte=%.3f, v=%+5.3f, psi=%+5.3f\n",
               t, state.x, state.y, ref_y, cte, state.v, state.psi);
    }
}

// 导出预测控制序列
static inline void mpc_control_with_plan(State_t current, Controls_t plan_out[MAX_HORIZON]) {
    Controls_t u_seq;
    Controls_t grad;

    // 初始化控制序列
    #pragma omp parallel for
    for (int i = 0; i < MAX_HORIZON; ++i) { 
        u_seq.delta[i] = 0.0f;
        u_seq.a[i] = 0.0f;
    }
    #pragma omp parallel for
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        compute_gradient(current, &u_seq, &grad);
        neon_update_controls(&u_seq, &grad);
        // 控制变化率约束
        for (int t = 1; t < MAX_HORIZON; ++t) {
            float delta_rate = u_seq.delta[t] - u_seq.delta[t - 1];
            if (fabsf(delta_rate) > MAX_DELTA_RATE * DT) {
                u_seq.delta[t] = u_seq.delta[t - 1] + SIGN(delta_rate) * MAX_DELTA_RATE * DT;
            }

            float a_rate = u_seq.a[t] - u_seq.a[t - 1];
            if (fabsf(a_rate) > MAX_A_RATE * DT) {
                u_seq.a[t] = u_seq.a[t - 1] + SIGN(a_rate) * MAX_A_RATE * DT;
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < MAX_HORIZON; ++i) {
        plan_out->delta[i] = u_seq.delta[i];
        plan_out->a[i] = u_seq.a[i];
    }
}
#endif