/*
 *  Copyright (C) 2018-2021 Texas Instruments Incorporated
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include <ti/osal/osal.h>
#include <ti/osal/DebugP.h>
#include <ti/board/board.h>
#include <ti/osal/TimerP.h>
#include "FreeRTOS.h"
#include <FREERTOS_log.h>
#include <timers.h>
#include "task.h"
#include "CPUMonitor.h"

#define MAIN_TASK_PRI  (configMAX_PRIORITIES-1)

#define MAIN_TASK_SIZE ((32 * 1024U)/sizeof(configSTACK_DEPTH_TYPE))
StackType_t gMainTaskStack[MAIN_TASK_SIZE] __attribute__((aligned(32)));
StaticTask_t gMainTaskObj;
TaskHandle_t gMainTask;

void test_freertos_main(void *args);
void c66xIntrConfig(void);

// 单位矩阵
void mat_identity(double *I_mat, size_t dim) {
    memset(I_mat, 0, sizeof(double) * dim * dim);
    for (size_t i = 0; i < dim; i++) I_mat[i * dim + i] = 1.0;
}

// 加法
void mat_add(double *A, double *B, double *Result, size_t dim) {
    for (size_t i = 0; i < dim * dim; i++) Result[i] = A[i] + B[i];
}

// 减法
void mat_sub(double *A, double *B, double *Result, size_t dim) {
    for (size_t i = 0; i < dim * dim; i++) Result[i] = A[i] - B[i];
}

// 乘法
void mat_mul(double *A, double *B, double *Result, size_t dim) {
    memset(Result, 0, sizeof(double) * dim * dim);
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            for (size_t k = 0; k < dim; k++) {
                Result[i * dim + j] += A[i * dim + k] * B[k * dim + j];
            }
        }
    }
}

// 对角阵求逆
void mat_inv_diag(double *A, double *Result, size_t dim) {
    for (size_t i = 0; i < dim * dim; i++) Result[i] = 0.0;
    for (size_t i = 0; i < dim; i++) {
        if (A[i * dim + i] != 0.0)
            Result[i * dim + i] = 1.0 / A[i * dim + i];
    }
}

// 矩阵乘法实现
void matrix_multiply(size_t n, size_t k, size_t m, double *A, double *B, double *C)
{
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < m; j++)
        {
            // C[i * m + j] = 0;
            for (size_t l = 0; l < k; l++)
            {
                C[i * m + j] += A[i * k + l] * B[l * m + j];
            }
        }
    }
}

int matrix_algo(size_t n,size_t k, size_t m)
{
    // size_t k, m;
    // m = n;
    // k = n;
    // 分配内存空间
    // uint32_t xStartTicks = (uint32_t)uiPortGetRunTimeCounterValue();
    // FREERTOS_log("[%u]matrix_algo begin!, size_t:%u\r\n", xStartTicks, n);
    // FREERTOS_log("matrix size: %u, Free mem:%u\r\n", n, xPortGetFreeHeapSize());

    double *A = (double *)pvPortMalloc(n * k * sizeof(double));
    double *B = (double *)pvPortMalloc(k * m * sizeof(double));
    double *C = (double *)pvPortMalloc(n * m * sizeof(double));
    if (A == NULL) {
        FREERTOS_log("Failed to allocate A!\r\n");
        return 1;
    }

    if (B == NULL) {
        FREERTOS_log("Failed to allocate B!\r\n");
        vPortFree(A);
        return 1;
    }

    if (C == NULL) {
        FREERTOS_log("Failed to allocate C!\r\n");
        vPortFree(A);
        vPortFree(B);
        return 1;
    }

    //FREERTOS_log("malloc pass!\r\n");

    // 使用当前时间作为随机数种子
    //srand(time(NULL));
	srand(1023);

    //FREERTOS_log("srand pass!\r\n");
    // 随机生成矩阵A和B的元素
    for (int i = 0; i < n * k; i++)
    {
        A[i] = 1.0*(rand() % 100);
    }
    //FREERTOS_log("rand A pass!\r\n");
    for (int i = 0; i < k * m; i++)
    {
        B[i] = 1.0*(rand() % 100);
    }
    for (int i = 0; i < n * m; i++)
    {
        C[i] = 0;
    }
   // for(int i=1;i<5;i++)
   // {
//	    printf("%f\n",A[i]);
//	    printf("%f\n",B[i]);
  //  }

    //FREERTOS_log("Random complete\r\n");

    // 执行矩阵乘法
    matrix_multiply(n, k, m, A, B, C);

    //FREERTOS_log("multi complete\r\n");


    // 输出结果矩阵C
    // printf("矩阵乘法的结果是:\n");
    // for (int i = 0; i < n*m; i++) {
    //     printf("%d ", C[i]);
    //     if ((i + 1) % m == 0) {
    //         printf("\n");
    //     }
    // }
    // 释放内存空间
    vPortFree(A);
    vPortFree(B);
    vPortFree(C);
    return 0;
}

// 迭代版 FFT
void fft_iterative(complex double *X, size_t N) {
    // 计算比特反转置换
    size_t logN = log2(N);
    for (size_t i = 0; i < N; i++) {
        size_t rev = 0;
        for (size_t j = 0; j < logN; j++) {
            if ((i >> j) & 1) rev |= (1 << (logN - 1 - j));
        }
        if (i < rev) {
            complex double temp = X[i];
            X[i] = X[rev];
            X[rev] = temp;
        }
    }

    // 迭代计算 FFT
    for (size_t len = 2; len <= N; len *= 2) {
        double angle = -2.0 * M_PI / len;
        for (size_t i = 0; i < N; i += len) {
            for (size_t j = 0; j < len / 2; j++) {
                double cos_part = cos(angle * j);
                double sin_part = sin(angle * j);
                complex double w = cos_part + sin_part * I;  // 计算 Twiddle Factor

                complex double u = X[i + j];
                complex double v = w * X[i + j + len / 2];

                X[i + j] = u + v;
                X[i + j + len / 2] = u - v;
            }
        }
    }
}

// FFT 计算封装
int fft_algo(size_t N)
{
    // 申请存储空间
    complex double *X = (complex double *)pvPortMalloc(N * sizeof(complex double));

    if (X == NULL) {
        FREERTOS_log("Failed to allocate X!\r\n");
        return 1;
    }
    // 初始化输入数据（随机数）
    srand(1023);
    for (size_t i = 0; i < N; i++) {
        double real_part = rand() % 100;
        double imag_part = rand() % 100;
        X[i] = real_part + imag_part * I;
    }

    // 执行迭代 FFT
    fft_iterative(X, N);

    // 释放内存
    vPortFree(X);
    return 0;
}

int kalman_simulate(size_t dim, size_t steps) {
    double *x = (double *)pvPortCalloc(dim, sizeof(double));          // 状态估计向量
    double *P = (double *)pvPortCalloc(dim * dim, sizeof(double));    // 协方差矩阵
    double *Q = (double *)pvPortCalloc(dim * dim, sizeof(double));    // 过程噪声
    double *R = (double *)pvPortCalloc(dim * dim, sizeof(double));    // 测量噪声
    double *K = (double *)pvPortMalloc(dim * dim * sizeof(double));    // 卡尔曼增益
    double *I_mat = (double *)pvPortMalloc(dim * dim * sizeof(double));    // 单位矩阵
    double *z = (double *)pvPortMalloc(dim * sizeof(double));          // 测量值
    double *true_state = (double *)pvPortMalloc(dim * sizeof(double)); // 真实状态

    double *temp1 = (double *)pvPortMalloc(dim * dim * sizeof(double));
    double *temp2 = (double *)pvPortMalloc(dim * dim * sizeof(double));
    double *temp_vec = (double *)pvPortMalloc(dim * sizeof(double));

    if(x == NULL || P == NULL || Q == NULL || R == NULL || K == NULL || I_mat == NULL || z == NULL || true_state == NULL || temp1 == NULL || temp2 == NULL || temp_vec == NULL) {
        FREERTOS_log("Failed to allocate memory!\r\n");
        return -1;
    }

    // 初始化
    for (size_t i = 0; i < dim; i++) {
        P[i * dim + i] = 1.0;
        Q[i * dim + i] = 0.5;
        R[i * dim + i] = 0.5;
        true_state[i] = 0.0;
    }
    mat_identity(I_mat, dim);

    // printf("Step\t");
    // for (size_t i = 0; i < dim; i++) printf("True[%zu]\t", i);
    // for (size_t i = 0; i < dim; i++) printf("Meas[%zu]\t", i);
    // for (size_t i = 0; i < dim; i++) printf("Est[%zu]\t", i);
    // printf("\n");

    for (size_t k = 0; k < steps; k++) {
        // 模拟真实状态和测量
        for (size_t i = 0; i < dim; i++) {
            true_state[i] += 1.0;
            z[i] = true_state[i] + ((rand() % 2000) / 1000.0 - 1.0);  // 噪声[-1, 1]
        }

        // 预测：P = P + Q
        mat_add(P, Q, P, dim);

        // K = P * inv(P + R)
        mat_add(P, R, temp1, dim);
        mat_inv_diag(temp1, temp2, dim);  // 对角矩阵快速求逆
        mat_mul(P, temp2, K, dim);

        // x = x + K * (z - x)
        for (size_t i = 0; i < dim; i++) temp_vec[i] = z[i] - x[i];
        for (size_t i = 0; i < dim; i++) {
            double sum = 0;
            for (size_t j = 0; j < dim; j++) {
                sum += K[i * dim + j] * temp_vec[j];
            }
            x[i] += sum;
        }

        // P = (I_mat - K) * P
        mat_sub(I_mat, K, temp1, dim);
        mat_mul(temp1, P, P, dim);

        // 输出
        // printf("%4zu\t", k + 1);
        // for (size_t i = 0; i < dim; i++) printf("%.2f\t", true_state[i]);
        // for (size_t i = 0; i < dim; i++) printf("%.2f\t", z[i]);
        // for (size_t i = 0; i < dim; i++) printf("%.2f\t", x[i]);
        // printf("\n");
    }

    // 释放内存
    vPortFree(x); vPortFree(P); vPortFree(Q); vPortFree(R); vPortFree(K); vPortFree(I_mat); vPortFree(z); vPortFree(true_state);
    vPortFree(temp1); vPortFree(temp2); vPortFree(temp_vec);
    return 0;
}

void LoadFunc(void * args){
    size_t n = (size_t)args;
    size_t next_n = n + 9;
    int ret;
    char buffer[42];
    uint32_t xStartTicks = (uint32_t)uiPortGetRunTimeCounterValue();
    
    //FREERTOS_log("[%u]LoadFunc:begin!\r\n", xStartTicks);
    // while(CPUMonitor_calcCounterDiff((uint32_t)uiPortGetRunTimeCounterValue(), xStartTicks) < 20000){
    // }

    ret = kalman_simulate(n, 1);

    if (!ret) {
        uint32_t xEndTicks = (uint32_t)uiPortGetRunTimeCounterValue();
        uint32_t duration = CPUMonitor_calcCounterDiff(xEndTicks, xStartTicks);
        snprintf(buffer, sizeof(buffer), "%lf", duration / (double)(1000 * 1000));
        //FREERTOS_log("[%u]LoadFunc:start!\r\n", xStartTicks);
        FREERTOS_log("%u,n^3,%s,Cortex-R5F,arm32,KF,1\r\n", n, buffer);
    }

    //FREERTOS_log("[%u]LoadFunc:done!\r\n", (uint32_t)uiPortGetRunTimeCounterValue());

    snprintf(buffer, sizeof(buffer), "Kalman_%u", next_n);
    if (next_n < 1000 && !ret) {
        BaseType_t xReturned;

        xReturned = xTaskCreate(
            LoadFunc,       /* Function that implements the task. */
            buffer,          /* Text name for the task. */
            MAIN_TASK_SIZE /3,      /* Stack size in words, not bytes. */
            ( void * ) next_n,    /* Parameter passed into the task. */
            MAIN_TASK_PRI - 1,/* Priority at which the task is created. */
            NULL);      /* Used to pass out the created task's handle. */

        if (xReturned != pdPASS) {
            FREERTOS_log("Failed to create task!\r\n");
        }
    } else if (next_n >= 1000) {
        FREERTOS_log("Maximum n reached!\r\n");
    }

    vTaskDelete(NULL);
}

void CPULoadFunc(struct tmrTimerControl * args){
    FREERTOS_log("[%u]CPU_Load:[%u%%]!!\r\n", (uint32_t)uiPortGetRunTimeCounterValue(), CPUMonitor_getCPULoad());
    CPUMonitor_reset();
}

void LoadTimer(struct tmrTimerControl * args){
    /* Create the task, storing the handle. */

    xTaskCreate(
                    LoadFunc,       /* Function that implements the task. */
                    "ALIAS",          /* Text name for the task. */
                    MAIN_TASK_SIZE,      /* Stack size in words, not bytes. */
                    ( void * ) 100,    /* Parameter passed into the task. */
                    MAIN_TASK_PRI - 1,/* Priority at which the task is created. */
                    NULL);      /* Used to pass out the created task's handle. */
}

void frertos_main(void *args)
{
#if defined(BUILD_MCU1_0)
    Board_initCfg boardCfg;
    Board_STATUS  status;

    boardCfg = BOARD_INIT_PINMUX_CONFIG |
               BOARD_INIT_UART_STDIO;

    status = Board_init(boardCfg);

    DebugP_assert(BOARD_SOK == status);
#endif
    // TimerHandle_t xTimers;
    // xTimers = xTimerCreate
    //             ( /* Just a text name, not used by the RTOS kernel. */
    //                 "CPUMonitor",
    //                 /* The timer period in ticks, must be greater than 0. */
    //                 pdMS_TO_TICKS(1000),
    //                 /* The timers will auto-reload themselves when they expire. */
    //                 pdTRUE,
    //                 /* The ID is used to store a count of the number of times the
    //                 timer has expired, which is initialised to 0. */
    //                 ( void * ) 0,
    //                 /* Each timer calls the same callback when it expires. */
    //                 CPULoadFunc
    //             );
    // if( xTimers == NULL )
    // {
    //     /* The timer was not created. */
    // }
    // else
    // {
    //     /* Start the timer. No block time is specified, and
    //     even if one was it would be ignored because the RTOS
    //     scheduler has not yet been started. */
    //     if( xTimerStart( xTimers, 0 ) != pdPASS )
    //     {
    //         /* The timer could not be set into the Active state. */
    //     }s
    // }

    // TimerHandle_t LoadGen;
    // LoadGen = xTimerCreate
    //         ( /* Just a text name, not used by the RTOS kernel. */
    //             "LoadTimer",
    //             /* The timer period in ticks, must be greater than 0. */
    //             pdMS_TO_TICKS(200),
    //             /* The timers will auto-reload themselves when they expire. */
    //             pdTRUE,
    //             /* The ID is used to store a count of the number of times the
    //             timer has expired, which is initialised to 0. */
    //             ( void * ) 0,
    //             /* Each timer calls the same callback when it expires. */
    //             LoadTimer
    //         );
    FREERTOS_log("input,time_complexity,time,cpu,arch,program,cache_on\r\n");

    xTaskCreate(
        LoadFunc,       /* Function that implements the task. */
        "ALIAS",          /* Text name for the task. */
        MAIN_TASK_SIZE/3,      /* Stack size in words, not bytes. */
        ( void * ) 10,    /* Parameter passed into the task. */
        MAIN_TASK_PRI - 1,/* Priority at which the task is created. */
        NULL);      /* Used to pass out the created task's handle. */

    // if( LoadGen == NULL )
    // {
    //     /* The timer was not created. */
    // }
    // else
    // {
    //     /* Start the timer. No block time is specified, and
    //     even if one was it would be ignored because the RTOS
    //     scheduler has not yet been started. */
    //     if( xTimerStart( LoadGen, 0 ) != pdPASS )
    //     {
    //         /* The timer could not be set into the Active state. */
    //     }
    // }
    vTaskDelete(NULL);
}

void timerIsr(void *arg){
    FREERTOS_log("[%u]timIsr!!!\r\n", (uint32_t)uiPortGetRunTimeCounterValue());
}

int32_t main()
{
#if !defined(BUILD_MCU1_0)
    Board_initCfg boardCfg;
    Board_STATUS  status;

    boardCfg = BOARD_INIT_PINMUX_CONFIG |
               BOARD_INIT_UART_STDIO;

    status = Board_init(boardCfg);

    DebugP_assert(BOARD_SOK == status);
#endif
    c66xIntrConfig();

    // int32_t id = TimerP_ANY;
    // TimerP_Params timerParams;
    // TimerP_Params_init(&timerParams);
    // timerParams.name = "m_test";
    // timerParams.periodType = TimerP_PeriodType_MICROSECS;
    // timerParams.runMode    = TimerP_RunMode_CONTINUOUS;
    // timerParams.startMode  = TimerP_StartMode_USER;
    // timerParams.period     = 10000;
    // TimerP_Handle handle = TimerP_create(id, (TimerP_Fxn)&timerIsr, &timerParams);
    // TimerP_start(handle);
    /* This task is created at highest priority, it should create more tasks and then delete itself */
    gMainTask = xTaskCreateStatic( frertos_main,   /* Pointer to the function that implements the task. */
                                  "freertos_main", /* Text name for the task.  This is to facilitate debugging only. */
                                  MAIN_TASK_SIZE,  /* Stack depth in units of StackType_t typically uint32_t on 32b CPUs */
                                  NULL,            /* We are not using the task parameter. */
                                  MAIN_TASK_PRI,   /* task priority, 0 is lowest priority, configMAX_PRIORITIES-1 is highest */
                                  gMainTaskStack,  /* pointer to stack base */
                                  &gMainTaskObj ); /* pointer to statically allocated task object memory */
    configASSERT(NULL != gMainTask);

	/* Start the scheduler to start the tasks executing. */
	vTaskStartScheduler();

	/* The following line should never be reached because vTaskStartScheduler()
	will only return if there was not enough FreeRTOS heap memory available to
	create the Idle and (if configured) Timer tasks.  Heap management, and
	techniques for trapping heap exhaustion, are described in the book text. */
    DebugP_assert(BFALSE);

    return 0;
}

void c66xIntrConfig(void)
{
#if defined (_TMS320C6X) && defined (SOC_J721E)
    /* On J721E C66x builds we define timer tick in the configuration file to
     * trigger event #21 for C66x_1(from DMTimer0) and #20 for C66x_2(from DMTimer1). 
     * Map DMTimer interrupts to these events through DMSC RM API.
     */
    #include <ti/drv/sciclient/sciclient.h>

    struct tisci_msg_rm_irq_set_req     rmIrqReq;
    struct tisci_msg_rm_irq_set_resp    rmIrqResp;

    rmIrqReq.valid_params           = TISCI_MSG_VALUE_RM_DST_ID_VALID |
                                      TISCI_MSG_VALUE_RM_DST_HOST_IRQ_VALID;
    rmIrqReq.src_index              = 0U;
#if defined (BUILD_C66X_1)
    rmIrqReq.src_id                 = TISCI_DEV_TIMER0;
    rmIrqReq.dst_id                 = TISCI_DEV_C66SS0_CORE0;
    rmIrqReq.dst_host_irq           = 21U;
#endif
#if defined (BUILD_C66X_2)
    rmIrqReq.src_id                 = TISCI_DEV_TIMER1;
    rmIrqReq.dst_id                 = TISCI_DEV_C66SS1_CORE0;
    rmIrqReq.dst_host_irq           = 20U;
#endif
    /* Unused params */
    rmIrqReq.global_event           = 0U;
    rmIrqReq.ia_id                  = 0U;
    rmIrqReq.vint                   = 0U;
    rmIrqReq.vint_status_bit_index  = 0U;
    rmIrqReq.secondary_host         = TISCI_MSG_VALUE_RM_UNUSED_SECONDARY_HOST;

    Sciclient_rmIrqSet(&rmIrqReq, &rmIrqResp, SCICLIENT_SERVICE_WAIT_FOREVER);
#endif

    return;
}


