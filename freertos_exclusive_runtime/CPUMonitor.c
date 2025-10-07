/*
 * Copyright (c) 2021, Texas Instruments Incorporated
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * *  Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * *  Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * *  Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/*
 *  ======== CPUMonitor_freertos.c ========
 */

/* ========================================================================== */
/*                             Include Files                                  */
/* ========================================================================== */

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <FreeRTOS.h>
#include <task.h>
#include <CPUMonitor.h>
#include <FREERTOS_log.h>
#include <task.h>
/* ========================================================================== */
/*                           Macros & Typedefs                                */
/* ========================================================================== */

/* None */

/* ========================================================================== */
/*                         Structure Declarations                             */
/* ========================================================================== */
extern TaskHandle_t xIdleTaskHandle = NULL;
typedef struct CPUMonitor_freertos_s
{   
    uint64_t          idlTskTime;
    uint64_t          totalTime;
    uint32_t          lastUpdate_idlTskTime;
    uint32_t          lastUpdate_totalTime;
} CPUMonitor_freertos;

/* ========================================================================== */
/*                          Function Declarations                             */
/* ========================================================================== */
static uint32_t CPUMonitor_calcPercentLoad(uint64_t threadTime, uint64_t totalTime);
uint32_t CPUMonitor_calcCounterDiff(uint32_t cur, uint32_t last);
/* ========================================================================== */
/*                            Global Variables                                */
/* ========================================================================== */

static CPUMonitor_freertos   gCPUMonitor_freertos;
static bool             gCPUMonitor_initDone = false;

/* ========================================================================== */
/*                          Function Definitions                              */
/* ========================================================================== */

void CPUMonitor_reset(void)
{
    vTaskSuspendAll();

    gCPUMonitor_freertos.idlTskTime = 0U;
    gCPUMonitor_freertos.totalTime = 0U;

    (void)xTaskResumeAll();

    return;
}

uint32_t CPUMonitor_getCPULoad(void)
{
    uint32_t cpuLoad;

    CPUMonitor_update();

    vTaskSuspendAll();

    cpuLoad = 100U - CPUMonitor_calcPercentLoad(gCPUMonitor_freertos.idlTskTime, gCPUMonitor_freertos.totalTime);

    (void)xTaskResumeAll();
    
    // CPUMonitor_reset();

    return cpuLoad;
}


void CPUMonitor_update(void)
{
    uint32_t            curTime;
    uint32_t            delta;
    // uint32_t            times;
    // uint32_t switch_context_inter[FSC_SIZE];
    // uint32_t context_used_inter[FSC_SIZE];

    vTaskSuspendAll();
    
    if(!gCPUMonitor_initDone)
    {
        (void)memset( (void *)&gCPUMonitor_freertos,0,sizeof(gCPUMonitor_freertos));
        gCPUMonitor_initDone = true;
    }

    /* Idle Task Update */
    curTime = (uint32_t)ulTaskGetIdleRunTimeCounter();
    delta = CPUMonitor_calcCounterDiff(curTime, gCPUMonitor_freertos.lastUpdate_idlTskTime);
    gCPUMonitor_freertos.lastUpdate_idlTskTime = curTime;

    gCPUMonitor_freertos.idlTskTime += delta;
    
    /* Total Time Update */
    curTime = (uint32_t)uiPortGetRunTimeCounterValue();
    delta = CPUMonitor_calcCounterDiff(curTime, gCPUMonitor_freertos.lastUpdate_totalTime);
    gCPUMonitor_freertos.lastUpdate_totalTime = curTime;
    gCPUMonitor_freertos.totalTime += delta;
    // times = switch_times;
    // for(uint32_t i=0;i<times;i++){
    //     switch_context_inter[i%FSC_SIZE] = switch_context[i%FSC_SIZE];
    //     context_used_inter[i%FSC_SIZE] = context_used[i%FSC_SIZE];
    // }
    // switch_times = 0;
    (void)xTaskResumeAll();
    // FREERTOS_log("switch for :[%u]times!!\r\n", times);
    // for(uint32_t i=0;i<times && i<FSC_SIZE;i++){
    //     if(context_used_inter[i%FSC_SIZE] == 1234567){
    //         FREERTOS_log("current task sleep!!\r\n");
    //     }else{
    //         FREERTOS_log("[timestamp:%u]last task uses:[%u]!!\r\n", switch_context_inter[i%FSC_SIZE], context_used_inter[i%FSC_SIZE]);
    //     }
    // }
    // FREERTOS_log("total_time:[%u]!!\r\n", delta);

    return;
}

/* ========================================================================================================================== */

uint32_t CPUMonitor_calcCounterDiff(uint32_t cur, uint32_t last)
{
    uint32_t delta;

    if(cur >= last)
    {
        delta = cur - last;
    }
    else
    {
        delta = ( 0xFFFFFFFFU - last ) + cur;
    }
    return delta;
}

static uint32_t CPUMonitor_calcPercentLoad(uint64_t threadTime, uint64_t totalTime)
{
    uint32_t percentLoad;

    percentLoad = (uint32_t)(threadTime  / (totalTime / 100U));

    return percentLoad;
}
