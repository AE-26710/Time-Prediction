/*
 * Copyright (c) 2015-2020, Texas Instruments Incorporated
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
/**
 *  \ingroup DRV_OSAL_MODULE
 *  \defgroup DRV_OSAL_LoadP LoadP
 *            LoadP interface
 *
 *  @{
 */

/** ============================================================================
 *  @file       LoadP.h
 *
 *  @brief      Load module to measure CPU Load and Task Load for the RTOS Porting Interface
 *
 *
 *  ============================================================================
 */

#ifndef CPUMONITOR__include
#define CPUMONITOR__include

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

extern void CPUMonitor_reset(void);

/*!
 *  @brief  Function to get CPU load in percentage
 *          
 *  @return CPU load in percentage
 */
extern uint32_t CPUMonitor_getCPULoad(void);

extern void CPUMonitor_update(void);

uint32_t CPUMonitor_calcCounterDiff(uint32_t cur, uint32_t last);

#endif /* ti_osal_LoadP__include */
/* @} */
