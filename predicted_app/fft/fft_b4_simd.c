#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#define PI 3.14159265358979323846
#define N (1024) // 假设 N 为 4 的幂

// ---------------- 读取/写入 WAV ----------------
typedef struct {
    char chunkID[4]; // "RIFF"
    uint32_t chunkSize;
    char format[4];  // "WAVE"
    char subchunk1ID[4]; // "fmt "
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char subchunk2ID[4]; // "data"
    uint32_t subchunk2Size;
} WAVHeader;

int16_t* pcm_data = NULL;

int read_wav(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return -1;
    WAVHeader header;
    if (fread(&header, sizeof(WAVHeader), 1, f) != 1) { fclose(f); return -1; }
    
    if (header.bitsPerSample != 16 || header.numChannels != 1) {
        fclose(f);
        return -2;
    }

    if (fread(pcm_data, sizeof(int16_t), N, f) != N) {
        fclose(f);
        return -3;
    }
    fclose(f);
    return 0;
}

int write_wav(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;
    WAVHeader header = {
        .chunkID = "RIFF",
        .chunkSize = 36 + N * 2,
        .format = "WAVE",
        .subchunk1ID = "fmt ",
        .subchunk1Size = 16,
        .audioFormat = 1,
        .numChannels = 1,
        .sampleRate = 44100,
        .byteRate = 44100 * 2,
        .blockAlign = 2,
        .bitsPerSample = 16,
        .subchunk2ID = "data",
        .subchunk2Size = N * 2
    };
    fwrite(&header, sizeof(header), 1, f);
    fwrite(pcm_data, sizeof(int16_t), N, f);
    fclose(f);
    return 0;
}

// ---------------- 复数类型 ----------------n
float *x_real;
float *x_imag;

// 为每个 stage 分别存储 twiddle（每个 stage 可以有不同的 batch 数量）
float **tw1_real_stages, **tw1_imag_stages;
float **tw2_real_stages, **tw2_imag_stages;
float **tw3_real_stages, **tw3_imag_stages;
int *tw_batches_per_stage;
int stages_count = 0;

static inline float32x4x2_t complex_mul_f32(float32x4x2_t a, float32x4x2_t b) {
    float32x4x2_t r;
    r.val[0] = vsubq_f32(vmulq_f32(a.val[0], b.val[0]), vmulq_f32(a.val[1], b.val[1])); // real
    r.val[1] = vaddq_f32(vmulq_f32(a.val[0], b.val[1]), vmulq_f32(a.val[1], b.val[0])); // imag
    return r;
}

int calc_log4(int n) {
    int cnt = 0;
    while (n > 1) { n >>= 2; cnt++; }
    return cnt;
}

void preCompute() {
    // 计算 stage 数
    stages_count = calc_log4(N);
    tw1_real_stages = (float**)malloc(sizeof(float*) * stages_count);
    tw1_imag_stages = (float**)malloc(sizeof(float*) * stages_count);
    tw2_real_stages = (float**)malloc(sizeof(float*) * stages_count);
    tw2_imag_stages = (float**)malloc(sizeof(float*) * stages_count);
    tw3_real_stages = (float**)malloc(sizeof(float*) * stages_count);
    tw3_imag_stages = (float**)malloc(sizeof(float*) * stages_count);
    tw_batches_per_stage = (int*)malloc(sizeof(int) * stages_count);

    int s_val = 4;
    for (int s_idx = 0; s_idx < stages_count; s_idx++, s_val <<= 2) {
        int m = s_val / 4;            // 每个组的 size
        int batches = (m + 3) / 4;    // 每 batch 4 个 j, 最后一 batch 可能不足 4
        tw_batches_per_stage[s_idx] = batches;

        tw1_real_stages[s_idx] = (float*)aligned_alloc(16, sizeof(float) * batches * 4);
        tw1_imag_stages[s_idx] = (float*)aligned_alloc(16, sizeof(float) * batches * 4);
        tw2_real_stages[s_idx] = (float*)aligned_alloc(16, sizeof(float) * batches * 4);
        tw2_imag_stages[s_idx] = (float*)aligned_alloc(16, sizeof(float) * batches * 4);
        tw3_real_stages[s_idx] = (float*)aligned_alloc(16, sizeof(float) * batches * 4);
        tw3_imag_stages[s_idx] = (float*)aligned_alloc(16, sizeof(float) * batches * 4);

        // 初始化为 0，安全起见
        for (int i = 0; i < batches * 4; i++) {
            tw1_real_stages[s_idx][i] = 0.0f;
            tw1_imag_stages[s_idx][i] = 0.0f;
            tw2_real_stages[s_idx][i] = 0.0f;
            tw2_imag_stages[s_idx][i] = 0.0f;
            tw3_real_stages[s_idx][i] = 0.0f;
            tw3_imag_stages[s_idx][i] = 0.0f;
        }

        int stride = N / s_val;
        int batch_idx = 0;
        for (int j = 0; j < m; j += 4) {
            float wr[4] = {0}, wi[4] = {0};
            float w2r[4] = {0}, w2i[4] = {0};
            float w3r[4] = {0}, w3i[4] = {0};
            for (int u = 0; u < 4; u++) {
                int jj = j + u;
                if (jj < m) {
                    int idx = jj * stride;
                    float angle1 = -2.0f * PI * idx / N;
                    float angle2 = 2.0f * angle1;
                    float angle3 = 3.0f * angle1;
                    wr[u] = cosf(angle1);
                    wi[u] = sinf(angle1);
                    w2r[u] = cosf(angle2);
                    w2i[u] = sinf(angle2);
                    w3r[u] = cosf(angle3);
                    w3i[u] = sinf(angle3);
                } else {
                    // padding for 不足 4 的部分
                    wr[u] = 1.0f; wi[u] = 0.0f;
                    w2r[u] = 1.0f; w2i[u] = 0.0f;
                    w3r[u] = 1.0f; w3i[u] = 0.0f;
                }
            }
            // 存储到 stage 的 twiddle buffer 中
            vst1q_f32(&tw1_real_stages[s_idx][batch_idx*4], vld1q_f32(wr));
            vst1q_f32(&tw1_imag_stages[s_idx][batch_idx*4], vld1q_f32(wi));
            vst1q_f32(&tw2_real_stages[s_idx][batch_idx*4], vld1q_f32(w2r));
            vst1q_f32(&tw2_imag_stages[s_idx][batch_idx*4], vld1q_f32(w2i));
            vst1q_f32(&tw3_real_stages[s_idx][batch_idx*4], vld1q_f32(w3r));
            vst1q_f32(&tw3_imag_stages[s_idx][batch_idx*4], vld1q_f32(w3i));
            batch_idx++;
        }
    }
}

void bit_reverse() {
    int log4n = calc_log4(N);
    for (int i = 0; i < N; i++) {
        int rev = 0;
        int temp = i;
        for (int j = 0; j < log4n; j++) {
            rev = (rev << 2) | (temp & 0x3);
            temp >>= 2;
        }
        if (rev > i) {
            float tmp_r = x_real[i];
            float tmp_i = x_imag[i];
            x_real[i] = x_real[rev];
            x_imag[i] = x_imag[rev];
            x_real[rev] = tmp_r;
            x_imag[rev] = tmp_i;
        }
    }
}

void fft() {
    bit_reverse();

    int s_val = 4;
    for (int s_idx = 0; s_idx < stages_count; s_idx++, s_val <<= 2) {
        int m = s_val / 4;
        int stride = N / s_val;
        int batches = tw_batches_per_stage[s_idx];

        for (int k = 0; k < N; k += s_val) {
            int batch_idx = 0;
            int j = 0;
            // 对于能整批的 part 使用 NEON
            for (; j + 3 < m; j += 4, batch_idx++) {
                int idx1 = k + j;
                int idx2 = idx1 + m;
                int idx3 = idx2 + m;
                int idx4 = idx3 + m;

                // Load twiddles (per-stage, per-batch)
                float32x4_t W1r = vld1q_f32(&tw1_real_stages[s_idx][batch_idx * 4]);
                float32x4_t W1i = vld1q_f32(&tw1_imag_stages[s_idx][batch_idx * 4]);
                float32x4_t W2r = vld1q_f32(&tw2_real_stages[s_idx][batch_idx * 4]);
                float32x4_t W2i = vld1q_f32(&tw2_imag_stages[s_idx][batch_idx * 4]);
                float32x4_t W3r = vld1q_f32(&tw3_real_stages[s_idx][batch_idx * 4]);
                float32x4_t W3i = vld1q_f32(&tw3_imag_stages[s_idx][batch_idx * 4]);

                // Load inputs
                float32x4_t A_r = vld1q_f32(&x_real[idx1]);
                float32x4_t A_i = vld1q_f32(&x_imag[idx1]);
                float32x4_t B_r = vld1q_f32(&x_real[idx2]);
                float32x4_t B_i = vld1q_f32(&x_imag[idx2]);
                float32x4_t C_r = vld1q_f32(&x_real[idx3]);
                float32x4_t C_i = vld1q_f32(&x_imag[idx3]);
                float32x4_t D_r = vld1q_f32(&x_real[idx4]);
                float32x4_t D_i = vld1q_f32(&x_imag[idx4]);

                // Complex mul
                float32x4x2_t B = complex_mul_f32((float32x4x2_t){B_r, B_i}, (float32x4x2_t){W1r, W1i});
                float32x4x2_t C = complex_mul_f32((float32x4x2_t){C_r, C_i}, (float32x4x2_t){W2r, W2i});
                float32x4x2_t D = complex_mul_f32((float32x4x2_t){D_r, D_i}, (float32x4x2_t){W3r, W3i});

                // Butterfly core
                float32x4_t T0r = vaddq_f32(A_r, C.val[0]);
                float32x4_t T0i = vaddq_f32(A_i, C.val[1]);
                float32x4_t T1r = vsubq_f32(A_r, C.val[0]);
                float32x4_t T1i = vsubq_f32(A_i, C.val[1]);
                float32x4_t T2r = vaddq_f32(B.val[0], D.val[0]);
                float32x4_t T2i = vaddq_f32(B.val[1], D.val[1]);
                float32x4_t T3r = vsubq_f32(B.val[0], D.val[0]);
                float32x4_t T3i = vsubq_f32(B.val[1], D.val[1]);

                float32x4_t jT3r = vnegq_f32(T3i);
                float32x4_t jT3i = T3r;

                vst1q_f32(&x_real[idx1], vaddq_f32(T0r, T2r));
                vst1q_f32(&x_imag[idx1], vaddq_f32(T0i, T2i));
                vst1q_f32(&x_real[idx2], vaddq_f32(T1r, jT3r));
                vst1q_f32(&x_imag[idx2], vaddq_f32(T1i, jT3i));
                vst1q_f32(&x_real[idx3], vsubq_f32(T0r, T2r));
                vst1q_f32(&x_imag[idx3], vsubq_f32(T0i, T2i));
                vst1q_f32(&x_real[idx4], vsubq_f32(T1r, jT3r));
                vst1q_f32(&x_imag[idx4], vsubq_f32(T1i, jT3i));
            }

            // 处理剩余 (m 为非 4 的倍数的尾部) - 标量回退
            for (; j < m; j++) {
                int idx1 = k + j;
                int idx2 = idx1 + m;
                int idx3 = idx2 + m;
                int idx4 = idx3 + m;

                // 计算 twiddle 索引
                int batch = j / 4;
                int pos_in_batch = j % 4;
                float W1r = tw1_real_stages[s_idx][batch*4 + pos_in_batch];
                float W1i = tw1_imag_stages[s_idx][batch*4 + pos_in_batch];
                float W2r = tw2_real_stages[s_idx][batch*4 + pos_in_batch];
                float W2i = tw2_imag_stages[s_idx][batch*4 + pos_in_batch];
                float W3r = tw3_real_stages[s_idx][batch*4 + pos_in_batch];
                float W3i = tw3_imag_stages[s_idx][batch*4 + pos_in_batch];

                float Br = x_real[idx2]*W1r - x_imag[idx2]*W1i;
                float Bi = x_real[idx2]*W1i + x_imag[idx2]*W1r;
                float Cr = x_real[idx3]*W2r - x_imag[idx3]*W2i;
                float Ci = x_real[idx3]*W2i + x_imag[idx3]*W2r;
                float Dr = x_real[idx4]*W3r - x_imag[idx4]*W3i;
                float Di = x_real[idx4]*W3i + x_imag[idx4]*W3r;

                float T0r = x_real[idx1] + Cr;
                float T0i = x_imag[idx1] + Ci;
                float T1r = x_real[idx1] - Cr;
                float T1i = x_imag[idx1] - Ci;
                float T2r = Br + Dr;
                float T2i = Bi + Di;
                float T3r = Br - Dr;
                float T3i = Bi - Di;

                float jT3r = -T3i;
                float jT3i = T3r;

                x_real[idx1] = T0r + T2r;
                x_imag[idx1] = T0i + T2i;
                x_real[idx2] = T1r + jT3r;
                x_imag[idx2] = T1i + jT3i;
                x_real[idx3] = T0r - T2r;
                x_imag[idx3] = T0i - T2i;
                x_real[idx4] = T1r - jT3r;
                x_imag[idx4] = T1i - jT3i;
            }
        }
    }
}

// ---------------- IFFT（同上但略有改动） ----------------
void ifft() {
    for (int i = 0; i < N; i++) {
        x_imag[i] = -x_imag[i];
    }
    fft();
    for (int i = 0; i < N; i++) {
        x_real[i] = x_real[i] / N;
        x_imag[i] = -x_imag[i] / N;
    }
}

// ---------------- 主程序 ----------------
int main() {
    pcm_data = (int16_t*)malloc(sizeof(int16_t) * N);
    if (!pcm_data) {
        printf("Failed to allocate pcm_data\n");
        return -1;
    }
    if (read_wav("test2.wav") != 0) {
        printf("无法读取 test2.wav\n");
        return -1;
    }

    x_real = aligned_alloc(16, sizeof(float) * N);
    x_imag = aligned_alloc(16, sizeof(float) * N);
    if (!x_real || !x_imag) { printf("aligned_alloc failed\n"); return -1; }

    for (int i = 0; i < N; i++) {
        x_real[i] = (float)pcm_data[i];
        x_imag[i] = 0.0f;
    }

    preCompute();
    clock_t start = clock();
    // 执行 FFT
    fft();
    clock_t end = clock();

    ifft();

    double time_spent = (double)(end - start) * 1000 / CLOCKS_PER_SEC;
    printf("FFT Time: %.6f ms\n", time_spent);

    // 验证结果
    double max_diff = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = fabs(x_real[i] - (double)pcm_data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max difference between original and reconstructed data: %f\n", max_diff);

    if (write_wav("output.wav") != 0) {
        printf("cannot generate output.wav\n");
        return -1;
    }
    free(pcm_data);
    free(x_real);
    free(x_imag);
    free(tw1_real_stages);
    free(tw1_imag_stages);
    free(tw2_real_stages);
    free(tw2_imag_stages);
    free(tw3_real_stages);
    free(tw3_imag_stages);
    free(tw_batches_per_stage);

    printf("done with generating output.wav\n");
    return 0;
}
