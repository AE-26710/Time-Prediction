
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>
#define PI 3.14159265358979323846
#define N (1024*64) // 

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
    fread(&header, sizeof(WAVHeader), 1, f);

    // 只支持 16bit 单通道
    if (header.bitsPerSample != 16 || header.numChannels != 1) {
        fclose(f);
        return -2;
    }

    fread(pcm_data, sizeof(int16_t), N, f);
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
// ---------------- 复数类型 ----------------

typedef struct {
    float* real;
    float* imag;
} Complex;

typedef struct {
    float real;
    float imag;
} Complex_;
Complex_ complex_add(Complex_ a, Complex_ b) {
    Complex_ r = {a.real + b.real, a.imag + b.imag};
    return r;
}

Complex_ complex_sub(Complex_ a, Complex_ b) {
    Complex_ r = {a.real - b.real, a.imag - b.imag};
    return r;
}

Complex_ complex_mul(Complex_ a, Complex_ b) {
    Complex_ r = {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
    return r;
}
//预计算旋转因子
// float *array_r;
// float *array_i;
// void preCompute(){
//     array_r = (float*)aligned_alloc(16, sizeof(float) * N);
//     array_i = (float*)aligned_alloc(16, sizeof(float) * N);
//     for (int k = 0; k < N/2; k++) {
//         float angle = -2 * PI * k / N;
//         array_r[k]= cos(angle);
//         array_i[k]= sin(angle);
//     }
// }
float **array_r, **array_i;

void preCompute() {
    array_r = (float**)malloc(sizeof(float*) * (int)(log2(N) + 1));
    array_i = (float**)malloc(sizeof(float*) * (int)(log2(N) + 1));
    
    int level = 0;
    for (int le = 2; le <= N; le <<= 1, level++) {
        int le2 = le / 2;
        int step = N / le;
        array_r[level] = (float*)malloc(sizeof(float) * le2);
        array_i[level] = (float*)malloc(sizeof(float) * le2);
        for (int k = 0; k < le2; k++) {
            int idx = k * step;
            double angle = -2 * PI * idx / N;
            array_r[level][k] = cos(angle);
            array_i[level][k] = sin(angle);
        }
    }
}
// ---------------- 快速傅里叶变换 FFT ----------------

void fft(float *x_real, float *x_imag) {
    int n = N;
    int i, j, k, m;
    int le, le2;
    float ur, ui;
    Complex_ u, t;
    int level=0;
    // 位反转重排（与原来一致）
    j = 0;
    for (i = 1; i < n; i++) {
        m = n >> 1;
        while (j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
        if (i < j) {
            float tmp_r = x_real[i];
            float tmp_i = x_imag[i];
            x_real[i] = x_real[j];
            x_imag[i] = x_imag[j];
            x_real[j] = tmp_r;
            x_imag[j] = tmp_i;
        }
    }

    for (le = 2; le <= n; le <<= 1,level++) {
        le2 = le / 2;

        for (j = 0; j < n; j += le) {
            if(le<=4){
                for (k = 0; k < le2; k++) {
                    int ip = j + k + le2;
                    
                    ur=array_r[level][k];
                    ui=array_i[level][k];
                    t = complex_mul((Complex_){ur, ui}, (Complex_){x_real[ip],x_imag[ip]});
                    u.real = x_real[j + k];
                    u.imag = x_imag[j + k];
                    Complex_ r1 = complex_add(u, t);
                    Complex_ r2 = complex_sub(u, t);
                    x_real[j + k] = r1.real;
                    x_imag[j + k] = r1.imag;
                    x_real[ip] = r2.real;
                    x_imag[ip] = r2.imag;

                    
                }
            }else{
                for (k = 0; k < le2; k += 4) {  // 每次处理4个点
                    int idx_base = k * (N / le);

                    // 旋转因子
                    float wr[4] = {
                        array_r[level][k],
                        array_r[level][k+1],
                        array_r[level][k+2],
                        array_r[level][k+3]
                    };
                    float wi[4] = {
                        array_i[level][k],
                        array_i[level][k+1],
                        array_i[level][k+2],
                        array_i[level][k+3]
                    };

                    float32x4_t wr_v = vld1q_f32(wr);
                    float32x4_t wi_v = vld1q_f32(wi);

                    // x[ip] = x[j + k + le2]
                    float32x4_t xr_ip = vld1q_f32(&x_real[j + k + le2]);
                    float32x4_t xi_ip = vld1q_f32(&x_imag[j + k + le2]);

                    // 复数乘法：t = w * x[ip]
                    float32x4_t t_real = vmlsq_f32(vmulq_f32(wr_v, xr_ip), wi_v, xi_ip); // wr*xr - wi*xi
                    float32x4_t t_imag = vmlaq_f32(vmulq_f32(wr_v, xi_ip), wi_v, xr_ip); // wr*xi + wi*xr

                    // x[j + k]
                    float32x4_t xr_u = vld1q_f32(&x_real[j + k]);
                    float32x4_t xi_u = vld1q_f32(&x_imag[j + k]);

                    // x[j + k] = u + t
                    float32x4_t xr_sum = vaddq_f32(xr_u, t_real);
                    float32x4_t xi_sum = vaddq_f32(xi_u, t_imag);

                    // x[ip] = u - t
                    float32x4_t xr_diff = vsubq_f32(xr_u, t_real);
                    float32x4_t xi_diff = vsubq_f32(xi_u, t_imag);

                    // 写回
                    vst1q_f32(&x_real[j + k], xr_sum);
                    vst1q_f32(&x_imag[j + k], xi_sum);
                    vst1q_f32(&x_real[j + k + le2], xr_diff);
                    vst1q_f32(&x_imag[j + k + le2], xi_diff);
                }

            }
        }
    }
}

// ---------------- IFFT（同上但略有改动） ----------------
void ifft(float *x_real, float *x_imag) {
    
    for (int i = 0; i < N; i++) {
        x_imag[i] = -x_imag[i];
    }
    fft(x_real,x_imag);
    for (int i = 0; i < N; i++) {
        x_real[i] = x_real[i]/N;
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
    preCompute();
    // 转换为复数
    Complex x;
    x.real = (float*)aligned_alloc(16, sizeof(float) * N);
    x.imag = (float*)aligned_alloc(16, sizeof(float) * N);

    for (int i = 0; i < N; i++) {
        x.real[i] = (float)pcm_data[i];
        x.imag[i] = 0.0f;
    }

    struct timespec start, end;

    // 获取起始时间
    clock_gettime(CLOCK_MONOTONIC, &start);
    // 执行 FFT
    fft(x.real,x.imag);
    // 获取结束时间
    clock_gettime(CLOCK_MONOTONIC, &end);

    double seconds = (end.tv_sec - start.tv_sec);
    double nanoseconds = (end.tv_nsec - start.tv_nsec) / 1e6;
    if (end.tv_nsec < start.tv_nsec) {
        seconds -= 1;
        nanoseconds += 1000;
    }

    double elapsed = seconds + nanoseconds;

    printf("FFT Time: %.4f ms\n", elapsed);

    ifft(x.real,x.imag);

    // 转换回 PCM
    for (int i = 0; i < N; i++) {
        double sample = round(x.real[i]);
        if (sample > 32767) sample = 32767;
        if (sample < -32768) sample = -32768;
        pcm_data[i] = (int16_t)sample;
    }
    double max_diff = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = cabs(x.real[i] - pcm_data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max difference between original and reconstructed data: %f\n", max_diff);
    
    
    if (write_wav("output.wav") != 0) {
        printf("cannot generate output.wav\n");
        return -1;
    }
    free(pcm_data);
    free(x.real);
    free(x.imag);
    free(array_i);
    free(array_r);
    printf("done with generating output.wav\n");
    return 0;
}

