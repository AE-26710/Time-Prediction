#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI 3.14159265358979323846
#define N (1024)

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
    double real;
    double imag;
} Complex;

Complex complex_add(Complex a, Complex b) {
    Complex r = {a.real + b.real, a.imag + b.imag};
    return r;
}

Complex complex_sub(Complex a, Complex b) {
    Complex r = {a.real - b.real, a.imag - b.imag};
    return r;
}

Complex complex_mul(Complex a, Complex b) {
    Complex r = {
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
    return r;
}

// 预计算旋转因子
double *array_r;
double *array_i;
void preCompute(){
    array_r = (double*)malloc(sizeof(double) * N);
    array_i = (double*)malloc(sizeof(double) * N);
    for (int k = 0; k < N; k++) {
        double angle = -2 * PI * k / N;
        array_r[k]= cos(angle);
        array_i[k]= sin(angle);
    }
}

// 位反转重排 - 基4
void bit_reverse_4(Complex *x) {
    int log4n = 0;
    for (int temp = N; temp > 1; temp >>= 2) log4n++;
    for (int i = 0; i < N; i++) {
        int rev = 0;
        int temp = i;
        for (int j = 0; j < log4n; j++) {
            rev = (rev << 2) | (temp & 0x3);
            temp >>= 2;
        }
        if (rev > i) {
            Complex tmp = x[i];
            x[i] = x[rev];
            x[rev] = tmp;
        }
    }
}

// ---------------- 基4 FFT ----------------
void fft(Complex *x) {
    bit_reverse_4(x); // 基4的位反转

    for (int s = 4; s <= N; s <<= 2) {
        int m = s / 4;
        for (int k = 0; k < N; k += s) {
            for (int j = 0; j < m; j++) {
                int idx1 = k + j;
                int idx2 = idx1 + m;
                int idx3 = idx2 + m;
                int idx4 = idx3 + m;

                int twiddle_stride = N / s;

                Complex W1 = { array_r[j * twiddle_stride], array_i[j * twiddle_stride] };
                Complex W2 = complex_mul(W1, W1);
                Complex W3 = complex_mul(W2, W1);

                Complex A = x[idx1];
                Complex B = complex_mul(W1, x[idx2]);
                Complex C = complex_mul(W2, x[idx3]);
                Complex D = complex_mul(W3, x[idx4]);

                Complex T0 = complex_add(A, C);
                Complex T1 = complex_sub(A, C);
                Complex T2 = complex_add(B, D);
                Complex T3 = complex_sub(B, D);

                x[idx1] = complex_add(T0, T2);
                x[idx2] = complex_add(T1, (Complex){ -T3.imag, T3.real });  // j*(B-D)
                x[idx3] = complex_sub(T0, T2);
                x[idx4] = complex_sub(T1, (Complex){ -T3.imag, T3.real });  // -j*(B-D)

            }
        }
    }
}

// ---------------- IFFT（同上但略有改动） ----------------
void ifft(Complex* x) {
    for (int i = 0; i < N; i++) {
        x[i].imag = -x[i].imag;
    }
    // preComputeIFFT();
    fft(x);
    for (int i = 0; i < N; i++) {
        x[i].real = x[i].real / N;
        x[i].imag = -x[i].imag / N;
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

    // 转换为复数
    Complex* x = (Complex*)malloc(sizeof(Complex) * N);
    for (int i = 0; i < N; i++) {
        x[i].real = pcm_data[i];
        x[i].imag = 0.0;
    }

    preCompute();
    clock_t start = clock();
    // 执行 FFT
    fft(x);
    clock_t end = clock();
    
    ifft(x);

    double time_spent = (double)(end - start) * 1000 / CLOCKS_PER_SEC;
    printf("FFT Time: %.6f ms\n", time_spent);

    // 转换回 PCM
    for (int i = 0; i < N; i++) {
        double sample = round(x[i].real);
        if (sample > 32767) sample = 32767;
        if (sample < -32768) sample = -32768;
        pcm_data[i] = (int16_t)sample;
    }
    // 验证结果
    double max_diff = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = cabs(x[i].real - pcm_data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max difference between original and reconstructed data: %f\n", max_diff);

    if (write_wav("output.wav") != 0) {
        printf("cannot generate output.wav\n");
        return -1;
    }
    free(pcm_data);
    free(x);
    free(array_i);
    free(array_r);
    printf("done with generating output.wav\n");
    return 0;
}