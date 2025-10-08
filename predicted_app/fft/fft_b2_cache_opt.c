#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <sys/mman.h>
#define PI 3.14159265358979323846
#define N (1024) // 

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

double **array_r, **array_i;

void preCompute() {
    array_r = (double**)malloc(sizeof(double*) * (int)(log2(N) + 1));
    array_i = (double**)malloc(sizeof(double*) * (int)(log2(N) + 1));
    
    int level = 0;
    for (int le = 2; le <= N; le <<= 1, level++) {
        int le2 = le / 2;
        int step = N / le;
        array_r[level] = (double*)malloc(sizeof(double) * le2);
        array_i[level] = (double*)malloc(sizeof(double) * le2);
        for (int k = 0; k < le2; k++) {
            int idx = k * step;
            double angle = -2 * PI * idx / N;
            array_r[level][k] = cos(angle);
            array_i[level][k] = sin(angle);
        }
    }
}
// ---------------- 快速傅里叶变换 FFT ----------------
void fft(Complex* x) {
    int n = N;
    int i, j, k, m;
    int le, le2;
    int level=0;
    double ur, ui, sr, si, tr, ti;
    Complex u, t;

    // 位反转重排
    j = 0;

    for (i = 1; i < n; i++) {
        m = n >> 1;
        while (j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
        if (i < j) {
            Complex temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
    }

    for (int le = 2; le <= n; le <<= 1,level++) {
        le2 = le / 2;
        for (j = 0; j < n; j += le) {
            for (k = 0; k < le2; k++) {

                double ur = array_r[level][k];
                double ui = array_i[level][k];
                int idx1 = j + k;
                int idx2 = idx1 + le2;
                
                double tr = ur * x[idx2].real - ui * x[idx2].imag;
                double ti = ur * x[idx2].imag + ui * x[idx2].real;

                double xr = x[idx1].real;
                double xi = x[idx1].imag;

                x[idx1].real = xr + tr;
                x[idx1].imag = xi + ti;

                x[idx2].real = xr - tr;
                x[idx2].imag = xi - ti;


            }
        }
    }
}

void ifft(Complex* x) {
    
    for (int i = 0; i < N; i++) {
        x[i].imag = -x[i].imag;
    }
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
    FILE *file1 = fopen("data.txt", "w");
    for (int i = 0; i < N; i++) {
        fprintf(file1, "%d\n", pcm_data[i]);
    }
    fclose(file1);
    preCompute();
    clock_t start = clock();
    // 执行 FFT
    fft(x);
    clock_t end = clock();
    ifft(x);
    
    double time_spent = (double)(end - start)*1000 / CLOCKS_PER_SEC;
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
        // double diff = fabs(round(creal(x[i])) - pcm_data[i]);
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
