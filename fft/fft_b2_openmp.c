#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <stdint.h>
#define PI 3.14159265358979323846

// WAV Header Structure
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

// WAV File Reading
int read_wav(const char* filename, int N) {
    FILE* f = fopen(filename, "rb");
    if (!f) return -1;
    WAVHeader header;
    fread(&header, sizeof(WAVHeader), 1, f);
    if (header.bitsPerSample != 16 || header.numChannels != 1) {
        fclose(f);
        return -2;
    }
    fread(pcm_data, sizeof(int16_t), N, f);
    fclose(f);
    return 0;
}

// WAV File Writing
int write_wav(const char* filename, int N) {
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;
    WAVHeader header = {
        .chunkID = {'R','I','F','F'},
        .chunkSize = 36 + N * 2,
        .format = {'W','A','V','E'},
        .subchunk1ID = {'f','m','t',' '},
        .subchunk1Size = 16,
        .audioFormat = 1,
        .numChannels = 1,
        .sampleRate = 44100,
        .byteRate = 44100 * 2,
        .blockAlign = 2,
        .bitsPerSample = 16,
        .subchunk2ID = {'d','a','t','a'},
        .subchunk2Size = N * 2
    };
    fwrite(&header, sizeof(header), 1, f);
    fwrite(pcm_data, sizeof(int16_t), N, f);
    fclose(f);
    return 0;
}

typedef double complex Complex;

// Bit-Reversal Function (In-Place)
void bit_reverse(int *rev, int n) {
    int log2n = 0;
    for (int temp = n; temp > 1; temp >>= 1) log2n++;
    for (int i = 0; i < n; i++) {
        int j = 0, k = i;
        for (int l = 0; l < log2n; l++) {
            j = (j << 1) | (k & 1);
            k >>= 1;
        }
        rev[i] = j;
    }
}

// Precompute Twiddle Factors
void preCompute(Complex *w, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        w[i] = cexp(-2.0 * PI * I * i / n);
    }
}

void bit_reverse_permute(Complex *x, int *rev, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            Complex temp = x[i];
            x[i] = x[rev[i]];
            x[rev[i]] = temp;
        }
    }
}

void fft(Complex *x, int *rev, Complex *w, int n) {
    bit_reverse_permute(x, rev, n);
    
    for (int s = 1; s <= log2(n); s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m2; j++) {
                Complex t = w[n / m * j] * x[k + j + m2];
                Complex u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m2] = u - t;
            }
        }
    }
}


void ifft(Complex *x, int *rev, Complex *w, int n) {
    bit_reverse_permute(x, rev, n);
    
    for (int s = 1; s <= log2(n); s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m2; j++) {
                Complex t = conj(w[n / m * j]) * x[k + j + m2];
                Complex u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m2] = u - t;
            }
        }
    }
    
    // #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] /= n;
    }
}

// Optimized Parallel FFT Function
void pfft(Complex *x, int *rev, Complex *w, int n) {
    bit_reverse_permute(x, rev, n);
    
    for (int s = 1; s <= log2(n); s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m2; j++) {
                Complex t = w[n / m * j] * x[k + j + m2];
                Complex u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m2] = u - t;
            }
        }
    }
}

// Optimized Parallel IFFT Function
void pifft(Complex *x, int *rev, Complex *w, int n) {
    bit_reverse_permute(x, rev, n);
    
    for (int s = 1; s <= log2(n); s++) {
        int m = 1 << s;
        int m2 = m >> 1;
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; k += m) {
            for (int j = 0; j < m2; j++) {
                Complex t = conj(w[n / m * j]) * x[k + j + m2];
                Complex u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m2] = u - t;
            }
        }
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        x[i] /= n;
    }
}

// Main Program
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Error: No Enough parameters (%s <num-of-threads> <power-of-transform-length>).\n", argv[0]);
        return 0;
    }

    int num_threads = atoi(argv[1]);
    int log2n = atoi(argv[2]);
    int n = 1 << log2n;

    omp_set_num_threads(num_threads);

    int *rev = (int *)malloc(n * sizeof(int));
    bit_reverse(rev, n);

    Complex *w = (Complex *)malloc(n * sizeof(Complex));
    preCompute(w, n);

    pcm_data = (int16_t*)malloc(sizeof(int16_t) * n);
    if (read_wav("test2.wav", n) != 0) {
        printf("无法读取 test2.wav\n");
        return -1;
    }

    Complex *x = (Complex *)malloc(n * sizeof(Complex));
    for (int i = 0; i < n; i++) {
        x[i] = pcm_data[i];
    }

    double t0 = omp_get_wtime();
    fft(x, rev, w, n);
    double t1 = omp_get_wtime();
    printf("Serial FFT Time: %f ms\n", (t1 - t0)*1000);

    double t4 = omp_get_wtime();
    ifft(x, rev, w, n);
    double t5 = omp_get_wtime();
    printf("Serial IFFT Time: %f ms\n", (t5 - t4)*1000);

    for (int i = 0; i < n; i++) {
        x[i] = pcm_data[i];
    }

    double t2 = omp_get_wtime();
    pfft(x, rev, w, n);
    double t3 = omp_get_wtime();
    printf("Parallel FFT Time: %f ms\n", (t3 - t2)*1000);

    double t6 = omp_get_wtime();
    pifft(x, rev, w, n);
    double t7 = omp_get_wtime();
    printf("Parallel IFFT Time: %f ms\n", (t7 - t6)*1000);

    // Verify Results
    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = cabs(x[i] - pcm_data[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Max difference between original and reconstructed data: %f\n", max_diff);

    // Save Results
    FILE *file = fopen("pcm_data1.txt", "w");
    for (int i = 0; i < n; i++) {
        fprintf(file, "%d\n", (int)round(creal(x[i])));
    }
    fclose(file);

    if (write_wav("output.wav", n) != 0) {
        printf("无法生成 output.wav\n");
        return -1;
    }

    printf("完成生成 output.wav\n");

    free(rev);
    free(w);
    free(pcm_data);
    free(x);

    return 0;
}