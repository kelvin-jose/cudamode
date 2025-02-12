#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void matrix_mult(float *A, float*B, float *C, const int M, const int K, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0;
        for(int i = 0; i < K; i++)
            sum += A[row * K + i] * B[i * N + col];
        C[row * N + col] = sum;
    }
}

void random_init(float *matrix, int M, int N) {
    for(int i = 0; i < M * N; i++)
        matrix[i] = (float)rand() / RAND_MAX;
}

int main() {
    float *h_matA, *h_matB, *h_matC, *d_matA, *d_matB, *d_matC;
    int M = 256, N = 512, K = 128;
    int matA_size = sizeof(float) * M * K;
    int matB_size = sizeof(float) * K * N;
    int matC_size = sizeof(float) * M * N;

    h_matA = (float*)malloc(matA_size);
    h_matB = (float*)malloc(matB_size);
    h_matC = (float*)malloc(matC_size);

    random_init(h_matA, M, K);
    random_init(h_matB, K, N);

    cudaMalloc((void**)&d_matA, matA_size);
    cudaMalloc((void**)&d_matB, matB_size);
    cudaMalloc((void**)&d_matC, matC_size);

    cudaMemcpy(d_matA, h_matA, matA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, matB_size, cudaMemcpyHostToDevice);

    dim3 block_size(32, 32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);

    matrix_mult<<<grid_size, block_size>>>(d_matA, d_matB, d_matC, M, K, N);

    cudaMemcpy(h_matC, d_matC, matC_size, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < M * N; i++) 
        printf("\n%f", h_matC[i]);

    free(h_matA);
    free(h_matB);
    free(h_matC);

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
    return 0;
}