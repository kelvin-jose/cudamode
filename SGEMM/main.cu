#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include "kernels.cuh"

void random_init(float *array, int M, int N) {
    for(int i = 0; i < M * N; i++)
        array[i] = (float)rand() / RAND_MAX;
}

int main() {
    srand(time(NULL));

    int M = 4096, N = 4096, K = 4096;

    float *h_matA, *h_matB, *h_matC, *h_matD;
    
    int matA_size = M * N * sizeof(float);
    int matB_size = N * K * sizeof(float);
    int matC_size = M * K * sizeof(float);
    int matD_size = M * K * sizeof(float);

    h_matA = (float*)malloc(matA_size);
    h_matB = (float*)malloc(matB_size);
    h_matC = (float*)malloc(matC_size);
    h_matD = (float*)malloc(matC_size);

    random_init(h_matA, M, N);
    random_init(h_matB, N, K);
    random_init(h_matC, M, K);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sec = 0.0;
    float *d_matA, *d_matB, *d_matC, *d_matD;

    cudaEventRecord(start);
    cudaMalloc((void**)&d_matA, matA_size);
    cudaMalloc((void**)&d_matB, matB_size);
    cudaMalloc((void**)&d_matC, matC_size);
    cudaMalloc((void**)&d_matD, matC_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sec, start, stop);
    printf(">> GPU memory allocation time: %.3f\n", sec);

    cudaEventRecord(start);
    cudaMemcpy(d_matA, h_matA, matA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, matB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matC, h_matC, matC_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sec, start, stop);
    printf(">> Host to GPU transfer time: %.3f\n", sec);

    sec = run_sgemm_naive(d_matA, d_matB, d_matC, d_matD, M, N, K, 0.7, 0.3);
    printf(">> Naive kernel execution time: %.3f\n", sec);
    cudaMemcpy(h_matD, d_matD, matC_size, cudaMemcpyDeviceToHost);

    sec = run_sgemm_coalesced(d_matA, d_matB, d_matC, d_matD, M, N, K, 0.7, 0.3);
    printf(">> Coalesced kernel execution time: %.3f\n", sec);
    cudaMemcpy(h_matD, d_matD, matC_size, cudaMemcpyDeviceToHost);

    sec = run_sgemm_shared_memory(d_matA, d_matB, d_matC, d_matD, M, N, K, 0.7, 0.3);
    printf(">> Shared memory kernel execution time: %.3f\n", sec);
    cudaMemcpy(h_matD, d_matD, matC_size, cudaMemcpyDeviceToHost);

    // cudaEventRecord(start);
    // cudaMemcpy(h_matD, d_matD, matC_size, cudaMemcpyDeviceToHost);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&sec, start, stop);
    // printf(">> GPU to host transfer time: %.3f\n", sec);

    // for(int i = 0; i < M * N; i++)
    //     printf("%f\n", h_matA[i]);
    // printf("----------------\n");
    // for(int i = 0; i < N * K; i++)
    //     printf("%f\n", h_matB[i]);
    // printf("----------------\n");
    // for(int i = 0; i < M * K; i++)
    //     printf("%f\n", h_matD[i]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_matA);
    free(h_matB);
    free(h_matC);
    free(h_matD);

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
    cudaFree(d_matD);

return 0;
}