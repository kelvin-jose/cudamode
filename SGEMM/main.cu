#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

void random_init(float *array, int M, int N) {
    for(int i = 0; i < M * N; i++)
        array[i] = (float)rand() / RAND_MAX;
}

int main() {
    srand(time(NULL));

    int M = 2, N = 2, K = 2;

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
    float *d_matA, *d_matB, *d_matC;

    cudaEventRecord(start);
    cudaMalloc((void**)&d_matA, matA_size);
    cudaMalloc((void**)&d_matB, matB_size);
    cudaMalloc((void**)&d_matC, matC_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sec, start, stop);
    printf(">> GPU memory allocation time: %f\n", sec);

    cudaEventRecord(start);
    cudaMemcpy(d_matA, h_matA, matA_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, matB_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matC, h_matC, matC_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sec, start, stop);
    printf(">> Host to GPU transfer time: %f\n", sec);

    // execute kernel here
    printf(">> Kernel execution time: %f\n", sec);

    cudaEventRecord(start);
    cudaMemcpy(h_matC, d_matC, matC_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sec, start, stop);
    printf(">> GPU to host transfer time: %f\n", sec);

    for(int i = 0; i < M * K; i++)
        printf("%f\n", h_matD[i]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_matA);
    free(h_matB);
    free(h_matC);
    free(h_matD);

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);

return 0;
}