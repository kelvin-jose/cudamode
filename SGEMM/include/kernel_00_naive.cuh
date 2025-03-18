#ifndef KERNEL_00_NAIVE
#define KERNEL_00_NAIVE

#include<cuda_runtime.h>

// MxN @ NxK + NxK
__global__ void sgemm_naive(float *matA, 
                            float *matB, 
                            float *matC, 
                            float *matD,
                            int M, 
                            int N, 
                            int K,
                            float alpha,
                            float beta) {
                        
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0;
        for(int i = 0; i < N; i++)
            sum += matA[row * N + i] * matB[i * K + col];
        
        matD[row * K + col] = alpha * sum + beta * matC[row * K + col];
    }
}

float run_sgemm_naive(float *matA, 
                      float *matB, 
                      float *matC, 
                      float *matD,
                      int M, 
                      int N, 
                      int K,
                      float alpha = 1.0,
                      float beta = 0.0) {
    dim3 block_size(32, 32);
    dim3 grid_size((K + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);
    cudaEvent_t start, stop;
    float sec = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sgemm_naive<<<grid_size, block_size>>>(matA, matB, matC, matD, M, N, K, 1, 0);
    cudaEventRecord(stop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sec, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

return sec;
}

#endif