#ifndef KERNEL_02_SHAREDMEM
#define KERNEL_02_SHAREDMEM
#define WRAPSIZE 32
#include<cuda_runtime.h>

// MxN @ NxK + NxK
__global__ void shared_memory(float *matA, 
                            float *matB, 
                            float *matC, 
                            float *matD,
                            int M, 
                            int N, 
                            int K,
                            float alpha,
                            float beta) {

    __shared__ float sharedA[WRAPSIZE * WRAPSIZE];
    __shared__ float sharedB[WRAPSIZE * WRAPSIZE];

    int col = blockIdx.x;
    int row = blockIdx.y;
    int trow = threadIdx.x / WRAPSIZE;
    int tcol = threadIdx.x % WRAPSIZE;

    matA += col * WRAPSIZE * N;
    matB += row * WRAPSIZE;
    matC += col * WRAPSIZE * K + row * WRAPSIZE;

    float sum = 0.0;
    for(int i = 0; i < N; i += WRAPSIZE) {
        sharedA[trow * WRAPSIZE + tcol] = matA[trow * N + tcol];
        sharedB[trow * WRAPSIZE + tcol] = matB[trow * K + tcol];

        __syncthreads();

        matA += WRAPSIZE;
        matB += WRAPSIZE * K;

        for(int j = 0; j < WRAPSIZE; j++) 
            sum += sharedA[trow * WRAPSIZE + j] * sharedB[j * WRAPSIZE + tcol];
        
        __syncthreads();

    }
        
    matC[trow * K + tcol] = alpha * sum + beta * matC[trow * K + tcol];

}

float run_sgemm_shared_memory(float *matA, 
    float *matB, 
    float *matC, 
    float *matD,
    int M, 
    int N, 
    int K,
    float alpha = 1.0,
    float beta = 0.0) {
dim3 block_size(32 * 32);
dim3 grid_size((K + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);
cudaEvent_t start, stop;
float sec = 0.0;

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
shared_memory<<<grid_size, block_size>>>(matA, matB, matC, matD, M, N, K, 1, 0);
cudaEventRecord(stop);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&sec, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);

return sec;
}

#endif