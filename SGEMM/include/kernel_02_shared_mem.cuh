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
        

    }
                         
    }

#endif