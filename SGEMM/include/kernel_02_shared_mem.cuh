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

    

    }

#endif