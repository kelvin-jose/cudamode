#include<cuda_runtime.h>
#include "utils.cuh"

__global__ void matrix_vector_mult(float *matrix, float *vector, float *result, int M, int N) {
    int block = blockIdx.x;
    if (block >= M)
        return;
    
    int thread = threadIdx.x;

    float sum = 0.0;

    for(int i = thread; i < N; i += blockDim.x) {
        sum += matrix[block * N + i] * vector[i];
    }

    sum = warp_reduce(sum);
    if (block == 0)
        result[block] = sum;
}

float run_matrix_vector_mult(float *matrix, float *vector, float *result, int M, int N) {

    dim3 block_size(WARP_SIZE);
    dim3 grid_size(M);

    cudaEvent_t start, stop;
    float sec = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_vector_mult<<<grid_size, block_size>>>(matrix, vector, result, M, N);
    cudaEventRecord(stop);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sec, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

return sec;
}