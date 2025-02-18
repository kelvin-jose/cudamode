#include<cuda_runtime.h>

#define WARP_SIZE 32

__global__ void matrix_vector_mult(float *matrix, float *vector, float *result, int M, int N) {
    
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