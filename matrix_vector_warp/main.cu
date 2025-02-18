#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<warp_op.cuh>

void random_init(float *array, int M, int N) {
    for(int i = 0; i < M * N; i++)
        array[i] = (float)rand() / RAND_MAX;
}

int main() {
    srand(time(NULL));

    int M = 2, N = 2;

    float *h_matrix, *h_vector, *h_result;
    
    int matrix_size = M * N * sizeof(float);
    int vector_size = N * sizeof(float);

    h_matrix = (float*)malloc(matrix_size);
    h_vector = (float*)malloc(vector_size);
    h_result = (float*)malloc(vector_size);

    random_init(h_matrix, M, N);
    random_init(h_vector, N, 1);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float sec = 0.0;
    float *d_matrix, *d_vector, *d_result;

    cudaEventRecord(start);
    cudaMalloc((void**)&d_matrix, matrix_size);
    cudaMalloc((void**)&d_vector, vector_size);
    cudaMalloc((void**)&d_result, vector_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sec, start, stop);
    printf(">> GPU memory allocation time: %f\n", sec);

    cudaEventRecord(start);
    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, vector_size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sec, start, stop);
    printf(">> Host to GPU transfer time: %f\n", sec);

    sec = run_matrix_vector_mult(d_matrix, d_vector, d_result, M, N);
    printf(">> Kernel execution time: %f\n", sec);

    cudaEventRecord(start);
    cudaMemcpy(h_result, d_result, vector_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sec, start, stop);
    printf(">> GPU to host transfer time: %f\n", sec);

    for(int i = 0; i < M*N; i++)
        printf("%f\n", h_matrix[i]);
    printf("----------------\n");
    for(int i = 0; i < N; i++)
        printf("%f\n", h_vector[i]);
    printf("----------------\n");
    for(int i = 0; i < N; i++)
        printf("%f\n", h_result[i]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(h_matrix);
    free(h_vector);
    free(h_result);

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

return 0;
}