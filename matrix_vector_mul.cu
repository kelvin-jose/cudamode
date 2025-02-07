#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void matrix_vector_mult(int *matrix, int *vector, int *output, size_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float sum = 0.0;
        for(int j = 0; j < N; j++)
            sum += matrix[i * N + j] * vector[j];
        output[i] = sum;
    }
}

int main() {
    srand(time(NULL));
    int N = 128;
    int *h_matrix, *h_vector, *h_output, *d_matrix, *d_vector, *d_output;
    size_t matrix_size = sizeof(int) * N * N;
    size_t vector_size = sizeof(int) * N;

    h_matrix = (int*)malloc(matrix_size);
    h_vector = (int*)malloc(vector_size);
    h_output = (int*)malloc(vector_size);

    for(int i = 0; i < N * N; i++) 
        h_matrix[i] = rand() % 10;
    
    for(int i = 0;i < N; i++)
        h_vector[i] = rand() % 10;

    cudaMalloc((void**)&d_matrix, matrix_size);
    cudaMalloc((void**)&d_vector, vector_size);
    cudaMalloc((void**)&d_output, vector_size);

    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, vector_size, cudaMemcpyHostToDevice);

    dim3 block_size(32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x);

    matrix_vector_mult<<<grid_size, block_size>>>(d_matrix, d_vector, d_output, N);

    printf("%d", h_matrix[10]);
    free(h_matrix);
    free(h_vector);
    cudaFree(d_matrix);
    cudaFree(d_vector);
    return 0;
}