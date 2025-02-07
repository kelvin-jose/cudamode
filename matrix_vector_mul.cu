#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

int main() {
    srand(time(NULL));
    int N = 128;
    int *h_matrix, *h_vector, *d_matrix, *d_vector;
    size_t matrix_size = sizeof(int) * N * N;
    size_t vector_size = sizeof(int) * N;

    h_matrix = (int*)malloc(matrix_size);
    h_vector = (int*)malloc(vector_size);

    for(int i = 0; i < N * N; i++) 
        h_matrix[i] = rand() % 10;
    
    for(int i = 0;i < N; i++)
        h_vector[i] = rand() % 10;

    cudaMalloc((void**)&d_matrix, matrix_size);
    cudaMalloc((void**)&d_vector, vector_size);

    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, vector_size, cudaMemcpyHostToDevice);


    printf("%d", h_matrix[10]);
    free(h_matrix);
    free(h_vector);
    cudaFree(d_matrix);
    cudaFree(d_vector);
    return 0;
}