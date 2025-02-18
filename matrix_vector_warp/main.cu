#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

void random_init(float *array, int M, int N) {
    for(int i = 0; i < M * N; i++)
        array[i] = (float)rand() / RAND_MAX;
}

void main() {
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

    

}