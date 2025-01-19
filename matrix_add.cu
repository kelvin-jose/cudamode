#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

int main() {
    srand(time(NULL));
    int M = 128, N = 128;
    int size = M * N;

    int *h_matA = (int*)malloc(size * sizeof(int));
    int *h_matB = (int*)malloc(size * sizeof(int));
    int *h_matC = (int*)malloc(size * sizeof(int));

    for(int i = 0;i < M * N; i++) {
        h_matA[i] = rand() % 100;
        h_matB[i] = rand() % 100;
    }

    int *d_matA, *d_matB, *d_matC;

    cudaMalloc((void**)&d_matA, size * sizeof(int));
    cudaMalloc((void**)&d_matB, size * sizeof(int));
    cudaMalloc((void**)&d_matC, size * sizeof(int));

    cudaMemcpy(d_matA, h_matA, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_mat, size * sizeof(int), cudaMemcpyHostToDevice);

    printf("%d %d", h_matA[0], h_matB[0]);
    free(h_matA);
    free(h_matB);

    return 0;
}