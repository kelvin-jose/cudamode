#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void matrix_add(int *matA, int *matB, int *matC, const int rows, const int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        matC[idx] = matA[idx] + matB[idx];
    }
    
}

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
    cudaMemcpy(d_matB, h_matB, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    matrix_add<<<gridSize, blockSize>>>(d_matA, d_matB, d_matC, M, N);
    
    cudaMemcpy(h_matC, d_matC, size * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < size; i++) {
        printf("%d + %d = %d\n", h_matA[i], h_matB[i], h_matC[i]);
    }
   
    free(h_matA);
    free(h_matB);

    return 0;
}