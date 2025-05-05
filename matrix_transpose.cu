#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

void pretty_print(float *A, int M, int N) {
    for(int i = 0; i < M; i++) {
        printf("\n");
        for(int j = 0; j < N; j++)
            printf("%f ", A[i * M + j]);
    }
}

__global__ void transpose(float *A, float *B, int M, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < M && y < N) 
        B[x * M + y] = A[y * N + x];
}

int main() {

    const int M = 4096, N = 4096;
    size_t size = M * N * sizeof(float);

    float *h_A, *h_B, *d_A, *d_B;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    
    for(int i = 0; i < M * N; i++)
        h_A[i] = (float)rand() / RAND_MAX;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 block_size(32, 32);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (M + block_size.y - 1) / block_size.y);

    transpose<<<grid_size, block_size>>>(d_A, d_B, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    pretty_print(h_A, M, N);
    pretty_print(h_B, M, N);
    return 0;
}