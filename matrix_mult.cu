#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

void random_init(float *matrix, int M, int N) {
    for(int i = 0; i < M * N; i++)
        matrix[i] = rand() % 10;
}

int main() {
    float *h_matA, *h_matB, *h_matC, *d_matA, *d_matB, *d_matC;
    int M = 2, N = 2, K = 2;
    int matAsize = sizeof(float) * M * K;
    int matBsize = sizeof(float) * K * N;
    int matCsize = sizeof(float) * M * N;

    h_matA = (float*)malloc(matAsize);
    h_matB = (float*)malloc(matBsize);
    h_matC = (float*)malloc(matCsize);

    random_init(h_matA, M, K);
    random_init(h_matB, K, N);
    for(int i = 0; i < M * K; i++) 
        printf("\n%f", h_matA[i]);

    return 0;
}