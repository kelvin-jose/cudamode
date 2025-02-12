#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

int main() {
    float *h_matA, *h_matB, *h_matC, *d_matA, *d_matB, *d_matC;
    int M = 2, N = 2, K = 2;
    int matAsize = sizeof(float) * M * K;
    int matBsize = sizeof(float) * K * N;
    int matCsize = sizeof(float) * M * N;

    h_matA = (float*)malloc(matAsize);
    h_matB = (float*)malloc(matBsize);
    h_matC = (float*)malloc(matCsize);
    

    return 0;
}