#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

void random_init(float *array, size_t size) {
    for(int i = 0; i < size; i++)
        array[i] = (float)rand() / RAND_MAX;
}

int main() {

    const int M = 4096, N = 4096;
    size_t size = M * N * sizeof(float);

    float *h_A, *h_B, *d_A, *d_B;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    
    for(int i = 0; i < size; i++)
        h_A[i] = (float)rand() / RAND_MAX;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    

    return 0;
}