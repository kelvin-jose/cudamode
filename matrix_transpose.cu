#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

int main() {

    const int M = 4096, N = 4096;
    size_t size = M * N * sizeof(float);

    float *h_A, *h_B, *d_A, *d_B;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    
    return 0;
}