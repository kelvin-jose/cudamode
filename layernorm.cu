#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

void random_init(float *array, size_t size) {
    for(int i = 0; i < size; i++)
        array[i] = (float)rand() / RAND_MAX;
}

int main() {

    const int batch_size = 2, dims = 4;
    float *h_A, *h_B, *h_gamma, *h_beta;
    float *d_A, *d_B, *d_gamma, *d_beta;
    size_t input_size = batch_size * dims * sizeof(float);
    size_t norm_params_size = dims * sizeof(float);

    h_A = (float*)malloc(input_size);
    h_B = (float*)malloc(input_size);
    h_gamma = (float*)malloc(norm_params_size);
    h_beta = (float*)malloc(norm_params_size);

    random_init(h_A, input_size);
    random_init(h_gamma, norm_params_size);
    random_init(h_beta, norm_params_size);

    cudaMalloc(&d_A, input_size);
    cudaMalloc(&d_B, input_size);
    cudaMalloc(&d_gamma, norm_params_size);
    cudaMalloc(&d_beta, norm_params_size);
    
    return 0;
}

