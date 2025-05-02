#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

#define EPSILON 1e-5

__global__ void layernorm(float *A, float *B, float *gamma, float *beta, const int batch_size, const int dims) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shared[];
    float *mean_var = shared;
    float *row_data = &shared[2];

    if (tid == 0) {
        mean_var[0] = 0.0f;
        mean_var[1] = 0.0f;
    }
    __syncthreads();

    if (tid < dims) 
    row_data[tid] = A[row * dims + tid];
    
    float sum = 0.0f;
    for(int i = 0; i < dims; i += blockDim.x)
        sum += row_data[i];

    atomicAdd(&mean_var[0], sum);
    __syncthreads();

    float mean = mean_var[0] / dims;

    float var_sum = 0.0f;
    for(int i = 0; i < dims; i += blockDim.x)
        var_sum += pow(row_data[i] - mean, 2);

    atomicAdd(&mean_var[1], var_sum);
    __syncthreads();

    float variance = mean_var[1] / dims;
    float std = sqrt(variance + EPSILON);

    if (tid < dims) {
        float norm = (row_data[tid] - mean) / std;
        if (gamma && beta)
            norm = norm * gamma[tid] + beta[tid];
        B[row * dims + tid] = norm;

    }
}

void random_init(float *array, size_t size) {
    for(int i = 0; i < size; i++)
        array[i] = (float)rand() / RAND_MAX;
}

int main() {

    const int batch_size = 8, dims = 4096;
    float *h_A, *h_B, *h_gamma, *h_beta;
    float *d_A, *d_B, *d_gamma, *d_beta;
    size_t input_size = batch_size * dims * sizeof(float);
    size_t norm_params_size = dims * sizeof(float);

    h_A = (float*)malloc(input_size);
    h_B = (float*)malloc(input_size);
    h_gamma = (float*)malloc(norm_params_size);
    h_beta = (float*)malloc(norm_params_size);

    random_init(h_A, batch_size * dims);
    random_init(h_gamma, dims);
    random_init(h_beta, dims);

    cudaMalloc((void**)&d_A, input_size);
    cudaMalloc((void**)&d_B, input_size);
    cudaMalloc((void**)&d_gamma, norm_params_size);
    cudaMalloc((void**)&d_beta, norm_params_size);

    cudaMemcpy(d_A, h_A, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, norm_params_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, norm_params_size, cudaMemcpyHostToDevice);

    const int block_size = 1024;
    size_t shared_mem = sizeof(float) * (2 + dims);

    layernorm<<<batch_size,block_size, shared_mem>>>(d_A, d_B, d_gamma, d_beta, batch_size, dims);
    cudaMemcpy(h_B, d_B, input_size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < dims; j++)
            printf("%.4f ", h_B[i * dims + j]);
        printf("\n");
    }

    free(h_A);
    free(h_B);
    free(h_gamma);
    free(h_beta);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    
    return 0;
}

