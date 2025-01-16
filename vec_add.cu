#include<stdio.h>
#include<cuda_runtime.h>

__global__ void add_vectors(const float *a, const float *b, float *c, const int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

int main() {
    
    // addition of two vectors in pure c
    int N = 100;
    float *a, *b, *c, *d;

    float *d_a, *d_b, *d_c;

    // allocate memory
    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    c = (float *)malloc(N * sizeof(float));
    d = (float *)malloc(N * sizeof(float));

    for(int i = 0; i<N; i++) {
        a[i] = 1.3 * i;
        b[i] = 2.5 * i;
        c[i] = a[i] + b[i];
    }

    for(int i=0; i<N; i++) 
        printf("%f\n", c[i]);
    
    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_c, N * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    add_vectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(d, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (c[i] == d[i])
            printf("%d\n", true);
        else
            printf("%d\n", false);
    }
    return 0;
}