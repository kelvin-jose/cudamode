#ifndef WARP_OP
#define WARP_OP

float run_matrix_vector_mult(float *matrix, float *vector, float *result, int M, int N);

__global__ void matrix_vector_mult(float *matrix, float *vector, float *result, int M, int N);

#endif