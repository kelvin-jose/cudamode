#ifndef UTILS
#define UTILS

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce(float sum) {
    for(int i = WARP_SIZE / 2; i > 0; i /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, i);
    return sum;
}

#endif