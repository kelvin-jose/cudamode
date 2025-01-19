#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

int main() {
    srand(time(NULL));
    int M = 128, N = 128;

    int *h_matA = (int*)malloc(M * N * sizeof(int));
    int *h_matB = (int*)malloc(M * N * sizeof(int));

    for(int i = 0;i < M * N; i++) {
        h_matA[i] = rand() % 100;
        h_matB[i] = rand() % 100;
    }

    printf("%d %d", h_matA[0], h_matB[0]);
    free(h_matA);
    free(h_matB);
    
    return 0;
}