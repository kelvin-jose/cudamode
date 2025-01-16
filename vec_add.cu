#include<stdio.h>

int main() {
    
    // addition of two vectors in pure c
    int N = 100;
    float *a, *b, *c;

    // allocate memory
    a = (float *)malloc(N * sizeof(float));
    b = (float *)malloc(N * sizeof(float));
    c = (float *)malloc(N * sizeof(float));

    for(int i = 0; i<N; i++) {
        a[i] = 1.3 * i;
        b[i] = 2.5 * i;
        c[i] = a[i] + b[i];
    }

    for(int i=0; i<N; i++) 
        printf("%f\n", c[i]);
    return 0;
}