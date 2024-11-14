#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

// Create a wrapper around malloc for save memory allocation, as per Karpathy. 

void* malloc_check(int size, char* file, int line) {
    void* ptr = malloc(size);
    if (ptr ==0) {
        fprintf(stderr, "Memory allocation failed at %s:%d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}