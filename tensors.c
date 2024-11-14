#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

// Create a wrapper around malloc for save memory allocation, as per Karpathy. 
// We use fprintf to write to the standard error instead of the standard output. 
// Program needs to terminate immediately upon failed malloc, so we use exit. 

void* malloc_check(int size, char* file, int line) {
    void* ptr = malloc(size);
    if (ptr ==0) {
        fprintf(stderr, "Memory allocation failed at %s:%d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

// define macro mallocCheck

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// Storage physically stores the values in contiguous memory. Can be referenced by multiple tensors. 

typedef struct {
    float* data;
    int size;
    int ref_count;
} Storage;

typedef struct {
    Tensor** inputs;
    Tensor* output;
    int num_inputs;
    void (*backward)(struct Function* self, Tensor* output);
    // Something to destroy resources?
} Function;

typedef struct {
    Storage* storage;
    int offset;
    int stride;
    int size;
    char* repr;
} GradientTensor;

typedef struct {
    Storage* storage;
    GradientTensor* grad;
    int offset;
    int stride;
    int size;
    char* repr;
    bool require_grad;
    bool is_leaf;
    Function* grad_fn;
} Tensor;

// Function to create actual memory for the arangement of tensors. 

Storage* create_storage(int size) {
    assert(size >=0);
    // first malloc a storage object.

    Storage* s = mallocCheck(sizeof(Storage));
    float* t = mallocCheck(size * sizeof(float));

    // If both succeeded set the data in s to t. 
    s->data = t;
    return s; 
}