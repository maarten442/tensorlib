#ifndef TENSORS_H
#define TENSORS_H

#include <stddef.h>
#include <stdbool.h>

typedef struct {
    float* data;
    int size;
    int ref_count;
} Storage;

typedef struct {
    Tensor** inputs;
    int num_inputs;
    Tensor* output; // can you output more tensors?
    int num_outputs;
    void (*backward)(struct Node*);
};

typedef struct {
    Storage* storage;
    float* grad; // question: does this need more information, or can it just hold the float pointer and reuse the strides?
    int offset;
    int ndims;
    int* strides;
    int* dimensions;
    bool require_grad;
    bool is_leaf;
    char* repr;
} Tensor;

#endif