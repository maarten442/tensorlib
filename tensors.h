#ifndef TENSORS_H
#define TENSORS_H

typedef struct {
    float* data;
    int size;
    int ref_count;
} Storage;

typedef struct Function Function;  // Forward declaration

typedef struct {
    Storage* storage;
    int offset;
    int* strides;
    int* dims;
    int ndims;
    char* repr;
} GradientTensor;

typedef struct {
    Storage* storage;
    GradientTensor* grad;
    int offset;
    int* strides;
    int* dims;
    int ndims;
    char* repr;
    bool require_grad;
    bool is_leaf;
    Function* grad_fn;
} Tensor;

struct Function {
    Tensor** inputs;
    Tensor* output;
    int num_inputs;
    void (*backward)(struct Function* self, Tensor* output);
};

#endif // TENSORS_H