#ifndef HEADER_H   // If HEADER_H is not defined
#define HEADER_H   // Define it

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
    int* strides; // array of strides for each dim
    int* dims;
    int ndims;
    char* repr;
    bool require_grad;
    bool is_leaf;
    Function* grad_fn;
} Tensor;

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

#endif // HEADER_H