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

// Function to create actual memory for the arangement of tensors. 

Storage* create_storage(int size) {
    assert(size >=0);
    // first malloc a storage object.

    Storage* s = mallocCheck(sizeof(Storage));
    float* t = mallocCheck(size * sizeof(float));

    // If both succeeded set the data in s to t. 
    s->data = t;
    s->ref_count = 1;
    s->size = size;
    return s; 
}

// Function to index into the array of floats and set an element

void set_storage(Storage* s, float item, int idx) {
    assert(idx >= 0);
    s->data[idx] = item;
}

// Function to index into the array of floats and get an element

float get_storage(Storage* s, int idx) {
    assert(idx >= 0);
    return s->data[idx];
}

// Increment reference count

void storage_incref(Storage* s) {
    s->ref_count++;
}

// Decrement reference count
// If the ref count is 0, no views on storage. Free the memory for the array and the Storage

void storage_decref(Storage* s) {
    s->ref_count--;
    if(s->ref_count == 0) {
        free(s->data);
        free(s);
    }
}

// Create actual tensors. Torch has many ways of doing this. Let's implement a bunch.

int logical_to_physical(Tensor* t, int* idx) {
    int result = t->offset;
    for(int i = 0; i < t->ndims; i++) {
        result += idx[i] * t->strides[i];
    }
    return result;   
}


float tensor_getitem(Tensor* t, int* idx, int idx_size) {
    assert(idx_size == t->ndims);
    // TODO add negative index suport
    int pidx = logical_to_physical(t, idx);
    return t->storage->data[pidx];
}


void tensor_setitem(Tensor* t, int* idx, int idx_size, float item) {
    // we need to pass the index size to ensure dimensionalities are the same. 

}
// torch.empty(size);

// Helper function to compute the dimensionality of the tensor. 

Tensor* tensor_empty(int size) {
    // Assert is already done in creating the memory.
    // Create underlying storage. 
    Storage* s = create_storage(size);
    Tensor* t = mallocCheck(sizeof(Tensor));

    // dimensionality
    t->dims = mallocCheck(sizeof(int)); 
    t->dims[0] = size;
    t->ndims = 1;
    t->offset = mallocCheck(sizeof(int));
    t->offset = 0;

    // gradient related
    t->grad = NULL;
    t->grad_fn = NULL;
    t->require_grad = false;
    t->is_leaf = false;

    // other
    t->repr = NULL;
    
}

// torch.arrange initially always creates a 1d tensor. 
// If we want to change dimensionality we need to reshape the tensor. 
Tensor* tensor_arrange(int size) {
    Tensor* t = tensor_empty(size);
    for(int i = 1; i <= size; i++) {

    }
}