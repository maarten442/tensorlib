#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include "tensors.h"
#include <string.h>
// Create a wrapper around malloc for save memory allocation, as per Karpathy. 
// We use fprintf to write to the standard error instead of the standard output. 
// Program needs to terminate immediately upon failed malloc, so we use exit. 

// TODO: fix partial allocation failures here. 
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


// Function to create actual memory for the arangement of tensors. 

Storage* create_storage(int size) {
    assert(size >=0);
    // first malloc a storage object.

    Storage* s = mallocCheck(sizeof(Storage));
    float* d = mallocCheck(size * sizeof(float));

    // If both succeeded set the data in s to t. 
    s->data = d;
    s->ref_count = 1;
    s->size = size;
    return s; 
}

// Function to index into the array of floats and set an element

void set_storage(Storage* s, float item, int idx) {
    assert(idx >= 0 && idx < s->size);
    s->data[idx] = item;
}

// Function to index into the array of floats and get an element

float get_storage(Storage* s, int idx) {
    assert(idx >= 0 && idx < s->size);
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
    return get_storage(t->storage, pidx);
}

void tensor_setitem(Tensor* t, int* idx, int idx_size, float item) {
    assert(idx_size == t->ndims);
    int pidx = logical_to_physical(t, idx);
    set_storage(t->storage, item, pidx);
}

// torch.empty(size);

Tensor* tensor_empty(int size) {
    // Assert is already done in creating the memory.
    // Create underlying storage. 
    Storage* s = create_storage(size);
    Tensor* t = mallocCheck(sizeof(Tensor));

    t->storage = s;
    // dimensionality

    t->dims = mallocCheck(sizeof(int)); 
    t->dims[0] = size; // 3 x 3 x 4 for example.
    t->ndims = 1;
    t->offset = 0;
    t->strides = mallocCheck(sizeof(int));
    t->strides[0] = 1;

    // gradient related
    t->grad = NULL;
    t->grad_fn = NULL;
    t->require_grad = false;
    t->is_leaf = true;

    // other
    t->repr = NULL;
    return t; 
}

// torch.arrange()

Tensor* tensor_arrange(int size) {
    Tensor* t = tensor_empty(size);
    for(int i = 0; i < size; i++) {
        tensor_setitem(t, &i, 1, (float) i);
    }
    t->repr = "Set from arrange";
    return t;
}

int* compute_strides(int* dims, int ndims) {
    int* strides = mallocCheck(ndims * sizeof(int));
    strides[ndims - 1] = 1;

    for(int i = ndims-2; i >= 0; i--) {
        strides[i] = dims[i+1] * strides[i+1];
    }
    return strides;
}

Tensor* reshape(Tensor* t, int* dims, int ndims) {
    // you get an array, you need to put it on the stack 
    Tensor* new_t = mallocCheck(sizeof(Tensor));
    new_t -> storage = t->storage;
    storage_incref(t->storage);

    new_t -> dims = dims; // give a heap pointer!
    new_t -> ndims = ndims;
    new_t -> strides = compute_strides(dims, ndims);
    new_t -> require_grad = false;
    new_t -> offset = t -> offset;
    new_t -> grad_fn = NULL;
    new_t -> is_leaf = true;
    new_t->repr = "I am reshaped!";

    return new_t;
}

// we need a function to free the tensor

void free_tensor(Tensor* t) {
    // First check if tensor exists
    if (t == NULL) {
        return;
    }

    // Handle storage reference counting
    if (t->storage != NULL) {  // Defensive check
        storage_decref(t->storage);
    }

    // Free all member pointers with NULL checks
    if (t->dims != NULL) free(t->dims);
    if (t->grad != NULL) free(t->grad);
    // if (t->repr != NULL) free(t->repr);
    if (t->strides != NULL) free(t->strides);
    if (t->grad_fn != NULL) free(t->grad_fn);

    // Finally free the tensor struct itself
    free(t);
}


// ***** IF YOU SET REPR USE A MALLOC BECAUSE STRING LITERALS GO TO THE 
// ***** DATA SECTION IN MEMORY AND CANNOT GET FREED!

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <size>\n", argv[0]);
        return 1;
    }
    int size = atoi(argv[1]);
    Tensor* t = tensor_arrange(size);
    printf("Hello from inside the tensor %s\n", t->repr);
    for(int i = 0; i<t->dims[0]; i++) {
        printf("%d ", (int) tensor_getitem(t, &i, 1));
    }
    printf("Reshaping the tensor\n");
    int* dims = mallocCheck(2 * sizeof(int));
    dims[0] = 3;
    dims[1] = 3;
    Tensor* t_new = reshape(t, dims, 2);
    for(int i = 0; i<t_new->dims[0]; i++) {
       for(int j =0; j<t_new->dims[1]; j++) {
        int idx[] = {i, j};
        printf("%d ", (int) tensor_getitem(t_new, idx, 2));
        }
        printf("\n");
    }
    int idx[] = {2, 2};
    printf("%d", (int) tensor_getitem(t_new, idx, 2));

    free_tensor(t);
    free_tensor(t_new);
}