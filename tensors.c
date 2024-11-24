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

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

// Utility functions, taken also from Karpathy

int ceil_div(int a, int b) {
    // integer division that rounds up, i.e. ceil(a / b)
    return (a + b - 1) / b;
}

int min(int a, int b) {
    return (a < b) ? a : b;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

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

int logical_to_physical(Tensor* t, int* idx, int idx_size) {
    assert(t->ndims == idx_size);
    int result = t->offset;

    for(int i = 0; i < t->ndims; i++) {
        if(idx[i] < 0 && idx[i] + t->dims[i] >= 0) {
            idx[i] = idx[i] + t->dims[i];
        }

        if(idx[i] > t->dims[i] || idx[i] < 0) {
            fprintf(stderr, "IndexError: index %d is out of bounds of %d\n", idx[i], t->dims[i]); 
            return NAN;
            }

        result += idx[i] * t->strides[i];
    }

    return result;   
}

float tensor_getitem(Tensor* t, int* idx, int idx_size) {
    assert(idx_size == t->ndims); 
    int pidx = logical_to_physical(t, idx, idx_size);
    return get_storage(t->storage, pidx);
}

void tensor_setitem(Tensor* t, int* idx, int idx_size, float item) {
    assert(idx_size == t->ndims);
    int pidx = logical_to_physical(t, idx, idx_size);
    set_storage(t->storage, item, pidx);
}

// torch.empty(size);

Tensor* tensor_empty(int ndims, int* dims) {
    // Assert is already done in creating the memory.
    // Create underlying storage.
    int mem_size = 1;
    for(int i = 0; i < ndims; i++) {
        mem_size *= dims[i];
    }

    Storage* s = create_storage(mem_size);
    Tensor* t = mallocCheck(sizeof(Tensor));

    t->storage = s;
    // dimensionality

    t->dims = mallocCheck(sizeof(int) * ndims); 
    memcpy(t->dims, dims, ndims * sizeof(int)); // 3 x 3 x 4 for example.
    t->ndims = ndims;
    t->offset = 0;
    t->strides = mallocCheck(sizeof(int) * ndims);
    memcpy(t->strides, compute_strides(t->dims, ndims), ndims * sizeof(int));

    // gradient related
    t->grad = NULL;
    t->grad_fn = NULL;
    t->require_grad = false;
    t->is_leaf = true;

    // other
    t->repr = NULL;
    return t; 
}

Tensor* tensor_empty_1d(int size) {
    int dims[] = {size};
    return tensor_empty(1, dims);
}

// torch.arrange()

Tensor* tensor_arrange(int size) {
    Tensor* t = tensor_empty_1d(size);
    for(int i = 0; i < size; i++) {
        tensor_setitem(t, &i, 1, (float) i);
    }
    t->repr = NULL;
    return t;
}

// Create a tensor with multidimensional initialization 
int* compute_strides(int* dims, int ndims) {
    int* strides = mallocCheck(ndims * sizeof(int));
    strides[ndims - 1] = 1;

    for(int i = ndims-2; i >= 0; i--) {
        strides[i] = dims[i+1] * strides[i+1];
    }
    return strides;
}

Tensor* tensor_arrange_multidimensional(int* dims, int ndims) {
    int size = 1;
    for (int i = 0; i<ndims; i++) {
        size*=dims[i];
    }

    Storage* storage = create_storage(size);
    Tensor* t = mallocCheck(sizeof(Tensor));

    // Initialize storage with sequential values
    for (int i = 0; i < size; i++) {
        set_storage(storage, (float)i, i);
    }
    t->storage = storage;    
    int* dimension = mallocCheck(ndims * sizeof(int));
    // copy the dims into this dimensions
    memcpy(dimension, dims, ndims * sizeof(int));
    t->dims = dimension;
    int* str = compute_strides(t->dims, ndims);
    int* strides = mallocCheck(ndims * sizeof(int));
    memcpy(strides, str, ndims * sizeof(int));
    t->strides = strides;
    t->ndims = ndims;
    t->offset = 0;
    // gradient related
    t->grad = NULL;
    t->grad_fn = NULL;
    t->require_grad = false;
    t->is_leaf = true;

    // other
    t->repr = NULL;

    return t;

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
    new_t -> repr = NULL;

    return new_t;
}

// we need a function to free the tensor
// Note that we create a new tensor first, and only after that we perform chekcs on the dims
// This is a tradeof for the complex case of multidimensional tensors

Tensor* tensor_slice(Tensor* t, int* start, int* end, int* step) {
    Tensor* new_t = mallocCheck(sizeof(Tensor));
    new_t -> storage = t -> storage;
    new_t -> offset = t -> offset + start[0] * t -> strides[0];
    new_t -> ndims = t -> ndims;
    new_t->dims = mallocCheck(t->ndims * sizeof(int));
    new_t->strides = mallocCheck(t->ndims * sizeof(int));
    for(int i = 0; i < t->ndims; i++) {
        if(start[i] < 0) {start[i] = start[i] + t->dims[i];}
        if(end[i] < 0) {end[i] = end[i] + t->dims[i];}
        if(step[i] < 0) {
            fprintf(stderr, "Steps cannot be negative");
            free(new_t);
            return tensor_empty_1d(0);
            }
        if(step[i] == 0) {
            fprintf(stderr, "Step cannot be 0");
            free(new_t);
            return tensor_empty_1d(0);
        }
        start[i] = min(max(start[i], 0), t->dims[i]);
        end[i] = min(max(end[i], 0), t->dims[i]);
        new_t -> dims[i] = ceil_div(end[i] - start[i], step[i]);
        new_t -> strides[i] = t->strides[i] * step[i]; 
    }
    storage_incref(t->storage);
    new_t -> repr = NULL;
    new_t -> require_grad = t -> require_grad;
    new_t -> grad_fn = NULL;
    new_t -> is_leaf = NULL;
    new_t -> grad = NULL;
    return new_t;
}

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
    if (t->repr != NULL) free(t->repr);
    if (t->strides != NULL) free(t->strides);
    if (t->grad_fn != NULL) free(t->grad_fn);

    // Finally free the tensor struct itself
    free(t);
}

// Tensor operations

Tensor* tensor_addf(Tensor* t, float f) {
    Tensor* new_t = tensor_arrange_multidimensional(t->storage->size);
    for(int i = 0; i < t->storage->size; i++) {
        float old_val = get_storage(t->storage, i);
        float new_val = old_val + f;
    }
    
    return new_t;
}

// recall the interface for tensor_setitem: void tensor_setitem(Tensor* t, int* idx, int idx_size, float item)
Tensor* add_tensors(Tensor* t_1, Tensor* t_2) {
    if (t_1->ndims != t_2->ndims) {
        fprintf(stderr, "The tensors must have the same number of dimensions!");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < t_1->ndims; i++) {
        if(t_1->dims[i] != t_2->dims[i]) {
            fprintf(stderr, "The tensors must have the same dimensionality!");
            exit(EXIT_FAILURE);
        } 
    }
    Tensor* new_t = tensor_arrange_multidimensional(t_1->dims, t_1->ndims);
    for(int i = 0; i<new_t->storage->size; i++) {
        set_storage(new_t->storage, t_1->storage->data[i] + t_2->storage->data[i], i);
    }
    // Use tensor getitem en tensor set item
    return new_t;
}

// ***** IF YOU SET REPR USE A MALLOC BECAUSE STRING LITERALS GO TO THE 
// ***** DATA SECTION IN MEMORY AND CANNOT GET FREED!

int main(int argc, char *argv[]) {
    // if (argc != 2) {
    //     printf("Usage: %s <size>\n", argv[0]);
    //     return 1;
    // }
    // int size = atoi(argv[1]);
    // Tensor* t = tensor_arrange(size);
    // printf("Hello from inside the tensor %s\n", t->repr);
    // for(int i = 0; i<t->dims[0]; i++) {
    //     printf("%d ", (int) tensor_getitem(t, &i, 1));
    // }
    // printf("Reshaping the tensor\n");
    // int* dims = mallocCheck(2 * sizeof(int));
    // dims[0] = 3;
    // dims[1] = 3;
    // Tensor* t_new = reshape(t, dims, 2);
    // for(int i = 0; i<t_new->dims[0]; i++) {
    //    for(int j =0; j<t_new->dims[1]; j++) {
    //     int idx[] = {i, j};
    //     printf("%d ", (int) tensor_getitem(t_new, idx, 2));
    //     }
    //     printf("\n");
    // }
    // int idx[] = {2, 2};
    // printf("%d", (int) tensor_getitem(t_new, idx, 2));

    // free_tensor(t);
    // free_tensor(t_new);
    int dims[] = {3, 3};
    int ndims = 2;
    int idx[] = {-1, -1};
    Tensor* t = tensor_arrange_multidimensional(dims, ndims);
    printf("%f\n", tensor_getitem(t, idx, 2));
    // Tensor* t_2 = tensor_arrange_multidimensional(dims, ndims);
    // Tensor* new = add_tensors(t, t_2);
    // for(int i = 0; i < new->dims[0]; i++) {
    //     for(int j = 0; j < new->dims[1]; j++) {
    //         int idx[] = {i, j};
    //         printf("%d ", (int)tensor_getitem(new, idx, 2));
    //     }
    //     printf("\n");
    // }
    // free_tensor(t);
    // free_tensor(t_2);
    // free_tensor(new);
}