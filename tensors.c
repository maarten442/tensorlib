#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include "tensors.h"

// -----------------------------------------------------------------------------------
// memory allocation, following @karpathy

void* malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__);

// -----------------------------------------------------------------------------------
// Storage takes care of the contiguous block of memory. Tensors have views on this memory.

Storage* storage_new(int size) {
    assert(size >= 0);
    Storage* storage = mallocCheck(sizeof(Storage));
    storage->data = mallocCheck(size *sizeof(float));
    storage->size = size;
    storage->ref_count = 1; // upon creation there is one reference.
    return storage;
}

void set_storage(int idx, float value, Storage* storage) {
    assert(idx > 0 && idx <= storage->size);
    storage->data[idx] = value;
}

float get_storage(int idx, Storage* storage) {
    assert(idx > 0 && idx <= storage->size);
    return storage->data[idx];
}

void incr_reference(Storage* storage) {
    storage->ref_count++;
}

void decr_reference(Storage* storage) {
    storage->ref_count--;
    if (storage->ref_count == 0) {
        free(storage->data);
        free(storage);
    }
}

// Tensor & tensor operations, views on the contiguous storage.