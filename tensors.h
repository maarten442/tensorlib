#ifndef TENSORS_H
#define TENSORS_H

typedef struct {
    float* data;
    int size;
    int ref_count;
} Storage;

#endif