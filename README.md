# tensorlib

The goal is to create a tensor library that supports the computation of automatic gradients. 

We need to create memory, and save tensors contiguously in memory. Views hold pointers to the actual tensor object. 

The goal is to first build these general capabilities, and later add the gradient computation. 

# TODO to get the base functionality right:

The add function currently adds the memory from tensors together, which is not good if you have different views on memory (the actual implementation of the tensor object).

- Add broadcasting of tensors
- Fix the add function with using set_tensor instead of the full underlying memory structure
- Add floating point addition to tensors
- Write tensor to string
- Write the relevant tensor operations: addition, multiplication, scalar multiplication. This is important for autograd.

I asked Claude to give me a plan for this, and I for sure need to implement the following:

Before implementing autograd, you'll need to add several key components:

Essential Tensor Operations:
Element-wise operations (I've provided multiplication, you already have addition)
Matrix multiplication (matmul)
Basic activation functions (ReLU, sigmoid, tanh)
Reduction operations (sum, mean)
Power/exponentiation operations
Division operations
Gradient-Related Structures:
Extend the Tensor struct to track gradient information
Add AutogradFunction structure for backward passes
Implement backward functions for each operation
Memory Management Updates:
Enhanced reference counting for gradients
Proper cleanup of autograd functions
Memory management for saved tensors
Key Features Needed:
Ability to enable/disable gradient computation
Gradient accumulation
Detaching tensors from computation graph
Zero_grad functionality
Backward pass implementation
The code I've provided gives you a starting point for these features. To fully implement autograd, you'll need to:

# Plan for gradients in computation graph:

- TBD