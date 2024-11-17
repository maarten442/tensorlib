# tensorlib
A minimal autograd library, implemented in c and wrapped in Python. 

# TODO to get the base functionality right:

The add function currently adds the memory from tensors together, which is not good if you have different views on memory (the actual implementation of the tensor object).

- Add broadcasting of tensors
- Fix the add function with using set_tensor instead of the full underlying memory structure

If this has been fixed we can implement the gradient function for tensor addition and write the python wrapper and build a simple computation graph to check if it works. 