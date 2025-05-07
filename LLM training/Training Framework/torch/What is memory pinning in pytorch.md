
### Introduction

+ Transferring data from the CPU to the GPU is fundamental in many PyTorch applications. It’s crucial for users to understand the most effective tools and options available for moving data between devices.
+ Two key methods for device-to-device data transfer in PyTorch: [`pin_memory()`](https://pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory "(in PyTorch v2.7)") and [`to()`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to "(in PyTorch v2.7)") with the `non_blocking=True` option.

#### Pinned Memory

+ A pinned (or page-locked or non-pageable) memory is a type of memory that cannot be swapped out to disk. It allows for faster and more predictable access times, but has the downside that it is more limited than the pageable memory (aka the main memory).

![[pinned_memory.png]]
#### CUDA and (non-)pageable memory

To understand how CUDA copies a tensor from CPU to CUDA, let’s consider the two scenarios above:

- If the memory is page-locked, the device can access the memory directly in the main memory. The memory addresses are well defined and functions that need to read these data can be significantly accelerated.
- If the memory is pageable, all the pages will have to be brought to the main memory before being sent to the GPU. This operation may take time and is less predictable than when executed on page-locked tensors.

More precisely, when CUDA sends pageable data from CPU to GPU, it must first create a page-locked copy of that data before making the transfer.

### Reference

> https://pytorch.org/tutorials/intermediate/pinmem_nonblock.html