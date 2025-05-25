### 1. introduction to torch profile [^1]

PyTorch includes a profiler API that is useful to identify the time and memory costs of various PyTorch operations in your code. Profiler can be easily integrated in your code, and the results can be printed as a table or returned in a JSON trace file.

```python
import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler
```

> [!NOTE] Depracated Attention
> An earlier version of the API in [`torch.autograd`](https://docs.pytorch.org/docs/stable/autograd.html#module-torch.autograd "torch.autograd") module is considered legacy and will be deprecated.
> Use `torch.profiler` instead.

[API reference to profile](https://docs.pytorch.org/docs/stable/profiler.html)

### 2. Demo

### 2.1 Module defination

 In this example, we build a custom module that performs two sub-tasks:
- A linear transformation on the input.
- Use the transformation result to get indices on a mask tensor.

```python
class MyModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MyModule, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, input, mask):
        with profiler.record_function("LINEAR PASS"):
            out = self.linear(input)

        with profiler.record_function("MASK INDICES"):
            threshold = out.sum(axis=1).mean().item()
            hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
            hi_idx = torch.from_numpy(hi_idx).cuda()
        return out, hi_idx
```

### 2.2 Run with profile
We initialize random input and mask tensors, and the model.

Before we run the profiler, we warm-up CUDA to ensure accurate performance benchmarking. We wrap the forward pass of our module in the `profiler.profile` context manager. The `with_stack=True` parameter appends the file and line number of the operation in the trace.

```python
model = MyModule(500, 10).cuda()
input = torch.rand(128, 500).cuda()
mask = torch.rand((500, 500, 500), dtype=torch.double).cuda()

# warm-up
model(input, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
    out, idx = model(input, mask)
```

💡Why warm up?

## 3. Main API

```python
# class
torch.profiler.profile(_*_, 
					_activities=None_,
					_schedule=None_, 
					_on_trace_ready=None_, 
					_record_shapes=False_, 
					_profile_memory=False_, 
					_with_stack=False_, 
					_with_flops=False_, 
					_with_modules=False_, 
					_experimental_config=None_, 
					_execution_trace_observer=None_, 
					_acc_events=False_, 
					_use_cuda=None_, 
					_custom_trace_id_callback=None_)
```

**Profiler context manager.**
Parameters:
- **activities** (_iterable_) – list of activity groups (CPU, CUDA) to use in profiling, supported values: `torch.profiler.ProfilerActivity.CPU`, `torch.profiler.ProfilerActivity.CUDA`, `torch.profiler.ProfilerActivity.XPU`. Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA or (when available) ProfilerActivity.XPU.
- **schedule** (_Callable_) – callable that takes step (int) as a single parameter and returns `ProfilerAction` value that specifies the profiler action to perform at each step.
- **on_trace_ready** (_Callable_) – callable that is called at each step when `schedule` returns `ProfilerAction.RECORD_AND_SAVE` during the profiling.
- **record_shapes** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")) – save information about operator’s input shapes.
- **profile_memory** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")) – track tensor memory allocation/deallocation.
- **with_stack** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")) – record source information (file and line number) for the ops.
- **with_flops** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")) – use formula to estimate the FLOPs (floating point operations) of specific operators (matrix multiplication and 2D convolution).
- **with_modules** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")) – record module hierarchy (including function names) corresponding to the callstack of the op. e.g. If module A’s forward call’s module B’s forward which contains an aten::add op, then aten::add’s module hierarchy is A.B Note that this support exist, at the moment, only for TorchScript models and not eager mode models.
- **experimental_config** (__ExperimentalConfig_) – A set of experimental options used for Kineto library features. Note, backward compatibility is not guaranteed.
- **execution_trace_observer** (_ExecutionTraceObserver_) – A PyTorch Execution Trace Observer object. [PyTorch Execution Traces](https://arxiv.org/pdf/2305.14516.pdf) offer a graph based representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators. When this argument is included the observer start() and stop() will be called for the same time window as PyTorch profiler. See the examples section below for a code sample.
- **acc_events** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")) – Enable the accumulation of FunctionEvents across multiple profiling cycles
- **use_cuda** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.13)")) –
[^1]: *mainly ref* https://docs.pytorch.org/tutorials/beginner/profiler.html
