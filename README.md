# Matrix Multiplication CUDA Kernel

This project demonstrates a custom CUDA kernel implementation for matrix multiplication, designed to handle both square and non-square matrices.

## Features
- Supports dynamic matrix sizes without requiring explicit input of dimensions.
- Handles both square and non-square matrices.

## Prerequisites
1. **Hardware**: NVIDIA GPU with CUDA support.
2. **Software**:
   - Python 3.7+
   - PyTorch with CUDA support installed.
   - NVIDIA CUDA toolkit.
   - Compiler for CUDA (e.g., `nvcc`).

## Files
- `matmul_kernel.cu`: Contains the CUDA kernel implementation and Python bindings for matrix multiplication.
- `main.py`: Example script to test and benchmark the CUDA kernel against PyTorch's `torch.matmul`.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the necessary dependencies:
   ```bash
   sudo apt-get install python3-pybind11
   pip install ninja
   ```
3. Build the CUDA kernel using PyTorch's extension loader:
   ```python
   from torch.utils.cpp_extension import load

   matmul_kernel = load(
       name="matmul_kernel",
       sources=["matmul_kernel.cu"],
       extra_cuda_cflags=["-O3"]
   )
   ```
4. Ensure the GPU device is properly configured and accessible.

## Usage
Example usage is provided below:

```python
import torch
import time

A = torch.randn(32, 64, device='cuda')  # MxK
B = torch.randn(64, 16, device='cuda')  # KxN

# Using PyTorch's matmul
start = time.time()
C_pytorch = torch.matmul(A, B)
print("PyTorch Time:", time.time() - start)

# Using custom CUDA kernel
start = time.time()
C = matmul_cuda(A, B)
print("CUDA Kernel Time:", time.time() - start)
```

### Expected Output
```
PyTorch Time: 0.10501384735107422
CUDA Kernel Time: 0.00020503997802734375
```

## Full Code
```python
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5"
!sudo apt-get install python3-pybind11
!pip install ninja

%%writefile matmul_kernel.cu
#include <torch/extension.h>

template <typename T>
__global__ void matmul_kernel(const T* A, const T* B, T* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template <typename T>
void matmul_launcher(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0); // Number of rows in A
    int K = A.size(1); // Number of columns in A
    int N = B.size(1); // Number of columns in B

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(A.data_ptr<T>(), B.data_ptr<T>(), C.data_ptr<T>(), M, N, K);
}

torch::Tensor matmul_binding(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Input matrices must be 2-dimensional");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions of matrices must match");

    auto C = torch::zeros({A.size(0), B.size(1)}, A.options()); // Create the result tensor on the same device
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_launcher", ([&] {
        matmul_launcher<scalar_t>(A, B, C);
    }));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul_binding, "Matrix multiplication kernel for dynamically sized matrices");
}

from torch.utils.cpp_extension import load

matmul_kernel = load(
    name="matmul_kernel",
    sources=["matmul_kernel.cu"],
    extra_cuda_cflags=["-O3"]
)

import time
A = torch.randn(32, 64, device='cuda')  # MxK
B = torch.randn(64, 16, device='cuda')  # KxN

start = time.time()
C_pytorch = torch.matmul(A, B)
print("PyTorch Time:", time.time() - start)

start = time.time()
C = matmul_cuda(A, B)
print("CUDA Kernel Time:", time.time() - start)
```

## Explanation of Results
- **PyTorch Time**: Time taken by PyTorch's highly optimized `torch.matmul`.
- **CUDA Kernel Time**: Time taken by the custom CUDA kernel. For smaller matrix sizes, the custom kernel may outperform PyTorch due to minimal overhead.

## Performance Insights
- For larger matrices, PyTorch's `torch.matmul` is likely to perform better due to advanced optimizations and better utilization of GPU resources.
- The custom kernel provides a clear understanding of GPU programming basics and is a starting point for further optimization.

## Limitations
- This implementation is for educational purposes and may not be as efficient as PyTorch for large-scale operations.
- Error handling for invalid inputs is basic and can be improved.




## License
This project is licensed under the MIT License. See the LICENSE file for details.

