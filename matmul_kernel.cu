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

