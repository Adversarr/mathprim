# mathprim: a light-weight Tensor(View) library

mathprim, or math's primitives, is a **glue-like** library for optimized numeric algorithms.

## Dependencies and Requirements

### Required Dependencies
- **C++ Compiler**: C++17 compatible compiler
- **CMake**: Version 3.23 or higher
- **Eigen3**: Required for matrix operations
- **BLAS**: OpenBLAS (Linux/Windows) or Accelerate (macOS)

### Optional Dependencies
- **CUDA Toolkit**: For GPU acceleration
- **OpenMP**: For CPU parallelization
- **CHOLMOD** (SuiteSparse): For direct sparse solvers
- **SuperLU**: For additional sparse matrix solvers
- **MKL**: Intel Math Kernel Library (optional alternative to OpenBLAS)
- **Python**: For Python bindings (with Development headers)
- **nanobind**: For Python bindings

### Version Requirements
- **C++**: Standard 17 or higher
- **CUDA**: Compatible with your GPU architecture (CUDA Architectures 86 used in bindings)
- **CMake**: 3.23 or higher

## Installation on Ubuntu

### Install Required Dependencies
```bash
# Install compiler and build tools
sudo apt-get update
sudo apt-get install -y build-essential cmake

# Install Eigen3 and OpenBLAS
sudo apt-get install -y libeigen3-dev libopenblas-dev
```

### Install Optional Dependencies
```bash
# Install SuiteSparse (CHOLMOD)
sudo apt-get install -y libsuitesparse-dev

# Install SuperLU
sudo apt-get install -y libsuperlu-dev

# Install benchmark and gtest (if testing and benchmarking are needed)
sudo apt-get install -y libbenchmark-dev libgtest-dev
```

## Installization on Arch based Linux

```bash
sudo pacman -S cmake eigen openblas # Required dependencies
sudo pacman -S suitesparse superlu # Optional dependencies
```

## Building from Source
```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

## Known Issues

1. SuperLU's ilu does not work properly.

# Architecture

Current Backends:
1. CPU
2. CUDA

Features:

0. Key features:
    1. zero cost abstraction
    2. Eigen support, with `Map` class,
1. BLAS: Optimized for speed.
    1. CPU: handmade, Eigen, cblas
    2. GPU: cuBLAS
2. Parallism:
    1. CPU: openmp, sequential
    2. GPU: cuda
3. Sparse matrices: (mainly on spd)
    1. CSR, CSC matrix spmv support.
    2. cusparse spmv support.
    3. direct: cholmod.
    4. iterative solver: cg/pcg
    5. preconditioners:
        1. diagonal: CPU/CUDA
        2. ic, ilu: for both CPU(Eigen's implementation), and CUDA(cuSPARSE).
        3. FSAI0: (crafted)
4. Fully customizable optimizers:
    1. GD/SGD/GD, with momentum(or nesterov momentum)
    2. AdamW
    3. l_bfgs: customizable...
        1. linesearcher (default to backtracking)
        2. preconditoner (default to scaled identity)

Future works:

1. torch bindings: especially for spd sparse matrices.
2. tiny nn libraries, ref. tiny-cuda-nn.
3. view's inplace operations, faster gemms on cpu/cuda
4. cuda: graph capture, multiple streams, ...

## Recipies

### View from existing buffer.

```cpp
shape_t<keep_dim, 3, 2> shape(4, 3, 2);
int p[24];
for (int i = 0; i < 24; ++i) {
  p[i] = i + 1;
}

auto view = make_view<device::cpu>(p, shape);
auto value0 = view(0, 0, 0);
```

### View from buffer.

```cpp
auto buf = make_buffer<int>(4, 3, 2);
auto view = buf.view();
for (int i = 0; i < 4; ++i) {
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 2; ++k) {
      buf.data()[i * 6 + j * 2 + k] = i * 6 + j * 2 + k + 1;
      EXPECT_EQ(view(i, j, k), i * 6 + j * 2 + k + 1);
    }
  }
}

for (const auto &vi : view) {
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 2; ++k) {
      EXPECT_EQ(view(0, j, k), j * 2 + k + 1);
    }
  }
  break;
}
```

### CUDA parallism

```cpp
__global__ void set_value(float *ptr, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    ptr[idx] = static_cast<float>(idx);
  }
}

// Create buffer on CUDA.
auto buf = make_buffer<float, device::cuda>(shape_t<-1, 4>(10, 4));
auto view = buf.view();
set_value<<<view.size(), 1>>>(buf.data(), buf.size());

// Parallism
par::cuda parfor;
parfor.run(view.shape(), [view]__device__(auto idx)  {
  auto [i, j] = idx;
  printf("Lambda view[%d, %d] = %f\n", i, j, view(i, j));
});
parfor.run(dshape<4>(10, 4, 1, 1), [view] __device__ (auto idx)  {
  auto [i, j, k, l] = idx;
  printf("Lambda view[%d, %d, %d, %d] = %f\n", i, j, k, l, view(i, j));
});
```