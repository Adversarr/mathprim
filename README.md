# mathprim: a light-weight TensorView and their algorithms (ex. experimental!)
Backends:
1. CPU: aligned allocated memory
2. GPU: CUDA

Algorithms:
1. BLAS (partial)
    1. CPU: handmade, Eigen, cblas
    2. GPU: cuBLAS
2. Parallism:
    1. CPU: openmp, sequential
    2. GPU: cuda, thrust
3. python bindings

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
auto buf = make_buffer<int>(make_dynamic_shape(4, 3, 2));
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
    for (int k = 0; k < 2; ++k) {2
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
parfor.run(dynamic_shape<4>(10, 4, 1, 1), [view] __device__ (auto idx)  {
  auto [i, j, k, l] = idx;
  printf("Lambda view[%d, %d, %d, %d] = %f\n", i, j, k, l, view(i, j));
});
```