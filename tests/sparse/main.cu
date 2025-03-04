#include "mathprim/parallel/cuda.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <mathprim/sparse/basic_sparse.hpp>
#include <mathprim/sparse/blas/cusparse.hpp>

// CUDA 错误检查宏
#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  }

// cuSPARSE 错误检查宏
#define CHECK_CUSPARSE(call)                                                   \
  {                                                                            \
    cusparseStatus_t status = call;                                            \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("cuSPARSE Error at %s:%d - %d\n", __FILE__, __LINE__, status);    \
      exit(1);                                                                 \
    }                                                                          \
  }

int main() {
  // 初始化 cuSPARSE 句柄
  cusparseHandle_t handle;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // 定义 CSR 矩阵示例 (3x3)
  const int rows = 3, cols = 3, nnz = 5;
  float h_csr_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  int h_csr_col_idx[] = {0, 1, 1, 2, 0}; // 列索引
  int h_csr_row_ptr[] = {0, 2, 4, 5};    // 行指针

  // 定义输入向量和输出向量 (全1向量)
  float h_x[] = {1.0f, 1.0f, 1.0f};
  float h_y[rows] = {0.0f};

  // 分配设备内存
  float *d_csr_values, *d_x, *d_y;
  int *d_csr_col_idx, *d_csr_row_ptr;

  CHECK_CUDA(cudaMalloc(&d_csr_values, nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_csr_col_idx, nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_csr_row_ptr, (rows + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_x, cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_y, rows * sizeof(float)));

  // 拷贝数据到设备
  CHECK_CUDA(cudaMemcpy(d_csr_values, h_csr_values, nnz * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csr_col_idx, h_csr_col_idx, nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr, (rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_y, 0, rows * sizeof(float))); // 初始化输出为0

  using namespace mathprim;
  auto sparse_view = sparse::basic_sparse_view<float, device::cuda,
                                               sparse::sparse_format::csr>(
      d_csr_values, d_csr_row_ptr, d_csr_col_idx, rows, cols, nnz,
      sparse::sparse_property::general);
  par::cuda cu;
  sparse::visit(sparse_view, cu, []__device__(auto i, auto j, auto v) {
    printf("A[%d, %d] = %.2f\n", i, j, v);
  });

  auto api =
      sparse::blas::cusparse<float, mathprim::sparse::sparse_format::csr>(
          sparse_view.as_const());
  auto x = view<device::cuda>(d_x, make_shape(cols));
  auto y = view<device::cuda>(d_y, make_shape(rows));
  api.gemv(1.0f, x, 0.0f, y);
  // 拷贝结果回主机
  CHECK_CUDA(
      cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

  // 打印结果
  printf("Result y = A*x:\n");
  for (int i = 0; i < rows; ++i) {
    printf("y[%d] = %.2f\n", i, h_y[i]);
  }

  api.spmm(1.0, x.reshape(cols, 1).as_const(), 0.0, y.reshape(rows, 1));
  // 拷贝结果回主机
  CHECK_CUDA(
      cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

  // 打印结果
  printf("Result y = A*x:\n");
  for (int i = 0; i < rows; ++i) {
    printf("y[%d] = %.2f\n", i, h_y[i]);
  }

  // 预期输出:
  // y[0] = 1*1 + 2*1 = 3.00
  // y[1] = 3*1 + 4*1 = 7.00
  // y[2] = 5*1 = 5.00

  CHECK_CUDA(cudaFree(d_csr_values));
  CHECK_CUDA(cudaFree(d_csr_col_idx));
  CHECK_CUDA(cudaFree(d_csr_row_ptr));
  CHECK_CUDA(cudaFree(d_x));
  CHECK_CUDA(cudaFree(d_y));

  return 0;
}