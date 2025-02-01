#include <mathprim/sparse/basic_sparse.hpp>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusparse.h>

// CUDA 错误检查宏
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// cuSPARSE 错误检查宏
#define CHECK_CUSPARSE(call) { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        printf("cuSPARSE Error at %s:%d - %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
}

int main() {
    // 初始化 cuSPARSE 句柄
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // 定义 CSR 矩阵示例 (3x3)
    const int rows = 3, cols = 3, nnz = 5;
    float h_csr_values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    int h_csr_col_idx[] = {0, 1, 1, 2, 0};   // 列索引
    int h_csr_row_ptr[] = {0, 2, 4, 5};      // 行指针

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
    CHECK_CUDA(cudaMemcpy(d_csr_values, h_csr_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_col_idx, h_csr_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csr_row_ptr, h_csr_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y, 0, rows * sizeof(float))); // 初始化输出为0

    // 创建矩阵描述符
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,                    // 矩阵描述符
        rows, cols, nnz,          // 行数、列数、非零元数
        d_csr_row_ptr,            // 行指针 (设备内存)
        d_csr_col_idx,            // 列索引 (设备内存)
        d_csr_values,             // 值数组 (设备内存)
        CUSPARSE_INDEX_32I,       // 行指针索引类型 (32位整数)
        CUSPARSE_INDEX_32I,       // 列索引类型 (32位整数)
        CUSPARSE_INDEX_BASE_ZERO, // 索引基址 (从0开始)
        CUDA_R_32F                // 数据类型 (32位浮点)
    ));

    // 创建向量描述符 (输入x和输出y)
    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(
        &vecX,                    // 向量描述符
        cols,                     // 向量长度
        d_x,                      // 数据指针 (设备内存)
        CUDA_R_32F                // 数据类型
    ));
    CHECK_CUSPARSE(cusparseCreateDnVec(
        &vecY,
        rows,
        d_y,
        CUDA_R_32F
    ));

    // 执行 SpMV: y = alpha * A * x + beta * y
    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    void* d_buffer = nullptr;

    // 获取所需缓冲区大小
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, // 矩阵不转置
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        CUDA_R_32F,                       // 计算数据类型
        CUSPARSE_SPMV_ALG_DEFAULT,        // 算法选择
        &bufferSize
    ));

    // 分配临时缓冲区
    CHECK_CUDA(cudaMalloc(&d_buffer, bufferSize));

    // 执行 SpMV 计算
    CHECK_CUSPARSE(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_buffer
    ));

    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // 打印结果
    printf("Result y = A*x:\n");
    for (int i = 0; i < rows; ++i) {
        printf("y[%d] = %.2f\n", i, h_y[i]);
    }
    // 预期输出:
    // y[0] = 1*1 + 2*1 = 3.00
    // y[1] = 3*1 + 4*1 = 7.00
    // y[2] = 5*1 = 5.00

    // 释放资源
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaFree(d_csr_values));
    CHECK_CUDA(cudaFree(d_csr_col_idx));
    CHECK_CUDA(cudaFree(d_csr_row_ptr));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}