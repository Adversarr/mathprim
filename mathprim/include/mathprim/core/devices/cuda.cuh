#pragma once
#ifndef MATHPRIM_ENABLE_CUDA
#error "This file should be included only when cuda is enabled."
#endif
#include <cuda_runtime.h>

#include "mathprim/core/buffer.hpp"
#include "mathprim/core/utils/cuda_utils.cuh"

namespace mathprim {

class cuda_error final : public std::runtime_error {
public:
  explicit cuda_error(cudaError_t error)
      : std::runtime_error(cudaGetErrorString(error)) {}
  cuda_error(const cuda_error &) = default;
  cuda_error(cuda_error &&) noexcept = default;
};

namespace device {
class cuda : public basic_device<cuda> {
public:
  static constexpr size_t alloc_alignment = 128;

  void *malloc_impl(size_t size) const {
    void *ptr = nullptr;
    if (const auto status = cudaMalloc(&ptr, size); status != cudaSuccess) {
      throw cuda_error(status);
    }
    MATHPRIM_ASSERT(ptr != nullptr);
    return ptr;
  }

  void free_impl(void *ptr) const noexcept {
    MATHPRIM_CUDA_CHECK_SUCCESS(cudaFree(ptr));
  }

  void memset_impl(void *ptr, int value, size_t size) const {
    if (const auto status = (cudaMemset(ptr, value, size));
        status != cudaSuccess) {
      throw cuda_error(status);
    }
  }

  const char *name_impl() const noexcept { return "cuda"; }
};

template <> struct device_traits<cuda> {
  static constexpr size_t alloc_alignment = 128;
};

template <> struct basic_memcpy<cuda, cpu> {
  void operator()(void *dst, const void *src, size_t size) const {
    if (const auto status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        status != cudaSuccess) {
      throw cuda_error(status);
    }
  }
};

template <> struct basic_memcpy<cpu, cuda> {
  void operator()(void *dst, const void *src, size_t size) const {
    if (const auto status = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        status != cudaSuccess) {
      throw cuda_error(status);
    }
  }
};

template <> struct basic_memcpy<cuda, cuda> {
  void operator()(void *dst, const void *src, size_t size) const {
    if (const auto status =
            cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
        status != cudaSuccess) {
      throw cuda_error(status);
    }
  }
};

} // namespace device

/**
 * @brief The default creator for a cuda buffer.
 *
 * @tparam T
 * @param shape
 * @return buffer, throw exception if failed.
 */
template <typename T, typename sshape>
continuous_buffer<T, sshape, device::cuda> make_cuda_buffer(const sshape &shape) {
  auto ptr = static_cast<T *>(device::cuda{}.malloc(sizeof(T) * mathprim::numel(shape)));
  return basic_buffer<T, sshape, default_stride_t<sshape>, device::cuda>(ptr, shape);
}

template <typename T,typename... Args,
          typename = std::enable_if_t<(internal::can_hold_v<Args> && ...)>>
auto make_cuda_buffer(Args... shape) {
  return make_buffer<T, device::cuda>(make_shape(shape...));
}


/**
 * @brief Create a pitched buffer, the last dimension is viewd as a struct.
 *
 */
template <typename T, index_t sshape_x, index_t sshape_y>
basic_buffer<T, shape_t<sshape_x, sshape_y>, stride_t<keep_dim, 1>, device::cuda> make_cuda_pitched_buffer(
    const shape_t<sshape_x, sshape_y> &shape) {
  T* ptr = nullptr;
  size_t pitch = 0;
  auto [height, width] = shape;
  size_t height_in_bytes = static_cast<size_t>(height) * sizeof(T);
  size_t width_in_bytes = static_cast<size_t>(width) * sizeof(T);

  auto error = cudaMallocPitch(&ptr, &pitch, width_in_bytes, height_in_bytes);
  if (error != cudaSuccess) {
    throw cuda_error(error);
  }

#if MATHPRIM_VERBOSE_MALLOC
  printf("cudaPitched: ptr=%p, pitch=%zu, width=%zu, height=%zu\n", ptr, pitch,
         width_in_bytes, height_in_bytes);
#endif

  using ret_type = basic_buffer<T, shape_t<sshape_x, sshape_y>, stride_t<keep_dim, 1>, device::cuda>;
  // Although pitch may not be aligned, free it is still safe.
  ret_type ret(ptr, shape, stride_t<keep_dim, 1>(pitch / sizeof(T), 1));

  // Expect pitch % sizeof(T) == 0
  if (pitch % sizeof(T) != 0) {
    throw std::runtime_error("Pitch is not a multiple of sizeof(T).");
  }

  return std::move(ret);
}

// Support for CUDA pitched.
template <typename T, index_t shape_x, index_t shape_y, index_t stride_x,
          index_t stride_y>
cudaPitchedPtr to_cuda_pitched_ptr(basic_view<T, shape_t<shape_x, shape_y>,
                                              stride_t<stride_x, stride_y>, device::cuda> view) {
  const auto [height, width] = view.shape();
  const auto [pitch, stride] = view.stride();
  cudaPitchedPtr ptr;
  ptr.ptr = view.data();
  // All the strides & shapes are in bytes.
  ptr.pitch = static_cast<size_t>(pitch) * sizeof(T);
  ptr.xsize = static_cast<size_t>(width * stride) * sizeof(T);
  ptr.ysize = static_cast<size_t>(height) * sizeof(T);
  return ptr;
}

template <typename T>
basic_view<T, dshape<2>, stride_t<keep_dim, 1>, device::cuda>
from_cuda_pitched_ptr(cudaPitchedPtr ptr) {
  T* data = static_cast<T*>(ptr.ptr);
  const auto width = static_cast<index_t>(ptr.xsize / sizeof(T));
  const auto height = static_cast<index_t>(ptr.ysize / sizeof(T));
  const auto pitch = static_cast<index_t>(ptr.pitch / sizeof(T));
  return view<device::cuda>(data, make_shape(height, width), stride_t<keep_dim, 1>(pitch, 1));
}

} // namespace mathprim
