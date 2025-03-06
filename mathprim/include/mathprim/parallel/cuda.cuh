#pragma once
#include <cuda/std/tuple>
#include <cuda/stream_ref>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "mathprim/core/devices/cuda.cuh"
#include "mathprim/parallel/parallel.hpp" // IWYU pragma: export

namespace mathprim {

namespace par {

namespace internal {

template <typename SgridT, typename SblockT,
          bool IsNaive = (SgridT::ndim <= 3 && SblockT::ndim <= 3)>
struct launcher;

template <typename CudaT>
MATHPRIM_PRIMFUNC void from_cuda(CudaT d, index_array<1> &array) {
  array[0] = d.x;
}

template <typename CudaT>
MATHPRIM_PRIMFUNC void from_cuda(CudaT d, index_array<2> &array) {
  array[0] = d.x;
  array[1] = d.y;
}

template <typename CudaT>
MATHPRIM_PRIMFUNC void from_cuda(CudaT d, index_array<3> &array) {
  array[0] = d.x;
  array[1] = d.y;
  array[2] = d.z;
}

template <typename CudaT>
MATHPRIM_PRIMFUNC void to_cuda(const index_array<1> &array, CudaT &d) {
  d.x = array[0];
  d.y = 1;
  d.z = 1;
}

template <typename CudaT>
MATHPRIM_PRIMFUNC void to_cuda(const index_array<2> &array, CudaT &d) {
  d.x = array[0];
  d.y = array[1];
  d.z = 1;
}

template <typename CudaT>
MATHPRIM_PRIMFUNC void to_cuda(const index_array<3> &array, CudaT &d) {
  d.x = array[0];
  d.y = array[1];
  d.z = array[2];
}

template <index_t NGrids, index_t NBlocks, typename Fn>
__global__ void do_work_naive(Fn fn) {
  index_array<NGrids> block_idx;
  from_cuda(blockIdx, block_idx);
  index_array<NBlocks> thread_idx;
  from_cuda(threadIdx, thread_idx);
  fn(block_idx, thread_idx);
}

template <index_t... Sgrids, index_t... Sblocks>
struct launcher<index_pack<Sgrids...>, index_pack<Sblocks...>, true> {
  template <typename Fn>
  void run(const index_pack<Sgrids...> &grids,
           const index_pack<Sblocks...> &blocks, Fn &&fn) const noexcept {
    dim3 grid_dim;
    to_cuda(grids.to_array(), grid_dim);
    dim3 block_dim;
    to_cuda(blocks.to_array(), block_dim);
    do_work_naive<sizeof...(Sgrids), sizeof...(Sblocks)>
        <<<grid_dim, block_dim, 0, stream_>>>(fn);
  }

  template <typename Fn>
  void run(const index_pack<Sgrids...> &grids, Fn &&fn) const noexcept {
    // TODO: Optimize.
    auto begin = thrust::make_counting_iterator(0);
    auto end = thrust::make_counting_iterator(grids.numel());
    auto streamed_policy = thrust::cuda::par_nosync.on(stream_);
    thrust::for_each(
        streamed_policy, begin, end,
        [fn, grids] __device__(index_t idx) { fn(ind2sub(grids, idx)); });
  }

  launcher() = default;
  explicit launcher(cudaStream_t stream) : stream_{stream} {}
  cudaStream_t stream_{nullptr};
};

template <index_t... Sgrids, index_t... Sblocks>
struct launcher<index_pack<Sgrids...>, index_pack<Sblocks...>, false> {
  template <typename Fn>
  void run(const index_pack<Sgrids...> &grids,
           const index_pack<Sblocks...> &blocks, Fn &&fn) const noexcept {
    auto total_grids = grids.numel(), total_blocks = blocks.numel();
    auto beg = thrust::make_counting_iterator(0);
    auto end = thrust::make_counting_iterator(total_grids * total_blocks);
    auto streamed_policy = thrust::cuda::par_nosync.on(stream_);
    thrust::for_each(
        streamed_policy, beg, end, [fn, grids, blocks] __device__(index_t idx) {
          auto block_idx = idx / total_blocks;
          auto thread_idx = idx % total_blocks;
          fn(ind2sub(grids, block_idx), ind2sub(blocks, thread_idx));
        });
  }

  template <typename Fn>
  void run(const index_pack<Sgrids...> &grids, Fn &&fn) const noexcept {
    auto total_grids = grids.numel();
    auto beg = thrust::make_counting_iterator(0);
    auto end = thrust::make_counting_iterator(total_grids);
    auto streamed_policy = thrust::cuda::par_nosync.on(stream_);
    thrust::for_each(
        streamed_policy, beg, end,
        [fn, grids] __device__(index_t idx) { fn(ind2sub(grids, idx)); });
  }

  launcher() = default;
  explicit launcher(cudaStream_t stream) : stream_{stream} {}
  cudaStream_t stream_{nullptr};
};

} // namespace internal

class cuda : public parfor<cuda> {
public:
  cuda() = default;
  explicit cuda(cudaStream_t stream) : stream_{stream} {}

  /**
   * @brief Launch a kernel with the given grid and block dimensions.
   *
   * @tparam Fn
   * @tparam sgrids
   * @tparam sblocks
   * @param grid_dim
   * @param block_dim
   * @param fn
   */
  template <typename Fn, index_t... Sgrids, index_t... Sblocks>
  void run_impl(const index_pack<Sgrids...> &grid_dim,
                const index_pack<Sblocks...> &block_dim,
                Fn &&fn) const noexcept {
    internal::launcher<index_pack<Sgrids...>, index_pack<Sblocks...>>{}.run(
        grid_dim, block_dim, fn);
  }

  /**
   * @brief Launch a kernel with the given grid dimensions.
   *
   * @tparam Fn
   * @tparam sgrids
   * @param grid_dim
   * @param fn
   */
  template <typename Fn, index_t... Sgrids>
  void run_impl(const index_pack<Sgrids...> &grid_dim, Fn &&fn) const noexcept {
    internal::launcher<index_pack<Sgrids...>, index_pack<>>{}.run(grid_dim, fn);
  }

  /// @brief Wait for all operations in the stream to complete, throws exception
  /// if any error occurs.
  void sync() const {
    auto err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
      throw cuda_error(err);
    }
  }

  /**
   * @brief Check if the stream is ready, returns true if all operations in the
   * stream have completed. Throws exception if any error occurs.
   *
   * @return true
   * @return false
   */
  bool ready() const {
    auto err = cudaStreamQuery(stream_);
    if (err == cudaSuccess) {
      return true;
    } else if (err == cudaErrorNotReady) {
      return false;
    } else {
      throw cuda_error(err);
    }
  }

  /**
   * @brief Get the underlying CUDA stream in use.
   * 
   * @return cudaStream_t 
   */
  cudaStream_t stream() const { return stream_; }

  /**
   * @brief Copy data from one view to another with cuda stream.
   *
   * @tparam T1
   * @tparam sshape1
   * @tparam sstride1
   * @tparam dev1 must be device::cuda or device::cpu
   * @tparam T2
   * @tparam sshape2
   * @tparam sstride2
   * @tparam dev2 must be device::cuda or device::cpu
   * @param dst
   * @param src
   */
  template <typename T1, typename Sshape1, typename Sstride1, typename Dev1,
            typename T2, typename Sshape2, typename Sstride2, typename Dev2>
  void copy(basic_view<T1, Sshape1, Sstride1, Dev1> dst,
            basic_view<T2, Sshape2, Sstride2, Dev2> src) {
    if (!src.is_contiguous() || !dst.is_contiguous()) {
      throw std::runtime_error(
          "The source or destination view is not contiguous.");
    }

    constexpr bool is_cuda1 = std::is_same_v<Dev1, device::cuda>,
                   is_cpu1 = std::is_same_v<Dev1, device::cpu>,
                   is_cuda2 = std::is_same_v<Dev2, device::cuda>,
                   is_cpu2 = std::is_same_v<Dev2, device::cpu>;
    static_assert(is_cuda1 || is_cpu1, "The src device is not supported.");
    static_assert(is_cuda2 || is_cpu2, "The dst device is not supported.");

    cudaMemcpyKind kind;
    if constexpr (is_cuda1 && is_cuda2) {
      kind = cudaMemcpyDeviceToDevice;
    } else if constexpr (is_cuda1 && is_cpu2) {
      kind = cudaMemcpyDeviceToHost;
    } else if constexpr (is_cpu1 && is_cuda2) {
      kind = cudaMemcpyHostToDevice;
    } else {
      kind = cudaMemcpyHostToHost;
    }

    const auto total = src.numel() * sizeof(T1);
    const auto avail = dst.numel() * sizeof(T2);
    if (avail < total) {
      throw std::runtime_error("The destination buffer is too small.");
    }
    cudaStream_t s = stream_;
    MATHPRIM_CUDA_CHECK_SUCCESS(
        cudaMemcpyAsync(dst.data(), src.data(), total, kind, s));
  }

  template <typename Fn, typename... VmapArgs>
  void vmap_impl(Fn &&fn, VmapArgs... args) {
    // now args is vmap_arg.
    auto size = (args.size(), ...); // Extract the size of each vmap_arg
    // Expects all vmap_args have the same size
    if (!((size == args.size()) && ...)) {
      throw std::runtime_error("vmap arguments must have the same size");
    }

    // Loop over the size of the vmap_arg
    auto vmap_shape = make_shape(size);
    // CUDA cannot capture the parameter pack, we use a tuple to store the args.
    // X: run(vmap_shape, [fn, args...] __device__(index_t i) { fn(args[i]...);
    // });

    auto atup = ::cuda::std::make_tuple(args...);
    run(vmap_shape, [fn, atup] __device__(index_t i) {
      // Unpack the tuple and call the function
      ::cuda::std::apply([&fn, i](auto &...args) { fn(args[i]...); }, atup);
    });
  }

private:
  cudaStream_t stream_{nullptr}; ///< CUDA stream: default stream
};

} // namespace par

} // namespace mathprim
