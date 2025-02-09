#pragma once
#include <cuda/stream_ref>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "mathprim/core/devices/cuda.cuh"
#include "mathprim/parallel/parallel.hpp" // IWYU pragma: export

namespace mathprim {

namespace par {

namespace internal {

template <typename sgrid_t, typename sblock_t,
          bool IsNaive = (sgrid_t::ndim <= 3 && sblock_t::ndim <= 3)>
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

template <index_t n_grids, index_t n_blocks, typename Fn>
__global__ void do_work_naive(Fn fn) {
  index_array<n_grids> block_idx;
  from_cuda(blockIdx, block_idx);
  index_array<n_blocks> thread_idx;
  from_cuda(threadIdx, thread_idx);
  fn(block_idx, thread_idx);
}

template <index_t... sgrids, index_t... sblocks>
struct launcher<index_pack<sgrids...>, index_pack<sblocks...>, true> {
  template <typename Fn>
  void run(const index_pack<sgrids...> &grids,
           const index_pack<sblocks...> &blocks, Fn &&fn) const noexcept {
    dim3 grid_dim;
    to_cuda(grids.to_array(), grid_dim);
    dim3 block_dim;
    to_cuda(blocks.to_array(), block_dim);

    do_work_naive<sizeof...(sgrids), sizeof...(sblocks)>
        <<<grid_dim, block_dim>>>(fn);
  }

  template <typename Fn>
  void run(const index_pack<sgrids...> &grids, Fn &&fn) const noexcept {
    auto begin = thrust::make_counting_iterator(0);
    auto end = thrust::make_counting_iterator(grids.numel());
    thrust::for_each(
        thrust::device, begin, end,
        [fn, grids] __device__(index_t idx) { fn(ind2sub(grids, idx)); });
  }
};

template <index_t... sgrids, index_t... sblocks>
struct launcher<index_pack<sgrids...>, index_pack<sblocks...>, false> {
  template <typename Fn>
  void run(const index_pack<sgrids...> &grids,
           const index_pack<sblocks...> &blocks, Fn &&fn) const noexcept {
    auto total_grids = grids.numel(), total_blocks = blocks.numel();
    auto beg = thrust::make_counting_iterator(0);
    auto end = thrust::make_counting_iterator(total_grids * total_blocks);
    thrust::for_each(
        thrust::device, beg, end, [fn, grids, blocks] __device__(index_t idx) {
          auto block_idx = idx / total_blocks;
          auto thread_idx = idx % total_blocks;
          fn(ind2sub(grids, block_idx), ind2sub(blocks, thread_idx));
        });
  }

  template <typename Fn>
  void run(const index_pack<sgrids...> &grids, Fn &&fn) const noexcept {
    auto total_grids = grids.numel();
    auto beg = thrust::make_counting_iterator(0);
    auto end = thrust::make_counting_iterator(total_grids);
    thrust::for_each(
        thrust::device, beg, end,
        [fn, grids] __device__(index_t idx) { fn(ind2sub(grids, idx)); });
  }
};

} // namespace internal

class cuda {
public:
  cuda() = default;
  explicit cuda(::cuda::stream_ref stream) : stream_{stream} {}

  template <typename Fn, index_t... sgrids, index_t... sblocks>
  void run(const index_pack<sgrids...> &grid_dim,
           const index_pack<sblocks...> &block_dim, Fn &&fn) const noexcept {
    internal::launcher<index_pack<sgrids...>, index_pack<sblocks...>>{}.run(
        grid_dim, block_dim, fn);
  }

  template <typename Fn, index_t... sgrids>
  void run(const index_pack<sgrids...> &grid_dim, Fn &&fn) const noexcept {
    internal::launcher<index_pack<sgrids...>, index_pack<>>{}.run(grid_dim, fn);
  }

private:
  ::cuda::stream_ref stream_; ///< CUDA stream: default stream
};

} // namespace par

} // namespace mathprim
