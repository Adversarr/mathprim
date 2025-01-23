#pragma once
#include <cuda/std/tuple>

#include "mathprim/core/dim.hpp"
#include "mathprim/core/parallel.hpp"
#include "mathprim/core/utils/cuda_utils.cuh"

namespace mathprim {

namespace internal {
template <typename Fn> __global__ void do_work(Fn fn, dim_t grid_dim, dim_t block_dim) {
  dim_t block_id
      = {static_cast<index_t>(blockIdx.x), static_cast<index_t>(blockIdx.y), static_cast<index_t>(blockIdx.z)};
  dim_t thread_id
      = {static_cast<index_t>(threadIdx.x), static_cast<index_t>(threadIdx.y), static_cast<index_t>(threadIdx.z)};

  for (index_t grid_w = 0; grid_w < to_valid_index(grid_dim.w_); ++grid_w) {
    for (index_t block_w = 0; block_w < to_valid_index(block_dim.w_); ++block_w) {
      block_id.w_ = grid_w;
      thread_id.w_ = block_w;

      fn(block_id, thread_id);
    }
  }
}

template <typename Fn> void foreach_index(const dim_t &grid_dim, const dim_t &block_dim, Fn fn) {
  dim3 grid{static_cast<unsigned int>(to_valid_index(grid_dim.x_)),
            static_cast<unsigned int>(to_valid_index(grid_dim.y_)),
            static_cast<unsigned int>(to_valid_index(grid_dim.z_))};
  dim3 block{static_cast<unsigned int>(to_valid_index(block_dim.x_)),
             static_cast<unsigned int>(to_valid_index(block_dim.y_)),
             static_cast<unsigned int>(to_valid_index(block_dim.z_))};
  do_work<<<grid, block>>>(fn, grid_dim, block_dim);
}

template <typename Fn> __global__ void do_work_cuda_supported_1d(Fn fn) {
  dim<1> block_id{static_cast<index_t>(blockIdx.x)};
  dim<1> thread_id{static_cast<index_t>(threadIdx.x)};
  fn(block_id, thread_id);
}

template <typename Fn> __global__ void do_work_cuda_supported_2d(Fn fn) {
  dim<2> block_id{static_cast<index_t>(blockIdx.x), static_cast<index_t>(blockIdx.y)};
  dim<2> thread_id{static_cast<index_t>(threadIdx.x), static_cast<index_t>(threadIdx.y)};
  fn(block_id, thread_id);
}

template <typename Fn> __global__ void do_work_cuda_supported_3d(Fn fn) {
  dim<3> block_id{static_cast<index_t>(blockIdx.x), static_cast<index_t>(blockIdx.y), static_cast<index_t>(blockIdx.z)};
  dim<3> thread_id{static_cast<index_t>(threadIdx.x), static_cast<index_t>(threadIdx.y),
                   static_cast<index_t>(threadIdx.z)};
  fn(block_id, thread_id);
}

}  // namespace internal

namespace par {

class cuda {
public:
  template <typename Fn> static void foreach_index(const dim_t &grid_dim, const dim_t &block_dim, Fn fn) {
    internal::foreach_index(grid_dim, block_dim, fn);
  }

  template <typename Fn> static void foreach_index(const dim<1> &grid_dim, const dim<1> &block_dim, Fn fn) {
    dim3 grid = to_cuda_dim(grid_dim);
    auto block = to_cuda_dim(block_dim);
    internal::do_work_cuda_supported_1d<<<grid, block>>>(fn);
  }

  template <typename Fn> static void foreach_index(const dim<2> &grid_dim, const dim<2> &block_dim, Fn fn) {
    dim3 grid = to_cuda_dim(grid_dim);
    auto block = to_cuda_dim(block_dim);
    internal::do_work_cuda_supported_2d<<<grid, block>>>(fn);
  }

  template <typename Fn> static void foreach_index(const dim<3> &grid_dim, const dim<3> &block_dim, Fn fn) {
    dim3 grid = to_cuda_dim(grid_dim);
    auto block = to_cuda_dim(block_dim);
    internal::do_work_cuda_supported_3d<<<grid, block>>>(fn);
  }
};

}  // namespace par

template <> struct parfor<par::cuda> {
  using impl = ::mathprim::par::cuda;
  template <typename Fn, index_t N> static void run(const dim<N> &grid_dim, const dim<N> &block_dim, Fn &&fn) {
    impl::foreach_index(grid_dim, block_dim, fn);
  }

  template <typename Fn, index_t N> static void run(const dim<N> &grid_dim, Fn &&fn) {
    // TODO: Optimize over blockDim.
    run(grid_dim, dim<N>{1}, [fn] MATHPRIM_DEVICE(const dim<N> &grid_id, const dim<N> & /* block_id */) {
      fn(grid_id);
    });
  }

  template <typename Fn, typename T, index_t N, device_t dev>
  static void for_each(const basic_view<T, N, dev>& buffer, Fn && fn) {
    run(buffer.shape(), [fn, buffer] MATHPRIM_DEVICE(const dim<N> &idx) {
      fn(buffer(idx));
    });
  }

  template <typename Fn, typename T, index_t N, device_t dev>
  static void for_each_indexed(const basic_view<T, N, dev>& buffer, Fn && fn) {
    run(buffer.shape(), [fn, buffer] MATHPRIM_DEVICE(const dim<N> &idx) {
      fn(idx, buffer(idx));
    });
  }

  template <typename Fn, typename... vmap_args> static void vmap(Fn &&fn, vmap_args &&...args) {
    static_assert(sizeof...(vmap_args) > 0, "must provide at least one argument");
    auto all_args = cuda::std::make_tuple(make_vmap_arg(std::forward<vmap_args>(args))...);

    parfor::run(dim<1>((args.size(), ...)), [all_args, fn] MATHPRIM_DEVICE(const dim<1> &idx) {
      auto apply_to_fn = [fn, i = idx.x_](auto &&...views) {
        fn(views[i]...);
      };
      cuda::std::apply(apply_to_fn, all_args);
    });
  }
};

using parfor_cuda = parfor<par::cuda>;  ///< Alias for parfor<par::cuda>

}  // namespace mathprim
