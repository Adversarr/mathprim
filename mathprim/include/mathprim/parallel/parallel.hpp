#pragma once
#include "mathprim/core/dim.hpp"

namespace mathprim::par {

template <typename IterT>
struct vmap_arg {
  using reference = typename IterT::reference;
  IterT beg_;
  IterT end_;

  MATHPRIM_PRIMFUNC explicit vmap_arg(IterT beg, IterT end) : beg_(beg), end_(end) {}
  vmap_arg(const vmap_arg&) = default;
  vmap_arg(vmap_arg&&) = default;
  vmap_arg& operator=(const vmap_arg&) = default;
  vmap_arg& operator=(vmap_arg&&) = default;

  MATHPRIM_PRIMFUNC IterT begin() const noexcept {
    return beg_;
  }

  MATHPRIM_PRIMFUNC IterT end() const noexcept {
    return end_;
  }

  MATHPRIM_PRIMFUNC reference operator[](index_t i) const noexcept {
    return beg_[i];
  }

  MATHPRIM_PRIMFUNC index_t size() const noexcept {
    return end_ - beg_;
  }
};

template <typename IterT>
vmap_arg<IterT> make_vmap_arg(IterT beg, IterT end) {
  return vmap_arg<IterT>(beg, end);
}

template <index_t BatchDim = 0, typename T, typename Sshape, typename Sstride, typename Dev>
vmap_arg<basic_view_iterator<T, Sshape, Sstride, Dev, BatchDim>> make_vmap_arg(
    const basic_view<T, Sshape, Sstride, Dev>& view) {
  return make_vmap_arg(view.begin(), view.end());
}

///////////////////////////////////////////////////////////////////////////////
/// Parallel for loop
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief CRTP base class for parallel for loop.
 * 
 * @tparam ParImpl 
 */
template <class ParImpl>
class parfor {
public:
  /**
   * @brief Launch a kernel with grid and block dimensions
   *
   * @tparam Fn
   * @tparam sgrids
   * @tparam sblocks
   * @param grid_dim
   * @param block_dim
   * @param fn
   */
  template <typename Fn, index_t... Sgrids, index_t... Sblocks>
  void run(const index_pack<Sgrids...>& grid_dim, const index_pack<Sblocks...>& block_dim, Fn&& fn) const noexcept {
    static_cast<const ParImpl*>(this)->run_impl(grid_dim, block_dim, std::forward<Fn>(fn));
  }

  /**
   * @brief Launch a kernel with grid dimensions
   *
   * @tparam Fn
   * @tparam sgrids
   * @param grid_dim
   * @param fn
   */
  template <typename Fn, index_t... Sgrids>
  void run(const index_pack<Sgrids...>& grid_dim, Fn&& fn) const noexcept {
    static_cast<const ParImpl*>(this)->run_impl(grid_dim, std::forward<Fn>(fn));
  }

  /**
   * @brief Launch kernel on given batch dimension.
   *
   * @tparam Fn
   * @tparam vmap_args
   * @param fn
   * @param args batch data.
   */
  template <typename Fn, typename... VmapArgs>
  void vmap(Fn&& fn, VmapArgs&&... args) {
    static_assert(sizeof...(VmapArgs) > 0, "must provide at least one argument");
    // ensure is a vmap_arg
    static_cast<ParImpl*>(this)->template vmap_impl<Fn>(std::forward<Fn>(fn),
                                                        make_vmap_arg(std::forward<VmapArgs>(args))...);
  }

protected:
  template <typename Fn, typename... VmapArgs>
  void vmap_impl(Fn&& fn, VmapArgs&&... args) {
    // now args is vmap_arg.
    auto size = (args.size(), ...);  // Extract the size of each vmap_arg
    // Expects all vmap_args have the same size
    if (!((size == args.size()) && ...)) {
      throw std::runtime_error("vmap arguments must have the same size");
    }

    // Loop over the size of the vmap_arg
    auto vmap_shape = make_shape(size);
    run(vmap_shape, [fn, args...](index_t i) {
      fn(args[i]...);
    });
  }
};

class seq : public parfor<seq> {
public:
  friend class parfor<seq>;
  template <typename Fn, index_t... Sgrids, index_t... Sblocks>
  void run_impl(const index_pack<Sgrids...>& grid_dim, const index_pack<Sblocks...>& block_dim,
                Fn&& fn) const noexcept {
    for (auto grid_id : grid_dim) {
      MATHPRIM_PRAGMA_UNROLL_HOST
      for (auto block_id : block_dim) {
        fn(grid_id, block_id);
      }
    }
  }

  template <typename Fn, index_t... Sgrids>
  void run_impl(const index_pack<Sgrids...>& grid_dim, Fn&& fn) const noexcept {
    MATHPRIM_PRAGMA_UNROLL_HOST
    for (auto grid_id : grid_dim) {
      fn(grid_id);
    }
  }
};

}  // namespace mathprim::par
