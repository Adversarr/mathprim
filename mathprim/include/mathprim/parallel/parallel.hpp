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

template <index_t batch_dim = 0, typename T, typename sshape, typename sstride, typename dev>
vmap_arg<basic_view_iterator<T, sshape, sstride, dev, batch_dim>> make_vmap_arg(
    const basic_view<T, sshape, sstride, dev>& view) {
  return make_vmap_arg(view.begin(), view.end());
}

template <class par_impl>
struct parfor {
  template <typename Fn, index_t... sgrids, index_t... sblocks>
  void run(const index_pack<sgrids...>& grid_dim, const index_pack<sblocks...>& block_dim, Fn fn) const noexcept {
    static_cast<const par_impl*>(this)->run_impl(grid_dim, block_dim, std::forward<Fn>(fn));
  }

  template <typename Fn, index_t... sgrids>
  void run(const index_pack<sgrids...>& grid_dim, Fn&& fn) const noexcept {
    static_cast<const par_impl*>(this)->run_impl(grid_dim, std::forward<Fn>(fn));
  }

  template <typename Fn, typename... vmap_args>
  void vmap(Fn&& fn, vmap_args&&... args) {
    static_assert(sizeof...(vmap_args) > 0, "must provide at least one argument");
    // ensure is a vmap_arg
    vmap_impl<Fn>(std::forward<Fn>(fn), make_vmap_arg(std::forward<vmap_args>(args))...);
  }

protected:
  template <typename Fn, typename... vmap_args>
  void vmap_impl(Fn&& fn, vmap_args&&... args) {
    // now args is vmap_arg.
    auto size = (args.size(), ...); // Extract the size of each vmap_arg
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
  template <typename Fn, index_t... sgrids, index_t... sblocks>
  void run_impl(const index_pack<sgrids...>& grid_dim, const index_pack<sblocks...>& block_dim,
                Fn&& fn) const noexcept {
    for (auto grid_id : grid_dim) {
      MATHPRIM_PRAGMA_UNROLL
      for (auto block_id : block_dim) {
        fn(grid_id, block_id);
      }
    }
  }

  template <typename Fn, index_t... sgrids>
  void run_impl(const index_pack<sgrids...>& grid_dim, Fn&& fn) const noexcept {
    MATHPRIM_PRAGMA_UNROLL
    for (auto grid_id : grid_dim) {
      fn(grid_id);
    }
  }
};

}  // namespace mathprim::par
