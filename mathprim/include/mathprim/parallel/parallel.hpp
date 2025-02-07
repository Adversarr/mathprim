#pragma once
#include "mathprim/core/dim.hpp"

namespace mathprim::par {

// template <typename T, index_t N, device_t dev, index_t batch_dim> struct vmap_arg {
//   const basic_view<T, N, dev> view_;
//   MATHPRIM_PRIMFUNC explicit vmap_arg(const basic_view<T, N, dev> &view) : view_(view) {}
//
//   MATHPRIM_PRIMFUNC auto operator[](index_t i) const noexcept {
//     return (view_.template slice<batch_dim>(i));
//   }
//
//   MATHPRIM_PRIMFUNC index_t size() const noexcept {
//     return view_.shape(batch_dim);
//   }
// };
//
// template <typename T, device_t dev, index_t batch_dim> struct vmap_arg<T, 1, dev, batch_dim> {
//   const basic_view<T, 1, dev> view_;
//   MATHPRIM_PRIMFUNC explicit vmap_arg(const basic_view<T, 1, dev> &view) : view_(view) {}
//
//   MATHPRIM_PRIMFUNC T &operator[](index_t i) const noexcept {
//     return (view_.template slice<batch_dim>(i));
//   }
//
//   MATHPRIM_PRIMFUNC index_t size() const noexcept {
//     return view_.shape(batch_dim);
//   }
// };
//
// template <index_t batch_dim = 0, typename T, index_t N, device_t dev>
// MATHPRIM_PRIMFUNC vmap_arg<T, N, dev, batch_dim> make_vmap_arg(basic_view<T, N, dev> view) {
//   return vmap_arg<T, N, dev, batch_dim>(view);
// }
//
// template <index_t batch_dim, typename T, index_t N, device_t dev>
// MATHPRIM_PRIMFUNC vmap_arg<T, N, dev, batch_dim> make_vmap_arg(vmap_arg<T, N, dev, batch_dim> arg) {
//   return vmap_arg<T, N, dev, batch_dim>(arg);
// }

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

  //   template <typename Fn, typename T, index_t N, device_t dev>
  //   static void for_each_indexed(basic_view<T, N, dev> buffer, Fn &&fn) {
  //     parfor::run(buffer.shape(), [f = std::forward<Fn>(fn), buffer](const dim<N> &idx) {
  //       f(idx, buffer(idx));
  //     });
  //   }

  //   template <typename Fn, typename... vmap_args>
  //   static void vmap(Fn &&fn, vmap_args &&...args) {
  //     static_assert(sizeof...(vmap_args) > 0, "must provide at least one argument");
  //     // ensure is a vmap_arg
  //     parfor::vmap_impl<Fn>(std::forward<Fn>(fn), make_vmap_arg(std::forward<vmap_args>(args))...);
  //   }

  // private:
  //   template <typename Fn, typename... vmap_args>
  //   static void vmap_impl(Fn &&fn, vmap_args &&...args) {
  //     // now args is vmap_arg.
  //     parfor::run(dim<1>((args.size(), ...)), [fn, args...](const dim<1> &idx) {
  //       fn((args[idx.x_])...);
  //     });
  //   }
};

class seq : public parfor<seq> {
public:
  template <typename Fn, index_t... sgrids, index_t... sblocks>
  void run_impl(const index_pack<sgrids...>& grid_dim, const index_pack<sblocks...>& block_dim,
                Fn&& fn) const noexcept {
    for (auto grid_id : grid_dim) {
      for (auto block_id : block_dim) {
        fn(grid_id, block_id);
      }
    }
  }

  template <typename Fn, index_t... sgrids>
  void run_impl(const index_pack<sgrids...>& grid_dim, Fn&& fn) const noexcept {
    for (auto grid_id : grid_dim) {
      fn(grid_id);
    }
  }
};

}  // namespace mathprim::par
