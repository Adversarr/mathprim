// Implements jax-like vmap function for vectorized operations
// The difference is that this vmap allow you to modify the input arguments
// but not allow you to compose the function with other functions
#pragma once

#include "mathprim/core/buffer_view.hpp"

namespace mathprim {

namespace internal {
template <typename T, index_t N, index_t batch_dim, device_t dev>
struct vmap_param {
  using value_type = basic_buffer_view<T, N, dev>;
  using elem_type = basic_buffer_view<T, N - 1, dev>;
  value_type buffer_;

  elem_type operator[](index_t i) const;
};
}  // namespace internal

}  // namespace mathprim
