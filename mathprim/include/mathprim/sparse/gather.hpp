#pragma once
#include "mathprim/core/buffer.hpp"
#include "mathprim/core/defines.hpp"
#include "mathprim/core/functional.hpp"
#include "mathprim/core/view.hpp"
#include "mathprim/parallel/parallel.hpp"
#include "mathprim/sparse/cvt.hpp"

namespace mathprim::sparse {

/**
 * @brief Holds all required information about:
 *     Dst[outer_ptrs_[i]:outer_ptrs_[i+1]] = sum_j weight[j] * Src[inner_inds[j]]
 *
 * This functionality behaves very similar to a CSR matrix-vector multiplication.
 */
template <typename Scalar, typename Device>
struct basic_gather_desc {
  using index_vector = contiguous_vector_view<index_t, Device>;
  using const_index_vector = contiguous_vector_view<const index_t, Device>;
  using weight_view = contiguous_vector_view<Scalar, Device>;
  using const_weight = contiguous_vector_view<const Scalar, Device>;

  MATHPRIM_PRIMFUNC basic_gather_desc() = default;
  MATHPRIM_PRIMFUNC basic_gather_desc(const_index_vector outer_ptrs, const_index_vector inner_inds,
                                      const_weight weight) noexcept :
      outer_ptrs_(outer_ptrs), inner_inds_(inner_inds), weight_(weight) {}
  MATHPRIM_PRIMFUNC basic_gather_desc(const_index_vector outer_ptrs, const_index_vector inner_inds) noexcept :
      outer_ptrs_(outer_ptrs), inner_inds_(inner_inds) {}
  MATHPRIM_INTERNAL_COPY(basic_gather_desc, default);
  MATHPRIM_INTERNAL_MOVE(basic_gather_desc, default);

  const_index_vector outer_ptrs_;  ///< Outer pointers: Range of inner indices for each outer index.
  const_index_vector inner_inds_;  ///< Inner indices: Indices of the source data.
  const_weight weight_;            ///< Weights: Weights for each inner index. If nullptr, all weights are 1.
};

/**
 * @brief An operator that suitable for parfor.
 */
template <typename Scalar, typename Device, index_t VectorizedDim>
struct basic_gather_operator : public par::basic_task<basic_gather_operator<Scalar, Device, VectorizedDim>> {
  using data_item = contiguous_view<Scalar, dshape<VectorizedDim>, Device>;
  using const_item = contiguous_view<const Scalar, dshape<VectorizedDim>, Device>;
  using data_view = batched<data_item>;
  using const_data = batched<const_item>;

  using desc_type = basic_gather_desc<Scalar, Device>;

  MATHPRIM_PRIMFUNC basic_gather_operator(data_view dst, const_data src, desc_type desc, Scalar alpha = 1) noexcept :
      dst_(dst), src_(src), desc_(desc), alpha_(alpha) {}

  MATHPRIM_INTERNAL_COPY(basic_gather_operator, default);
  MATHPRIM_INTERNAL_MOVE(basic_gather_operator, default);

  data_view dst_;
  const_data src_;
  desc_type desc_;
  Scalar alpha_{1};

  template <typename ParImpl>
  void run_impl(const par::parfor<ParImpl> &parallel) const noexcept {
    index_t dst_size = dst_.shape(0);
    parallel.run(dst_size, *this);
  }

  MATHPRIM_PRIMFUNC void operator()(index_t outer) {
    const index_t start = desc_.outer_ptrs_(outer);
    const index_t end = desc_.outer_ptrs_(outer + 1);
    for (index_t i = start; i < end; ++i) {
      Scalar alpha = (desc_.weight_ ? desc_.weight_(i) : 1) * alpha_;
      functional::madd<data_item, const_item> madd(alpha);
      madd(dst_(outer), src_(desc_.inner_inds_(i)));
    }
  }
};

template <typename Scalar, typename Device>
struct basic_gather_operator<Scalar, Device, 0> : public par::basic_task<basic_gather_operator<Scalar, Device, 0>> {
  using data_view = contiguous_vector_view<Scalar, Device>;
  using const_data = contiguous_vector_view<const Scalar, Device>;
  using desc_type = basic_gather_desc<Scalar, Device>;

  MATHPRIM_PRIMFUNC basic_gather_operator(data_view dst, const_data src, desc_type desc, Scalar alpha = 1) noexcept :
      dst_(dst), src_(src), desc_(desc), alpha_(alpha) {}

  MATHPRIM_INTERNAL_COPY(basic_gather_operator, default);
  MATHPRIM_INTERNAL_MOVE(basic_gather_operator, default);
  data_view dst_;
  const_data src_;
  desc_type desc_;
  Scalar alpha_{1};

  template <typename ParImpl>
  void run_impl(const par::parfor<ParImpl> &parallel) const noexcept {
    index_t dst_size = dst_.shape(0);
    parallel.run(dst_size, *this);
  }

  MATHPRIM_PRIMFUNC void operator()(index_t outer) const noexcept {
    Scalar out = dst_(outer);
    const index_t start = desc_.outer_ptrs_(outer);
    const index_t end = desc_.outer_ptrs_(outer + 1);
    for (index_t i = start; i < end; ++i) {
      Scalar alpha = (desc_.weight_ ? desc_.weight_(i) : 1) * alpha_;
      out += alpha * src_[desc_.inner_inds_(i)];
    }

    dst_(outer) = out;
  }
};

/**
 * @brief Holds the buffer.
 */
template <typename Scalar, typename Device>
class basic_gather_info {
public:
  using desc_type = basic_gather_desc<Scalar, Device>;
  using index_vector_buffer = to_buffer_t<typename desc_type::index_vector>;
  using weight_buffer = to_buffer_t<typename desc_type::weight_view>;

  basic_gather_info(index_t num_outputs, index_t num_inputs, bool has_weights = false) noexcept :
      num_outputs_(num_outputs), num_inputs_(num_inputs), has_weights_(has_weights) {}
  MATHPRIM_INTERNAL_MOVE(basic_gather_info, default);
  MATHPRIM_INTERNAL_COPY(basic_gather_info, delete);

  desc_type desc() const noexcept {
    if (has_weight()) {
      return {outer_.view(), inner_.view(), weight_.view()};
    } else {
      return {outer_.view(), inner_.view()};
    }
  }

  bool has_weight() const noexcept { return has_weights_; }

  template <typename Iter>
  void set_from_triplets(Iter beg, Iter end) {
    // Build CSR matrix and check the indices.
    auto sparse_coo = make_from_triplets<Scalar>(beg, end, num_outputs_, num_inputs_);
    auto sparse_csr = make_from_coos<Scalar, sparse_format::csr>(sparse_coo);
    visit(sparse_csr.view(), par::seq(), [&](index_t dst, index_t src, Scalar /* weight */) {
      if (dst >= num_outputs_ || src >= num_inputs_) {
        throw std::runtime_error("Invalid index in the sparse matrix.");
      }
    });

    // Copy the data.
    outer_ = make_buffer<index_t, Device>(sparse_csr.outer_ptrs().shape());
    inner_ = make_buffer<index_t, Device>(sparse_csr.inner_indices().shape());
    copy(outer_.view(), sparse_csr.outer_ptrs());
    copy(inner_.view(), sparse_csr.inner_indices());

    if (has_weights_) {
      weight_ = make_buffer<Scalar, Device>(sparse_csr.values().shape());
      copy(weight_.view(), sparse_csr.values());
    }
  }

  basic_sparse_view<const Scalar, Device, sparse_format::csr> view_as_csr() const {
    if (!weight_) {
      throw std::runtime_error("The gather info does not have weights.");
    }

    return {weight_.view(), outer_.view(),  inner_.view(),           num_outputs_,
            num_inputs_,    inner_.numel(), sparse_property::general};
  }

private:
  index_vector_buffer outer_;
  index_vector_buffer inner_;
  weight_buffer weight_;
  index_t num_outputs_{0};
  index_t num_inputs_{0};
  bool has_weights_{false};
};

}  // namespace mathprim::sparse
