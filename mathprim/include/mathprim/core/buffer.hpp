#pragma once
#include <type_traits>

#include "mathprim/core/defines.hpp"
#include "mathprim/core/dim.hpp"
#include "mathprim/core/view.hpp"

namespace mathprim {

namespace internal {

template <typename T>
static constexpr bool is_trival_v = std::is_trivial_v<T>;

template <typename T>
static constexpr bool no_cvref_v = std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>, T>;

template <typename T>
static constexpr bool is_buffer_supported_v = internal::is_trival_v<T> && internal::no_cvref_v<T>;

template <typename SshapeFrom, typename SstrideFrom, typename SshapeTo, typename SstrideTo>
static constexpr bool is_buffer_castable_v
    = is_castable_v<SshapeFrom, SshapeTo> && is_castable_v<SstrideFrom, SstrideTo>;

}  // namespace internal

template <typename Scalar, typename Sshape, typename Sstride, typename Dev>
class basic_buffer {
public:
  using shape_at_compile_time = Sshape;
  using stride_at_compile_time = Sstride;
  static_assert(internal::is_buffer_supported_v<Scalar>, "Unsupported buffer type.");

  template <typename, typename, typename, typename>
  friend class basic_buffer;  // ok, they are friends.

  // Destructor: default and noexcept, all FREE errors will indicates a fatal error.
  ~basic_buffer() noexcept = default;

  // Move constructor: allow implicit cast from a buffer with same shape and stride at runtime.
  template <typename Sshape2, typename Sstride2,
            typename = std::enable_if_t<internal::is_buffer_castable_v<Sshape2, Sstride2, Sshape, Sstride>>>
  basic_buffer(basic_buffer<Scalar, Sshape2, Sstride2, Dev> &&other) noexcept :  // NOLINT(google-explicit-constructor)
      shape_(other.shape()), stride_(other.stride()), data_(std::move(other.data_)) {}

  // Move assignment
  template <typename Sshape2, typename Sstride2,
            typename = std::enable_if_t<internal::is_buffer_castable_v<Sshape2, Sstride2, Sshape, Sstride>>>
  basic_buffer &operator=(basic_buffer<Scalar, Sshape2, Sstride2, Dev> &&other) noexcept {
    if (this != &other) {
      shape_ = other.shape();
      stride_ = other.stride();
      data_ = std::move(other.data_);
    }
    return *this;
  }

  // For external allocated buffers. not responsible for the allocation but responsible for deallocation
  basic_buffer(Scalar *data, const Sshape &shape) : basic_buffer(data, shape, make_default_stride<Scalar>(shape)) {}
  basic_buffer(Scalar *data, const Sshape &shape, const Sstride &stride) :
      shape_(shape), stride_(stride), data_(data) {}

  // Disable copy constructor and all assignment.
  basic_buffer() noexcept = default;  // empty buffer, waiting for move assignement.
  basic_buffer(const basic_buffer &) = delete;
  basic_buffer &operator=(const basic_buffer &) = delete;

  // swap
  template <typename Sshape2, typename Sstride2,
            typename = std::enable_if_t<internal::is_buffer_castable_v<Sshape2, Sstride2, Sshape, Sstride>>>
  void swap(basic_buffer<Scalar, Sshape2, Sstride2, Dev> &other) noexcept {
    std::swap(data_, other.data_);
    internal::swap_impl(shape_.dyn_, other.shape_.dyn_);
    internal::swap_impl(stride_.dyn_, other.stride_.dyn_);
  }

  bool valid() const noexcept {
    return data_ != nullptr;
  }

  explicit operator bool() const noexcept {
    return valid();
  }

  // Shape of buffer.
  const Sshape &shape() const noexcept {
    return shape_;
  }

  index_t shape(index_t i) const noexcept {
    return shape_.at(i);
  }

  // Stride of buffer.
  const Sstride &stride() const noexcept {
    return stride_;
  }

  index_t stride(index_t i) const noexcept {
    return stride_.at(i);
  }

  // The valid ndim of the buffer.
  index_t ndim() const noexcept {
    return mathprim::ndim(shape_);
  }

  // The number of elements in the buffer.
  index_t numel() const noexcept {
    return mathprim::numel(shape_);
  }

  // The size of the buffer.
  index_t size() const noexcept {
    return numel();
  }

  // The physical size of the buffer.
  index_t physical_size() const noexcept {
    return stride_.template get<0>() * shape_.template get<0>();
  }

  // Underlying data pointer.
  Scalar *data() noexcept {
    return data_.get();
  }

  const Scalar *data() const noexcept {
    return data_.get();
  }

  using view_type = basic_view<Scalar, Sshape, Sstride, Dev>;
  using const_view_type = basic_view<const Scalar, Sshape, Sstride, Dev>;
  using iterator = basic_view_iterator<Scalar, Sshape, Sstride, Dev, 0>;
  using const_iterator = basic_view_iterator<const Scalar, Sshape, Sstride, Dev, 0>;

  // default view, implemented in view.hpp
  view_type view() noexcept {
    return view_type(data_.get(), shape_, stride_);
  }
  const_view_type view() const noexcept {
    return const_view();
  }

  const_view_type const_view() const noexcept {
    return const_view_type(data_.get(), shape_, stride_);
  }

  iterator begin() noexcept {
    return view().begin();
  }

  const_iterator begin() const noexcept {
    return view().begin();
  }

  iterator end() noexcept {
    return view().end();
  }

  const_iterator end() const noexcept {
    return view().end();
  }

  void fill_bytes(int value) {
    if (!data_) {
      throw std::runtime_error("Fill bytes on an empty buffer.");
    }
    Dev{}.memset(data_.get(), value, sizeof(Scalar) * numel());
  }

  // Return if the underlying data is contiguous.
  bool is_contiguous() const noexcept {
    return stride_ == make_default_stride<Scalar>(shape_);
  }

  /**
   * @brief Return a new buffer with the same shape and stride, but different device.
   *        It asserts the buffer is contiguous.
   *
   * @tparam Device2
   * @return basic_buffer<Scalar, Sshape, Sstride, Device2>
   */
  template <typename Device2>
  basic_buffer<Scalar, Sshape, Sstride, Device2> to(const Device2 & = {}) const;

  /**
   * @brief Return a new buffer with the same shape and stride, but different device.
   *        It asserts the buffer is contiguous.
   * 
   * @return basic_buffer<Scalar, Sshape, Sstride, Dev> 
   */
  basic_buffer<Scalar, Sshape, Sstride, Dev> clone() const;

private:
  struct no_destructor_deleter {
    inline void operator()(Scalar *ptr) noexcept {
      Dev{}.free(ptr);
    }
  };

  Sshape shape_;
  Sstride stride_;
  std::unique_ptr<Scalar, no_destructor_deleter> data_;
};

template <typename Scalar, typename Sshape, typename Dev>
using continuous_buffer = basic_buffer<Scalar, Sshape, default_stride_t<Sshape>, Dev>;

/**
 * @brief The default creator for a buffer.
 *
 * @tparam T
 * @param shape
 * @return buffer, throw exception if failed.
 */
template <typename Scalar, typename Dev = device::cpu, typename Sshape>
continuous_buffer<Scalar, Sshape, Dev> make_buffer(const Sshape &shape) {
  auto ptr = static_cast<Scalar *>(Dev{}.malloc(sizeof(Scalar) * mathprim::numel(shape)));
  return basic_buffer<Scalar, Sshape, default_stride_t<Sshape>, Dev>(ptr, shape);
}

/**
 * @brief Create a continuous buffer, but no static information.
 *
 */
template <typename Scalar, typename Dev = device::cpu, typename... Args,
          typename = std::enable_if_t<(internal::can_hold_v<Args> && ...)>>
auto make_buffer(Args... shape) {
  return make_buffer<Scalar, Dev>(make_shape(shape...));
}

template <typename Scalar, typename Sshape, typename Sstride, typename Dev>
template <typename Device2>
basic_buffer<Scalar, Sshape, Sstride, Device2> basic_buffer<Scalar, Sshape, Sstride, Dev>::to(const Device2 &) const {
  if (!is_contiguous()) {
    throw std::runtime_error("Cannot convert a non-contiguous buffer to another device.");
  }

  // clone a new buffer with given device, preserving its shape and stride.
  auto buf = make_buffer<Scalar, Device2>(shape_);
  copy(buf.view(), view());
  return buf;
}

template <typename Scalar, typename Sshape, typename Sstride, typename Dev>
basic_buffer<Scalar, Sshape, Sstride, Dev> basic_buffer<Scalar, Sshape, Sstride, Dev>::clone() const {
  if (!is_contiguous()) {
    throw std::runtime_error("Cannot clone a non-contiguous buffer directly.");
  }

  auto buf = make_buffer<Scalar, Dev>(shape_);
  copy(buf.view(), view());
  return buf;
}

///////////////////////////////////////////////////////////////////////////////
/// Memcpy between buffer/views.
///////////////////////////////////////////////////////////////////////////////
template <typename T1, typename Sshape1, typename Sstride1, typename Dev1,  //
          typename T2, typename Sshape2, typename Sstride2, typename Dev2>
void copy(const basic_buffer<T1, Sshape1, Sstride1, Dev1> &dst, const basic_buffer<T2, Sshape2, Sstride2, Dev2> &src,
          bool enforce_same_shape = true) {
  return copy(dst.view(), src.const_view(), enforce_same_shape);
}

template <typename T1, typename Sshape1, typename Sstride1, typename Dev1,  //
          typename T2, typename Sshape2, typename Sstride2, typename Dev2>
void copy(const basic_view<T1, Sshape1, Sstride1, Dev1> &dst, const basic_buffer<T2, Sshape2, Sstride2, Dev2> &src,
          bool enforce_same_shape = true) {
  return copy(dst, src.const_view(), enforce_same_shape);
}
template <typename T1, typename Sshape1, typename Sstride1, typename Dev1,  //
          typename T2, typename Sshape2, typename Sstride2, typename Dev2>
void copy(const basic_buffer<T1, Sshape1, Sstride1, Dev1> &dst, const basic_view<T2, Sshape2, Sstride2, Dev2> &src,
          bool enforce_same_shape = true) {
  return copy(dst.view(), src, enforce_same_shape);
}
}  // namespace mathprim
