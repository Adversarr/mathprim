#pragma once
#include "mathprim/core/view.hpp"

namespace mathprim {

/**
 * @brief Descriptor of a parameter item in any optimization problem.
 *
 * @tparam Scalar
 * @tparam Device
 */
template <typename Scalar, typename Device>
class parameter_item {
public:
  using scalar_type = Scalar;
  using device_type = Device;
  using view_type = contiguous_view<Scalar, dshape<1>, Device>;
  using const_buffer_type = contiguous_view<const Scalar, dshape<1>, Device>;

  parameter_item() = default;
  parameter_item(const parameter_item&) = default;
  parameter_item(parameter_item&&) noexcept = default;
  parameter_item& operator=(const parameter_item&) = default;
  parameter_item& operator=(parameter_item&&) noexcept = default;

  /**
   * @brief Construct a new parameter item object
   *
   * @param value view to the value of the parameter
   * @param gradient
   * @param name optional, name of the parameter
   */
  parameter_item(view_type value, view_type gradient, std::string name = {}) :
      value_(value), gradient_(gradient), name_(name) {
    MATHPRIM_INTERNAL_CHECK_VALID_VIEW(value);
    MATHPRIM_INTERNAL_CHECK_VALID_VIEW(gradient);
  }

  /**
   * @brief Construct a new parameter item object
   *
   * @param value view to the value of the parameter
   * @param name optional, name of the parameter
   */
  parameter_item(view_type value, std::string name = {}) :  // NOLINT(google-explicit-constructor)
      value_(value), name_(name) {
    MATHPRIM_INTERNAL_CHECK_VALID_VIEW(value);
  }

  /// @brief Get the value of the parameter
  const view_type& value() const noexcept { return value_; }

  /// @brief Get the gradient of the parameter
  const view_type& gradient() const noexcept { return gradient_; }

  /// @brief Get the name of the parameter
  const std::string& name() const noexcept { return name_; }

  /// @brief Get the offset of the parameter in fused_gradients.
  index_t offset() const noexcept { return offset_; }

  void set_gradient(view_type gradient) noexcept { gradient_ = gradient; }

  void set_offset(index_t offset) noexcept { offset_ = offset; }

  view_type value_;
  view_type gradient_;
  index_t offset_{};
  std::string name_;
};
 
}  // namespace mathprim