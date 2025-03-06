#pragma once
#include <type_traits>
#include <utility>

#include "mathprim/core/defines.hpp"
namespace mathprim::singletons::internal {

template <typename Derived, typename Handle>
class basic_singleton {
  // Handle must be safe to copy, e.g. a pointer to a real handle struct. (similar to a unique_ptr)
  static_assert(std::is_nothrow_move_constructible_v<Handle>, "Handle must be move constructible.");
  static_assert(std::is_nothrow_move_assignable_v<Handle> || std::is_nothrow_copy_assignable_v<Handle>,
                "Handle must be nothrow assignable.");
  static_assert(std::is_default_constructible_v<Handle>, "Handle must be default constructible.");
  static_assert(std::is_convertible_v<Handle, bool>, "Handle must be convertible to bool.");

  // We enforce the derived class have these methods to handle the singleton.
  // void create_impl(Handle& handle) noexcept;
  // void destroy_impl(Handle& handle) noexcept;

public:
  /**
   * @brief Return the managed handle.
   *
   * @return Handle&
   */
  static Handle& get() noexcept {
    return instance().handle_;
  }

  /**
   * @brief Initialize the singleton with an external handle.
   *
   * @param handle The external handle.
   * @param responsibility whether I should destroy the handle.
   * @return Handle
   */
  static void init_external(Handle handle, bool responsibility = false) noexcept {
    instance(std::move(handle), responsibility);
  }

  // Although public, you should not rely on this function.
  static basic_singleton& instance(Handle ext_handle = {}, bool responsibility = false) noexcept {
    static basic_singleton instance(std::move(ext_handle), responsibility);
    return instance;
  }

private:
  explicit MATHPRIM_NOINLINE basic_singleton(Handle handle, bool responsibility) {
    if (static_cast<bool>(handle)) /* external handle is valid */ {
      handle_ = std::move(handle);
      responsibility_ = responsibility;
    } else /* create my new one. */ {
      static_cast<Derived*>(this)->create_impl(handle_);
      responsibility_ = true;
    }
  }

  MATHPRIM_NOINLINE ~basic_singleton() noexcept {
    static_cast<Derived*>(this)->destroy_impl(handle_);
    handle_ = Handle{};  // reset handle to default state
  }

  Handle handle_{};              ///< The global handle: default to zero-init
  bool responsibility_ = false;  ///< Responsibility for destroy.
};

}  // namespace mathprim::singletons::internal