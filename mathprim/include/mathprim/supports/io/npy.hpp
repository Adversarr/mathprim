#pragma once
#include <ostream>
#include <sstream>
#include <vector>

#include "mathprim/core/buffer.hpp"
#include "mathprim/core/utils/common.hpp"
#include "mathprim/core/view.hpp"
namespace mathprim::io {

template <typename Scalar, index_t Ndim>
class numpy {
public:
  using buffer_type = contiguous_buffer<Scalar, dshape<Ndim>, device::cpu>;
  using view_type = contiguous_view<Scalar, dshape<Ndim>, device::cpu>;
  using const_view_type = contiguous_view<const Scalar, dshape<Ndim>, device::cpu>;

  void write(std::ostream& os, const const_view_type& view);

  buffer_type read(std::istream& is);
};

template <typename Scalar, index_t Ndim>
void numpy<Scalar, Ndim>::write(std::ostream& os, const const_view_type& view) {
  const char magic[] = "\x93NUMPY";
  const uint8_t version_major = 1;
  const uint8_t version_minor = 0;
  auto shape = view.shape();

  // build header content
  std::string dtype;
  if constexpr (std::is_same_v<Scalar, float>) {
    dtype = "<f4";
  } else if constexpr (std::is_same_v<Scalar, double>) {
    dtype = "<f8";
  } else {
    static_assert(mathprim::internal::always_false_v<Scalar>, "Unsupported scalar type");
  }
  std::string fortran_order = "False";
  std::string shape_str = "(";
  for (index_t i = 0; i < ndim(shape); ++i) {
    shape_str += std::to_string(shape[i]);
    if (i != ndim(shape) - 1)
      shape_str += ", ";
  }
  shape_str += ")";

  std::string dict = "{'descr': '" + dtype + "', 'fortran_order': " + fortran_order + ", 'shape': " + shape_str + ", }";

  // len(header) must be multiple of 64
  uint16_t header_len = dict.size() + 1;
  uint16_t padding_len = 64 - ((header_len + 10) % 64);
  if (padding_len == 64)
    padding_len = 0;
  header_len += padding_len;

  // Magic
  os.write(magic, 6);

  // Versioning
  os.write(reinterpret_cast<const char*>(&version_major), 1);
  os.write(reinterpret_cast<const char*>(&version_minor), 1);

  ///<<< Begin Header
  // Length
  os.write(reinterpret_cast<const char*>(&header_len), 2);

  // Content
  os.write(dict.c_str(), dict.size());

  // Padding
  std::string padding(padding_len, ' ');
  os.write(padding.c_str(), padding_len);
  os.write("\n", 1);
  ///>>> End of Header

  // Write tensor
  size_t total_elements = view.numel();
  auto data = view.data();
  os.write(reinterpret_cast<const char*>(data), total_elements * sizeof(Scalar));
}
template <typename Scalar, index_t Ndim>
typename numpy<Scalar, Ndim>::buffer_type numpy<Scalar, Ndim>::read(std::istream& is) {
  // Magic string and version
  char magic[6];
  is.read(magic, 6);
  MATHPRIM_INTERNAL_CHECK_THROW(std::string(magic, 6) == "\x93NUMPY", std::runtime_error,
                                "Invalid .npy file: invalid magic string");

  uint8_t version_major, version_minor;
  is.read(reinterpret_cast<char*>(&version_major), 1);
  is.read(reinterpret_cast<char*>(&version_minor), 1);
  MATHPRIM_INTERNAL_CHECK_THROW((version_major == 1 && version_minor == 0), std::runtime_error,
                                "Unsupported .npy file version");

  // Read header length
  uint16_t header_len;
  is.read(reinterpret_cast<char*>(&header_len), 2);

  // Read header data
  std::vector<char> header_data(header_len);
  is.read(header_data.data(), header_len);

  // Parse header
  std::string header_str(header_data.data(), header_len);
  std::stringstream header(header_str);
  header.ignore(header_str.find('{') + 1);  // Skip to the start of the dictionary
  std::string descr, fortran_order, shape_str;
  char c;
  while (header.get(c) && c != '}') {
    std::string key;
    while (c != ':' && c != '}') {
      key += c;
      header.get(c);
    }
    key.erase(key.find_last_not_of(' ') + 1);
    key.erase(0, key.find_first_not_of(' '));
    if (key == "'descr'") {
      header.get(c);  // Skip space
      header.get(c);  // Skip first quote
      while (header.get(c) && c != '\'') {
        descr += c;
      }
    } else if (key == "'fortran_order'") {
      header.get(c);  // Skip space
      while (header.get(c) && c != ',') {
        fortran_order += c;
      }

      MATHPRIM_INTERNAL_CHECK_THROW(fortran_order == "False", std::runtime_error,
                                    "Unsupported fortran_order != False in .npy file");
    } else if (key == "'shape'") {
      header.get(c);  // Skip space
      while (header.get(c) && c != ')') {
        shape_str += c;
      }
      shape_str += ')';
    } else if (key == "") {
      // Done.
      break;
    } else {
      MATHPRIM_INTERNAL_CHECK_THROW(false, std::runtime_error, "Unsupported key in .npy header [[[" + key + "]]]");
    }
    header.get(c);  // Skip comma or closing brace
  }

  // Extract shape
  std::vector<index_t> shape;
  std::stringstream shape_stream(shape_str);
  shape_stream.ignore(1);  // Skip '('
  index_t dim;
  while (shape_stream >> dim) {
    shape.push_back(dim);
    shape_stream.ignore(1);  // Skip ','
  }

  // Validate dtype
  if constexpr (std::is_same_v<Scalar, float>) {
    MATHPRIM_INTERNAL_CHECK_THROW(descr == "<f4", std::runtime_error, "Mismatched dtype: expected '<f4' for float");
  } else if constexpr (std::is_same_v<Scalar, double>) {
    MATHPRIM_INTERNAL_CHECK_THROW(descr == "<f8", std::runtime_error, "Mismatched dtype: expected '<f8' for double");
  } else {
    static_assert(internal::always_false_v<Scalar>, "Unsupported scalar type");
  }

  // Allocate buffer
  dshape<Ndim> shape_mp;
  for (index_t i = 0; i < ndim(shape_mp); ++i) {
    shape_mp[i] = shape[i];
  }
  buffer_type buffer = make_buffer<Scalar>(shape_mp);
  size_t total_elements = buffer.numel();

  // Read binary data
  is.read(reinterpret_cast<char*>(buffer.data()), total_elements * sizeof(Scalar));

  return buffer;
}
}  // namespace mathprim::io