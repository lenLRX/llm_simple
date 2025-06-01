#pragma once

#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <iostream>
#include <sstream>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

#include "defs.hpp"

bool LoadBinaryFile(const char *path, void *buffer, size_t data_size);

template <typename TensorType>
std::string TensorShapeToStr(TensorType &tensor) {
  auto &d = tensor.dimensions();
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < d.size(); ++i) {
    ss << d[i];
    if (i < d.size() - 1) {
      ss << ",";
    }
  }
  ss << ")";
  return ss.str();
}

int64_t get_current_us();

uint16_t fp32_to_bfloat16(uint32_t fp32_bits);

bool has_awq_quantization(const nlohmann::json &j);

template <typename EigenTy, int ndim>
std::string
print_tensor_typed(EigenTy *ptr,
                   const Eigen::array<Eigen::Index, ndim> &tensor_dim,
                   const Eigen::array<Eigen::Index, ndim> &print_extend) {

  Eigen::TensorMap<
      Eigen::Tensor<EigenTy, ndim, Eigen::RowMajor | Eigen::DontAlign>>
      t_map(static_cast<EigenTy *>(ptr), tensor_dim);

  Eigen::array<Eigen::Index, ndim> print_offsets{};
  Eigen::Tensor<EigenTy, ndim, Eigen::RowMajor | Eigen::DontAlign> print_slice =
      t_map.slice(print_offsets, print_extend);
  std::stringstream ss;
  ss << print_slice;
  return ss.str();
}

template <int ndim>
std::string print_tensor(void *ptr, DataType dtype,
                         const Eigen::array<Eigen::Index, ndim> &tensor_dim,
                         const Eigen::array<Eigen::Index, ndim> &print_extend) {
  switch (dtype) {
  case DT_FLOAT16:
    return print_tensor_typed<Eigen::half, ndim>(
        static_cast<Eigen::half *>(ptr), tensor_dim, print_extend);
    break;
  case DT_BFLOAT16:
    return print_tensor_typed<Eigen::bfloat16, ndim>(
        static_cast<Eigen::bfloat16 *>(ptr), tensor_dim, print_extend);
    break;
  case DT_FLOAT32:
    return print_tensor_typed<float, ndim>(static_cast<float *>(ptr),
                                           tensor_dim, print_extend);
    break;
  default:
    break;
  }

  return fmt::format("print_tensor unsupported dtype: {}", dtype);
}

template <typename IndexTy, int ndim>
std::string print_tensor(void *ptr, DataType dtype,
                         const std::array<IndexTy, ndim> &tensor_dim_s64,
                         const std::array<IndexTy, ndim> &print_extend_s64) {
  Eigen::array<Eigen::Index, ndim> tensor_dim;
  Eigen::array<Eigen::Index, ndim> print_extend;
  std::transform(tensor_dim_s64.begin(), tensor_dim_s64.end(),
                 tensor_dim.begin(),
                 [](size_t x) { return static_cast<Eigen::Index>(x); });
  std::transform(print_extend_s64.begin(), print_extend_s64.end(),
                 print_extend.begin(),
                 [](size_t x) { return static_cast<Eigen::Index>(x); });
  return print_tensor<2>(ptr, dtype, tensor_dim, print_extend);
}


