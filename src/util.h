#pragma once

#include <chrono>
#include <sstream>
#include <stddef.h>
#include <stdint.h>
#include <string>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

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

