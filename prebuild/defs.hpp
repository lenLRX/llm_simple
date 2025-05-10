#pragma once

#include <cstdint>
#include <stdlib.h>

enum DeviceType { DEV_CPU = 0, DEV_GPU, DEV_NPU };

enum DataType : int32_t {
  DT_UINT8 = 0,
  DT_INT8,
  DT_UINT32,
  DT_INT32,
  DT_FLOAT16,
  DT_BFLOAT16,
  DT_FLOAT32,
  DT_UINT64,
  DT_INT64,
};

inline static size_t SizeOfTensor(size_t size, DataType dt) {
  switch (dt) {
  case DT_INT8:
  case DT_UINT8:
    return size * sizeof(uint8_t);
    break;
  case DT_INT32:
  case DT_UINT32:
    return size * sizeof(uint32_t);
  case DT_FLOAT16:
  case DT_BFLOAT16:
    return size * sizeof(uint16_t);
  case DT_FLOAT32:
    return size * sizeof(uint32_t);
  case DT_UINT64:
  case DT_INT64:
    return size * sizeof(uint64_t);
  default:
    break;
  }
  return -1;
}
