#include <fstream>
#include <spdlog/spdlog.h>

#include "util.h"

bool LoadBinaryFile(const char *path, void *buffer, size_t data_size) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    spdlog::critical("failed to open {}", path);
    return false;
  }

  ifs.read((char *)buffer, data_size);

  if (ifs.fail()) {
    spdlog::critical("failed to read {} of size {}", path, data_size);
    return false;
  }

  spdlog::debug("loaded binary {} size {}", path, data_size);

  return true;
}

int64_t get_current_us() {
  auto now = std::chrono::system_clock::now();

  auto micros = std::chrono::duration_cast<std::chrono::microseconds>(
      now.time_since_epoch());
  return micros.count();
}

uint16_t fp32_to_bfloat16(uint32_t fp32_bits) {
  // Step 1: 分解FP32的组成部分
  const uint32_t sign = (fp32_bits >> 31) & 0x1;      // 符号位
  const uint32_t exponent = (fp32_bits >> 23) & 0xFF; // 指数部分
  const uint32_t mantissa = fp32_bits & 0x007FFFFF; // 原尾数（不含隐含的1）

  // Step 2: 处理尾数舍入（含隐含的1，共24位）
  const uint32_t full_mantissa =
      mantissa | 0x00800000; // 添加隐含的1构成24位有效数
  const uint32_t trunc_bits = full_mantissa & 0x00FFFFFF; // 确保24位有效数

  // 提取需要保留的高8位和需要截断的低16位
  const uint32_t mant_high = (trunc_bits >> 16) & 0xFF; // 高8位（含隐含的1）
  const uint32_t mant_low = trunc_bits & 0xFFFF; // 低16位用于舍入判断

  // Round to Nearest, Ties to Even (RNTE)
  uint32_t rounded_mant = mant_high;
  if (mant_low > 0x8000) { // 大于中间值，进位
    rounded_mant += 1;
  } else if (mant_low == 0x8000) { // 等于中间值，判断奇偶
    if ((mant_high & 0x1) != 0) {  // 奇数则进位
      rounded_mant += 1;
    }
  } // 小于中间值直接截断

  // Step 3: 处理尾数进位溢出（如0xFF -> 0x100）
  uint32_t new_exponent = exponent;
  if (rounded_mant > 0xFF) { // 进位导致尾数溢出
    new_exponent += 1;       // 指数+1
    rounded_mant = 0x80;     // 尾数重置为隐含的1 + 0x00（即0x80）
  }

  // Step 4: 组合BFloat16的二进制
  // 若指数溢出（>0xFF），则结果为无穷大（这里简化为饱和处理）
  const uint32_t bf16_exponent = (new_exponent > 0xFF) ? 0xFF : new_exponent;
  const uint32_t bf16_mantissa =
      (rounded_mant & 0x7F); // 取低7位（去掉隐含的1）

  return ((sign << 15) | (bf16_exponent << 7) | bf16_mantissa);
}

bool has_awq_quantization(const nlohmann::json &j) {
  return j.contains("quantization_config") &&
         j["quantization_config"].is_object() &&
         j["quantization_config"].value("quant_method", "") == "awq";
}
