#include "npu_op_test_util.h"

void make_random_float(float *buffer, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  for (size_t i = 0; i < size; ++i) {
    buffer[i] = dis(gen);
  }
}

void make_random_float_uint4(float *buffer, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 15);

  for (size_t i = 0; i < size; ++i) {
    buffer[i] = static_cast<float>(dis(gen));
  }
}

bool all_close(float *output_buffer, float *golden_buffer, size_t size,
               float abs_err, float relative_err) {
  for (size_t i = 0; i < size; ++i) {
    float a = output_buffer[i];
    float b = golden_buffer[i];

    float abs_diff = std::fabs(a - b);
    float max_abs_val = std::max(std::fabs(a), std::fabs(b));
    if (abs_diff > abs_err && (abs_diff / max_abs_val) > relative_err) {
      std::cout << "all_close failed, output [" << i << "] :" << a << " vs "
                << b << std::endl;
      return false;
    }
  }
  return true;
}

bool all_close_inf(float *output_buffer, float *golden_buffer, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    float a = output_buffer[i];
    float b = golden_buffer[i];

    if (std::isinf(b)) {
      if (std::isinf(a) && std::signbit(a) == std::signbit(b)) {
        continue;
      }
      std::cout << "all_close failed, output [" << i << "] :" << a << " vs "
                << b << std::endl;
      return false;
    }

    float abs_diff = std::fabs(a - b);
    float max_abs_val = std::max(std::fabs(a), std::fabs(b));

    if (abs_diff > 0.001f && (abs_diff / max_abs_val) > 0.001f) {
      std::cout << "all_close failed, output [" << i << "] :" << a << " vs "
                << b << std::endl;
      return false;
    }
  }
  return true;
}

void read_binary(const char *path, void *data, size_t size) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    std::cout << "failed to open " << path << std::endl;
  }
  ifs.read((char *)data, size);
}

void write_binary(const char *path, void *data, size_t size) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    std::cout << "failed to open " << path << std::endl;
  }
  ofs.write((char *)data, size);
}
