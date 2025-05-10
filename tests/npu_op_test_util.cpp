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

void make_random_bytes(void* ptr, std::size_t size) {
    unsigned char* byte_ptr = static_cast<unsigned char*>(ptr);
    std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<unsigned short> distribution(0, 255);

    for (std::size_t i = 0; i < size; ++i) {
        byte_ptr[i] = static_cast<unsigned char>(distribution(generator));
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

bool all_close2(float *output_buffer, float *golden_buffer, size_t size,
                float abs_err, float relative_err) {
  size_t failed = 0;
  for (size_t i = 0; i < size; ++i) {
    float a = output_buffer[i];
    float b = golden_buffer[i];

    float abs_diff = std::fabs(a - b);
    float max_abs_val = std::max(std::fabs(a), std::fabs(b));
    if (abs_diff > abs_err && (abs_diff / max_abs_val) > relative_err) {
      failed += 1;
    }
  }
  return (static_cast<double>(failed) / static_cast<double>(size)) <
         relative_err;
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

void InitFreqCIS(float *freq_cis, int head_dim, int max_seq_len) {
  const float theta = 10000.0f;
  int freq_len = head_dim / 2;
  float *freq = new float[freq_len];

  for (int i = 0; i < freq_len; ++i) {
    freq[i] =
        1.0f /
        (powf(theta, static_cast<float>(i * 2) / static_cast<float>(head_dim)));
  }

  float *t = new float[max_seq_len];
  for (int i = 0; i < max_seq_len; ++i) {
    t[i] = static_cast<float>(i);
  }

  float *freq_outer = new float[freq_len * max_seq_len];

  // max_seq_len row, freq_len column
  for (int i = 0; i < max_seq_len; ++i) {
    for (int j = 0; j < freq_len; ++j) {
      freq_outer[i * freq_len + j] = t[i] * freq[j];
    }
  }

  for (int i = 0; i < max_seq_len * freq_len; ++i) {
    freq_cis[i * 2] = std::cos(freq_outer[i]);
    freq_cis[i * 2 + 1] = std::sin(freq_outer[i]);
  }

  delete[] freq;
  delete[] t;
  delete[] freq_outer;
}



