#pragma once
#include "npu_op_test_util.h"

class GemmAWQ4BitOpTest : public OpTestBase<GemmAWQ4BitOpTest> {
public:
  void Init(size_t max_m, size_t max_n, size_t max_k, bool bias);
  bool Run(size_t m, size_t n, size_t k);
  void CleanUp();

  size_t max_m{0};
  size_t max_n{0};
  size_t max_k{0};
  bool bias{false};

  size_t max_lhs_buffer_size{0};
  size_t max_weight_buffer_size{0};
  size_t max_zero_buffer_size{0};
  size_t max_scale_buffer_size{0};
  size_t max_output_buffer_size{0};
  size_t max_bias_buffer_size{0};

  size_t max_ffn_hidden{0};
  size_t group_size{128};

  void *dev_lhs_f16{nullptr};
  void *dev_weight_s4{nullptr};
  void *dev_zero_fp16{nullptr};
  void *dev_scale_fp16{nullptr};
  void *dev_output_f16{nullptr};
  void *dev_bias_f32{nullptr};

  float *host_lhs{nullptr};
  float *host_rhs{nullptr};
  float *host_rhs_nz{nullptr};
  float *host_zero{nullptr};
  float *host_scale{nullptr};
  float *host_output{nullptr};
  float *host_bias{nullptr};
  Eigen::half *host_lhs_f16{nullptr};
  uint8_t *host_weight_s4{nullptr};
  Eigen::half *host_zero_f16{nullptr};
  Eigen::half *host_scale_f16{nullptr};
  Eigen::half *host_output_f16{nullptr};
  float *golden_fp32{nullptr};
};
