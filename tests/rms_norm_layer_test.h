#pragma once

#include "npu_op_test_util.h"

class RMSNormOpTest : public OpTestBase<RMSNormOpTest> {
public:
  void Init(size_t max_first_dim, size_t max_last_dim);
  bool Run(size_t first_dim, size_t last_dim, float eps);
  void CleanUp();

  size_t max_first_dim{0};
  size_t max_last_dim{0};

  void *dev_input_f16{nullptr};
  void *dev_output_f16{nullptr};
  void *dev_weight_f16{nullptr};

  float *host_input{nullptr};
  float *host_output{nullptr};
  Eigen::half *host_input_f16{nullptr};
  Eigen::half *host_output_f16{nullptr};
  float *golden_fp32{nullptr};
  float *weight_fp32{nullptr};
  Eigen::half *weight_fp16{nullptr};
};
