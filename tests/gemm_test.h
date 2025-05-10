#pragma once

#include "npu_ops.h"
#include "npu_op_test_util.h"

template <typename EigenTy>
class GemmOpTest : public OpTestBase<GemmOpTest<EigenTy>> {
public:
  void Init(size_t max_m, size_t max_n, size_t max_k, bool bias);
  bool Run(size_t m, size_t n, size_t k);
  void CleanUp();

  size_t max_m{0};
  size_t max_n{0};
  size_t max_k{0};
  bool bias{false};

  size_t max_lhs_buffer_size{0};
  size_t max_rhs_buffer_size{0};
  size_t max_bias_buffer_size{0};
  size_t max_output_buffer_size{0};


  void *dev_lhs{nullptr};
  void *dev_rhs{nullptr};
  void *dev_bias{nullptr};
  void *dev_output{nullptr};

  float *host_lhs{nullptr};
  float *host_rhs{nullptr};
  float *host_bias{nullptr};
  float *host_output{nullptr};
  EigenTy *host_lhs_b16{nullptr};
  EigenTy *host_rhs_b16{nullptr};
  EigenTy *host_rhs_nz_b16{nullptr};

  EigenTy *host_output_b16{nullptr};
  EigenTy *host_golden_b16{nullptr};
  float *golden_fp32{nullptr};
};
