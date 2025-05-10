#pragma once

#include "npu_op_test_util.h"

template <typename EigenTy>
class RoPESingleOpTest : public OpTestBase<RoPESingleOpTest<EigenTy>> {
public:
  void Init(size_t max_seq_len, size_t head_dim, size_t head_num, bool is_neox);
  bool Run(size_t offset, size_t test_size);
  void CleanUp();

  size_t max_seq_len{0};
  size_t head_dim{0};
  size_t head_num{0};
  bool is_neox{true};

  void *dev_input_f16{nullptr};
  void *dev_output_f16{nullptr};
  void *dev_freq_cis{nullptr};

  float *host_input{nullptr};
  float *host_output{nullptr};
  EigenTy *host_input_f16{nullptr};
  EigenTy *host_output_f16{nullptr};
  float *golden_fp32{nullptr};
  float *host_freq_cis{nullptr};
};
