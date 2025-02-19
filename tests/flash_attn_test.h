#pragma once
#include "npu_op_test_util.h"

class FlashAttentionOpTest : public OpTestBase<FlashAttentionOpTest> {
public:
  void Init(size_t max_m, size_t max_n, size_t head_num, size_t head_dim);
  bool Run(size_t m, size_t n, size_t offset);
  void CleanUp();

  size_t max_m{0};
  size_t max_n{0};

  size_t head_dim{0};
  size_t head_num{0};
  size_t hidden_dim{0};

  size_t max_q_buffer_size;
  size_t max_k_buffer_size;
  size_t max_qk_buffer_size;
  size_t max_pv_buffer_size;
  size_t max_softmax_buffer_size;

  size_t max_v_buffer_size;
  size_t max_o_buffer_size;

  void *dev_q_f16{nullptr};
  void *dev_k_f16{nullptr};
  void *dev_v_f16{nullptr};
  void *dev_o_f16{nullptr};

  float *host_q{nullptr};
  float *host_k{nullptr};
  float *host_qk{nullptr};
  float *host_pv{nullptr};
  float *host_mask{nullptr};
  float *host_softmax_output{nullptr};


  float *host_v{nullptr};
  float *host_o{nullptr};
  Eigen::half *host_q_f16{nullptr};
  Eigen::half *host_k_f16{nullptr};
  Eigen::half *host_v_f16{nullptr};
  Eigen::half *host_o_f16{nullptr};

  float *golden_output{nullptr};

};

