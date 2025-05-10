#pragma once

#include "npu_op_test_util.h"

class EmbeddingOpTest : public OpTestBase<EmbeddingOpTest> {
public:
  void Init(size_t max_index_num, size_t vocab_size, size_t hidden_size);
  bool Run(size_t test_size);
  void CleanUp();


  size_t max_index_num{0};
  size_t vocab_size{0};
  size_t hidden_dim{0};

  void *dev_index_s32{nullptr};
  void *dev_weight_u16{nullptr};
  void *dev_output_u16{nullptr};

  int32_t *host_input{nullptr};
  uint16_t *host_weight{nullptr};
  uint16_t *host_output{nullptr};
  uint16_t *golden_u16{nullptr};
};
