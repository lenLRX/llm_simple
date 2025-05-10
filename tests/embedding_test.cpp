#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

#include <acl/acl.h>
#include <random>

#include <Eigen/Core>
#include <boost/program_options.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "embedding_test.h"
#include "npu_op_test_util.h"
#include "npu_ops.h"

void EmbeddingOpTest::Init(size_t max_index_num, size_t vocab_size,
                           size_t hidden_dim) {
  this->max_index_num = max_index_num;
  this->vocab_size = vocab_size;
  this->hidden_dim = hidden_dim;

  host_input = new int32_t[max_index_num];
  host_weight = new uint16_t[vocab_size * hidden_dim];
  host_output = new uint16_t[max_index_num * hidden_dim];
  golden_u16 = new uint16_t[max_index_num * hidden_dim];

  CHECK_ACL(aclrtMalloc((void **)&dev_index_s32,
                        max_index_num * sizeof(int32_t),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_weight_u16,
                        vocab_size * hidden_dim * sizeof(uint16_t),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_output_u16,
                        max_index_num * hidden_dim * sizeof(uint16_t),
                        ACL_MEM_MALLOC_HUGE_FIRST));

  make_random_bytes((void *)host_weight,
                    vocab_size * hidden_dim * sizeof(uint16_t));
  CHECK_ACL(aclrtMemcpy(
      dev_weight_u16, vocab_size * hidden_dim * sizeof(uint16_t), host_weight,
      vocab_size * hidden_dim * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE));
}

bool EmbeddingOpTest::Run(size_t test_size) {
  spdlog::info("{} vocab_size {} hidden_dim {} test_size {}",
               __PRETTY_FUNCTION__, vocab_size, hidden_dim, test_size);

  std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<int32_t> distribution(0, vocab_size - 1);

  for (std::size_t i = 0; i < test_size; ++i) {
    host_input[i] = static_cast<int32_t>(distribution(generator));
  }

  CHECK_ACL(aclrtMemcpy(dev_index_s32, test_size * sizeof(int32_t), host_input,
                        test_size * sizeof(int32_t),
                        ACL_MEMCPY_HOST_TO_DEVICE));

  npu_embedding_layer(dev_output_u16, dev_weight_u16, dev_index_s32, test_size,
                      hidden_dim, DT_FLOAT16, stream);

  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(
      host_output, test_size * hidden_dim * sizeof(uint16_t), dev_output_u16,
      test_size * hidden_dim * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));

  for (int i = 0; i < test_size; ++i) {
    int32_t idx = host_input[i];
    for (int j = 0; j < hidden_dim; ++j) {
      auto a = host_weight[idx * hidden_dim + j];
      auto b = host_output[i * hidden_dim + j];
      if (a != b) {
        std::cout << "all_close failed, index " << idx << " output [" << i
                  << "," << j << "] :" << a << " vs " << b << std::endl;
        return false;
      }
    }
  }

  return true;
}

void EmbeddingOpTest::CleanUp() {

  delete[] host_input;
  delete[] host_output;
  delete[] host_weight;
  delete[] golden_u16;

  CHECK_ACL(aclrtFree(dev_index_s32));
  CHECK_ACL(aclrtFree(dev_output_u16));
  CHECK_ACL(aclrtFree(dev_weight_u16));
}
