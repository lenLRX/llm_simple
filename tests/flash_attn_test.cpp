#include <cmath>
#include <gtest/gtest.h>
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

#include "defs.hpp"
#include "flash_attn_test.h"
#include "npu_op_test_util.h"
#include "npu_ops.h"

void FlashAttentionOpTest::Init(size_t max_m, size_t max_n, size_t head_num,
                                size_t head_dim) {
  this->max_m = max_m;
  this->max_n = max_n;
  this->head_dim = head_dim;
  this->head_num = head_num;
  this->hidden_dim = head_dim * head_num;

  max_q_buffer_size = max_m * hidden_dim;
  max_k_buffer_size = max_n * hidden_dim;
  max_qk_buffer_size = max_m * max_n * head_num;
  max_pv_buffer_size = max_m * hidden_dim;
  max_softmax_buffer_size = max_m * max_n * head_num;
  max_v_buffer_size = max_n * hidden_dim;
  max_o_buffer_size = max_m * hidden_dim;

  host_q = new float[max_q_buffer_size];
  host_k = new float[max_k_buffer_size];
  host_qk = new float[max_qk_buffer_size];
  host_pv = new float[max_pv_buffer_size];
  host_mask = new float[max_m * max_n];
  host_softmax_output = new float[max_softmax_buffer_size];
  host_v = new float[max_v_buffer_size];
  host_o = new float[max_o_buffer_size];

  host_q_f16 = new Eigen::half[max_q_buffer_size];
  host_k_f16 = new Eigen::half[max_k_buffer_size];
  host_v_f16 = new Eigen::half[max_v_buffer_size];
  host_o_f16 = new Eigen::half[max_o_buffer_size];

  golden_output = new float[max_o_buffer_size];

  CHECK_ACL(aclrtMalloc((void **)&dev_q_f16,
                        max_q_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_k_f16,
                        max_k_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_v_f16,
                        max_v_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_o_f16,
                        max_o_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
}

bool FlashAttentionOpTest::Run(size_t m, size_t n, size_t offset) {
  spdlog::info(
      "FlashAttentionOpTest::Run {{m={},n={},offset={},hidden_dim={}}}", m, n,
      offset, hidden_dim);

  size_t q_element_cnt = m * hidden_dim;
  size_t k_element_cnt = n * hidden_dim;
  size_t v_element_cnt = n * hidden_dim;
  size_t o_element_cnt = m * n;

  make_random_float(host_q, q_element_cnt);
  make_random_float(host_k, k_element_cnt);
  make_random_float(host_v, v_element_cnt);

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_q_map((float *)host_q, m, head_num, head_dim);
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_k_map((float *)host_k, n, head_num, head_dim);
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_v_map((float *)host_v, n, head_num, head_dim);

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_q_fp16_map((Eigen::half *)host_q_f16, m, head_num, head_dim);
  input_q_fp16_map = input_q_map.cast<Eigen::half>();
  input_q_map = input_q_fp16_map.cast<float>();

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_k_fp16_map((Eigen::half *)host_k_f16, n, head_num, head_dim);
  input_k_fp16_map = input_k_map.cast<Eigen::half>();
  input_k_map = input_k_fp16_map.cast<float>();

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_v_fp16_map((Eigen::half *)host_v_f16, n, head_num, head_dim);
  input_v_fp16_map = input_v_map.cast<Eigen::half>();
  input_v_map = input_v_fp16_map.cast<float>();

  float qk_scale = 1 / sqrtf(static_cast<float>(head_dim));

  // (bs, nh, seqlen, hd) @ (bs, nh, hd, cache_len+seqlen) => bs, nh, seqlen,
  // cache_len+seqlen
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      q_matmul_k_map(static_cast<float *>(host_qk), head_num, m, n);

  Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign> q_emb_trans(
      head_num, m, head_dim);
  q_emb_trans = input_q_map.shuffle(Eigen::array<int, 3>({1, 0, 2}));
  Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign> k_emb_trans(
      head_num, head_dim, n);
  k_emb_trans = input_k_map.shuffle(Eigen::array<int, 3>({1, 2, 0}));

  // todo make mask
  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      mask_map(static_cast<float *>(host_mask), m, n);

  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      if (col > (row + offset)) {
        host_mask[row * n + col] = -std::numeric_limits<float>::infinity();
      } else {
        host_mask[row * n + col] = 0.0f;
      }
    }
  }

  // tensor contraction does not support batch matmul
  // need a for loop to bmm
  // https://gitlab.com/libeigen/eigen/-/issues/2449
  Eigen::array<Eigen::IndexPair<int>, 1> qk_product_dims = {
      Eigen::IndexPair<int>(1, 0)};
  for (int i = 0; i < head_num; ++i) {
    q_matmul_k_map.chip<0>(i) = q_emb_trans.chip<0>(i).contract(
                                    k_emb_trans.chip<0>(i), qk_product_dims) +
                                mask_map;
  }

  q_matmul_k_map = (q_matmul_k_map * q_matmul_k_map.constant(qk_scale));

  auto hs = head_num * m;

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      softmax_input_map(static_cast<float *>(host_qk), hs, n);

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      softmax_output_map(static_cast<float *>(host_softmax_output), head_num, m,
                         n);

  auto softmax_input_max = softmax_input_map.maximum(Eigen::array<size_t, 1>{1})
                               .eval()
                               .reshape(Eigen::array<size_t, 2>{hs, 1})
                               .broadcast(Eigen::array<size_t, 2>{1, n});

  auto softmax_input_diff =
      (softmax_input_map - softmax_input_max).exp().eval();

  auto softmax_input_sum = softmax_input_diff.sum(Eigen::array<size_t, 1>{1})
                               .eval()
                               .reshape(Eigen::array<size_t, 2>{hs, 1})
                               .broadcast(Eigen::array<size_t, 2>{1, n});

  softmax_output_map = (softmax_input_diff / softmax_input_sum)
                           .reshape(std::array<size_t, 3>{head_num, m, n});

  // (seq_length, n_heads, head_dim) -> (n_heads, seq_length, head_dim)
  auto vmap_trans = input_v_map.shuffle(Eigen::array<int, 3>({1, 0, 2}));

  Eigen::array<Eigen::IndexPair<int>, 1> output_product_dims = {
      Eigen::IndexPair<int>(1, 0)};
  // tmp_output: (n_heads, seq_length, head_dim)
  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      tmp_output_map(static_cast<float *>(host_pv), head_num, m, head_dim);
  for (int i = 0; i < head_num; ++i) {
    tmp_output_map.chip<0>(i) = softmax_output_map.chip<0>(i).contract(
        vmap_trans.chip<0>(i), output_product_dims);
  }

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      tmp_output_tensor_map(static_cast<float *>(golden_output), m, hidden_dim);

  // tmp_output: (n_heads, seq_length, head_dim) -> (seq_length, n_heads,
  // head_dim)
  tmp_output_tensor_map =
      tmp_output_map.shuffle(Eigen::array<int, 3>({1, 0, 2}))
          .reshape(std::array<size_t, 2>{m, hidden_dim});

  CHECK_ACL(aclrtMemcpy(dev_q_f16, m * hidden_dim * sizeof(aclFloat16),
                        host_q_f16, m * hidden_dim * sizeof(aclFloat16),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_k_f16, n * hidden_dim * sizeof(aclFloat16),
                        host_k_f16, n * hidden_dim * sizeof(aclFloat16),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_v_f16, n * hidden_dim * sizeof(aclFloat16),
                        host_v_f16, n * hidden_dim * sizeof(aclFloat16),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemset(dev_o_f16, m * hidden_dim * sizeof(aclFloat16), 0,
                        m * hidden_dim * sizeof(aclFloat16)));
  spdlog::info("launch kernel");
  npu_flash_attn_layer(dev_o_f16, dev_q_f16, dev_k_f16, dev_v_f16, m, n, offset,
                       head_num, head_dim, DT_FLOAT16, stream);

  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(host_o_f16, m * hidden_dim * sizeof(aclFloat16),
                        dev_o_f16, m * hidden_dim * sizeof(aclFloat16),
                        ACL_MEMCPY_DEVICE_TO_HOST));

  write_binary("fa_input_q_f16.bin", host_q_f16,
               m * hidden_dim * sizeof(aclFloat16));
  write_binary("fa_input_k_f16.bin", host_k_f16,
               n * hidden_dim * sizeof(aclFloat16));
  write_binary("fa_input_v_f16.bin", host_v_f16,
               n * hidden_dim * sizeof(aclFloat16));

  write_binary("fa_out_f16.bin", host_o_f16,
               m * hidden_dim * sizeof(aclFloat16));
  write_binary("fa_golden_out_fp32.bin", golden_output,
               m * hidden_dim * sizeof(aclFloat16));

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      output_fp32_map((float *)host_o, m, hidden_dim);

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
      output_fp16_map((Eigen::half *)host_o_f16, m, hidden_dim);
  output_fp32_map = output_fp16_map.cast<float>();

  return all_close(host_o, golden_output, m * hidden_dim);
}

void FlashAttentionOpTest::CleanUp() {
  delete[] host_q;
  delete[] host_k;
  delete[] host_qk;
  delete[] host_pv;
  delete[] host_mask;
  delete[] host_softmax_output;
  delete[] host_v;
  delete[] host_o;
  delete[] golden_output;
  delete[] host_q_f16;
  delete[] host_k_f16;
  delete[] host_v_f16;
  delete[] host_o_f16;

  CHECK_ACL(aclrtFree(dev_q_f16));
  CHECK_ACL(aclrtFree(dev_k_f16));
  CHECK_ACL(aclrtFree(dev_v_f16));
  CHECK_ACL(aclrtFree(dev_o_f16));
}
