#include <cmath>
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

#include "npu_op_test_util.h"
#include "npu_ops.h"
#include "rope_single_layer_test.h"

template <typename EigenTy>
void RoPESingleOpTest<EigenTy>::Init(size_t max_seq_len, size_t head_dim,
                                     size_t head_num, bool is_neox) {
  this->max_seq_len = max_seq_len;
  this->head_dim = head_dim;
  this->head_num = head_num;

  int freq_cis_size = head_dim * max_seq_len;
  host_freq_cis = new float[freq_cis_size];
  InitFreqCIS(host_freq_cis, head_dim, max_seq_len);

  size_t max_buffer_size = head_num * head_dim * max_seq_len;

  host_input = new float[max_buffer_size];
  host_output = new float[max_buffer_size];
  host_input_f16 = new EigenTy[max_buffer_size];
  host_output_f16 = new EigenTy[max_buffer_size];
  golden_fp32 = new float[max_buffer_size];

  CHECK_ACL(aclrtMalloc((void **)&dev_input_f16,
                        max_buffer_size * sizeof(EigenTy),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_output_f16,
                        max_buffer_size * sizeof(EigenTy),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_freq_cis, freq_cis_size * sizeof(float),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(dev_freq_cis, freq_cis_size * sizeof(float),
                        host_freq_cis, freq_cis_size * sizeof(float),
                        ACL_MEMCPY_HOST_TO_DEVICE));
}

template <typename EigenTy>
bool RoPESingleOpTest<EigenTy>::Run(size_t offset, size_t seq_len) {
  spdlog::info("{} max_seq_len {} head_num {} head_dim {} offset {} seq_len {}",
               __PRETTY_FUNCTION__, max_seq_len, head_num, head_dim, offset,
               seq_len);

  make_random_float(host_input, head_dim * head_num * seq_len);

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      golden_fp32_map((float *)golden_fp32, seq_len, head_num, head_dim);

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_map((float *)host_input, seq_len, head_num, head_dim);

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      freq_cis_map((float *)host_freq_cis + offset * head_dim, seq_len, 1,
                   head_dim);

  Eigen::TensorMap<
      Eigen::Tensor<EigenTy, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_fp16_map((EigenTy *)host_input_f16, seq_len, head_num, head_dim);
  input_fp16_map = input_map.cast<EigenTy>();
  input_map = input_fp16_map.template cast<float>();

  int freq_len = head_dim / 2;
  int hidden_dim = head_num * head_dim;

  for (int s = 0; s < seq_len; ++s) {
    for (int n = 0; n < head_num; ++n) {
      for (int f = 0; f < freq_len; ++f) {
        float fc = host_freq_cis[(s + offset) * freq_len * 2 + 2 * f];
        float fd = host_freq_cis[(s + offset) * freq_len * 2 + 2 * f + 1];

        int hidden_offset = s * hidden_dim + n * head_dim;

        float qa = host_input[hidden_offset + (is_neox ? f : (2 * f))];
        float qb = host_input[hidden_offset + (is_neox ? (freq_len + f) : (2 * f + 1))];


        golden_fp32[hidden_offset + (is_neox ? f : (2 * f))] = qa * fc - qb * fd;
        golden_fp32[hidden_offset + (is_neox ? (freq_len + f) : (2 * f + 1))] = qa * fd + qb * fc;

      }
    }
  }

  golden_fp32_map = golden_fp32_map.template cast<EigenTy>().template cast<float>();

  CHECK_ACL(aclrtMemcpy(
      dev_input_f16, seq_len * head_num * head_dim * sizeof(EigenTy),
      host_input_f16, seq_len * head_num * head_dim * sizeof(EigenTy),
      ACL_MEMCPY_HOST_TO_DEVICE));

  npu_rope_single_layer(dev_output_f16, dev_freq_cis, dev_input_f16, offset,
                        seq_len, head_num, head_dim * head_num, is_neox,
                        GetDataType<EigenTy>(), this->stream);
  CHECK_ACL(aclrtSynchronizeStream(this->stream));

  CHECK_ACL(aclrtMemcpy(
      host_output_f16, seq_len * head_num * head_dim * sizeof(EigenTy),
      dev_output_f16, seq_len * head_num * head_dim * sizeof(EigenTy),
      ACL_MEMCPY_DEVICE_TO_HOST));

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      output_fp32_map((float *)host_output, seq_len, head_num, head_dim);

  Eigen::TensorMap<
      Eigen::Tensor<EigenTy, 3, Eigen::RowMajor | Eigen::DontAlign>>
      output_fp16_map((EigenTy *)host_output_f16, seq_len, head_num, head_dim);
  output_fp32_map = output_fp16_map.template cast<float>();
  return all_close(host_output, golden_fp32, seq_len * head_num * head_dim);
}

template <typename EigenTy> void RoPESingleOpTest<EigenTy>::CleanUp() {

  delete[] host_input;
  delete[] host_output;
  delete[] host_input_f16;
  delete[] host_output_f16;
  delete[] host_freq_cis;
  delete[] golden_fp32;

  CHECK_ACL(aclrtFree(dev_input_f16));
  CHECK_ACL(aclrtFree(dev_output_f16));
  CHECK_ACL(aclrtFree(dev_freq_cis));
}

template class RoPESingleOpTest<Eigen::half>;
template class RoPESingleOpTest<Eigen::bfloat16>;
