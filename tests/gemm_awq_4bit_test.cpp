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

#include "gemm_awq_4bit_test.h"
#include "npu_op_test_util.h"
#include "npu_ops.h"

void GemmAWQ4BitOpTest::Init(size_t max_m, size_t max_n, size_t max_k,
                             bool bias) {
  this->max_m = max_m;
  this->max_n = max_n;
  this->max_k = max_k;
  this->bias = bias;

  max_lhs_buffer_size = max_m * max_k;
  max_weight_buffer_size = max_k * max_n;
  max_zero_buffer_size = max_k * max_n / group_size;
  max_scale_buffer_size = max_k * max_n / group_size;
  max_output_buffer_size = max_k * max_n;
  max_bias_buffer_size = max_n;

  host_lhs = new float[max_lhs_buffer_size];
  host_rhs = new float[max_weight_buffer_size];
  host_rhs_nz = new float[max_weight_buffer_size];
  host_zero = new float[max_zero_buffer_size];
  host_scale = new float[max_scale_buffer_size];
  host_output = new float[max_output_buffer_size];
  host_bias = new float[max_bias_buffer_size];
  host_lhs_f16 = new Eigen::half[max_lhs_buffer_size];
  host_weight_s4 = new uint8_t[max_weight_buffer_size / 2];
  host_zero_f16 = new Eigen::half[max_zero_buffer_size];
  host_scale_f16 = new Eigen::half[max_scale_buffer_size];
  host_output_f16 = new Eigen::half[max_output_buffer_size];
  golden_fp32 = new float[max_output_buffer_size];

  CHECK_ACL(aclrtMalloc((void **)&dev_lhs_f16,
                        max_lhs_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_weight_s4,
                        max_weight_buffer_size / 2 * sizeof(uint8_t),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_zero_fp16,
                        max_zero_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_scale_fp16,
                        max_scale_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_output_f16,
                        max_output_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_bias_f32,
                        max_bias_buffer_size * sizeof(float),
                        ACL_MEM_MALLOC_HUGE_FIRST));
}

bool GemmAWQ4BitOpTest::Run(size_t m, size_t n, size_t k) {
  spdlog::info("GemmAWQ4BitOpTest::Run {{{},{},{}}}, bias={}", m, n, k, bias);

  size_t n1 = n / 16;
  size_t lhs_element_cnt = m * k;
  size_t rhs_element_cnt = n * k;
  size_t num_group = k / group_size;
  size_t zero_scale_cnt = n * num_group;
  size_t output_element_cnt = m * n;
  make_random_float(host_lhs, lhs_element_cnt);
  make_random_float_uint4(host_rhs, rhs_element_cnt);
  make_random_float_uint4(host_zero, zero_scale_cnt);
  make_random_float(host_scale, zero_scale_cnt);
  make_random_float(host_bias, n);

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      bias_fp32_map((float *)host_bias, 1, n);

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      golden_fp32_map((float *)golden_fp32, m, n);

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      input_lhs_map((float *)host_lhs, m, k);

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
      input_lhs_fp16_map((Eigen::half *)host_lhs_f16, m, k);
  input_lhs_fp16_map = input_lhs_map.cast<Eigen::half>();
  input_lhs_map = input_lhs_fp16_map.cast<float>();

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_rhs_map((float *)host_rhs, num_group, group_size, n);

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_rhs_nz_map((float *)host_rhs_nz, k / 16, n, 16);
  input_rhs_nz_map =
      input_rhs_map.reshape(Eigen::array<size_t, 3>{k / 16, 16, n})
          .shuffle(Eigen::array<size_t, 3>({0, 2, 1}));

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_zero_fp32_map((float *)host_zero, num_group, 1, n);

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_zero_fp16_map((Eigen::half *)host_zero_f16, num_group, 1, n);

  input_zero_fp16_map = input_zero_fp32_map.cast<Eigen::half>();
  input_zero_fp32_map = input_zero_fp16_map.cast<float>();

  // move weight offset to zero
  input_zero_fp16_map = input_zero_fp16_map -
                        input_zero_fp32_map.constant(8.0f).cast<Eigen::half>();

  Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_scale_fp32_map((float *)host_scale, num_group, 1, n);

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_scale_fp16_map((Eigen::half *)host_scale_f16, num_group, 1, n);

  input_scale_fp16_map =
      (input_scale_fp32_map / input_scale_fp32_map.constant(16.0f))
          .cast<Eigen::half>();
  input_scale_fp32_map = input_scale_fp16_map.cast<float>();

  auto float_to_u4 = [](float x) -> uint8_t {
    return (static_cast<uint8_t>(x) + 8) & 0xf;
  };

  // (x, 4, 64, 2) -> (x, 64, 4, 2)
  for (int i1 = 0; i1 < rhs_element_cnt / 512; ++i1) {
    int i1_stride_u8 = 4 * 64 * 2;
    int i1_stride_s4 = 4 * 64;
    for (int i2 = 0; i2 < 4; ++i2) {
      int i2_stride_u8 = 64 * 2;
      int i2_stride_s4 = 1;
      for (int i3 = 0; i3 < 64; ++i3) {
        int i3_stride_u8 = 2;
        int i3_stride_s4 = 4;
        int u8_offset =
            i1 * i1_stride_u8 + i2 * i2_stride_u8 + i3 * i3_stride_u8;
        host_weight_s4[i1 * i1_stride_s4 + i2 * i2_stride_s4 +
                       i3 * i3_stride_s4] =
            (float_to_u4(host_rhs_nz[u8_offset])) |
            (float_to_u4(host_rhs_nz[u8_offset + 1]) << 4);
      }
    }
  }

  CHECK_ACL(aclrtMemcpy(dev_lhs_f16, lhs_element_cnt * sizeof(aclFloat16),
                        host_lhs_f16, lhs_element_cnt * sizeof(aclFloat16),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_weight_s4, rhs_element_cnt * sizeof(uint8_t) / 2,
                        host_weight_s4, rhs_element_cnt * sizeof(uint8_t) / 2,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_zero_fp16, zero_scale_cnt * sizeof(aclFloat16),
                        host_zero_f16, zero_scale_cnt * sizeof(aclFloat16),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_scale_fp16, zero_scale_cnt * sizeof(aclFloat16),
                        host_scale_f16, zero_scale_cnt * sizeof(aclFloat16),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  if (bias) {
    CHECK_ACL(aclrtMemcpy(dev_bias_f32, n * sizeof(float), host_bias,
                          n * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    npu_matmul_nz_awq_4bit_bias_layer(
        dev_output_f16, dev_lhs_f16, dev_weight_s4, dev_zero_fp16,
        dev_scale_fp16, dev_bias_f32, m, n, k, DT_FLOAT16, stream);
  } else {

    npu_matmul_nz_awq_4bit_layer(dev_output_f16, dev_lhs_f16, dev_weight_s4,
                                 dev_zero_fp16, dev_scale_fp16, m, n, k,
                                 DT_FLOAT16, stream);
  }
  Eigen::array<size_t, 3> brc_dim = {1, group_size, 1};

  auto tmp_expr = ((input_rhs_map - input_zero_fp32_map.broadcast(brc_dim))
                       .cast<Eigen::half>()
                       .cast<float>() *
                   (input_scale_fp32_map.broadcast(brc_dim)))
                      .reshape(Eigen::array<size_t, 2>{k, n});

  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
      Eigen::IndexPair<int>(1, 0)};
  golden_fp32_map = input_lhs_map.contract(
      tmp_expr.cast<Eigen::half>().cast<float>(), product_dims);

  if (bias) {
    Eigen::array<size_t, 2> brc_dim = {m, 1};

    golden_fp32_map = golden_fp32_map + bias_fp32_map.broadcast(brc_dim);
  }

  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(
      host_output_f16, output_element_cnt * sizeof(aclFloat16), dev_output_f16,
      output_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      output_fp32_map((float *)host_output, m, n);

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
      output_fp16_map((Eigen::half *)host_output_f16, m, n);
  output_fp32_map = output_fp16_map.cast<float>();

  return all_close(host_output, golden_fp32, output_element_cnt);
}

void GemmAWQ4BitOpTest::CleanUp() {
  delete[] host_lhs;
  delete[] host_rhs;
  delete[] host_zero;
  delete[] host_scale;
  delete[] host_output;
  delete[] host_lhs_f16;
  delete[] host_weight_s4;
  delete[] host_zero_f16;
  delete[] host_scale_f16;
  delete[] host_output_f16;
  delete[] host_bias;
  delete[] golden_fp32;

  CHECK_ACL(aclrtFree(dev_lhs_f16));
  CHECK_ACL(aclrtFree(dev_weight_s4));
  CHECK_ACL(aclrtFree(dev_zero_fp16));
  CHECK_ACL(aclrtFree(dev_scale_fp16));
  CHECK_ACL(aclrtFree(dev_output_f16));
  CHECK_ACL(aclrtFree(dev_bias_f32));
}
