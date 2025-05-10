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

#include "gemm_test.h"
#include "npu_op_test_util.h"
#include "npu_ops.h"

template <typename EigenTy>
void GemmOpTest<EigenTy>::Init(size_t max_m, size_t max_n, size_t max_k,
                               bool bias) {
  this->max_m = max_m;
  this->max_n = max_n;
  this->max_k = max_k;
  this->bias = bias;

  max_lhs_buffer_size = max_m * max_k;
  max_rhs_buffer_size = max_k * max_n;
  max_bias_buffer_size = max_n;
  max_output_buffer_size = max_m * max_n;

  host_lhs = new float[max_lhs_buffer_size];
  host_rhs = new float[max_rhs_buffer_size];
  host_bias = new float[max_bias_buffer_size];
  host_output = new float[max_output_buffer_size];
  host_lhs_b16 = new EigenTy[max_lhs_buffer_size];
  host_rhs_b16 = new EigenTy[max_rhs_buffer_size];
  host_rhs_nz_b16 = new EigenTy[max_rhs_buffer_size];
  host_output_b16 = new EigenTy[max_output_buffer_size];
  host_golden_b16 = new EigenTy[max_output_buffer_size];
  golden_fp32 = new float[max_output_buffer_size];

  CHECK_ACL(aclrtMalloc((void **)&dev_lhs,
                        max_lhs_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_rhs,
                        max_rhs_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_bias,
                        max_bias_buffer_size * sizeof(float),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_output,
                        max_output_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
}

template <typename EigenTy>
bool GemmOpTest<EigenTy>::Run(size_t m, size_t n, size_t k) {
  spdlog::info("{} {{m={},n={},k={},bias={}}} dtype {}", __PRETTY_FUNCTION__, m,
               n, k, bias, GetDataType<EigenTy>());

  size_t n1 = n / 16;
  size_t lhs_element_cnt = m * k;
  size_t rhs_element_cnt = n * k;
  size_t output_element_cnt = m * n;
  size_t bias_element_cnt = n;
  make_random_float(host_lhs, lhs_element_cnt);
  make_random_float(host_bias, bias_element_cnt);
  make_random_float(host_rhs, rhs_element_cnt);

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      bias_fp32_map((float *)host_bias, 1, n);

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      golden_fp32_map((float *)golden_fp32, m, n);

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      input_lhs_map((float *)host_lhs, m, k);

  Eigen::TensorMap<
      Eigen::Tensor<EigenTy, 2, Eigen::RowMajor | Eigen::DontAlign>>
      input_lhs_b16_map((EigenTy *)host_lhs_b16, m, k);
  input_lhs_b16_map = input_lhs_map.cast<EigenTy>();
  input_lhs_map = input_lhs_b16_map.template cast<float>();

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      input_rhs_map((float *)host_rhs, k, n);

  Eigen::TensorMap<
      Eigen::Tensor<EigenTy, 2, Eigen::RowMajor | Eigen::DontAlign>>
      input_rhs_b16_map((EigenTy *)host_rhs_b16, k, n);
  input_rhs_b16_map = input_rhs_map.cast<EigenTy>();
  input_rhs_map = input_rhs_b16_map.template cast<float>();

  Eigen::TensorMap<
      Eigen::Tensor<EigenTy, 3, Eigen::RowMajor | Eigen::DontAlign>>
      input_rhs_b16_nz_map((EigenTy *)host_rhs_nz_b16, n / 16, k, 16);
  input_rhs_b16_nz_map =
      input_rhs_b16_map.reshape(Eigen::array<size_t, 3>{k, n / 16, 16})
          .shuffle(Eigen::array<size_t, 3>({1, 0, 2}));

  // TODO init bias
  CHECK_ACL(aclrtMemcpy(dev_lhs, lhs_element_cnt * sizeof(aclFloat16),
                        host_lhs_b16, lhs_element_cnt * sizeof(aclFloat16),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_rhs, rhs_element_cnt * sizeof(aclFloat16),
                        host_rhs_nz_b16, rhs_element_cnt * sizeof(aclFloat16),
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(dev_bias, bias_element_cnt * sizeof(float), host_bias,
                        bias_element_cnt * sizeof(float),
                        ACL_MEMCPY_HOST_TO_DEVICE))

  aclrtEvent start_event, end_event;
  CHECK_ACL(aclrtCreateEvent(&start_event));
  CHECK_ACL(aclrtCreateEvent(&end_event));
  CHECK_ACL(aclrtRecordEvent(start_event, this->stream));

  if (bias) {
    npu_matmul_bias_nz_layer(dev_output, dev_lhs, dev_rhs, dev_bias, m, n, k,
                             GetDataType<EigenTy>(), this->stream);
  } else {
    npu_matmul_nz_layer(dev_output, dev_lhs, dev_rhs, m, n, k,
                        GetDataType<EigenTy>(), this->stream);
  }
  CHECK_ACL(aclrtRecordEvent(end_event, this->stream));
  CHECK_ACL(aclrtSynchronizeStream(this->stream));
  float duration_ms;
  CHECK_ACL(aclrtEventElapsedTime(&duration_ms, start_event, end_event));
  CHECK_ACL(aclrtDestroyEvent(start_event));
  CHECK_ACL(aclrtDestroyEvent(end_event));

  spdlog::info("kernel duration {}ms", duration_ms);

  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
      Eigen::IndexPair<int>(1, 0)};

  golden_fp32_map = input_lhs_map.contract(input_rhs_map, product_dims);
  if (bias) {
    Eigen::array<size_t, 2> brc_dim = {m, 1};

    golden_fp32_map = golden_fp32_map + bias_fp32_map.broadcast(brc_dim);
  }

  Eigen::TensorMap<
      Eigen::Tensor<EigenTy, 2, Eigen::RowMajor | Eigen::DontAlign>>
      golden_b16_map((EigenTy *)host_golden_b16, m, n);
  golden_b16_map = golden_fp32_map.cast<EigenTy>();
  golden_fp32_map = golden_b16_map.template cast<float>();

  CHECK_ACL(aclrtSynchronizeStream(this->stream));

  CHECK_ACL(aclrtMemcpy(
      host_output_b16, output_element_cnt * sizeof(aclFloat16), dev_output,
      output_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      output_fp32_map((float *)host_output, m, n);

  Eigen::TensorMap<
      Eigen::Tensor<EigenTy, 2, Eigen::RowMajor | Eigen::DontAlign>>
      output_b16_map((EigenTy *)host_output_b16, m, n);
  output_fp32_map = output_b16_map.template cast<float>();

  return all_close(host_output, golden_fp32, output_element_cnt, 0.01, 0.01);
}

template <typename EigenTy> void GemmOpTest<EigenTy>::CleanUp() {
  delete[] host_lhs;
  delete[] host_rhs;
  delete[] host_bias;
  delete[] host_output;
  delete[] host_lhs_b16;
  delete[] host_rhs_b16;
  delete[] host_rhs_nz_b16;
  delete[] host_output_b16;
  delete[] host_golden_b16;
  delete[] golden_fp32;

  CHECK_ACL(aclrtFree(dev_lhs));
  CHECK_ACL(aclrtFree(dev_rhs));
  CHECK_ACL(aclrtFree(dev_bias));
  CHECK_ACL(aclrtFree(dev_output));
}

template class GemmOpTest<Eigen::half>;

template class GemmOpTest<Eigen::bfloat16>;
