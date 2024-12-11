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
#include "rms_norm_layer_test.h"

void RMSNormOpTest::Init(size_t max_first_dim, size_t max_last_dim) {
  this->max_first_dim = max_first_dim;
  this->max_last_dim = max_last_dim;

  size_t max_buffer_size = max_first_dim * max_last_dim;

  host_input = new float[max_buffer_size];
  host_output = new float[max_buffer_size];
  host_input_f16 = new Eigen::half[max_buffer_size];
  host_output_f16 = new Eigen::half[max_buffer_size];
  golden_fp32 = new float[max_buffer_size];
  weight_fp32 = new float[max_last_dim];
  weight_fp16 = new Eigen::half[max_last_dim];

  CHECK_ACL(aclrtMalloc((void **)&dev_input_f16,
                        max_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_output_f16,
                        max_buffer_size * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&dev_weight_f16,
                        max_last_dim * sizeof(aclFloat16),
                        ACL_MEM_MALLOC_HUGE_FIRST));
}

bool RMSNormOpTest::Run(size_t first_dim, size_t last_dim, float eps) {
  spdlog::info("RMSNormOpTest::Run {{{},{}}} eps:{}", first_dim, last_dim, eps);

  if (last_dim % 16 != 0) {
    spdlog::critical("last dim {} is not aligned to 16!", last_dim);
    return false;
  }

  if (first_dim > max_first_dim) {
    spdlog::critical("{} > {}", first_dim, max_first_dim);
    return false;
  }

  if (last_dim > max_last_dim) {
    spdlog::critical("{} > {}", last_dim, max_last_dim);
    return false;
  }

  int total_element_cnt = first_dim * last_dim;
  make_random_float(host_input, total_element_cnt);
  make_random_float(weight_fp32, last_dim);

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      golden_fp32_map((float *)golden_fp32, first_dim, last_dim);

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      input_map((float *)host_input, first_dim, last_dim);

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
      input_fp16_map((Eigen::half *)host_input_f16, first_dim, last_dim);
  input_fp16_map = input_map.cast<Eigen::half>();
  input_map = input_fp16_map.cast<float>();

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      weight_map((float *)weight_fp32, 1, last_dim);

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
      weight_fp16_map((Eigen::half *)weight_fp16, 1, last_dim);
  weight_fp16_map = weight_map.cast<Eigen::half>();
  weight_map = weight_fp16_map.cast<float>();

  std::array<long, 1> mean_dims = {1};
  Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign> mean =
      (input_map * input_map)
          .mean(mean_dims)
          .eval()
          .reshape(std::array<long, 2>{(long)first_dim, 1L});
  Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>
      sqrt_mean_add_eps =
          (mean + mean.constant(eps))
              .sqrt()
              .eval()
              .reshape(std::array<long int, 2>{(long int)first_dim, 1});
  golden_fp32_map =
      (input_map /
       sqrt_mean_add_eps.broadcast(std::array<size_t, 2>{1, last_dim}) *
       weight_map.broadcast(std::array<size_t, 2>{first_dim, 1}));

  CHECK_ACL(aclrtMemcpy(dev_input_f16, total_element_cnt * sizeof(aclFloat16),
                        host_input_f16, total_element_cnt * sizeof(aclFloat16),
                        ACL_MEMCPY_HOST_TO_DEVICE));

  CHECK_ACL(aclrtMemcpy(dev_weight_f16, last_dim * sizeof(aclFloat16), weight_fp16,
                        last_dim * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

  npu_rmsnorm_layer(dev_output_f16, dev_weight_f16, dev_input_f16, first_dim,
                    last_dim, eps, DT_FLOAT16, stream);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  CHECK_ACL(aclrtMemcpy(host_output_f16, total_element_cnt * sizeof(aclFloat16),
                        dev_output_f16, total_element_cnt * sizeof(aclFloat16),
                        ACL_MEMCPY_DEVICE_TO_HOST));

  Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
      output_fp32_map((float *)host_output, first_dim, last_dim);

  Eigen::TensorMap<
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
      output_fp16_map((Eigen::half *)host_output_f16, first_dim, last_dim);
  output_fp32_map = output_fp16_map.cast<float>();
  return all_close(host_output, golden_fp32, total_element_cnt);
}

void RMSNormOpTest::CleanUp() {

  delete[] host_input;
  delete[] host_output;
  delete[] host_input_f16;
  delete[] host_output_f16;
  delete[] golden_fp32;
  delete[] weight_fp32;
  delete[] weight_fp16;

  CHECK_ACL(aclrtFree(dev_input_f16));
  CHECK_ACL(aclrtFree(dev_output_f16));
  CHECK_ACL(aclrtFree(dev_weight_f16));
}


