#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#include <acl/acl.h>
#include <gtest/gtest.h>
#include <random>

#include <Eigen/Core>
#include <fmt/format.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "gemm_awq_4bit_test.h"
#include "npu_op_test_util.h"
#include "npu_ops.h"
#include "rms_norm_layer_test.h"

#define CHECK_ACL_GTEST(x)                                                     \
  do {                                                                         \
    aclError __ret = x;                                                        \
    if (__ret != ACL_ERROR_NONE) {                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret        \
                << std::endl;                                                  \
      ASSERT_TRUE(false);                                                      \
    }                                                                          \
  } while (0);

class ACLEnvironment : public ::testing::Environment {
public:
  ~ACLEnvironment() override {}

  // Override this to define how to set up the environment.
  void SetUp() override {
    CHECK_ACL_GTEST(aclInit(nullptr));
    CHECK_ACL_GTEST(aclrtSetDevice(deviceId));
    CHECK_ACL_GTEST(aclrtCreateContext(&context, deviceId));
  }

  // Override this to define how to tear down the environment.
  void TearDown() override {
    CHECK_ACL_GTEST(aclrtDestroyContext(context));
    CHECK_ACL_GTEST(aclrtResetDevice(deviceId));
    CHECK_ACL_GTEST(aclFinalize());
  }
  aclrtContext context;
  int32_t deviceId{0};
};

class ACLTimer {
public:
  ACLTimer(const std::string &desc, aclrtStream stream)
      : desc(desc), stream(stream) {
    CHECK_ACL(aclrtCreateEvent(&start_event));
    CHECK_ACL(aclrtCreateEvent(&end_event));
  }
  ~ACLTimer() {
    CHECK_ACL(aclrtDestroyEvent(start_event));
    CHECK_ACL(aclrtDestroyEvent(end_event));
  }

  void Start() { CHECK_ACL(aclrtRecordEvent(start_event, stream)); }

  void Stop() {
    CHECK_ACL(aclrtRecordEvent(end_event, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
  }
  void Print() {
    float duration_ms;
    CHECK_ACL(aclrtEventElapsedTime(&duration_ms, start_event, end_event));
    std::cout << desc << " duration: " << duration_ms << "ms" << std::endl;
  }

private:
  std::string desc;
  aclrtStream stream;
  aclrtEvent start_event;
  aclrtEvent end_event;
};

testing::Environment *const acl_env =
    testing::AddGlobalTestEnvironment(new ACLEnvironment);

constexpr size_t max_seq_len = 2048;
constexpr size_t head_num = 40;
constexpr size_t head_dim = 128;
constexpr size_t hidden_dim = head_dim * head_num;
constexpr size_t multiple_of = 256;
constexpr size_t ffn_hidden_unaligned = (4 * (hidden_dim * 2) / 3);
constexpr size_t ffn_hidden =
    (ffn_hidden_unaligned + multiple_of - 1) / multiple_of * multiple_of;

TEST(NpuOpsTest, SoftmaxLayer) {
  aclrtStream stream = nullptr;
  CHECK_ACL_GTEST(aclrtCreateStream(&stream));

  size_t max_buffer_size = head_num * max_seq_len * max_seq_len;
  void *dev_input_f16;
  void *dev_output_f16;

  float *host_input = new float[max_buffer_size];
  float *host_output = new float[max_buffer_size];
  Eigen::half *host_input_f16 = new Eigen::half[max_buffer_size];
  Eigen::half *host_output_f16 = new Eigen::half[max_buffer_size];
  float *golden_fp32 = new float[max_buffer_size];
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_input_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_output_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));

  for (int curr_pos = 1; curr_pos <= max_seq_len; ++curr_pos) {
    // for (int curr_size = 1; curr_size <= max_seq_len; ++curr_size) {
    for (int curr_size : {1, 7}) {
      std::cout << "test softmax: "
                << "curr_pos: " << curr_pos << " curr_size: " << curr_size
                << "\n";
      int hs = head_num * curr_size;
      int total_element_cnt = hs * curr_pos;

      make_random_float(host_input, total_element_cnt);

      Eigen::TensorMap<
          Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
          golden_fp32_map((float *)golden_fp32, hs, curr_pos);

      Eigen::TensorMap<
          Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
          input_map((float *)host_input, hs, curr_pos);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          input_fp16_map((Eigen::half *)host_input_f16, hs, curr_pos);
      input_fp16_map = input_map.cast<Eigen::half>();
      input_map = input_fp16_map.cast<float>();

      auto input_max = input_map.maximum(Eigen::array<int, 1>{1})
                           .eval()
                           .reshape(Eigen::array<int, 2>{hs, 1})
                           .broadcast(Eigen::array<int, 2>{1, curr_pos});

      auto input_diff = (input_map - input_max).exp().eval();

      auto input_sum = input_diff.sum(Eigen::array<int, 1>{1})
                           .eval()
                           .reshape(Eigen::array<int, 2>{hs, 1})
                           .broadcast(Eigen::array<int, 2>{1, curr_pos});

      golden_fp32_map = input_diff / input_sum;

      CHECK_ACL_GTEST(aclrtMemcpy(
          dev_input_f16, total_element_cnt * sizeof(aclFloat16), host_input_f16,
          total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

      npu_softmax_layer(dev_output_f16, dev_input_f16, hs, curr_pos, DT_FLOAT16,
                        stream);
      CHECK_ACL_GTEST(aclrtSynchronizeStream(stream));

      CHECK_ACL_GTEST(
          aclrtMemcpy(host_output_f16, total_element_cnt * sizeof(aclFloat16),
                      dev_output_f16, total_element_cnt * sizeof(aclFloat16),
                      ACL_MEMCPY_DEVICE_TO_HOST));

      Eigen::TensorMap<
          Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
          output_fp32_map((float *)host_output, hs, curr_pos);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          output_fp16_map((Eigen::half *)host_output_f16, hs, curr_pos);
      output_fp32_map = output_fp16_map.cast<float>();

      ASSERT_TRUE(all_close(host_output, golden_fp32, total_element_cnt))
          << "curr_pos: " << curr_pos << " curr_size: " << curr_size;
    }
  }

  delete[] host_input;
  delete[] host_output;
  delete[] host_input_f16;
  delete[] host_output_f16;
  delete[] golden_fp32;

  CHECK_ACL_GTEST(aclrtFree(dev_input_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_output_f16));

  CHECK_ACL_GTEST(aclrtDestroyStream(stream));
}

TEST(NpuOpsTest, RMSNormLayer) {
  float eps = 1e-5;

  RMSNormOpTest op_test;
  op_test.Init(max_seq_len, hidden_dim);

  for (int seq_len = 1; seq_len <= max_seq_len; ++seq_len) {
    ASSERT_TRUE(op_test.Run(seq_len, hidden_dim, eps))
        << "seq_len: " << seq_len;
  }
  op_test.CleanUp();
}

TEST(NpuOpsTest, GemmNzLayer) {
  aclrtStream stream = nullptr;
  CHECK_ACL_GTEST(aclrtCreateStream(&stream));

  size_t max_n = 32000;

  size_t max_lhs_buffer_size = hidden_dim * hidden_dim;
  size_t max_rhs_buffer_size = hidden_dim * max_n;
  size_t max_output_buffer_size = hidden_dim * max_n;
  void *dev_lhs_f16;
  void *dev_rhs_f16;
  void *dev_output_f16;

  float *host_lhs = new float[max_lhs_buffer_size];
  float *host_rhs = new float[max_rhs_buffer_size];
  float *host_output = new float[max_output_buffer_size];
  Eigen::half *host_lhs_f16 = new Eigen::half[max_lhs_buffer_size];
  Eigen::half *host_rhs_f16 = new Eigen::half[max_rhs_buffer_size];
  Eigen::half *host_rhs_f16_nz = new Eigen::half[max_rhs_buffer_size];
  Eigen::half *host_output_f16 = new Eigen::half[max_output_buffer_size];
  float *golden_fp32 = new float[max_output_buffer_size];

  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_lhs_f16,
                              max_lhs_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_rhs_f16,
                              max_rhs_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_output_f16,
                              max_output_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> m_dis(2, hidden_dim);

  std::vector<size_t> m_list = {1};
  constexpr int rnd_m_num = 10;
  for (int i = 0; i < rnd_m_num; ++i) {
    m_list.push_back(m_dis(gen));
  }

  // for (size_t m = 1; m <= hidden_dim; ++m) {
  for (size_t m : m_list) {
    for (size_t n : {(size_t)hidden_dim, max_n}) {
      size_t n1 = n / 16;
      std::cout << "test gemm m: " << m << " n: " << n << "\n";
      size_t lhs_element_cnt = m * hidden_dim;
      size_t rhs_element_cnt = n * hidden_dim;
      size_t output_element_cnt = m * n;
      make_random_float(host_lhs, lhs_element_cnt);
      make_random_float(host_rhs, rhs_element_cnt);

      Eigen::TensorMap<
          Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
          golden_fp32_map((float *)golden_fp32, m, n);

      Eigen::TensorMap<
          Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
          input_lhs_map((float *)host_lhs, m, hidden_dim);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          input_lhs_fp16_map((Eigen::half *)host_lhs_f16, m, hidden_dim);
      input_lhs_fp16_map = input_lhs_map.cast<Eigen::half>();
      input_lhs_map = input_lhs_fp16_map.cast<float>();

      Eigen::TensorMap<
          Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
          input_rhs_map((float *)host_rhs, hidden_dim, n);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          input_rhs_fp16_map((Eigen::half *)host_rhs_f16, hidden_dim, n);
      input_rhs_fp16_map = input_rhs_map.cast<Eigen::half>();
      input_rhs_map = input_rhs_fp16_map.cast<float>();

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
          input_rhs_fp16_nz_map((Eigen::half *)host_rhs_f16_nz, n1, hidden_dim,
                                16);

      input_rhs_fp16_nz_map =
          input_rhs_fp16_map
              .reshape(Eigen::array<int, 3>{hidden_dim, (int)n1, 16})
              .shuffle(Eigen::array<int, 3>({1, 0, 2}));

      CHECK_ACL_GTEST(aclrtMemcpy(
          dev_lhs_f16, lhs_element_cnt * sizeof(aclFloat16), host_lhs_f16,
          lhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
      CHECK_ACL_GTEST(aclrtMemcpy(
          dev_rhs_f16, rhs_element_cnt * sizeof(aclFloat16), host_rhs_f16_nz,
          rhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

      npu_matmul_nz_layer(dev_output_f16, dev_lhs_f16, dev_rhs_f16, m, n,
                          hidden_dim, DT_FLOAT16, stream);

      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
          Eigen::IndexPair<int>(1, 0)};
      golden_fp32_map = input_lhs_map.contract(input_rhs_map, product_dims);

      CHECK_ACL_GTEST(aclrtSynchronizeStream(stream));

      CHECK_ACL_GTEST(
          aclrtMemcpy(host_output_f16, output_element_cnt * sizeof(aclFloat16),
                      dev_output_f16, output_element_cnt * sizeof(aclFloat16),
                      ACL_MEMCPY_DEVICE_TO_HOST));

      Eigen::TensorMap<
          Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
          output_fp32_map((float *)host_output, m, n);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          output_fp16_map((Eigen::half *)host_output_f16, m, n);
      output_fp32_map = output_fp16_map.cast<float>();

      ASSERT_TRUE(all_close(host_output, golden_fp32, output_element_cnt))
          << "m: " << m << " n: " << n;
    }
  }

  delete[] host_lhs;
  delete[] host_rhs;
  delete[] host_output;
  delete[] host_lhs_f16;
  delete[] host_rhs_f16;
  delete[] host_rhs_f16_nz;
  delete[] host_output_f16;
  delete[] golden_fp32;

  CHECK_ACL_GTEST(aclrtFree(dev_lhs_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_rhs_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_output_f16));

  CHECK_ACL_GTEST(aclrtDestroyStream(stream));
}

TEST(NpuOpsTest, SiluMul) {
  aclrtStream stream = nullptr;
  CHECK_ACL_GTEST(aclrtCreateStream(&stream));

  size_t max_buffer_size = ffn_hidden * max_seq_len;
  void *dev_input_0_f16;
  void *dev_input_1_f16;
  void *dev_output_f16;

  float *host_input_0 = new float[max_buffer_size];
  float *host_input_1 = new float[max_buffer_size];
  float *host_output = new float[max_buffer_size];
  Eigen::half *host_input_0_f16 = new Eigen::half[max_buffer_size];
  Eigen::half *host_input_1_f16 = new Eigen::half[max_buffer_size];
  Eigen::half *host_output_f16 = new Eigen::half[max_buffer_size];
  float *golden_fp32 = new float[max_buffer_size];

  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_input_0_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_input_1_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_output_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));

  for (int seq_len = 1; seq_len <= max_seq_len; ++seq_len) {
    std::cout << "test silu_mul: "
              << "seq_len: " << seq_len << "\n";
    int total_element_cnt = seq_len * ffn_hidden;
    make_random_float(host_input_0, total_element_cnt);
    make_random_float(host_input_1, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        golden_fp32_map((float *)golden_fp32, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_0_map((float *)host_input_0, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_1_map((float *)host_input_1, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_0_fp16_map((Eigen::half *)host_input_0_f16, total_element_cnt);
    input_0_fp16_map = input_0_map.cast<Eigen::half>();
    input_0_map = input_0_fp16_map.cast<float>();

    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_1_fp16_map((Eigen::half *)host_input_1_f16, total_element_cnt);
    input_1_fp16_map = input_1_map.cast<Eigen::half>();
    input_1_map = input_1_fp16_map.cast<float>();

    golden_fp32_map =
        (input_0_map / ((-input_0_map).exp() + input_0_map.constant(1))) *
        input_1_map;
    CHECK_ACL_GTEST(
        aclrtMemcpy(dev_input_0_f16, total_element_cnt * sizeof(aclFloat16),
                    host_input_0_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_GTEST(
        aclrtMemcpy(dev_input_1_f16, total_element_cnt * sizeof(aclFloat16),
                    host_input_1_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_HOST_TO_DEVICE));

    npu_silu_mul_layer(dev_output_f16, dev_input_0_f16, dev_input_1_f16,
                       total_element_cnt, DT_FLOAT16, stream);
    CHECK_ACL_GTEST(aclrtSynchronizeStream(stream));

    CHECK_ACL_GTEST(aclrtMemcpy(
        host_output_f16, total_element_cnt * sizeof(aclFloat16), dev_output_f16,
        total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_fp32_map((float *)host_output, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_fp16_map((Eigen::half *)host_output_f16, total_element_cnt);
    output_fp32_map = output_fp16_map.cast<float>();

    ASSERT_TRUE(all_close(host_output, golden_fp32, total_element_cnt))
        << "seq_len: " << seq_len;
  }

  delete[] host_input_0;
  delete[] host_input_1;
  delete[] host_output;
  delete[] host_input_0_f16;
  delete[] host_input_1_f16;
  delete[] host_output_f16;
  delete[] golden_fp32;

  CHECK_ACL_GTEST(aclrtFree(dev_input_0_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_input_1_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_output_f16));

  CHECK_ACL_GTEST(aclrtDestroyStream(stream));
}

TEST(NpuOpsTest, Add) {
  aclrtStream stream = nullptr;
  CHECK_ACL_GTEST(aclrtCreateStream(&stream));

  size_t max_buffer_size = ffn_hidden * max_seq_len;
  void *dev_input_0_f16;
  void *dev_input_1_f16;
  void *dev_output_f16;

  float *host_input_0 = new float[max_buffer_size];
  float *host_input_1 = new float[max_buffer_size];
  float *host_output = new float[max_buffer_size];
  Eigen::half *host_input_0_f16 = new Eigen::half[max_buffer_size];
  Eigen::half *host_input_1_f16 = new Eigen::half[max_buffer_size];
  Eigen::half *host_output_f16 = new Eigen::half[max_buffer_size];
  float *golden_fp32 = new float[max_buffer_size];

  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_input_0_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_input_1_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_output_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));

  for (int seq_len = 1; seq_len <= max_seq_len; ++seq_len) {
    std::cout << "test add: "
              << "seq_len: " << seq_len << "\n";
    int total_element_cnt = seq_len * ffn_hidden;
    make_random_float(host_input_0, total_element_cnt);
    make_random_float(host_input_1, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        golden_fp32_map((float *)golden_fp32, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_0_map((float *)host_input_0, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_1_map((float *)host_input_1, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_0_fp16_map((Eigen::half *)host_input_0_f16, total_element_cnt);
    input_0_fp16_map = input_0_map.cast<Eigen::half>();
    input_0_map = input_0_fp16_map.cast<float>();

    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_1_fp16_map((Eigen::half *)host_input_1_f16, total_element_cnt);
    input_1_fp16_map = input_1_map.cast<Eigen::half>();
    input_1_map = input_1_fp16_map.cast<float>();

    CHECK_ACL_GTEST(
        aclrtMemcpy(dev_input_0_f16, total_element_cnt * sizeof(aclFloat16),
                    host_input_0_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_GTEST(
        aclrtMemcpy(dev_input_1_f16, total_element_cnt * sizeof(aclFloat16),
                    host_input_1_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_HOST_TO_DEVICE));

    npu_add_layer(dev_output_f16, dev_input_0_f16, dev_input_1_f16,
                  total_element_cnt, DT_FLOAT16, stream);

    golden_fp32_map = input_0_map + input_1_map;
    CHECK_ACL_GTEST(aclrtSynchronizeStream(stream));

    CHECK_ACL_GTEST(aclrtMemcpy(
        host_output_f16, total_element_cnt * sizeof(aclFloat16), dev_output_f16,
        total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_fp32_map((float *)host_output, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_fp16_map((Eigen::half *)host_output_f16, total_element_cnt);
    output_fp32_map = output_fp16_map.cast<float>();

    ASSERT_TRUE(all_close(host_output, golden_fp32, total_element_cnt))
        << "seq_len: " << seq_len;
  }

  delete[] host_input_0;
  delete[] host_input_1;
  delete[] host_output;
  delete[] host_input_0_f16;
  delete[] host_input_1_f16;
  delete[] host_output_f16;
  delete[] golden_fp32;

  CHECK_ACL_GTEST(aclrtFree(dev_input_0_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_input_1_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_output_f16));

  CHECK_ACL_GTEST(aclrtDestroyStream(stream));
}

TEST(NpuOpsTest, BMMQKTransCausual) {
  aclrtStream stream = nullptr;
  CHECK_ACL_GTEST(aclrtCreateStream(&stream));

  float qk_scale = 1 / sqrtf(static_cast<float>(head_dim));

  size_t max_lhs_buffer_size = max_seq_len * hidden_dim;
  size_t max_rhs_buffer_size = max_seq_len * hidden_dim;
  size_t max_output_buffer_size = head_num * max_seq_len * max_seq_len;

  void *dev_lhs_f16;
  void *dev_rhs_f16;
  void *dev_output_f16;

  float *host_lhs = new float[max_lhs_buffer_size];
  float *host_rhs = new float[max_rhs_buffer_size];
  float *host_output = new float[max_output_buffer_size];
  float *host_mask = new float[max_output_buffer_size];
  Eigen::half *host_lhs_f16 = new Eigen::half[max_lhs_buffer_size];
  Eigen::half *host_rhs_f16 = new Eigen::half[max_rhs_buffer_size];
  Eigen::half *host_output_f16 = new Eigen::half[max_output_buffer_size];
  float *golden_fp32 = new float[max_output_buffer_size];

  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_lhs_f16,
                              max_lhs_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_rhs_f16,
                              max_rhs_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_output_f16,
                              max_output_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> cur_size_dis(2, max_seq_len);

  std::vector<size_t> cur_size_list = {1};

  constexpr int rnd_m_num = 10;
  constexpr int rnd_n_num = 10;
  for (int i = 0; i < rnd_m_num; ++i) {
    cur_size_list.push_back(cur_size_dis(gen));
  }

  for (size_t cur_size : cur_size_list) {
    std::uniform_int_distribution<size_t> cur_pos_dis(cur_size, max_seq_len);
    std::vector<size_t> cur_pos_list;
    for (int i = 0; i < rnd_n_num; ++i) {
      cur_pos_list.push_back(cur_pos_dis(gen));
    }
    for (size_t cur_pos : cur_pos_list) {
      size_t prev_pos = cur_pos - cur_size;
      std::cout << "test cur_size: " << cur_size << " cur_pos: " << cur_pos
                << "\n";

      size_t lhs_element_cnt = head_num * cur_size * head_dim;
      size_t rhs_element_cnt = head_num * cur_pos * head_dim;
      size_t output_element_cnt = head_num * cur_size * cur_pos;
      make_random_float(host_lhs, lhs_element_cnt);
      make_random_float(host_rhs, rhs_element_cnt);

      for (int row = 0; row < cur_size; ++row) {
        for (int col = 0; col < cur_pos; ++col) {
          if (col > (row + prev_pos)) {
            host_mask[row * cur_pos + col] =
                -std::numeric_limits<float>::infinity();
          } else {
            host_mask[row * cur_pos + col] = 0;
          }
        }
      }

      Eigen::TensorMap<
          Eigen::Tensor<float, 2, Eigen::RowMajor | Eigen::DontAlign>>
          mask_map(host_mask, cur_size, cur_pos);

      Eigen::TensorMap<
          Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
          golden_fp32_map((float *)golden_fp32, head_num, cur_size, cur_pos);

      Eigen::TensorMap<
          Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
          input_lhs_map((float *)host_lhs, cur_size, head_num, head_dim);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
          input_lhs_fp16_map((Eigen::half *)host_lhs_f16, cur_size, head_num,
                             head_dim);
      input_lhs_fp16_map = input_lhs_map.cast<Eigen::half>();
      input_lhs_map = input_lhs_fp16_map.cast<float>();

      Eigen::TensorMap<
          Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
          input_rhs_map((float *)host_rhs, cur_pos, head_num, head_dim);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
          input_rhs_fp16_map((Eigen::half *)host_rhs_f16, cur_pos, head_num,
                             head_dim);
      input_rhs_fp16_map = input_rhs_map.cast<Eigen::half>();
      input_rhs_map = input_rhs_fp16_map.cast<float>();

      CHECK_ACL_GTEST(aclrtMemcpy(
          dev_lhs_f16, lhs_element_cnt * sizeof(aclFloat16), host_lhs_f16,
          lhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
      CHECK_ACL_GTEST(aclrtMemcpy(
          dev_rhs_f16, rhs_element_cnt * sizeof(aclFloat16), host_rhs_f16,
          rhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

      npu_batch_matmul_qk_trans_causual_layer(
          dev_output_f16, dev_lhs_f16, dev_rhs_f16, head_num, cur_size, cur_pos,
          head_dim, prev_pos, qk_scale, DT_FLOAT16, stream);
      Eigen::array<Eigen::IndexPair<int>, 1> qk_product_dims = {
          Eigen::IndexPair<int>(1, 0)};
      for (int i = 0; i < head_num; ++i) {
        golden_fp32_map.chip<0>(i) =
            input_lhs_map.shuffle(Eigen::array<int, 3>({1, 0, 2}))
                .chip<0>(i)
                .contract(input_rhs_map.shuffle(Eigen::array<int, 3>({1, 2, 0}))
                              .chip<0>(i),
                          qk_product_dims) +
            mask_map;
      }

      golden_fp32_map = golden_fp32_map * golden_fp32_map.constant(qk_scale);

      CHECK_ACL_GTEST(aclrtSynchronizeStream(stream));
      CHECK_ACL_GTEST(
          aclrtMemcpy(host_output_f16, output_element_cnt * sizeof(aclFloat16),
                      dev_output_f16, output_element_cnt * sizeof(aclFloat16),
                      ACL_MEMCPY_DEVICE_TO_HOST));

      Eigen::TensorMap<
          Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
          output_fp32_map((float *)host_output, head_num, cur_size, cur_pos);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
          output_fp16_map((Eigen::half *)host_output_f16, head_num, cur_size,
                          cur_pos);
      output_fp32_map = output_fp16_map.cast<float>();

      ASSERT_TRUE(all_close_inf(host_output, golden_fp32, output_element_cnt))
          << "test cur_size: " << cur_size << " cur_pos: " << cur_pos;
    }
  }

  delete[] host_lhs;
  delete[] host_rhs;
  delete[] host_output;
  delete[] host_mask;
  delete[] host_lhs_f16;
  delete[] host_rhs_f16;
  delete[] host_output_f16;
  delete[] golden_fp32;

  CHECK_ACL_GTEST(aclrtFree(dev_lhs_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_rhs_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_output_f16));

  CHECK_ACL_GTEST(aclrtDestroyStream(stream));
}

TEST(NpuOpsTest, BMMTransV) {
  aclrtStream stream = nullptr;
  CHECK_ACL_GTEST(aclrtCreateStream(&stream));

  size_t max_lhs_buffer_size = head_num * max_seq_len * max_seq_len;
  size_t max_rhs_buffer_size = head_num * max_seq_len * hidden_dim;
  size_t max_output_buffer_size = head_num * max_seq_len * hidden_dim;

  void *dev_lhs_f16;
  void *dev_rhs_f16;
  void *dev_output_f16;

  float *host_lhs = new float[max_lhs_buffer_size];
  float *host_rhs = new float[max_rhs_buffer_size];
  float *host_output = new float[max_output_buffer_size];
  Eigen::half *host_lhs_f16 = new Eigen::half[max_lhs_buffer_size];
  Eigen::half *host_rhs_f16 = new Eigen::half[max_rhs_buffer_size];
  Eigen::half *host_output_f16 = new Eigen::half[max_output_buffer_size];
  float *golden_fp32 = new float[max_output_buffer_size];
  float *golden_trans_fp32 = new float[max_output_buffer_size];

  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_lhs_f16,
                              max_lhs_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_rhs_f16,
                              max_rhs_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_output_f16,
                              max_output_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> cur_size_dis(2, max_seq_len);

  std::vector<size_t> cur_size_list = {1};
  // std::vector<size_t> cur_size_list;

  constexpr int rnd_m_num = 10;
  constexpr int rnd_n_num = 10;
  for (int i = 0; i < rnd_m_num; ++i) {
    cur_size_list.push_back(cur_size_dis(gen));
  }

  for (size_t cur_size : cur_size_list) {
    std::uniform_int_distribution<size_t> cur_pos_dis(cur_size, max_seq_len);
    std::vector<size_t> cur_pos_list;
    for (int i = 0; i < rnd_n_num; ++i) {
      cur_pos_list.push_back(cur_pos_dis(gen));
    }
    for (size_t cur_pos : cur_pos_list) {
      std::cout << "test cur_size: " << cur_size << " cur_pos: " << cur_pos
                << "\n";

      size_t lhs_element_cnt = head_num * cur_size * cur_pos;
      size_t rhs_element_cnt = head_num * cur_pos * head_dim;
      size_t output_element_cnt = head_num * cur_size * head_dim;
      make_random_float(host_lhs, lhs_element_cnt);
      make_random_float(host_rhs, rhs_element_cnt);

      Eigen::TensorMap<
          Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
          golden_fp32_map((float *)golden_fp32, head_num, cur_size, head_dim);

      Eigen::TensorMap<
          Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
          golden_trans_fp32_map((float *)golden_trans_fp32, cur_size, head_num,
                                head_dim);

      Eigen::TensorMap<
          Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
          input_lhs_map((float *)host_lhs, head_num, cur_size, cur_pos);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
          input_lhs_fp16_map((Eigen::half *)host_lhs_f16, head_num, cur_size,
                             cur_pos);
      input_lhs_fp16_map = input_lhs_map.cast<Eigen::half>();
      input_lhs_map = input_lhs_fp16_map.cast<float>();

      Eigen::TensorMap<
          Eigen::Tensor<float, 3, Eigen::RowMajor | Eigen::DontAlign>>
          input_rhs_map((float *)host_rhs, cur_pos, head_num, head_dim);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
          input_rhs_fp16_map((Eigen::half *)host_rhs_f16, cur_pos, head_num,
                             head_dim);
      input_rhs_fp16_map = input_rhs_map.cast<Eigen::half>();
      input_rhs_map = input_rhs_fp16_map.cast<float>();

      CHECK_ACL_GTEST(aclrtMemcpy(
          dev_lhs_f16, lhs_element_cnt * sizeof(aclFloat16), host_lhs_f16,
          lhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
      CHECK_ACL_GTEST(aclrtMemcpy(
          dev_rhs_f16, rhs_element_cnt * sizeof(aclFloat16), host_rhs_f16,
          rhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

      npu_batch_matmul_trans_v_layer(dev_output_f16, dev_lhs_f16, dev_rhs_f16,
                                     head_num, cur_size, head_dim, cur_pos,
                                     1.0f, DT_FLOAT16, stream);

      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
          Eigen::IndexPair<int>(1, 0)};
      for (int i = 0; i < head_num; ++i) {
        golden_fp32_map.chip<0>(i) = input_lhs_map.chip<0>(i).contract(
            input_rhs_map.shuffle(Eigen::array<int, 3>({1, 0, 2})).chip<0>(i),
            product_dims);
      }

      golden_trans_fp32_map =
          golden_fp32_map.shuffle(Eigen::array<int, 3>({1, 0, 2}));

      CHECK_ACL_GTEST(aclrtSynchronizeStream(stream));
      CHECK_ACL_GTEST(
          aclrtMemcpy(host_output_f16, output_element_cnt * sizeof(aclFloat16),
                      dev_output_f16, output_element_cnt * sizeof(aclFloat16),
                      ACL_MEMCPY_DEVICE_TO_HOST));

      Eigen::TensorMap<
          Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
          output_fp32_map((float *)host_output, output_element_cnt);

      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
          output_fp16_map((Eigen::half *)host_output_f16, output_element_cnt);
      output_fp32_map = output_fp16_map.cast<float>();

      ASSERT_TRUE(all_close(host_output, golden_trans_fp32, output_element_cnt))
          << "test cur_size: " << cur_size << " cur_pos: " << cur_pos;
    }
  }

  delete[] host_lhs;
  delete[] host_rhs;
  delete[] host_output;
  delete[] host_lhs_f16;
  delete[] host_rhs_f16;
  delete[] host_output_f16;
  delete[] golden_fp32;
  delete[] golden_trans_fp32;

  CHECK_ACL_GTEST(aclrtFree(dev_lhs_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_rhs_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_output_f16));

  CHECK_ACL_GTEST(aclrtDestroyStream(stream));
}

TEST(NpuOpsTest, GemmAWQ4Bit) {
  aclrtStream stream = nullptr;

  GemmAWQ4BitOpTest op_test;
  op_test.Init(hidden_dim, ffn_hidden, ffn_hidden);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> m_dis(2, hidden_dim);

  std::vector<size_t> m_list = {1};
  constexpr int rnd_m_num = 10;
  for (int i = 0; i < rnd_m_num; ++i) {
    m_list.push_back(m_dis(gen));
  }

  for (size_t m : m_list) {
    for (size_t k : {(size_t)hidden_dim, (size_t)ffn_hidden}) {
      for (size_t n : {(size_t)hidden_dim, (size_t)ffn_hidden}) {

        ASSERT_TRUE(op_test.Run(m, n, k))
            << "m: " << m << " n: " << n << " k: " << k;

      } // loop n
    }   // loop k
  }
}

static void InitFreqCIS(float *freq_cis) {
  const float theta = 10000.0f;
  int head_dim = hidden_dim / head_num;
  int freq_len = head_dim / 2;
  float *freq = new float[freq_len];

  for (int i = 0; i < freq_len; ++i) {
    freq[i] =
        1.0f /
        (powf(theta, static_cast<float>(i * 2) / static_cast<float>(head_dim)));
  }

  float *t = new float[max_seq_len];
  for (int i = 0; i < max_seq_len; ++i) {
    t[i] = static_cast<float>(i);
  }

  float *freq_outer = new float[freq_len * max_seq_len];

  // max_seq_len row, freq_len column
  for (int i = 0; i < max_seq_len; ++i) {
    for (int j = 0; j < freq_len; ++j) {
      freq_outer[i * freq_len + j] = t[i] * freq[j];
    }
  }

  for (int i = 0; i < max_seq_len * freq_len; ++i) {
    freq_cis[i * 2] = std::cos(freq_outer[i]);
    freq_cis[i * 2 + 1] = std::sin(freq_outer[i]);
  }

  delete[] freq;
  delete[] t;
  delete[] freq_outer;
}

TEST(NpuOpsTest, RopeLayer) {
  aclrtStream stream = nullptr;
  CHECK_ACL_GTEST(aclrtCreateStream(&stream));

  size_t max_buffer_size = hidden_dim * max_seq_len;
  size_t rope_dim = hidden_dim * max_seq_len;
  size_t freq_len = head_dim / 2;
  void *dev_input_q_f16;
  void *dev_input_k_f16;
  void *dev_output_q_f16;
  void *dev_output_k_f16;
  void *dev_freq_cis;

  float *host_input_q = new float[max_buffer_size];
  float *host_input_k = new float[max_buffer_size];
  float *host_output_q = new float[max_buffer_size];
  float *host_output_k = new float[max_buffer_size];
  Eigen::half *host_input_q_f16 = new Eigen::half[max_buffer_size];
  Eigen::half *host_input_k_f16 = new Eigen::half[max_buffer_size];
  Eigen::half *host_output_q_f16 = new Eigen::half[max_buffer_size];
  Eigen::half *host_output_k_f16 = new Eigen::half[max_buffer_size];
  float *golden_q_fp32 = new float[max_buffer_size];
  float *golden_k_fp32 = new float[max_buffer_size];
  float *host_freq_cis = new float[rope_dim];
  InitFreqCIS(host_freq_cis);

  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_input_q_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_input_k_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_output_q_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_output_k_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_freq_cis, rope_dim * sizeof(float),
                              ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK_ACL_GTEST(aclrtMemcpy(dev_freq_cis, rope_dim * sizeof(float),
                              host_freq_cis, rope_dim * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));

  for (int seq_len = 1; seq_len <= max_seq_len; ++seq_len) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> start_pos_dis(0, max_seq_len - seq_len);
    int start_pos = start_pos_dis(gen);

    std::cout << "test rope: "
              << "seq_len: " << seq_len << " start pos " << start_pos << "\n";
    int total_element_cnt = seq_len * hidden_dim;
    make_random_float(host_input_q, total_element_cnt);
    make_random_float(host_input_k, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_q_fp32_map((float *)host_input_q, total_element_cnt);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_q_fp16_map((Eigen::half *)host_input_q_f16, total_element_cnt);
    input_q_fp16_map = input_q_fp32_map.cast<Eigen::half>();
    input_q_fp32_map = input_q_fp16_map.cast<float>();

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_k_fp32_map((float *)host_input_k, total_element_cnt);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_k_fp16_map((Eigen::half *)host_input_k_f16, total_element_cnt);
    input_k_fp16_map = input_k_fp32_map.cast<Eigen::half>();
    input_k_fp32_map = input_k_fp16_map.cast<float>();

    for (int s = 0; s < seq_len; ++s) {
      for (int n = 0; n < head_num; ++n) {
        for (int f = 0; f < freq_len; ++f) {
          float fc = host_freq_cis[(s + start_pos) * freq_len * 2 + 2 * f];
          float fd = host_freq_cis[(s + start_pos) * freq_len * 2 + 2 * f + 1];

          int hidden_offset = s * hidden_dim + n * head_dim;

          float qa = host_input_q[hidden_offset + 2 * f];
          float qb = host_input_q[hidden_offset + 2 * f + 1];

          float ka = host_input_k[hidden_offset + 2 * f];
          float kb = host_input_k[hidden_offset + 2 * f + 1];

          golden_q_fp32[hidden_offset + 2 * f] = qa * fc - qb * fd;
          golden_q_fp32[hidden_offset + 2 * f + 1] = qa * fd + qb * fc;

          golden_k_fp32[hidden_offset + 2 * f] = ka * fc - kb * fd;
          golden_k_fp32[hidden_offset + 2 * f + 1] = ka * fd + kb * fc;
        }
      }
    }

    CHECK_ACL_GTEST(
        aclrtMemcpy(dev_input_q_f16, total_element_cnt * sizeof(aclFloat16),
                    host_input_q_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_GTEST(
        aclrtMemcpy(dev_input_k_f16, total_element_cnt * sizeof(aclFloat16),
                    host_input_k_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_HOST_TO_DEVICE));

    npu_rope_layer(dev_output_q_f16, dev_output_k_f16, dev_freq_cis,
                   dev_input_q_f16, dev_input_k_f16, start_pos, seq_len,
                   head_num, hidden_dim, false, DT_FLOAT16, stream);
    CHECK_ACL_GTEST(aclrtSynchronizeStream(stream));

    CHECK_ACL_GTEST(
        aclrtMemcpy(host_output_q_f16, total_element_cnt * sizeof(aclFloat16),
                    dev_output_q_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL_GTEST(
        aclrtMemcpy(host_output_k_f16, total_element_cnt * sizeof(aclFloat16),
                    dev_output_k_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_DEVICE_TO_HOST));

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_q_fp32_map((float *)host_output_q, total_element_cnt);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_q_fp16_map((Eigen::half *)host_output_q_f16, total_element_cnt);
    output_q_fp32_map = output_q_fp16_map.cast<float>();

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_k_fp32_map((float *)host_output_k, total_element_cnt);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_k_fp16_map((Eigen::half *)host_output_k_f16, total_element_cnt);
    output_k_fp32_map = output_k_fp16_map.cast<float>();

    ASSERT_TRUE(all_close(golden_q_fp32, host_output_q, total_element_cnt))
        << "seq_len: " << seq_len;

    ASSERT_TRUE(all_close(golden_k_fp32, host_output_k, total_element_cnt))
        << "seq_len: " << seq_len;
  }

  delete[] host_input_q;
  delete[] host_input_k;
  delete[] host_output_q;
  delete[] host_output_k;
  delete[] host_input_q_f16;
  delete[] host_input_k_f16;
  delete[] host_output_q_f16;
  delete[] host_output_k_f16;
  delete[] golden_q_fp32;
  delete[] golden_k_fp32;
  delete[] host_freq_cis;

  CHECK_ACL_GTEST(aclrtFree(dev_input_q_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_input_k_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_output_q_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_output_k_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_freq_cis));

  CHECK_ACL_GTEST(aclrtDestroyStream(stream));
}

TEST(NpuOpsTest, RopeSingleLayer) {
  aclrtStream stream = nullptr;
  CHECK_ACL_GTEST(aclrtCreateStream(&stream));

  size_t max_buffer_size = hidden_dim * max_seq_len;
  size_t rope_dim = hidden_dim * max_seq_len;
  size_t freq_len = head_dim / 2;
  void *dev_input_q_f16;
  void *dev_input_k_f16;
  void *dev_output_q_f16;
  void *dev_output_k_f16;
  void *dev_freq_cis;

  float *host_input_q = new float[max_buffer_size];
  float *host_input_k = new float[max_buffer_size];
  float *host_output_q = new float[max_buffer_size];
  float *host_output_k = new float[max_buffer_size];
  Eigen::bfloat16 *host_input_q_f16 = new Eigen::bfloat16[max_buffer_size];
  Eigen::bfloat16 *host_input_k_f16 = new Eigen::bfloat16[max_buffer_size];
  Eigen::bfloat16 *host_output_q_f16 = new Eigen::bfloat16[max_buffer_size];
  Eigen::bfloat16 *host_output_k_f16 = new Eigen::bfloat16[max_buffer_size];
  float *golden_q_fp32 = new float[max_buffer_size];
  float *golden_k_fp32 = new float[max_buffer_size];
  float *host_freq_cis = new float[rope_dim];
  InitFreqCIS(host_freq_cis);

  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_input_q_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_input_k_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_output_q_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_output_k_f16,
                              max_buffer_size * sizeof(aclFloat16),
                              ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL_GTEST(aclrtMalloc((void **)&dev_freq_cis, rope_dim * sizeof(float),
                              ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK_ACL_GTEST(aclrtMemcpy(dev_freq_cis, rope_dim * sizeof(float),
                              host_freq_cis, rope_dim * sizeof(float),
                              ACL_MEMCPY_HOST_TO_DEVICE));

  for (int seq_len = 1; seq_len <= max_seq_len; ++seq_len) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> start_pos_dis(0, max_seq_len - seq_len);
    int start_pos = start_pos_dis(gen);

    std::cout << "test rope: "
              << "seq_len: " << seq_len << " start pos " << start_pos << "\n";
    int total_element_cnt = seq_len * hidden_dim;
    make_random_float(host_input_q, total_element_cnt);
    make_random_float(host_input_k, total_element_cnt);

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_q_fp32_map((float *)host_input_q, total_element_cnt);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::bfloat16, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_q_fp16_map((Eigen::bfloat16 *)host_input_q_f16, total_element_cnt);
    input_q_fp16_map = input_q_fp32_map.cast<Eigen::bfloat16>();
    input_q_fp32_map = input_q_fp16_map.cast<float>();

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_k_fp32_map((float *)host_input_k, total_element_cnt);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::bfloat16, 1, Eigen::RowMajor | Eigen::DontAlign>>
        input_k_fp16_map((Eigen::bfloat16 *)host_input_k_f16, total_element_cnt);
    input_k_fp16_map = input_k_fp32_map.cast<Eigen::bfloat16>();
    input_k_fp32_map = input_k_fp16_map.cast<float>();

    for (int s = 0; s < seq_len; ++s) {
      for (int n = 0; n < head_num; ++n) {
        for (int f = 0; f < freq_len; ++f) {
          float fc = host_freq_cis[(s + start_pos) * freq_len * 2 + 2 * f];
          float fd = host_freq_cis[(s + start_pos) * freq_len * 2 + 2 * f + 1];

          int hidden_offset = s * hidden_dim + n * head_dim;

          float qa = host_input_q[hidden_offset + 2 * f];
          float qb = host_input_q[hidden_offset + 2 * f + 1];

          float ka = host_input_k[hidden_offset + 2 * f];
          float kb = host_input_k[hidden_offset + 2 * f + 1];

          golden_q_fp32[hidden_offset + 2 * f] = qa * fc - qb * fd;
          golden_q_fp32[hidden_offset + 2 * f + 1] = qa * fd + qb * fc;

          golden_k_fp32[hidden_offset + 2 * f] = ka * fc - kb * fd;
          golden_k_fp32[hidden_offset + 2 * f + 1] = ka * fd + kb * fc;
        }
      }
    }

    CHECK_ACL_GTEST(
        aclrtMemcpy(dev_input_q_f16, total_element_cnt * sizeof(aclFloat16),
                    host_input_q_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL_GTEST(
        aclrtMemcpy(dev_input_k_f16, total_element_cnt * sizeof(aclFloat16),
                    host_input_k_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_HOST_TO_DEVICE));

    npu_rope_single_layer(dev_output_q_f16, dev_freq_cis,
                   dev_input_q_f16, start_pos, seq_len,
                   head_num, hidden_dim, true, DT_BFLOAT16, stream);
    CHECK_ACL_GTEST(aclrtSynchronizeStream(stream));

    CHECK_ACL_GTEST(
        aclrtMemcpy(host_output_q_f16, total_element_cnt * sizeof(aclFloat16),
                    dev_output_q_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL_GTEST(
        aclrtMemcpy(host_output_k_f16, total_element_cnt * sizeof(aclFloat16),
                    dev_output_k_f16, total_element_cnt * sizeof(aclFloat16),
                    ACL_MEMCPY_DEVICE_TO_HOST));

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_q_fp32_map((float *)host_output_q, total_element_cnt);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::bfloat16, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_q_fp16_map((Eigen::bfloat16 *)host_output_q_f16, total_element_cnt);
    output_q_fp32_map = output_q_fp16_map.cast<float>();

    Eigen::TensorMap<
        Eigen::Tensor<float, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_k_fp32_map((float *)host_output_k, total_element_cnt);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::bfloat16, 1, Eigen::RowMajor | Eigen::DontAlign>>
        output_k_fp16_map((Eigen::bfloat16 *)host_output_k_f16, total_element_cnt);
    output_k_fp32_map = output_k_fp16_map.cast<float>();

    ASSERT_TRUE(all_close(golden_q_fp32, host_output_q, total_element_cnt))
        << "seq_len: " << seq_len;
  }

  delete[] host_input_q;
  delete[] host_input_k;
  delete[] host_output_q;
  delete[] host_output_k;
  delete[] host_input_q_f16;
  delete[] host_input_k_f16;
  delete[] host_output_q_f16;
  delete[] host_output_k_f16;
  delete[] golden_q_fp32;
  delete[] golden_k_fp32;
  delete[] host_freq_cis;

  CHECK_ACL_GTEST(aclrtFree(dev_input_q_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_input_k_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_output_q_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_output_k_f16));
  CHECK_ACL_GTEST(aclrtFree(dev_freq_cis));

  CHECK_ACL_GTEST(aclrtDestroyStream(stream));
}
