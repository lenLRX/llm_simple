#include <iostream>
#include <cmath>
#include <limits>
#include <string>

#include <gtest/gtest.h>
#include <acl/acl.h>
#include <random>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <fmt/format.h>


#include "npu_ops.h"


#define CHECK_ACL(x)                                                           \
  do {                                                                         \
    aclError __ret = x;                                                        \
    if (__ret != ACL_ERROR_NONE) {                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret        \
                << std::endl;                                                  \
      ASSERT_TRUE(false);                                                      \
    }                                                                          \
  } while (0);

#define CHECK_ACL2(x)                                                          \
  do {                                                                         \
    aclError __ret = x;                                                        \
    if (__ret != ACL_ERROR_NONE) {                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret        \
                << std::endl;                                                  \
    }                                                                          \
  } while (0);

class ACLEnvironment : public ::testing::Environment {
 public:
  ~ACLEnvironment() override {}

  // Override this to define how to set up the environment.
  void SetUp() override {
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
  }

  // Override this to define how to tear down the environment.
  void TearDown() override {
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
  }
  aclrtContext context;
  int32_t deviceId{0};
};

class ACLTimer {
public:
    ACLTimer(const std::string& desc, aclrtStream stream):desc(desc), stream(stream) {
        CHECK_ACL2(aclrtCreateEvent(&start_event));
        CHECK_ACL2(aclrtCreateEvent(&end_event));
    }
    ~ACLTimer() {
        CHECK_ACL2(aclrtDestroyEvent(start_event));
        CHECK_ACL2(aclrtDestroyEvent(end_event));
    }

    void Start() {
        CHECK_ACL2(aclrtRecordEvent(start_event, stream));
    }

    void Stop() {
        CHECK_ACL2(aclrtRecordEvent(end_event, stream));
        CHECK_ACL2(aclrtSynchronizeStream(stream));
    }
    void Print() {
        float duration_ms;
        CHECK_ACL2(aclrtEventElapsedTime(&duration_ms, start_event, end_event));
        std::cout << desc << " duration: " << duration_ms << "ms" << std::endl;
    }
private:
    std::string desc;
    aclrtStream stream;
    aclrtEvent start_event;
    aclrtEvent end_event;
};

testing::Environment* const acl_env =
    testing::AddGlobalTestEnvironment(new ACLEnvironment);

constexpr size_t max_seq_len = 2048;
constexpr size_t head_num = 32;
constexpr size_t head_dim = 128;
constexpr size_t hidden_dim = 4096;
constexpr size_t ffn_hidden = 11008;


void make_random_float(float* buffer, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (size_t i = 0; i < size; ++i) {
      buffer[i] = dis(gen);
    }
}

void make_random_float_uint4(float* buffer, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);

    for (size_t i = 0; i < size; ++i) {
      buffer[i] = static_cast<float>(dis(gen));
    }
}


bool all_close(float* output_buffer, float* golden_buffer, size_t size,
              float abs_err = 0.001f, float relative_err =  0.001f) {
  for (size_t i = 0; i < size; ++i) {
      float a = output_buffer[i];
      float b = golden_buffer[i];
      
      float abs_diff = std::fabs(a - b);
      float max_abs_val = std::max(std::fabs(a), std::fabs(b));
      if (abs_diff > abs_err && (abs_diff / max_abs_val) > relative_err) {
        std::cout << "all_close failed, output [" << i << "] :" << a << " vs " << b << std::endl;
        return false;
      }
  }
  return true;
}

bool all_close_inf(float* output_buffer, float* golden_buffer, size_t size) {
  for (size_t i = 0; i < size; ++i) {
      float a = output_buffer[i];
      float b = golden_buffer[i];

      if (std::isinf(b)) {
        if (std::isinf(a) && std::signbit(a) == std::signbit(b)) {
          continue;
        }
        std::cout << "all_close failed, output [" << i << "] :" << a << " vs " << b << std::endl;
        return false;
      }
      
      float abs_diff = std::fabs(a - b);
      float max_abs_val = std::max(std::fabs(a), std::fabs(b));

      if (abs_diff > 0.001f && (abs_diff / max_abs_val) > 0.001f) {
        std::cout << "all_close failed, output [" << i << "] :" << a << " vs " << b << std::endl;
        return false;
      }
  }
  return true;
}


TEST(NpuOpsTest, SoftmaxLayer) {
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    size_t max_buffer_size = head_num * max_seq_len * max_seq_len;
    void* dev_input_f16;
    void* dev_output_f16;

    float* host_input = new float[max_buffer_size];
    float* host_output = new float[max_buffer_size];
    Eigen::half* host_input_f16 = new Eigen::half[max_buffer_size];
    Eigen::half* host_output_f16 = new Eigen::half[max_buffer_size];
    float* golden_fp32 = new float[max_buffer_size];
    CHECK_ACL(aclrtMalloc((void**)&dev_input_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&dev_output_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));

    for (int curr_pos = 1; curr_pos <= max_seq_len; ++curr_pos) {
        //for (int curr_size = 1; curr_size <= max_seq_len; ++curr_size) {
        for (int curr_size : {1, 7}) {
          std::cout << "test softmax: " << "curr_pos: " << curr_pos << " curr_size: " << curr_size << "\n";
          int hs = head_num * curr_size;
          int total_element_cnt = hs * curr_pos;

          make_random_float(host_input, total_element_cnt);

          Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
          golden_fp32_map((float*)golden_fp32, hs, curr_pos);

          Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
          input_map((float*)host_input, hs, curr_pos);

          Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
          input_fp16_map((Eigen::half*)host_input_f16, hs, curr_pos);
          input_fp16_map = input_map.cast<Eigen::half>();
          input_map = input_fp16_map.cast<float>();

          auto input_max = input_map.maximum(Eigen::array<int, 1>{1}).eval()
              .reshape(Eigen::array<int, 2>{hs, 1}).broadcast(Eigen::array<int, 2>{1, curr_pos});

          auto input_diff = (input_map - input_max).exp().eval();

          auto input_sum = input_diff.sum(Eigen::array<int, 1>{1}).eval()
              .reshape(Eigen::array<int, 2>{hs, 1}).broadcast(Eigen::array<int, 2>{1, curr_pos});

          golden_fp32_map = input_diff/input_sum;

          CHECK_ACL(aclrtMemcpy(dev_input_f16, total_element_cnt * sizeof(aclFloat16), host_input_f16,
                                total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

          npu_softmax_layer(dev_output_f16,
                            dev_input_f16,
                            hs,
                            curr_pos,
                            DT_FLOAT16,
                            stream);
          CHECK_ACL(aclrtSynchronizeStream(stream));

          CHECK_ACL(aclrtMemcpy(host_output_f16, total_element_cnt * sizeof(aclFloat16), dev_output_f16, total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

          Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
          output_fp32_map((float*)host_output, hs, curr_pos);

          Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
          output_fp16_map((Eigen::half*)host_output_f16, hs, curr_pos);
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
          
    CHECK_ACL(aclrtFree(dev_input_f16));
    CHECK_ACL(aclrtFree(dev_output_f16));

    CHECK_ACL(aclrtDestroyStream(stream));
}


TEST(NpuOpsTest, RMSNormLayer) {
  aclrtStream stream = nullptr;
  float eps = 1e-5;
  CHECK_ACL(aclrtCreateStream(&stream));

  size_t max_buffer_size = hidden_dim * max_seq_len;
  void* dev_input_f16;
  void* dev_output_f16;
  void* dev_weight_f16;

  float* host_input = new float[max_buffer_size];
  float* host_output = new float[max_buffer_size];
  Eigen::half* host_input_f16 = new Eigen::half[max_buffer_size];
  Eigen::half* host_output_f16 = new Eigen::half[max_buffer_size];
  float* golden_fp32 = new float[max_buffer_size];
  float* weight_fp32 = new float[hidden_dim];
  Eigen::half* weight_fp16 = new Eigen::half[hidden_dim];


  CHECK_ACL(aclrtMalloc((void**)&dev_input_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_output_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_weight_f16, hidden_dim * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));

  for (int seq_len = 1; seq_len <= max_seq_len; ++seq_len) {
    std::cout << "test rmsnorm: " << "seq_len: " << seq_len << "\n";
    int total_element_cnt = seq_len * hidden_dim;
    make_random_float(host_input, total_element_cnt);
    make_random_float(weight_fp32, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    golden_fp32_map((float*)golden_fp32, seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_map((float*)host_input, seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_fp16_map((Eigen::half*)host_input_f16, seq_len, hidden_dim);
    input_fp16_map = input_map.cast<Eigen::half>();
    input_map = input_fp16_map.cast<float>();

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    weight_map((float*)weight_fp32, 1, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    weight_fp16_map((Eigen::half*)weight_fp16, 1, hidden_dim);
    weight_fp16_map = weight_map.cast<Eigen::half>();
    weight_map = weight_fp16_map.cast<float>();

    std::array<long,1> mean_dims         = {1};
    Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign> mean = (input_map*input_map).mean(mean_dims).eval().reshape(std::array<long,2>{seq_len, 1});
    Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign> sqrt_mean_add_eps = (mean + mean.constant(eps)).sqrt().eval().reshape(std::array<long,2>{seq_len, 1});
    golden_fp32_map = (input_map / sqrt_mean_add_eps.broadcast(std::array<size_t, 2>{1, hidden_dim})
      * weight_map.broadcast(std::array<size_t, 2>{(size_t)seq_len, 1}));

    CHECK_ACL(aclrtMemcpy(dev_input_f16, total_element_cnt * sizeof(aclFloat16), host_input_f16,
                                total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
    
    CHECK_ACL(aclrtMemcpy(dev_weight_f16, hidden_dim * sizeof(float), weight_fp16,
                                hidden_dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

    npu_rmsnorm_layer(dev_output_f16,
                      dev_weight_f16,
                      dev_input_f16,
                      seq_len,
                      hidden_dim,
                      eps,
                      DT_FLOAT16,
                      stream);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(host_output_f16, total_element_cnt * sizeof(aclFloat16), dev_output_f16, total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_fp32_map((float*)host_output, seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_fp16_map((Eigen::half*)host_output_f16, seq_len, hidden_dim);
    output_fp32_map = output_fp16_map.cast<float>();

    ASSERT_TRUE(all_close(host_output, golden_fp32, total_element_cnt))
        << "seq_len: " << seq_len;

  }

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

  CHECK_ACL(aclrtDestroyStream(stream));
}



TEST(NpuOpsTest, GemmNzLayer) {
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  size_t max_n = 32000;

  size_t max_lhs_buffer_size = hidden_dim * hidden_dim;
  size_t max_rhs_buffer_size = hidden_dim * max_n;
  size_t max_output_buffer_size = hidden_dim * max_n;
  void* dev_lhs_f16;
  void* dev_rhs_f16;
  void* dev_output_f16;

  float* host_lhs = new float[max_lhs_buffer_size];
  float* host_rhs = new float[max_rhs_buffer_size];
  float* host_output = new float[max_output_buffer_size];
  Eigen::half* host_lhs_f16 = new Eigen::half[max_lhs_buffer_size];
  Eigen::half* host_rhs_f16 = new Eigen::half[max_rhs_buffer_size];
  Eigen::half* host_rhs_f16_nz = new Eigen::half[max_rhs_buffer_size];
  Eigen::half* host_output_f16 = new Eigen::half[max_output_buffer_size];
  float* golden_fp32 = new float[max_output_buffer_size];


  CHECK_ACL(aclrtMalloc((void**)&dev_lhs_f16, max_lhs_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_rhs_f16, max_rhs_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_output_f16, max_output_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> m_dis(2, hidden_dim);

  std::vector<size_t> m_list = {1};
  constexpr int rnd_m_num = 10;
  for (int i = 0; i < rnd_m_num; ++i) {
    m_list.push_back(m_dis(gen));
  }

  //for (size_t m = 1; m <= hidden_dim; ++m) {
  for (size_t m: m_list) {
    for (size_t n : {(size_t)hidden_dim, max_n}) {
      size_t n1 = n/16;
      std::cout << "test gemm m: " << m << " n: " << n << "\n";
      size_t lhs_element_cnt = m * hidden_dim;
      size_t rhs_element_cnt = n * hidden_dim;
      size_t output_element_cnt = m * n;
      make_random_float(host_lhs, lhs_element_cnt);
      make_random_float(host_rhs, rhs_element_cnt);

      Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
      golden_fp32_map((float*)golden_fp32, m, n);

      Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
      input_lhs_map((float*)host_lhs, m, hidden_dim);

      Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
      input_lhs_fp16_map((Eigen::half*)host_lhs_f16, m, hidden_dim);
      input_lhs_fp16_map = input_lhs_map.cast<Eigen::half>();
      input_lhs_map = input_lhs_fp16_map.cast<float>();

      Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
      input_rhs_map((float*)host_rhs, hidden_dim, n);

      Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
      input_rhs_fp16_map((Eigen::half*)host_rhs_f16, hidden_dim, n);
      input_rhs_fp16_map = input_rhs_map.cast<Eigen::half>();
      input_rhs_map = input_rhs_fp16_map.cast<float>();

      Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      input_rhs_fp16_nz_map((Eigen::half*)host_rhs_f16_nz, n1, hidden_dim, 16);

      input_rhs_fp16_nz_map = input_rhs_fp16_map.reshape(Eigen::array<int, 3>{hidden_dim, (int)n1, 16})
        .shuffle(Eigen::array<int, 3>({1, 0, 2}));

      CHECK_ACL(aclrtMemcpy(dev_lhs_f16, lhs_element_cnt * sizeof(aclFloat16), host_lhs_f16,
                                lhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
      CHECK_ACL(aclrtMemcpy(dev_rhs_f16, rhs_element_cnt * sizeof(aclFloat16), host_rhs_f16_nz,
                                rhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

      npu_matmul_nz_layer(dev_output_f16,
                          dev_lhs_f16,
                          dev_rhs_f16,
                          m, n, hidden_dim,
                          DT_FLOAT16,
                          stream);
      
      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
      golden_fp32_map = input_lhs_map.contract(input_rhs_map, product_dims);

      CHECK_ACL(aclrtSynchronizeStream(stream));

      CHECK_ACL(aclrtMemcpy(host_output_f16, output_element_cnt * sizeof(aclFloat16), dev_output_f16, output_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

      Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
      output_fp32_map((float*)host_output, m, n);

      Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
      output_fp16_map((Eigen::half*)host_output_f16, m, n);
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
          
  CHECK_ACL(aclrtFree(dev_lhs_f16));
  CHECK_ACL(aclrtFree(dev_rhs_f16));
  CHECK_ACL(aclrtFree(dev_output_f16));

  CHECK_ACL(aclrtDestroyStream(stream));
}


TEST(NpuOpsTest, SiluMul) {
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  size_t max_buffer_size = ffn_hidden * max_seq_len;
  void* dev_input_0_f16;
  void* dev_input_1_f16;
  void* dev_output_f16;

  float* host_input_0 = new float[max_buffer_size];
  float* host_input_1 = new float[max_buffer_size];
  float* host_output = new float[max_buffer_size];
  Eigen::half* host_input_0_f16 = new Eigen::half[max_buffer_size];
  Eigen::half* host_input_1_f16 = new Eigen::half[max_buffer_size];
  Eigen::half* host_output_f16 = new Eigen::half[max_buffer_size];
  float* golden_fp32 = new float[max_buffer_size];


  CHECK_ACL(aclrtMalloc((void**)&dev_input_0_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_input_1_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_output_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));

  for (int seq_len = 1; seq_len <= max_seq_len; ++seq_len) {
    std::cout << "test silu_mul: " << "seq_len: " << seq_len << "\n";
    int total_element_cnt = seq_len * ffn_hidden;
    make_random_float(host_input_0, total_element_cnt);
    make_random_float(host_input_1, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    golden_fp32_map((float*)golden_fp32, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_0_map((float*)host_input_0, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_1_map((float*)host_input_1, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_0_fp16_map((Eigen::half*)host_input_0_f16, total_element_cnt);
    input_0_fp16_map = input_0_map.cast<Eigen::half>();
    input_0_map = input_0_fp16_map.cast<float>();

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_1_fp16_map((Eigen::half*)host_input_1_f16, total_element_cnt);
    input_1_fp16_map = input_1_map.cast<Eigen::half>();
    input_1_map = input_1_fp16_map.cast<float>();

    golden_fp32_map = (input_0_map / ((-input_0_map).exp() + input_0_map.constant(1))) * input_1_map;
    CHECK_ACL(aclrtMemcpy(dev_input_0_f16, total_element_cnt * sizeof(aclFloat16), host_input_0_f16,
                                total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_input_1_f16, total_element_cnt * sizeof(aclFloat16), host_input_1_f16,
                                total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

    npu_silu_mul_layer(dev_output_f16,
                      dev_input_0_f16,
                      dev_input_1_f16,
                      total_element_cnt,
                      DT_FLOAT16,
                      stream);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(host_output_f16, total_element_cnt * sizeof(aclFloat16), dev_output_f16, total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    output_fp32_map((float*)host_output, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    output_fp16_map((Eigen::half*)host_output_f16, total_element_cnt);
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
          
  CHECK_ACL(aclrtFree(dev_input_0_f16));
  CHECK_ACL(aclrtFree(dev_input_1_f16));
  CHECK_ACL(aclrtFree(dev_output_f16));

  CHECK_ACL(aclrtDestroyStream(stream));
}



TEST(NpuOpsTest, Add) {
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  size_t max_buffer_size = ffn_hidden * max_seq_len;
  void* dev_input_0_f16;
  void* dev_input_1_f16;
  void* dev_output_f16;

  float* host_input_0 = new float[max_buffer_size];
  float* host_input_1 = new float[max_buffer_size];
  float* host_output = new float[max_buffer_size];
  Eigen::half* host_input_0_f16 = new Eigen::half[max_buffer_size];
  Eigen::half* host_input_1_f16 = new Eigen::half[max_buffer_size];
  Eigen::half* host_output_f16 = new Eigen::half[max_buffer_size];
  float* golden_fp32 = new float[max_buffer_size];


  CHECK_ACL(aclrtMalloc((void**)&dev_input_0_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_input_1_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_output_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));

  for (int seq_len = 1; seq_len <= max_seq_len; ++seq_len) {
    std::cout << "test add: " << "seq_len: " << seq_len << "\n";
    int total_element_cnt = seq_len * ffn_hidden;
    make_random_float(host_input_0, total_element_cnt);
    make_random_float(host_input_1, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    golden_fp32_map((float*)golden_fp32, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_0_map((float*)host_input_0, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_1_map((float*)host_input_1, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_0_fp16_map((Eigen::half*)host_input_0_f16, total_element_cnt);
    input_0_fp16_map = input_0_map.cast<Eigen::half>();
    input_0_map = input_0_fp16_map.cast<float>();

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_1_fp16_map((Eigen::half*)host_input_1_f16, total_element_cnt);
    input_1_fp16_map = input_1_map.cast<Eigen::half>();
    input_1_map = input_1_fp16_map.cast<float>();

    CHECK_ACL(aclrtMemcpy(dev_input_0_f16, total_element_cnt * sizeof(aclFloat16), host_input_0_f16,
                                total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_input_1_f16, total_element_cnt * sizeof(aclFloat16), host_input_1_f16,
                                total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

    npu_add_layer(dev_output_f16,
                      dev_input_0_f16,
                      dev_input_1_f16,
                      total_element_cnt,
                      DT_FLOAT16,
                      stream);
    
    golden_fp32_map = input_0_map + input_1_map;
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(host_output_f16, total_element_cnt * sizeof(aclFloat16), dev_output_f16, total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    output_fp32_map((float*)host_output, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    output_fp16_map((Eigen::half*)host_output_f16, total_element_cnt);
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
          
  CHECK_ACL(aclrtFree(dev_input_0_f16));
  CHECK_ACL(aclrtFree(dev_input_1_f16));
  CHECK_ACL(aclrtFree(dev_output_f16));

  CHECK_ACL(aclrtDestroyStream(stream));
}


TEST(NpuOpsTest, BMMQKTransCausual) {
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  float qk_scale = 1/sqrtf(static_cast<float>(head_dim));

  size_t max_lhs_buffer_size = max_seq_len * hidden_dim;
  size_t max_rhs_buffer_size = max_seq_len * hidden_dim;
  size_t max_output_buffer_size = head_num * max_seq_len * max_seq_len;

  void* dev_lhs_f16;
  void* dev_rhs_f16;
  void* dev_output_f16;

  float* host_lhs = new float[max_lhs_buffer_size];
  float* host_rhs = new float[max_rhs_buffer_size];
  float* host_output = new float[max_output_buffer_size];
  float* host_mask = new float[max_output_buffer_size];
  Eigen::half* host_lhs_f16 = new Eigen::half[max_lhs_buffer_size];
  Eigen::half* host_rhs_f16 = new Eigen::half[max_rhs_buffer_size];
  Eigen::half* host_output_f16 = new Eigen::half[max_output_buffer_size];
  float* golden_fp32 = new float[max_output_buffer_size];


  CHECK_ACL(aclrtMalloc((void**)&dev_lhs_f16, max_lhs_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_rhs_f16, max_rhs_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_output_f16, max_output_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> cur_size_dis(2, max_seq_len);

  std::vector<size_t> cur_size_list = {1};

  constexpr int rnd_m_num = 10;
  constexpr int rnd_n_num = 10;
  for (int i = 0; i < rnd_m_num; ++i) {
    cur_size_list.push_back(cur_size_dis(gen));
  }

  for (size_t cur_size: cur_size_list) {
    std::uniform_int_distribution<size_t> cur_pos_dis(cur_size, max_seq_len);
    std::vector<size_t> cur_pos_list;
    for (int i = 0; i < rnd_n_num; ++i) {
      cur_pos_list.push_back(cur_pos_dis(gen));
    }
    for (size_t cur_pos: cur_pos_list) {
      size_t prev_pos = cur_pos - cur_size;
      std::cout << "test cur_size: " << cur_size << " cur_pos: " << cur_pos << "\n";

      size_t lhs_element_cnt = head_num * cur_size * head_dim;
      size_t rhs_element_cnt = head_num * cur_pos * head_dim;
      size_t output_element_cnt = head_num * cur_size * cur_pos;
      make_random_float(host_lhs, lhs_element_cnt);
      make_random_float(host_rhs, rhs_element_cnt);

      for (int row = 0; row < cur_size; ++row) {
        for (int col = 0; col < cur_pos; ++col) {
          if (col > (row+prev_pos)) {
            host_mask[row*cur_pos+col] = -std::numeric_limits<float>::infinity();
          }
          else {
            host_mask[row*cur_pos+col] = 0;
          }
        }
      }

      Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
      mask_map(host_mask, cur_size, cur_pos);
      
      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      golden_fp32_map((float*)golden_fp32, head_num, cur_size, cur_pos);

      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      input_lhs_map((float*)host_lhs, cur_size, head_num, head_dim);

      Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      input_lhs_fp16_map((Eigen::half*)host_lhs_f16, cur_size, head_num, head_dim);
      input_lhs_fp16_map = input_lhs_map.cast<Eigen::half>();
      input_lhs_map = input_lhs_fp16_map.cast<float>();

      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      input_rhs_map((float*)host_rhs, cur_pos, head_num, head_dim);

      Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      input_rhs_fp16_map((Eigen::half*)host_rhs_f16, cur_pos, head_num, head_dim);
      input_rhs_fp16_map = input_rhs_map.cast<Eigen::half>();
      input_rhs_map = input_rhs_fp16_map.cast<float>();

      CHECK_ACL(aclrtMemcpy(dev_lhs_f16, lhs_element_cnt * sizeof(aclFloat16), host_lhs_f16,
                                lhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
      CHECK_ACL(aclrtMemcpy(dev_rhs_f16, rhs_element_cnt * sizeof(aclFloat16), host_rhs_f16,
                                rhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
      
      npu_batch_matmul_qk_trans_causual_layer(dev_output_f16,
                                              dev_lhs_f16,
                                              dev_rhs_f16,
                                              head_num,
                                              cur_size,
                                              cur_pos,
                                              head_dim,
                                              prev_pos,
                                              qk_scale,
                                              DT_FLOAT16,
                                              stream);
      Eigen::array<Eigen::IndexPair<int>, 1> qk_product_dims = { Eigen::IndexPair<int>(1, 0) };
      for (int i = 0; i < head_num; ++i) {
        golden_fp32_map.chip<0>(i) = input_lhs_map.shuffle(Eigen::array<int, 3>({1, 0, 2}))
          .chip<0>(i).contract(input_rhs_map.shuffle(Eigen::array<int, 3>({1, 2, 0})).chip<0>(i), qk_product_dims) + mask_map;
      }

      golden_fp32_map = golden_fp32_map * golden_fp32_map.constant(qk_scale);

      CHECK_ACL(aclrtSynchronizeStream(stream));
      CHECK_ACL(aclrtMemcpy(host_output_f16, output_element_cnt * sizeof(aclFloat16),
                            dev_output_f16, output_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      output_fp32_map((float*)host_output, head_num, cur_size, cur_pos);

      Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      output_fp16_map((Eigen::half*)host_output_f16, head_num, cur_size, cur_pos);
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
          
  CHECK_ACL(aclrtFree(dev_lhs_f16));
  CHECK_ACL(aclrtFree(dev_rhs_f16));
  CHECK_ACL(aclrtFree(dev_output_f16));

  CHECK_ACL(aclrtDestroyStream(stream));
}


TEST(NpuOpsTest, BMMTransV) {
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  size_t max_lhs_buffer_size = head_num * max_seq_len * max_seq_len;
  size_t max_rhs_buffer_size = head_num * max_seq_len * hidden_dim;
  size_t max_output_buffer_size = head_num * max_seq_len * hidden_dim;

  void* dev_lhs_f16;
  void* dev_rhs_f16;
  void* dev_output_f16;

  float* host_lhs = new float[max_lhs_buffer_size];
  float* host_rhs = new float[max_rhs_buffer_size];
  float* host_output = new float[max_output_buffer_size];
  Eigen::half* host_lhs_f16 = new Eigen::half[max_lhs_buffer_size];
  Eigen::half* host_rhs_f16 = new Eigen::half[max_rhs_buffer_size];
  Eigen::half* host_output_f16 = new Eigen::half[max_output_buffer_size];
  float* golden_fp32 = new float[max_output_buffer_size];
  float* golden_trans_fp32 = new float[max_output_buffer_size];


  CHECK_ACL(aclrtMalloc((void**)&dev_lhs_f16, max_lhs_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_rhs_f16, max_rhs_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_output_f16, max_output_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> cur_size_dis(2, max_seq_len);

  std::vector<size_t> cur_size_list = {1};
  //std::vector<size_t> cur_size_list;

  constexpr int rnd_m_num = 10;
  constexpr int rnd_n_num = 10;
  for (int i = 0; i < rnd_m_num; ++i) {
    cur_size_list.push_back(cur_size_dis(gen));
  }

  for (size_t cur_size: cur_size_list) {
    std::uniform_int_distribution<size_t> cur_pos_dis(cur_size, max_seq_len);
    std::vector<size_t> cur_pos_list;
    for (int i = 0; i < rnd_n_num; ++i) {
      cur_pos_list.push_back(cur_pos_dis(gen));
    }
    for (size_t cur_pos: cur_pos_list) {
      std::cout << "test cur_size: " << cur_size << " cur_pos: " << cur_pos << "\n";

      size_t lhs_element_cnt = head_num * cur_size * cur_pos;
      size_t rhs_element_cnt = head_num * cur_pos * head_dim;
      size_t output_element_cnt = head_num * cur_size * head_dim;
      make_random_float(host_lhs, lhs_element_cnt);
      make_random_float(host_rhs, rhs_element_cnt);

      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      golden_fp32_map((float*)golden_fp32, head_num, cur_size, head_dim);

      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      golden_trans_fp32_map((float*)golden_trans_fp32, cur_size, head_num, head_dim);

      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      input_lhs_map((float*)host_lhs, head_num, cur_size, cur_pos);

      Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      input_lhs_fp16_map((Eigen::half*)host_lhs_f16, head_num, cur_size, cur_pos);
      input_lhs_fp16_map = input_lhs_map.cast<Eigen::half>();
      input_lhs_map = input_lhs_fp16_map.cast<float>();

      Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      input_rhs_map((float*)host_rhs, cur_pos, head_num, head_dim);

      Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
      input_rhs_fp16_map((Eigen::half*)host_rhs_f16, cur_pos, head_num, head_dim);
      input_rhs_fp16_map = input_rhs_map.cast<Eigen::half>();
      input_rhs_map = input_rhs_fp16_map.cast<float>();

      CHECK_ACL(aclrtMemcpy(dev_lhs_f16, lhs_element_cnt * sizeof(aclFloat16), host_lhs_f16,
                                lhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
      CHECK_ACL(aclrtMemcpy(dev_rhs_f16, rhs_element_cnt * sizeof(aclFloat16), host_rhs_f16,
                                rhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
      
      npu_batch_matmul_trans_v_layer(dev_output_f16,
                                      dev_lhs_f16,
                                      dev_rhs_f16,
                                      head_num,
                                      cur_size,
                                      head_dim,
                                      cur_pos,
                                      1.0f,
                                      DT_FLOAT16,
                                      stream);

      Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
      for (int i = 0; i < head_num; ++i) {
        golden_fp32_map.chip<0>(i) = input_lhs_map.chip<0>(i).contract(
          input_rhs_map.shuffle(Eigen::array<int, 3>({1, 0, 2})).chip<0>(i), product_dims);
      }

      golden_trans_fp32_map = golden_fp32_map.shuffle(Eigen::array<int, 3>({1, 0, 2}));      

      CHECK_ACL(aclrtSynchronizeStream(stream));
      CHECK_ACL(aclrtMemcpy(host_output_f16, output_element_cnt * sizeof(aclFloat16),
                            dev_output_f16, output_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

      Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
      output_fp32_map((float*)host_output, output_element_cnt);

      Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
      output_fp16_map((Eigen::half*)host_output_f16, output_element_cnt);
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
          
  CHECK_ACL(aclrtFree(dev_lhs_f16));
  CHECK_ACL(aclrtFree(dev_rhs_f16));
  CHECK_ACL(aclrtFree(dev_output_f16));

  CHECK_ACL(aclrtDestroyStream(stream));
}

TEST(NpuOpsTest, GemmAWQ4Bit) {
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  size_t max_n = ffn_hidden;
  size_t group_size = 128;

  size_t max_lhs_buffer_size = ffn_hidden * ffn_hidden;
  size_t max_weight_buffer_size = ffn_hidden * max_n;
  size_t max_zero_buffer_size = ffn_hidden * max_n / group_size;
  size_t max_scale_buffer_size = ffn_hidden * max_n / group_size;
  size_t max_output_buffer_size = ffn_hidden * max_n;

  void* dev_lhs_f16;
  void* dev_weight_s4;
  void* dev_zero_fp16;
  void* dev_scale_fp16;
  void* dev_output_f16;

  float* host_lhs = new float[max_lhs_buffer_size];
  float* host_rhs = new float[max_weight_buffer_size];
  float* host_rhs_nz = new float[max_weight_buffer_size];
  float* host_zero = new float[max_zero_buffer_size];
  float* host_scale = new float[max_scale_buffer_size];
  float* host_output = new float[max_output_buffer_size];
  Eigen::half* host_lhs_f16 = new Eigen::half[max_lhs_buffer_size];
  uint8_t* host_weight_s4 = new uint8_t[max_weight_buffer_size/2];
  Eigen::half* host_zero_f16 = new Eigen::half[max_zero_buffer_size];
  Eigen::half* host_scale_f16 = new Eigen::half[max_scale_buffer_size];
  Eigen::half* host_output_f16 = new Eigen::half[max_output_buffer_size];
  float* golden_fp32 = new float[max_output_buffer_size];


  CHECK_ACL(aclrtMalloc((void**)&dev_lhs_f16, max_lhs_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_weight_s4, max_weight_buffer_size/2 * sizeof(uint8_t), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_zero_fp16, max_zero_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_scale_fp16, max_scale_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_output_f16, max_output_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> m_dis(2, hidden_dim);

  std::vector<size_t> m_list = {1};
  constexpr int rnd_m_num = 10;
  for (int i = 0; i < rnd_m_num; ++i) {
    m_list.push_back(m_dis(gen));
  }

  
  for (size_t m: m_list) {
    for (size_t k : {(size_t)hidden_dim, (size_t)ffn_hidden}) {
      for (size_t n : {(size_t)hidden_dim, (size_t)ffn_hidden}) {
        size_t n1 = n/16;
        size_t lhs_element_cnt = m * k;
        size_t rhs_element_cnt = n * k;
        size_t num_group = k / group_size;
        size_t zero_scale_cnt = n * num_group;
        size_t output_element_cnt = m * n;
        make_random_float(host_lhs, lhs_element_cnt);
        make_random_float_uint4(host_rhs, rhs_element_cnt);
        make_random_float_uint4(host_zero, zero_scale_cnt);
        make_random_float(host_scale, zero_scale_cnt);
        
        /*
        for (int mi = 0; mi < m ; ++mi) {
          for (int hi = 0; hi < k; ++hi) {
            if (mi == hi) {
              host_lhs[mi*k + hi] = 1.0f;
            }
            else {
              host_lhs[mi*k + hi] = 0.0f;
            }
          }
        }

        for (int ni = 0; ni < n ; ++ni) {
          for (int hi = 0; hi < k; ++hi) {
            if (ni == hi) {
              host_rhs[hi*n + ni] = 1.0f;
            }
            else {
              host_rhs[hi*n + ni] = 0.0f;
            }
          }
        }

        for (int zi = 0; zi < zero_scale_cnt; ++zi) {
          //host_zero[zi] = 0;
          //host_scale[zi] = 1.0f;
        }
        */
        
        Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        golden_fp32_map((float*)golden_fp32, m, n);

        Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        input_lhs_map((float*)host_lhs, m, k);

        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        input_lhs_fp16_map((Eigen::half*)host_lhs_f16, m, k);
        input_lhs_fp16_map = input_lhs_map.cast<Eigen::half>();
        input_lhs_map = input_lhs_fp16_map.cast<float>();

        Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        input_rhs_map((float*)host_rhs, num_group, group_size, n);

        Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        input_rhs_nz_map((float*)host_rhs_nz, k / 16, n, 16);
        input_rhs_nz_map = input_rhs_map.reshape(Eigen::array<size_t, 3>{k / 16, 16, n}).shuffle(Eigen::array<size_t, 3>({0, 2, 1}));

        Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        input_zero_fp32_map((float*)host_zero, num_group, 1, n);

        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        input_zero_fp16_map((Eigen::half*)host_zero_f16, num_group, 1, n);

        input_zero_fp16_map = input_zero_fp32_map.cast<Eigen::half>();
        input_zero_fp32_map = input_zero_fp16_map.cast<float>();

        // move weight offset to zero
        input_zero_fp16_map = input_zero_fp16_map - input_zero_fp32_map.constant(8.0f).cast<Eigen::half>();

        Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        input_scale_fp32_map((float*)host_scale, num_group, 1, n);
        
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        input_scale_fp16_map((Eigen::half*)host_scale_f16, num_group, 1, n);

        input_scale_fp16_map = (input_scale_fp32_map / input_scale_fp32_map.constant(16.0f)).cast<Eigen::half>();
        input_scale_fp32_map = input_scale_fp16_map.cast<float>();

        auto float_to_u4 = [](float x)->uint8_t {
          return (static_cast<uint8_t>(x) + 8)&0xf;
        };

        // (x, 4, 64, 2) -> (x, 64, 4, 2)
        for (int i1 = 0; i1 < rhs_element_cnt/512; ++i1) {
            int i1_stride_u8 = 4 * 64 * 2;
            int i1_stride_s4 = 4 * 64;
            for (int i2 = 0; i2 < 4; ++i2) {
                int i2_stride_u8 = 64 * 2;
                int i2_stride_s4 = 1;
                for (int i3 = 0; i3 < 64; ++i3) {
                    int i3_stride_u8 = 2;
                    int i3_stride_s4 = 4;
                    int u8_offset = i1 * i1_stride_u8 + i2 * i2_stride_u8 + i3 * i3_stride_u8;
                    host_weight_s4[i1 * i1_stride_s4 + i2 * i2_stride_s4 + i3 * i3_stride_s4]
                        = (float_to_u4(host_rhs_nz[u8_offset])) | (float_to_u4(host_rhs_nz[u8_offset + 1]) << 4);
                }
            }
        }

        CHECK_ACL(aclrtMemcpy(dev_lhs_f16, lhs_element_cnt * sizeof(aclFloat16), host_lhs_f16,
                                  lhs_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dev_weight_s4, rhs_element_cnt * sizeof(uint8_t)/2, host_weight_s4,
                                  rhs_element_cnt * sizeof(uint8_t)/2, ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dev_zero_fp16, zero_scale_cnt * sizeof(aclFloat16), host_zero_f16,
                                  zero_scale_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
        CHECK_ACL(aclrtMemcpy(dev_scale_fp16, zero_scale_cnt * sizeof(aclFloat16), host_scale_f16,
                                  zero_scale_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
        
        ACLTimer timer(fmt::format("gemm awq 4bit m: {} n: {} k: {}", m, n, k), stream);

        timer.Start();
        npu_matmul_nz_awq_4bit_layer(dev_output_f16,
                            dev_lhs_f16,
                            dev_weight_s4,
                            dev_zero_fp16,
                            dev_scale_fp16,
                            m, n, k,
                            DT_FLOAT16,
                            stream);
        timer.Stop();
        timer.Print();
        Eigen::array<size_t, 3> brc_dim = {1, group_size, 1};

        auto tmp_expr = ((input_rhs_map - input_zero_fp32_map.broadcast(brc_dim)).cast<Eigen::half>().cast<float>() *
            (input_scale_fp32_map.broadcast(brc_dim))).reshape(Eigen::array<size_t, 2>{k, n});
        
        Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
        golden_fp32_map = input_lhs_map.contract(tmp_expr.cast<Eigen::half>().cast<float>(), product_dims);

        
        CHECK_ACL(aclrtSynchronizeStream(stream));

        CHECK_ACL(aclrtMemcpy(host_output_f16, output_element_cnt * sizeof(aclFloat16), dev_output_f16, output_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

        Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        output_fp32_map((float*)host_output, m, n);

        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        output_fp16_map((Eigen::half*)host_output_f16, m, n);
        output_fp32_map = output_fp16_map.cast<float>();

        ASSERT_TRUE(all_close(host_output, golden_fp32, output_element_cnt))
          << "m: " << m << " n: " << n << " k: " << k;

      } // loop n
    } // loop k
  }

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
  delete[] golden_fp32;
          
  CHECK_ACL(aclrtFree(dev_lhs_f16));
  CHECK_ACL(aclrtFree(dev_weight_s4));
  CHECK_ACL(aclrtFree(dev_zero_fp16));
  CHECK_ACL(aclrtFree(dev_scale_fp16));
  CHECK_ACL(aclrtFree(dev_output_f16));

  CHECK_ACL(aclrtDestroyStream(stream));
}

static void InitFreqCIS(float* freq_cis) {
    const float theta = 10000.0f;
    int head_dim = hidden_dim / head_num;
    int freq_len = head_dim / 2;
    float* freq = new float[freq_len];

    for (int i = 0; i < freq_len; ++i) {
        freq[i] = 1.0f / (powf(theta, static_cast<float>(i *2) / static_cast<float>(head_dim)));
    }

    float* t = new float[max_seq_len];
    for (int i = 0; i < max_seq_len; ++i) {
        t[i] = static_cast<float>(i);
    }

    float* freq_outer = new float[freq_len*max_seq_len];

    // max_seq_len row, freq_len column
    for (int i = 0; i < max_seq_len; ++i) {
        for (int j = 0; j < freq_len; ++j) {
            freq_outer[i*freq_len + j] = t[i] * freq[j];
        }
    }

    for (int i = 0; i < max_seq_len * freq_len; ++i) {
        freq_cis[i*2] = std::cos(freq_outer[i]);
        freq_cis[i*2+1] = std::sin(freq_outer[i]);
    }

    delete[] freq;
    delete[] t;
    delete[] freq_outer;

}

TEST(NpuOpsTest, RopeLayer) {
  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));

  size_t max_buffer_size = hidden_dim * max_seq_len;
  size_t rope_dim = hidden_dim * max_seq_len;
  size_t freq_len = head_dim / 2;
  void* dev_input_q_f16;
  void* dev_input_k_f16;
  void* dev_output_q_f16;
  void* dev_output_k_f16;
  void* dev_freq_cis;

  float* host_input_q = new float[max_buffer_size];
  float* host_input_k = new float[max_buffer_size];
  float* host_output_q = new float[max_buffer_size];
  float* host_output_k = new float[max_buffer_size];
  Eigen::half* host_input_q_f16 = new Eigen::half[max_buffer_size];
  Eigen::half* host_input_k_f16 = new Eigen::half[max_buffer_size];
  Eigen::half* host_output_q_f16 = new Eigen::half[max_buffer_size];
  Eigen::half* host_output_k_f16 = new Eigen::half[max_buffer_size];
  float* golden_q_fp32 = new float[max_buffer_size];
  float* golden_k_fp32 = new float[max_buffer_size];
  float* host_freq_cis = new float[rope_dim];
  InitFreqCIS(host_freq_cis);

  CHECK_ACL(aclrtMalloc((void**)&dev_input_q_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_input_k_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_output_q_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_output_k_f16, max_buffer_size * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void**)&dev_freq_cis, rope_dim * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
 
  CHECK_ACL(aclrtMemcpy(dev_freq_cis, rope_dim * sizeof(float), host_freq_cis,
                        rope_dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));

  for (int seq_len = 1; seq_len <= max_seq_len; ++seq_len) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> start_pos_dis(0, max_seq_len - seq_len);
    int start_pos = start_pos_dis(gen);

    std::cout << "test rope: " << "seq_len: " << seq_len << " start pos " << start_pos << "\n";
    int total_element_cnt = seq_len * hidden_dim;
    make_random_float(host_input_q, total_element_cnt);
    make_random_float(host_input_k, total_element_cnt);

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_q_fp32_map((float*)host_input_q, total_element_cnt);
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_q_fp16_map((Eigen::half*)host_input_q_f16, total_element_cnt);
    input_q_fp16_map = input_q_fp32_map.cast<Eigen::half>();
    input_q_fp32_map = input_q_fp16_map.cast<float>();

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_k_fp32_map((float*)host_input_k, total_element_cnt);
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    input_k_fp16_map((Eigen::half*)host_input_k_f16, total_element_cnt);
    input_k_fp16_map = input_k_fp32_map.cast<Eigen::half>();
    input_k_fp32_map = input_k_fp16_map.cast<float>();

    for (int s = 0; s < seq_len; ++s) {
        for (int n = 0; n < head_num; ++n) {
            for (int f = 0;f < freq_len; ++f) {
                float fc = host_freq_cis[(s + start_pos)*freq_len*2 + 2*f];
                float fd = host_freq_cis[(s + start_pos)*freq_len*2 + 2*f+1];

                int hidden_offset = s * hidden_dim + n * head_dim;

                float qa = host_input_q[hidden_offset + 2*f];
                float qb = host_input_q[hidden_offset + 2*f+1];

                float ka = host_input_k[hidden_offset + 2*f];
                float kb = host_input_k[hidden_offset + 2*f+1];

                golden_q_fp32[hidden_offset + 2*f] = qa * fc - qb * fd;
                golden_q_fp32[hidden_offset + 2*f + 1] = qa * fd + qb * fc;

                golden_k_fp32[hidden_offset + 2*f] = ka * fc - kb * fd;
                golden_k_fp32[hidden_offset + 2*f + 1] = ka * fd + kb * fc;
            }
        }
    }

    CHECK_ACL(aclrtMemcpy(dev_input_q_f16, total_element_cnt * sizeof(aclFloat16), host_input_q_f16,
                                total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(dev_input_k_f16, total_element_cnt * sizeof(aclFloat16), host_input_k_f16,
                                total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

    npu_rope_layer(dev_output_q_f16,
                   dev_output_k_f16,
                   dev_freq_cis,
                   dev_input_q_f16,
                   dev_input_k_f16,
                   start_pos,
                   seq_len,
                   head_num,
                   hidden_dim,
                   false,
                   DT_FLOAT16,
                   stream);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(host_output_q_f16, total_element_cnt * sizeof(aclFloat16), dev_output_q_f16, total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(host_output_k_f16, total_element_cnt * sizeof(aclFloat16), dev_output_k_f16, total_element_cnt * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    output_q_fp32_map((float*)host_output_q, total_element_cnt);
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    output_q_fp16_map((Eigen::half*)host_output_q_f16, total_element_cnt);
    output_q_fp32_map = output_q_fp16_map.cast<float>();

    Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    output_k_fp32_map((float*)host_output_k, total_element_cnt);
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    output_k_fp16_map((Eigen::half*)host_output_k_f16, total_element_cnt);
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

          
  CHECK_ACL(aclrtFree(dev_input_q_f16));
  CHECK_ACL(aclrtFree(dev_input_k_f16));
  CHECK_ACL(aclrtFree(dev_output_q_f16));
  CHECK_ACL(aclrtFree(dev_output_k_f16));
  CHECK_ACL(aclrtFree(dev_freq_cis));

  CHECK_ACL(aclrtDestroyStream(stream));
}