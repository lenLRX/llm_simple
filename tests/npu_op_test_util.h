#pragma once

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <acl/acl.h>
#include <random>

#include <Eigen/Core>
#include <boost/program_options.hpp>
#include <fmt/format.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <spdlog/spdlog.h>

#define CHECK_ACL(x)                                                           \
  do {                                                                         \
    aclError __ret = x;                                                        \
    if (__ret != ACL_ERROR_NONE) {                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret        \
                << std::endl;                                                  \
    }                                                                          \
  } while (0);

void make_random_float(float *buffer, size_t size);

void make_random_float_uint4(float *buffer, size_t size);

bool all_close(float *output_buffer, float *golden_buffer, size_t size,
               float abs_err = 0.001f, float relative_err = 0.001f);

bool all_close_inf(float *output_buffer, float *golden_buffer, size_t size);

void read_binary(const char *path, void *data, size_t size);
void write_binary(const char *path, void *data, size_t size);

class OpTestTensor {
public:
  void *host_buffer{nullptr};
  void *dev_buffer{nullptr};
};

template <typename Derived> class OpTestBase {
public:
  template <typename... Args> void Init(Args... args) {
    CHECK_ACL(aclrtCreateStream(&stream));
    static_cast<Derived>(this)->Init(args...);
  }

  template <typename... Args> bool Run(Args... args) {
    return static_cast<Derived>(this)->Run(args...);
  }

  void CleanUp() {
    static_cast<Derived>(this)->CleanUp();
    CHECK_ACL(aclrtDestroyStream(stream));
  }

  aclrtStream stream = nullptr;
};