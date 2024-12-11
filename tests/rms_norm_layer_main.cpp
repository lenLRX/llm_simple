#include "rms_norm_layer_test.h"

namespace po = boost::program_options;

int main(int argc, char **argv) {
  aclrtContext context;
  int32_t deviceId{0};
  int first_dim;
  int last_dim;
  float eps;

  po::options_description desc("RMSNormOpTest options");
  desc.add_options()("help", "produce help message")(
      "first_dim", po::value<int>(&first_dim)->default_value(2048),
      "first_dim. default:2048")("last_dim",
                                 po::value<int>(&last_dim)->default_value(2048),
                                 "last_dim. default:2048")(
      "eps", po::value<float>(&eps)->default_value(1e-5), "eps. default:1e-5");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }
  po::notify(vm);

  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(deviceId));
  CHECK_ACL(aclrtCreateContext(&context, deviceId));

  RMSNormOpTest op_test;
  op_test.Init(first_dim, last_dim);
  bool test_result = op_test.Run(first_dim, last_dim, eps);
  op_test.CleanUp();

  CHECK_ACL(aclrtDestroyContext(context));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());

  if (test_result) {
    spdlog::info("test success");
  } else {
    spdlog::error("test failed");
  }
}
