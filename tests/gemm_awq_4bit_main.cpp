#include "gemm_awq_4bit_test.h"

namespace po = boost::program_options;

int main(int argc, char **argv) {
  aclrtContext context;
  int32_t deviceId{0};
  int m;
  int n;
  int k;
  bool bias;

  po::options_description desc("GemmAWQ4BitOpTest options");
  desc.add_options()("help", "produce help message")                    //
      ("m", po::value<int>(&m)->default_value(2048), "m. default:2048") //
      ("n", po::value<int>(&n)->default_value(2048), "n. default:2048") //
      ("k", po::value<int>(&k)->default_value(2048), "k. default:2048")
      ("bias", po::value<bool>(&bias)->default_value(false), "bias. default:false");


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

  GemmAWQ4BitOpTest op_test;
  op_test.Init(m, n, k, bias);
  bool test_result = op_test.Run(m, n, k);
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
