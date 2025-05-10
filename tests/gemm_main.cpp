#include "gemm_test.h"
#include <string>

namespace po = boost::program_options;

template <typename EigenTy> bool TestFn(int m, int n, int k, bool bias) {
  aclrtContext context;
  int32_t deviceId{0};

  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(deviceId));
  CHECK_ACL(aclrtCreateContext(&context, deviceId));

  GemmOpTest<EigenTy> op_test;
  op_test.Init(m, n, k, bias);
  bool test_result = op_test.Run(m, n, k);
  op_test.CleanUp();

  CHECK_ACL(aclrtDestroyContext(context));
  CHECK_ACL(aclrtResetDevice(deviceId));
  CHECK_ACL(aclFinalize());
  return test_result;
}

int main(int argc, char **argv) {
  aclrtContext context;
  int32_t deviceId{0};
  int m;
  int n;
  int k;
  bool bias;
  std::string dtype_str;

  po::options_description desc("GemmAWQ4BitOpTest options");
  desc.add_options()("help", "produce help message")                    //
      ("m", po::value<int>(&m)->default_value(2048), "m. default:2048") //
      ("n", po::value<int>(&n)->default_value(2048), "n. default:2048") //
      ("k", po::value<int>(&k)->default_value(2048), "k. default:2048") //
      ("bias", po::value<bool>(&bias)->default_value(false),
       "bias, default:false") //
      ("dtype", po::value<std::string>(&dtype_str)->required(),
       "dtype. float16 or bfloat16");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }
  po::notify(vm);

  bool test_result = false;

  if (dtype_str == "float16") {
    test_result = TestFn<Eigen::half>(m, n, k, bias);
  } else if (dtype_str == "bfloat16") {
    test_result = TestFn<Eigen::bfloat16>(m, n, k, bias);
  }

  if (test_result) {
    spdlog::info("test success");
  } else {
    spdlog::error("test failed");
  }
}
