#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <string>
#include <map>

#include "tokenizer.hpp"
#include "llama2_model.hpp"
#include "acl_util.hpp"

namespace po = boost::program_options;

static std::map<std::string, spdlog::level::level_enum>
log_level_name_to_enum {
    {"trace", spdlog::level::trace},
    {"debug", spdlog::level::debug},
    {"info", spdlog::level::info},
    {"warning", spdlog::level::warn},
    {"error", spdlog::level::err},
    {"critical", spdlog::level::critical},
    {"off", spdlog::level::off}
};

int main(int argc, char** argv) {
    try {
        Llama2Model model;
        std::string str_device_type;
        std::string str_prompt;
        std::string str_level;
        std::string profiling_output_path;

        po::options_description desc("llama2 inference options");
        desc.add_options()
        ("help", "produce help message")
        ("max_seq_len", po::value<int>(&model.max_seq_len)->default_value(16), "max sequence length of tokens. default:128")
        ("tokenizer", po::value<std::string>(&model.config.tok_path)->required(), "path to tokenizer")
        ("weight", po::value<std::string>(&model.config.model_path)->required(), "path to model weight")
        ("config", po::value<std::string>(&model.config.config_path)->required(), "path to model config")
        ("device_type", po::value<std::string>(&str_device_type)->required(), "device type, cpu/gpu")
        ("prompt", po::value<std::string>(&str_prompt)->required(), "prompt str")
        ("log_level", po::value<std::string>(&str_level), "log level:[trace,debug,info,warning,error,critical,off]")
        ("profiling_output", po::value<std::string>(&profiling_output_path), "profiling_output_file xx.json")
        ("debug_print", po::value<bool>(&model.debug_print), "print tensor value to debug");


        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        po::notify(vm);

        if (vm.count("log_level")) {
            if (!log_level_name_to_enum.count(str_level)) {
                std::cout << "invalid log_level:" << str_level << "\n";
                return 1;
            }
            spdlog::set_level(log_level_name_to_enum[str_level]);
        }
        else {
            // default level is info
            spdlog::set_level(spdlog::level::info);
        }

        if (str_device_type == "cpu") {
            model.device_type = DEV_CPU;
        }
        else if (str_device_type == "gpu") {
            model.device_type = DEV_GPU;
        }
        else if (str_device_type == "npu") {
            CHECK_ACL(aclInit(nullptr));
            model.device_type = DEV_NPU;
        }
        else {
            spdlog::critical("invalid device type {}", str_device_type);
            return 1;
        }


        if (boost::filesystem::exists(model.config.tok_path) &&
            boost::filesystem::is_regular_file(model.config.tok_path)) {
            if (!model.tokenizer.Init(model.config.tok_path)) {
                return 1;
            }
        }
        else {
            spdlog::error("invalid tokenizer path {}", model.config.tok_path);
            return 1;
        }

        if (boost::filesystem::exists(model.config.model_path) &&
            boost::filesystem::is_directory(model.config.model_path)) {
        }
        else {
            spdlog::error("invalid model_weight path {}", model.config.model_path);
            return 1;
        }

        if (boost::filesystem::exists(model.config.config_path) &&
            boost::filesystem::is_regular_file(model.config.config_path)) {
        }
        else {
            spdlog::error("invalid config path {}", model.config.config_path);
            return 1;
        }

        if (!model.Init()) {
            spdlog::error("failed to init model");
            return 1;
        }

        if (vm.count("profiling_output")) {
            if (!model.profiler.StartLogging(profiling_output_path)) {
                spdlog::error("failed to init profiler, check {} is writable", profiling_output_path);
                return 1;
            }
            model.is_profiling = true;
        }

        model.Forward(str_prompt);

        if (vm.count("profiling_output")) {
            model.profiler.Finish();
        }
    }
    catch (std::exception& e) {
        spdlog::error("{}", e.what());
        return 1;
    }
    
}
