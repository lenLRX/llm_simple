#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <fstream>
#include <sstream>
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
        std::string prompt_file_path;
        std::string str_level;
        std::string profiling_output_path;
        std::string reverse_promt;
        std::string str_quant_method;

        po::options_description desc("llama2 inference options");
        desc.add_options()
        ("help", "produce help message")
        ("max_seq_len", po::value<int>(&model.max_seq_len)->default_value(2048), "max sequence length of tokens. default:2048")
        ("max_gen_token", po::value<int>(&model.max_gen_len)->default_value(2048), "max generate of tokens. default:2048")
        ("tokenizer", po::value<std::string>(&model.config.tok_path)->required(), "path to tokenizer")
        ("weight", po::value<std::string>(&model.config.model_path)->required(), "path to model weight")
        ("config", po::value<std::string>(&model.config.config_path)->required(), "path to model config")
        ("device_type", po::value<std::string>(&str_device_type)->required(), "device type, cpu/gpu")
        ("prompt", po::value<std::string>(&str_prompt), "prompt str")
        ("prompt_file", po::value<std::string>(&prompt_file_path), "prompt file")
        ("log_level", po::value<std::string>(&str_level), "log level:[trace,debug,info,warning,error,critical,off]")
        ("profiling_output", po::value<std::string>(&profiling_output_path), "profiling_output_file xx.json")
        ("debug_print", po::value<bool>(&model.debug_print)->default_value(false), "print tensor value to debug")
        ("temperature", po::value<float>(&model.temperature)->default_value(0.6), "sample temperature, default: 0.6")
        ("top_p", po::value<float>(&model.top_p)->default_value(0.9), "sample top_p, default: 0.9")
        ("reverse_promt", po::value<std::string>(&reverse_promt), "reverse_promt in interactive mode")
        ("i", "interactive mode")
        ("quant_method", po::value<std::string>(&str_quant_method), "quant_method: current support: awq_4bit")
        ("quant_group_size", po::value<int>(&model.quant_group_size)->default_value(-1), "group size in quant")
        ("rope_is_neox_style", po::value<bool>(&model.rope_is_neox_style)->default_value(false), "rope embedding style, defalut: false");


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

        if (vm.count("prompt_file")) {
            if (vm.count("prompt")) {
                spdlog::warn("prompt_file overwrite prompt string");
            }
            std::ifstream prompt_file(prompt_file_path.c_str());
            if (!prompt_file) {
                spdlog::critical("failed to open prompt_file {}", prompt_file_path);
                return 1;
            }
            std::stringstream ss;
            ss << prompt_file.rdbuf();
            str_prompt = ss.str();
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

        if (str_quant_method == "awq_4bit") {
            model.q_type = QuantType::AWQ_4B;
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

        // interactive mode
        if (vm.count("i")) {
            if (!vm.count("reverse_promt")) {
                spdlog::critical("reverse_promt must be specified in interactive mode");
                return 1;
            }
            model.Chat(str_prompt, reverse_promt);
        }
        // text completion mode
        else {
            model.TextCompletion(str_prompt);
        }

        if (vm.count("profiling_output")) {
            model.profiler.Finish();
        }
    }
    catch (std::exception& e) {
        spdlog::error("{}", e.what());
        return 1;
    }
    
}
