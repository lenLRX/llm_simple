#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <string>

#include "tokenizer.hpp"
#include "llama2_model.hpp"

namespace po = boost::program_options;

int main(int argc, char** argv) {
    try {
        Llama2Model model;
        std::string str_device_type;
        std::string str_prompt;

        po::options_description desc("llama2 inference options");
        desc.add_options()
        ("help", "produce help message")
        ("tokenizer", po::value<std::string>(&model.config.tok_path)->required(), "path to tokenizer")
        ("weight", po::value<std::string>(&model.config.model_path)->required(), "path to model weight")
        ("config", po::value<std::string>(&model.config.config_path)->required(), "path to model config")
        ("device_type", po::value<std::string>(&str_device_type)->required(), "device type, cpu/gpu")
        ("prompt", po::value<std::string>(&str_prompt)->required(), "prompt str");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        po::notify(vm);

        if (str_device_type == "cpu") {
            model.device_type = DEV_CPU;
        }
        else if (str_device_type == "gpu") {
            model.device_type = DEV_GPU;
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

        model.Forward(str_prompt);
    }
    catch (std::exception& e) {
        spdlog::error("{}", e.what());
        return 1;
    }
    
}
