#include <fstream>
#include <string>
#include <sstream>
#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>

#include "llama2_model.hpp"
#include "llama2_layer_cpu.hpp"
#include "device.hpp"
#include "util.h"


bool Llama2Model::Init() {
    std::ifstream config_fs(config.config_path.c_str());
    config_fs >> config.config;

    spdlog::info("using json config\n{}", config.config.dump(4));
    hidden_dim = config.config["dim"].get<int>();
    n_heads = config.config["n_heads"].get<int>();
    n_layers = config.config["n_layers"].get<int>();
    norm_eps = config.config["norm_eps"].get<float>();

    embedding_layer.Init(this);

    transformer_layers.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        transformer_layers[i].Init(this, i);
    }
    
    return true;
}


std::string Llama2Model::Forward(const std::string& input_seq) {
    auto tokens = tokenizer.Encode(input_seq, true, false);
    size_t input_token_size = tokens.size();
    spdlog::info("Llama2Model::Forward input \"{}\" token_size {}", input_seq, input_token_size);
    auto input_token = Tensor::MakeCPUTensor(input_token_size, DT_UINT32);
    memcpy(input_token->data_ptr, tokens.data(), sizeof(uint32_t) * input_token_size);
    embedding_layer.Forward(input_token, input_token_size);




    return "ok";
}


std::shared_ptr<Tensor> Llama2EmbeddingLayer::Forward(std::shared_ptr<Tensor> input, size_t seq_len) {
    return impl->Forward(input, seq_len);
}


bool Llama2EmbeddingLayer::Init(Llama2Model* model) {
    switch (model->device_type)
    {
    case DEV_CPU:
        impl = new Llama2EmbeddingLayerCPUImpl();
        break;
    
    default:
        spdlog::critical("invalid device type");
        return false;
        break;
    }
    return impl->Init(model);
}


void Llama2EmbeddingLayer::UnInit() {

}


 Llama2EmbeddingLayerImpl::~Llama2EmbeddingLayerImpl() {

 }


bool Llama2EmbeddingLayerImpl::Init(Llama2Model* model) {
    nwords_size = model->tokenizer.n_words;
    hidden_dim = model->hidden_dim;
    weight_size = sizeof(uint16_t) * nwords_size * model->hidden_dim;

    embedding_weight = (uint8_t*)malloc(weight_size);
    if (embedding_weight == nullptr) {
        spdlog::critical("oom!");
        return false;
    }

    auto weight_path = boost::filesystem::path(model->config.model_path) / "model.embed_tokens.weight.bin";

    if (!LoadBinaryFile(weight_path.c_str(), embedding_weight, weight_size)) {
        return false;
    }
    return true;
}

void Llama2EmbeddingLayerImpl::UnInit() {
    free(embedding_weight);
}


Llama2EmbeddingLayer::~Llama2EmbeddingLayer() {
    delete impl;
}


RMSNormLayerImpl::~RMSNormLayerImpl() {

}

bool RMSNormLayerImpl::Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm) {
    hidden_dim = model->hidden_dim;
    eps = model->norm_eps;

    weight_size = sizeof(uint16_t) * model->hidden_dim;
    norm_weight = (uint8_t*)malloc(weight_size);
    if (norm_weight == nullptr) {
        spdlog::critical("oom!");
        return false;
    }

    auto weight_path = boost::filesystem::path(model->config.model_path); 
    if (last_norm) {
        //model.norm.weight.bin
        weight_path = weight_path / "model.norm.weight.bin";
    }
    else {
        std::stringstream ss;
        ss << "model.layers." << layer_no << ".input_layernorm.weight.bin";
        weight_path = weight_path / ss.str();
    }

    if (!LoadBinaryFile(weight_path.c_str(), norm_weight, weight_size)) {
        return false;
    }
    return true;
}

void RMSNormLayerImpl::UnInit() {
    free(norm_weight);
}


RMSNormLayer::~RMSNormLayer() {
    delete impl;
}


std::shared_ptr<Tensor> RMSNormLayer::Forward(std::shared_ptr<Tensor> input, size_t seq_len) {
    return impl->Forward(input, seq_len);
}


bool RMSNormLayer::Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm) {
    switch (model->device_type)
    {
    case DEV_CPU:
        impl = new RMSNormLayerCPUImpl();
        break;
    
    default:
        spdlog::critical("invalid device type");
        return false;
        break;
    }
    return impl->Init(model, layer_no, pre_norm, last_norm);
}


void RMSNormLayer::UnInit() {

}

bool Llamma2TransformerLayerImpl::Init(Llama2Model* model, int layer_no) {
    return true;
}

void Llamma2TransformerLayerImpl::UnInit() {

}


Llamma2TransformerLayer::~Llamma2TransformerLayer() {
    delete impl;
}


std::shared_ptr<Tensor> Llamma2TransformerLayer::Forward(std::shared_ptr<Tensor> input, size_t seq_len) {
    auto pre_norm_out = pre_norm.Forward(input, seq_len);
    auto post_norm_out = post_norm.Forward(pre_norm_out, seq_len);
    return post_norm_out;
}


bool Llamma2TransformerLayer::Init(Llama2Model* model, int layer_no) {
    if (!pre_norm.Init(model, layer_no, true, false)) {
        return false;
    }
    switch (model->device_type)
    {
    case DEV_CPU:
        impl = new Llamma2TransformerLayerCPUImpl();
        return impl->Init(model, layer_no);
        break;
    
    default:
        spdlog::critical("invalid device type");
        break;
    }
    if (!post_norm.Init(model, layer_no, false, false)) {
        return false;
    }
    return false;
}

void Llamma2TransformerLayer::UnInit() {

}


MatmulLayerImpl::~MatmulLayerImpl() {

}


bool MatmulLayerImpl::Init(Llama2Model* model, const std::string& weight_path) {

}


void MatmulLayerImpl::UnInit() {
    
}


MatmulLayer::~MatmulLayer() {

}

bool MatmulLayer::Init(Llama2Model* model, const std::string& weight_path) {

}

void MatmulLayer::UnInit() {

}

