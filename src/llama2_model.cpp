#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>

#include "llama2_model.hpp"
#include "llama2_layer_cpu.hpp"
#include "llama2_layer_npu.hpp"
#include "device.hpp"
#include "util.h"


Llama2InferenceCtx::Llama2InferenceCtx(Llama2Model* model, size_t cur_pos, size_t prev_pos)
:model(model), cur_pos(cur_pos), prev_pos(prev_pos) {
    cur_size = cur_pos - prev_pos;
}


bool Llama2Model::Init() {
    if (device_type == DEV_NPU) {
        aclrtContext context;
        int32_t deviceId = 0;
        CHECK_ACL(aclrtSetDevice(deviceId));
        CHECK_ACL(aclrtCreateContext(&context, deviceId));
        CHECK_ACL(aclrtCreateStream(&model_stream));
    }
    std::ifstream config_fs(config.config_path.c_str());
    config_fs >> config.config;

    spdlog::info("using json config\n{}", config.config.dump(4));
    hidden_dim = config.config["dim"].get<int>();
    n_heads = config.config["n_heads"].get<int>();
    n_layers = config.config["n_layers"].get<int>();
    norm_eps = config.config["norm_eps"].get<float>();
    multiple_of = config.config["multiple_of"].get<float>();

    InitFreqCIS();

    embedding_layer.Init(this);
    causual_mask_layer.Init(this);
    transformer_layers.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        spdlog::info("loading layer {}/{}", i, n_layers);
        transformer_layers[i].Init(this, i);
    }

    last_norm.Init(this, -1, false, true);
    last_mm.Init(this, (boost::filesystem::path(config.model_path) / "lm_head.weight.bin").c_str(), tokenizer.n_words, hidden_dim);
    argmax_layer.Init(this);
    top_p_layer.Init(this);
    
    return true;
}


void Llama2Model::Chat(const std::string& input_seq, const std::string& reverse_prompt) {
    auto tokens = tokenizer.Encode(input_seq, true, false);
    auto reverse_prompt_size = reverse_prompt.size();
    int input_token_size = tokens.size();
    spdlog::debug("promt token size {}", input_token_size);
    size_t curr_string_size = input_seq.size();

    std::cout << input_seq;

    int prev_pos = 0;
    int cur_pos = input_token_size;

    bool is_interacting = true;

    do {
        if (is_interacting) {
            std::string user_input;
            std::getline(std::cin, user_input);
            user_input = user_input + "\n";
            //spdlog::info("user_input {}", user_input);
            auto user_input_tokens = tokenizer.Encode(user_input, false, false);
            auto user_input_tokens_size = user_input_tokens.size();
            
            tokens.insert(tokens.end(), user_input_tokens.begin(), user_input_tokens.end());
            cur_pos += user_input_tokens_size;
            curr_string_size += user_input.size();
            is_interacting = false;
        }

        if (cur_pos > max_seq_len) {
            std::cout << std::endl;
            spdlog::info("cur_pos {} greater than max_seq_len {}, end generation",  cur_pos, prev_pos);
            return;
        }

        Llama2InferenceCtx ctx(this, cur_pos, prev_pos);
        if (device_type == DEV_NPU) {
            ctx.npu_stream = model_stream;
        }
        spdlog::debug("cur_pos {} prev_pos {} curr size {}",  cur_pos, prev_pos, ctx.cur_size);
        auto input_token = Tensor::MakeCPUTensor(ctx.cur_size, DT_UINT32);
        memcpy(input_token->data_ptr, tokens.data() + prev_pos, sizeof(uint32_t) * ctx.cur_size);
        input_token = input_token->to(device_type);
        auto h = embedding_layer.Forward(input_token, ctx);

        if (debug_print) {
            auto print_h = h->to(DEV_CPU);
            Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
            embed_map(static_cast<Eigen::half*>(print_h->data_ptr), ctx.cur_size, hidden_dim);

            Eigen::array<Eigen::Index, 2> offsets = {0, 0};
            Eigen::array<Eigen::Index, 2> extents = {ctx.cur_size, 4};
            Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = embed_map.slice(offsets, extents);
            std::cout << "embed output \n" << print_slice << "\n";
            //break;
        }

        
        auto causlmask = causual_mask_layer.Forward(ctx);

        for (int i = 0; i < n_layers; ++i) {
            h = transformer_layers[i].Forward(h, causlmask, ctx);

            if (debug_print) {
                auto print_h = h->to(DEV_CPU);
                Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
                h_map(static_cast<Eigen::half*>(print_h->data_ptr), ctx.cur_size, hidden_dim);
                Eigen::array<Eigen::Index, 2> offsets = {0, 0};
                Eigen::array<Eigen::Index, 2> extents = {ctx.cur_size, 4};
                Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = h_map.slice(offsets, extents);
                std::cout << "h_map " << i <<" output \n" << print_slice << "\n";
            }
            //break;
        }
        //break;

        h = last_norm.Forward(h, ctx);

        if (debug_print) {
            CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
            auto print_h = h->to(DEV_CPU);
            Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
            h_map(static_cast<Eigen::half*>(print_h->data_ptr), ctx.cur_size, hidden_dim);
            Eigen::array<Eigen::Index, 2> offsets = {0, 0};
            Eigen::array<Eigen::Index, 2> extents = {ctx.cur_size, 288};
            Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = h_map.slice(offsets, extents);
            std::cout << "h_map after norm output \n" << print_slice << "\n";
        }


        h = last_mm.Forward(h, ctx);

        if (debug_print) {
            CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
            auto print_h = h->to(DEV_CPU);
            Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
            h_map(static_cast<Eigen::half*>(print_h->data_ptr), ctx.cur_size, tokenizer.n_words);
            Eigen::array<Eigen::Index, 2> offsets = {0, 0};
            Eigen::array<Eigen::Index, 2> extents = {ctx.cur_size, 4};
            Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = h_map.slice(offsets, extents);
            std::cout << "h_map last mm output \n" << print_slice << "\n";
        }

        if (device_type == DEV_NPU) {
            CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
        }

        h = h->to(DEV_CPU);
        int next_tok;
        if (temperature > 0.0f) {
            next_tok = top_p_layer.Forward(h, ctx);
        }
        else {
            auto max_pos = argmax_layer.Forward(h, ctx);
            next_tok = static_cast<int32_t*>(max_pos->data_ptr)[ctx.cur_size-1];
        }
        if (next_tok == tokenizer.eos_id) {
            is_interacting = true;
        }
        spdlog::debug("cur_pos {} max_seq_len {} next_tok {}",  cur_pos, max_seq_len, next_tok);
        tokens.push_back(next_tok);
        
        auto full_string = tokenizer.Decode(tokens);
        std::string new_str = full_string.substr(curr_string_size+1);
        curr_string_size += new_str.size();

        int rstring_offset = full_string.size() - reverse_prompt_size;
        if (rstring_offset >= 0) {
            bool match_reverse_prompt = true;
            for (int ri = 0; ri < reverse_prompt_size;  ++ri) {
                if (full_string[rstring_offset + ri] != reverse_prompt[ri]) {
                    match_reverse_prompt = false;
                }
            }

            if (match_reverse_prompt) {
                is_interacting = true;
            }
        }

        std::cout << new_str << std::flush;

        prev_pos = cur_pos;
        ++cur_pos;
    } while(cur_pos < max_seq_len);

    std::cout << std::endl;
}


void Llama2Model::TextCompletion(const std::string& input_seq) {
    auto tokens = tokenizer.Encode(input_seq, true, false);
    tokens.reserve(max_seq_len);
    int input_token_size = tokens.size();
    spdlog::debug("Llama2Model::TextCompletion input \"{}\" promt size {} token_size {}",  input_seq, input_token_size, max_seq_len);
    size_t curr_string_size = input_seq.size();

    int output_seq_len = input_token_size;

    std::cout << "input prompt:\n" << input_seq;

    int prev_pos = 0;

    for (int cur_pos = input_token_size; cur_pos < max_seq_len; ++cur_pos) {
        spdlog::debug("cur_pos {} prev_pos {}",  cur_pos, prev_pos);
        Llama2InferenceCtx ctx(this, cur_pos, prev_pos);
        if (device_type == DEV_NPU) {
            ctx.npu_stream = model_stream;
        }
        auto input_token = Tensor::MakeCPUTensor(ctx.cur_size, DT_UINT32);
        memcpy(input_token->data_ptr, tokens.data() + prev_pos, sizeof(uint32_t) * ctx.cur_size);
        input_token = input_token->to(device_type);
        auto h = embedding_layer.Forward(input_token, ctx);

        if (debug_print) {
            auto print_h = h->to(DEV_CPU);
            Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
            embed_map(static_cast<Eigen::half*>(print_h->data_ptr), ctx.cur_size, hidden_dim);

            Eigen::array<Eigen::Index, 2> offsets = {0, 0};
            Eigen::array<Eigen::Index, 2> extents = {ctx.cur_size, 4};
            Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = embed_map.slice(offsets, extents);
            std::cout << "embed output \n" << print_slice << "\n";
            //break;
        }

        
        auto causlmask = causual_mask_layer.Forward(ctx);

        for (int i = 0; i < n_layers; ++i) {
            h = transformer_layers[i].Forward(h, causlmask, ctx);

            if (debug_print) {
                auto print_h = h->to(DEV_CPU);
                Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
                h_map(static_cast<Eigen::half*>(print_h->data_ptr), ctx.cur_size, hidden_dim);
                Eigen::array<Eigen::Index, 2> offsets = {0, 0};
                Eigen::array<Eigen::Index, 2> extents = {ctx.cur_size, 4};
                Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = h_map.slice(offsets, extents);
                std::cout << "h_map " << i <<" output \n" << print_slice << "\n";
            }
            //break;
        }
        //break;

        h = last_norm.Forward(h, ctx);

        if (debug_print) {
            CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
            auto print_h = h->to(DEV_CPU);
            Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
            h_map(static_cast<Eigen::half*>(print_h->data_ptr), ctx.cur_size, hidden_dim);
            Eigen::array<Eigen::Index, 2> offsets = {0, 0};
            Eigen::array<Eigen::Index, 2> extents = {ctx.cur_size, 288};
            Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = h_map.slice(offsets, extents);
            std::cout << "h_map after norm output \n" << print_slice << "\n";
        }


        h = last_mm.Forward(h, ctx);

        if (debug_print) {
            CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
            auto print_h = h->to(DEV_CPU);
            Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
            h_map(static_cast<Eigen::half*>(print_h->data_ptr), ctx.cur_size, tokenizer.n_words);
            Eigen::array<Eigen::Index, 2> offsets = {0, 0};
            Eigen::array<Eigen::Index, 2> extents = {ctx.cur_size, 4};
            Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = h_map.slice(offsets, extents);
            std::cout << "h_map last mm output \n" << print_slice << "\n";
        }

        if (device_type == DEV_NPU) {
            CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
        }

        h = h->to(DEV_CPU);
        int next_tok;
        if (temperature > 0.0f) {
            next_tok = top_p_layer.Forward(h, ctx);
        }
        else {
            auto max_pos = argmax_layer.Forward(h, ctx);
            next_tok = static_cast<int32_t*>(max_pos->data_ptr)[ctx.cur_size-1];
        }
        if (next_tok == tokenizer.eos_id) {
            break;
        }
        spdlog::debug("cur_pos {} max_seq_len {} next_tok {}",  cur_pos, max_seq_len, next_tok);
        tokens.push_back(next_tok);
        ++output_seq_len;

        auto output_token = tokens;
        output_token.resize(output_seq_len);
        auto next_tok_str = tokenizer.Decode(output_token);
        spdlog::debug("Llama2Model::Forward input \"{}\" output string {}", input_seq, next_tok_str);

        auto full_string = tokenizer.Decode(tokens);
        std::string new_str = full_string.substr(curr_string_size+1);
        curr_string_size += new_str.size();

        std::cout << new_str << std::flush;

        prev_pos = cur_pos;
    }

    std::cout << std::endl;

    auto output_token = tokens;
    output_token.resize(output_seq_len);

    auto next_tok_str = tokenizer.Decode(output_token);

    spdlog::debug("Llama2Model::TextCompletion input \"{}\" output string {}", input_seq, next_tok_str);
}

std::string Llama2Model::GetCurrTokenString(size_t prev_string_size, const std::vector<int>& tokens) {
    auto full_string = tokenizer.Decode(tokens);
    return full_string.substr(prev_string_size+1);
}

bool Llama2Model::InitFreqCIS() {
    const float theta = 10000.0f;
    int head_dim = hidden_dim / n_heads;
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

    freq_cis = new float[freq_len*max_seq_len*2];

    for (int i = 0; i < max_seq_len * freq_len; ++i) {
        freq_cis[i*2] = std::cos(freq_outer[i]);
        freq_cis[i*2+1] = std::sin(freq_outer[i]);
    }

    if (device_type == DEV_NPU) {
        float* old_freq_cis = freq_cis;
        size_t freq_cis_size = freq_len*max_seq_len*2*sizeof(float);
        CHECK_ACL(aclrtMalloc((void **)&freq_cis, freq_cis_size, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemcpy(freq_cis, freq_cis_size, old_freq_cis, freq_cis_size, ACL_MEMCPY_HOST_TO_DEVICE));
        delete[] old_freq_cis;
    }

    delete[] freq;
    delete[] t;
    delete[] freq_outer;
    return true;
}


std::shared_ptr<Tensor> Llama2EmbeddingLayer::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    return impl->Forward(input, ctx);
}


bool Llama2EmbeddingLayer::Init(Llama2Model* model) {
    switch (model->device_type)
    {
    case DEV_NPU:
        impl = new Llama2EmbeddingLayerNPUImpl();
        break;
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
    pad_id = model->tokenizer.pad_id;

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
        if (pre_norm) {
            ss << "model.layers." << layer_no << ".input_layernorm.weight.bin";
        }
        else {
            ss << "model.layers." << layer_no << ".post_attention_layernorm.weight.bin";
        }

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


std::shared_ptr<Tensor> RMSNormLayer::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    return impl->Forward(input, ctx);
}


bool RMSNormLayer::Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm) {
    switch (model->device_type)
    {
    case DEV_NPU:
        impl = new RMSNormLayerNPUImpl();
        break;
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
    hidden_dim = model->hidden_dim;
    head_dim = model->hidden_dim / model->n_heads;
    n_heads = model->n_heads;
    ffn_hidden = (4 * hidden_dim * 2) / 3;
    ffn_hidden = (ffn_hidden + model->multiple_of - 1) / model->multiple_of * model->multiple_of;
    max_seq_len = model->max_seq_len;
    return true;
}

void Llamma2TransformerLayerImpl::UnInit() {

}


Llamma2TransformerLayer::~Llamma2TransformerLayer() {
    delete impl;
}


std::shared_ptr<Tensor> Llamma2TransformerLayer::Forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> mask, Llama2InferenceCtx& ctx) {
    return impl->Forward(input, mask, ctx);
}


bool Llamma2TransformerLayer::Init(Llama2Model* model, int layer_no) {
    switch (model->device_type)
    {
    case DEV_NPU:
        impl = new Llamma2TransformerLayerNPUImpl();
        break;
    case DEV_CPU:
        impl = new Llamma2TransformerLayerCPUImpl();
        break;
    
    default:
        spdlog::critical("invalid device type");
        return false;
        break;
    }

    return impl->Init(model, layer_no);
}

void Llamma2TransformerLayer::UnInit() {

}


ArgMaxLayerImpl::~ArgMaxLayerImpl() {

}

bool ArgMaxLayerImpl::Init(Llama2Model* model) {
    // TODO: change name
    hidden_dim = model->tokenizer.n_words;
    return true;
}

void ArgMaxLayerImpl::UnInit() {

}



ArgMaxLayer::~ArgMaxLayer() {

}

std::shared_ptr<Tensor>
ArgMaxLayer::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    return impl->Forward(input, ctx);
}

bool ArgMaxLayer::Init(Llama2Model* model) {
    switch (model->device_type)
    {
    case DEV_NPU:
    case DEV_CPU:
        impl = new ArgMaxLayerCPUImpl();
        break;
    
    default:
        spdlog::critical("invalid device type");
        return false;
        break;
    }
    return impl->Init(model);
}

void ArgMaxLayer::UnInit() {

}


SampleTopPLayerImpl::~SampleTopPLayerImpl() {

}


bool SampleTopPLayerImpl::Init(Llama2Model* model) {
    temperature = model->temperature;
    top_p = model->top_p;
    vocab_size = model->tokenizer.n_words;
    return true;
}

void SampleTopPLayerImpl::UnInit() {

}


SampleTopPLayer::~SampleTopPLayer() {

}

int
SampleTopPLayer::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    return impl->Forward(input, ctx);
}

bool SampleTopPLayer::Init(Llama2Model* model) {
    switch (model->device_type)
    {
    case DEV_NPU:
    case DEV_CPU:
        impl = new SampleTopPLayerCPUImpl();
        break;
    
    default:
        spdlog::critical("invalid device type");
        return false;
        break;
    }
    return impl->Init(model);
}

void SampleTopPLayer::UnInit() {

}



SoftmaxLayerImpl::~SoftmaxLayerImpl() {

}

    
bool SoftmaxLayerImpl::Init(Llama2Model* model) {
    hidden_dim = model->hidden_dim;
    n_heads = model->n_heads;
    eps = 1E-5;
    return true;
}

void SoftmaxLayerImpl::UnInit() {

}



SoftmaxLayer::~SoftmaxLayer() {

}

std::shared_ptr<Tensor>
SoftmaxLayer::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    return impl->Forward(input, ctx);
}


bool SoftmaxLayer::Init(Llama2Model* model) {
    switch (model->device_type)
    {
    case DEV_NPU:
        impl = new SoftmaxLayerNPUImpl();
        break;
    case DEV_CPU:
        impl = new SoftmaxLayerCPUImpl();
        break;
    
    default:
        spdlog::critical("invalid device type");
        return false;
        break;
    }
    return impl->Init(model);
}

void SoftmaxLayer::UnInit() {

}


CausualMaskLayerImpl::~CausualMaskLayerImpl() {

}

bool CausualMaskLayerImpl::Init(Llama2Model* model) {
    return true;
}


void CausualMaskLayerImpl::UnInit() {

}


CausualMaskLayer::~CausualMaskLayer() {

}

std::shared_ptr<Tensor>
CausualMaskLayer::Forward(Llama2InferenceCtx& ctx) {
    return impl->Forward(ctx);
}

bool CausualMaskLayer::Init(Llama2Model* model) {
    switch (model->device_type)
    {
    case DEV_NPU:
    case DEV_CPU:
        impl = new CausualMaskLayerCPUImpl();
        break;
    
    default:
        spdlog::critical("invalid device type");
        return false;
        break;
    }
    return impl->Init(model);
}

void CausualMaskLayer::UnInit() {
}



MatmulLayerImpl::~MatmulLayerImpl() {

}


bool MatmulLayerImpl::Init(Llama2Model* model, const std::string& weight_path, size_t n, size_t k) {
    this->n = n;
    this->k = k;
    weight_size = sizeof(uint16_t) * n * k;

    weight = (uint8_t*)malloc(weight_size);
    if (weight == nullptr) {
        spdlog::critical("oom!");
        return false;
    }

    if (!LoadBinaryFile(weight_path.c_str(), weight, weight_size)) {
        return false;
    }
    return true;
}


void MatmulLayerImpl::UnInit() {
    
}


MatmulLayer::~MatmulLayer() {

}

std::shared_ptr<Tensor> MatmulLayer::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    return impl->Forward(input, ctx);
}

bool MatmulLayer::Init(Llama2Model* model, const std::string& weight_path, size_t n, size_t k) {
    switch (model->device_type)
    {
    case DEV_NPU:
        impl = new MatmulLayerNPUImpl();
        break;
    case DEV_CPU:
        impl = new MatmulLayerCPUImpl();
        break;
    
    default:
        spdlog::critical("invalid device type");
        return false;
        break;
    }
    return impl->Init(model, weight_path, n, k);
}

void MatmulLayer::UnInit() {

}


RoPELayerImpl::~RoPELayerImpl() {

}

bool RoPELayerImpl::Init(Llama2Model* model, const std::string& weight_path) {
    hidden_dim = model->hidden_dim;
    head_dim = model->hidden_dim / model->n_heads;
    n_heads = model->n_heads;
    rope_dim = model->max_seq_len * model->hidden_dim;
    weight_size = sizeof(float) * rope_dim;

    freqs_cis = model->freq_cis;

    return true;
}

void RoPELayerImpl::UnInit() {

}


RoPELayer::~RoPELayer() {

}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RoPELayer::Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k, Llama2InferenceCtx& ctx) {
    return impl->Forward(input_q, input_k, ctx);
}


bool RoPELayer::Init(Llama2Model* model, const std::string& weight_path) {
    switch (model->device_type)
    {
    case DEV_NPU:
        impl = new RoPELayerNPUImpl();
        break;
    case DEV_CPU:
        impl = new RoPELayerCPUImpl();
        break;
    
    default:
        spdlog::critical("invalid device type");
        return false;
        break;
    }
    return impl->Init(model, weight_path);
}

void RoPELayer::UnInit() {

}

