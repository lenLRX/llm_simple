#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <tuple>
#include <nlohmann/json.hpp>

#include "acl_util.hpp"

#include "tokenizer.hpp"
#include "device.hpp"
#include "profiling.hpp"


class Llama2Config {
public:
    std::string tok_path;
    std::string model_path;
    std::string config_path;
    nlohmann::json config;
};

enum class QuantType {
    NoQuant,
    AWQ_4B
};


class Llama2Model;

class Llama2InferenceCtx {
public:
    Llama2InferenceCtx(Llama2Model* model, size_t cur_pos, size_t prev_pos);
    Llama2Model* model;
    size_t cur_pos;
    size_t prev_pos;
    size_t cur_size;
    aclrtStream npu_stream{nullptr};
};


class Llama2EmbeddingLayerImpl {
public:
    virtual ~Llama2EmbeddingLayerImpl();
    // N -> [N, hidden_dim]
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) = 0;
    virtual bool Init(Llama2Model* model);
    virtual void UnInit();

    size_t nwords_size;
    size_t weight_size;
    size_t hidden_dim;
    int pad_id;
    uint8_t* embedding_weight{nullptr};
};


class Llama2EmbeddingLayer {
public:
    
    ~Llama2EmbeddingLayer();
    // N -> [N, hidden_dim]
    std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx);
    bool Init(Llama2Model* model);
    void UnInit();
    
    Llama2EmbeddingLayerImpl* impl{nullptr};
};


class RMSNormLayerImpl {
public:
    virtual ~RMSNormLayerImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) = 0;
    virtual bool Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm);
    virtual void UnInit();

    size_t hidden_dim;
    size_t weight_size;
    float eps;
    uint8_t* norm_weight{nullptr};
};


class RMSNormLayer {
public:
    ~RMSNormLayer();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx);
    bool Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm);
    void UnInit();
    RMSNormLayerImpl* impl{nullptr};
};


class RoPELayerImpl {
public:
    virtual ~RoPELayerImpl();
    virtual std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k, Llama2InferenceCtx& ctx) = 0;
    virtual bool Init(Llama2Model* model, const std::string& weight_path);
    virtual void UnInit();

    size_t hidden_dim;
    size_t n_heads;
    size_t head_dim;
    size_t rope_dim;
    size_t weight_size;
    bool rope_is_neox_style;
    float* freqs_cis{nullptr};
};


class RoPELayer {
public:
    ~RoPELayer();
    virtual std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k, Llama2InferenceCtx& ctx);
    bool Init(Llama2Model* model, const std::string& weight_path);
    void UnInit();
    RoPELayerImpl* impl{nullptr};
};


class ArgMaxLayerImpl {
public:
    virtual ~ArgMaxLayerImpl();
    virtual std::shared_ptr<Tensor>
        Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) = 0;
    virtual bool Init(Llama2Model* model);
    virtual void UnInit();

    size_t hidden_dim;
};


class ArgMaxLayer {
public:
    ~ArgMaxLayer();
    virtual std::shared_ptr<Tensor>
        Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx);
    bool Init(Llama2Model* model);
    void UnInit();
    ArgMaxLayerImpl* impl{nullptr};
};

class SampleTopPLayerImpl {
public:
    virtual ~SampleTopPLayerImpl();
    virtual int
        Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) = 0;
    virtual bool Init(Llama2Model* model);
    virtual void UnInit();

    float temperature;
    float top_p;
    size_t vocab_size;
};


class SampleTopPLayer {
public:
    ~SampleTopPLayer();
    virtual int
        Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx);
    bool Init(Llama2Model* model);
    void UnInit();
    SampleTopPLayerImpl* impl{nullptr};
};


class SoftmaxLayerImpl {
public:
    virtual ~SoftmaxLayerImpl();
    virtual std::shared_ptr<Tensor>
        Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) = 0;
    virtual bool Init(Llama2Model* model);
    virtual void UnInit();

    size_t hidden_dim;
    size_t n_heads;
    float eps;
};


class SoftmaxLayer {
public:
    ~SoftmaxLayer();
    virtual std::shared_ptr<Tensor>
        Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx);
    bool Init(Llama2Model* model);
    void UnInit();
    SoftmaxLayerImpl* impl{nullptr};
};


class CausualMaskLayerImpl {
public:
    virtual ~CausualMaskLayerImpl();
    virtual std::shared_ptr<Tensor>
        Forward(Llama2InferenceCtx& ctx) = 0;
    virtual bool Init(Llama2Model* model);
    virtual void UnInit();
};


class CausualMaskLayer {
public:
    ~CausualMaskLayer();
    virtual std::shared_ptr<Tensor>
        Forward(Llama2InferenceCtx& ctx);
    bool Init(Llama2Model* model);
    void UnInit();
    CausualMaskLayerImpl* impl{nullptr};
};


class MatmulLayerImpl {
public:
    virtual ~MatmulLayerImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) = 0;
    virtual bool Init(Llama2Model* model, const std::string& weight_path, size_t n, size_t k);
    virtual bool InitAWQ(Llama2Model* model, const std::string& weight_path,
                         const std::string& zero_path, const std::string& scale_path, size_t n, size_t k, QuantType quant_type);
    virtual void UnInit();

    size_t n;
    size_t k;
    size_t weight_size;
    size_t zero_size;
    size_t scale_size;
    uint8_t* weight{nullptr};
    uint8_t* qzeros{nullptr};
    uint8_t* qscales{nullptr};

    QuantType qtype{QuantType::NoQuant};
};


class MatmulLayer {
public:
    ~MatmulLayer();
    std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx);
    bool Init(Llama2Model* model, const std::string& weight_path, size_t n, size_t k);
    bool InitAWQ(Llama2Model* model, const std::string& weight_path,
                 const std::string& zero_path, const std::string& scale_path, size_t n, size_t k, QuantType quant_type);
    void UnInit();
    MatmulLayerImpl* impl{nullptr};
};


class Llamma2TransformerLayerImpl {
public:
    virtual ~Llamma2TransformerLayerImpl() = default;
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> mask, Llama2InferenceCtx& ctx) = 0;
    virtual bool Init(Llama2Model* model, int layer_no);
    virtual void UnInit();

    size_t ffn_hidden;
    size_t hidden_dim;
    size_t head_dim;
    size_t n_heads;
    size_t max_seq_len;
};


class Llamma2TransformerLayer {
public:
    ~Llamma2TransformerLayer();
    std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> mask, Llama2InferenceCtx& ctx);
    bool Init(Llama2Model* model, int layer_no);
    void UnInit();

    Llamma2TransformerLayerImpl* impl{nullptr};
};


class Llama2Model {
public:
    Llama2Config config;
    Tokenizer tokenizer;

    bool Init();

    bool InitFreqCIS();

    void Chat(const std::string& input_seq, const std::string& reverse_prompt);
    void TextCompletion(const std::string& input_seq);
    std::string GetCurrTokenString(size_t prev_string_size, const std::vector<int>& tokens);

    DeviceType device_type;
    int hidden_dim;
    int n_heads;
    int n_layers;
    int multiple_of;
    int max_seq_len;
    int max_gen_len;
    float norm_eps;
    float temperature{0.0f};
    float top_p{0.0f};
    QuantType q_type{QuantType::NoQuant};
    int quant_group_size{-1};
    bool rope_is_neox_style;

    aclrtStream model_stream;

    Llama2EmbeddingLayer embedding_layer;
    CausualMaskLayer causual_mask_layer;
    std::vector<Llamma2TransformerLayer> transformer_layers;
    RMSNormLayer last_norm;
    MatmulLayer last_mm;
    ArgMaxLayer argmax_layer;
    SampleTopPLayer top_p_layer;

    float* freq_cis{nullptr};

    bool debug_print{false};
    bool is_profiling{false};
    AppProfiler profiler;
};


