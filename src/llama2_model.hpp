#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <tuple>
#include <nlohmann/json.hpp>

#include "tokenizer.hpp"
#include "device.hpp"


class Llama2Config {
public:
    std::string tok_path;
    std::string model_path;
    std::string config_path;
    nlohmann::json config;
};


class Llama2Model;


class Llama2EmbeddingLayerImpl {
public:
    virtual ~Llama2EmbeddingLayerImpl();
    // N -> [N, hidden_dim]
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) = 0;
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
    std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len);
    bool Init(Llama2Model* model);
    void UnInit();
    
    Llama2EmbeddingLayerImpl* impl{nullptr};
};


class RMSNormLayerImpl {
public:
    virtual ~RMSNormLayerImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) = 0;
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
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len);
    bool Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm);
    void UnInit();
    RMSNormLayerImpl* impl{nullptr};
};


class RoPELayerImpl {
public:
    virtual ~RoPELayerImpl();
    virtual std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k, size_t seq_len) = 0;
    virtual bool Init(Llama2Model* model, const std::string& weight_path);
    virtual void UnInit();

    size_t hidden_dim;
    size_t rope_dim;
    size_t weight_size;
    float* freqs_cis{nullptr};
};


class RoPELayer {
public:
    ~RoPELayer();
    virtual std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k, size_t seq_len);
    bool Init(Llama2Model* model, const std::string& weight_path);
    void UnInit();
    RoPELayerImpl* impl{nullptr};
};


class ArgMaxLayerImpl {
public:
    virtual ~ArgMaxLayerImpl();
    virtual std::shared_ptr<Tensor>
        Forward(std::shared_ptr<Tensor> input, size_t seq_len) = 0;
    virtual bool Init(Llama2Model* model);
    virtual void UnInit();

    size_t hidden_dim;
};


class ArgMaxLayer {
public:
    ~ArgMaxLayer();
    virtual std::shared_ptr<Tensor>
        Forward(std::shared_ptr<Tensor> input, size_t seq_len);
    bool Init(Llama2Model* model);
    void UnInit();
    ArgMaxLayerImpl* impl{nullptr};
};


class SoftmaxLayerImpl {
public:
    virtual ~SoftmaxLayerImpl();
    virtual std::shared_ptr<Tensor>
        Forward(std::shared_ptr<Tensor> input, size_t seq_len) = 0;
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
        Forward(std::shared_ptr<Tensor> input, size_t seq_len);
    bool Init(Llama2Model* model);
    void UnInit();
    SoftmaxLayerImpl* impl{nullptr};
};


class MatmulLayerImpl {
public:
    virtual ~MatmulLayerImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) = 0;
    virtual bool Init(Llama2Model* model, const std::string& weight_path, size_t n, size_t k);
    virtual void UnInit();

    size_t n;
    size_t k;
    size_t weight_size;
    uint8_t* weight{nullptr};
};


class MatmulLayer {
public:
    ~MatmulLayer();
    std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len);
    bool Init(Llama2Model* model, const std::string& weight_path, size_t n, size_t k);
    void UnInit();
    MatmulLayerImpl* impl{nullptr};
};


class Llamma2TransformerLayerImpl {
public:
    virtual ~Llamma2TransformerLayerImpl() = default;
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) = 0;
    virtual bool Init(Llama2Model* model, int layer_no);
    virtual void UnInit();

    size_t ffn_hidden;
    size_t hidden_dim;
    size_t head_dim;
    size_t n_heads;
};


class Llamma2TransformerLayer {
public:
    ~Llamma2TransformerLayer();
    std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len);
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

    std::string Forward(const std::string& input_seq);

    DeviceType device_type;
    int hidden_dim;
    int n_heads;
    int n_layers;
    int multiple_of;
    int max_seq_len{8192};
    float norm_eps;

    Llama2EmbeddingLayer embedding_layer;
    std::vector<Llamma2TransformerLayer> transformer_layers;
    RMSNormLayer last_norm;
    MatmulLayer last_mm;
    ArgMaxLayer argmax_layer;

    float* freq_cis{nullptr};
};


