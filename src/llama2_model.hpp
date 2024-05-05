#pragma once

#include <string>
#include <vector>
#include <cstdint>
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


class MatmulLayerImpl {
public:
    virtual ~MatmulLayerImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) = 0;
    virtual bool Init(Llama2Model* model, const std::string& weight_path);
    virtual void UnInit();

    size_t n;
    size_t k;
    uint8_t* weight{nullptr};
};


class MatmulLayer {
public:
    ~MatmulLayer();
    bool Init(Llama2Model* model, const std::string& weight_path);
    void UnInit();
    MatmulLayerImpl* impl{nullptr};
};


class Llamma2TransformerLayerImpl {
public:
    virtual ~Llamma2TransformerLayerImpl() = default;
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) = 0;
    virtual bool Init(Llama2Model* model, int layer_no);
    virtual void UnInit();
};


class Llamma2TransformerLayer {
public:
    ~Llamma2TransformerLayer();
    std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len);
    bool Init(Llama2Model* model, int layer_no);
    void UnInit();

    RMSNormLayer pre_norm;
    RMSNormLayer post_norm;
    Llamma2TransformerLayerImpl* impl{nullptr};
};



class Llama2Model {
public:
    Llama2Config config;
    Tokenizer tokenizer;

    bool Init();

    std::string Forward(const std::string& input_seq);

    DeviceType device_type;
    int hidden_dim;
    int n_heads;
    int n_layers;
    float norm_eps;

    Llama2EmbeddingLayer embedding_layer;
    std::vector<Llamma2TransformerLayer> transformer_layers;

};


