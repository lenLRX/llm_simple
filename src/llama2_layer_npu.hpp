#pragma once

#include "llama2_model.hpp"


class Llama2EmbeddingLayerNPUImpl: public Llama2EmbeddingLayerImpl {
public:
    virtual ~Llama2EmbeddingLayerNPUImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) override;
    virtual bool Init(Llama2Model* model) override;
    virtual void UnInit() override;
};


class RMSNormLayerNPUImpl: public RMSNormLayerImpl {
public:
    virtual ~RMSNormLayerNPUImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) override;
    virtual bool Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm) override;
    virtual void UnInit() override;
};

class SoftmaxLayerNPUImpl: public SoftmaxLayerImpl {
public:
    virtual ~SoftmaxLayerNPUImpl();
    virtual std::shared_ptr<Tensor>
        Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) override;
    virtual bool Init(Llama2Model* model) override;
    virtual void UnInit() override;
};

class RoPELayerNPUImpl: public RoPELayerImpl {
public:
    virtual ~RoPELayerNPUImpl();
    virtual std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k, Llama2InferenceCtx& ctx) override;
    virtual bool Init(Llama2Model* model, const std::string& weight_path) override;
    virtual void UnInit() override;
};



class Llamma2TransformerLayerNPUImpl: public Llamma2TransformerLayerImpl {
public:
    virtual ~Llamma2TransformerLayerNPUImpl();
    std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> mask, Llama2InferenceCtx& ctx) override;
    virtual bool Init(Llama2Model* model, int layer_no) override;
    virtual void UnInit() override;


    RMSNormLayer pre_norm;
    RMSNormLayer post_norm;
    MatmulLayer q_proj;
    MatmulLayer k_proj;
    MatmulLayer v_proj;
    MatmulLayer o_proj;

    MatmulLayer gate_proj; // w1
    MatmulLayer down_proj; // w2
    MatmulLayer up_proj; // w3

    RoPELayer rope_emb;
    SoftmaxLayer softmax;

    std::shared_ptr<Tensor> k_cache;
    std::shared_ptr<Tensor> v_cache;
};


class MatmulLayerNPUImpl: public MatmulLayerImpl {
public:
    virtual ~MatmulLayerNPUImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) override;
    virtual bool Init(Llama2Model* model, const std::string& weight_path, size_t n, size_t k) override;
    virtual bool InitAWQ(Llama2Model* model, const std::string& weight_path,
                         const std::string& zero_path, const std::string& scale_path, size_t n, size_t k, QuantType quant_type) override;
    virtual void UnInit() override;
};

