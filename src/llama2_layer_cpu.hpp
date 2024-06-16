#pragma once

#include "llama2_model.hpp"


class Llama2EmbeddingLayerCPUImpl: public Llama2EmbeddingLayerImpl {
public:
    virtual ~Llama2EmbeddingLayerCPUImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) override;
    virtual bool Init(Llama2Model* model) override;
    virtual void UnInit() override;
};


class Llamma2TransformerLayerCPUImpl: public Llamma2TransformerLayerImpl {
public:
    virtual ~Llamma2TransformerLayerCPUImpl();
    std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) override;
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
};


class RMSNormLayerCPUImpl: public RMSNormLayerImpl {
public:
    virtual ~RMSNormLayerCPUImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) override;
    virtual bool Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm) override;
    virtual void UnInit() override;
};


class ArgMaxLayerCPUImpl: public ArgMaxLayerImpl {
public:
    virtual ~ArgMaxLayerCPUImpl();
    virtual std::shared_ptr<Tensor>
        Forward(std::shared_ptr<Tensor> input, size_t seq_len) override;
    virtual bool Init(Llama2Model* model) override;
    virtual void UnInit() override;
};



class SoftmaxLayerCPUImpl: public SoftmaxLayerImpl {
public:
    virtual ~SoftmaxLayerCPUImpl();
    virtual std::shared_ptr<Tensor>
        Forward(std::shared_ptr<Tensor> input, size_t seq_len) override;
    virtual bool Init(Llama2Model* model) override;
    virtual void UnInit() override;
};



class MatmulLayerCPUImpl: public MatmulLayerImpl {
public:
    virtual ~MatmulLayerCPUImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) override;
    virtual bool Init(Llama2Model* model, const std::string& weight_path, size_t n, size_t k) override;
    virtual void UnInit() override;
};


class RoPELayerCPUImpl: public RoPELayerImpl {
public:
    virtual ~RoPELayerCPUImpl();
    virtual std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
        Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k, size_t seq_len) override;
    virtual bool Init(Llama2Model* model, const std::string& weight_path) override;
    virtual void UnInit() override;
};

