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
};


class RMSNormLayerCPUImpl: public RMSNormLayerImpl {
public:
    virtual ~RMSNormLayerCPUImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) override;
    virtual bool Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm) override;
    virtual void UnInit() override;
};



class MatmulLayerCPUImpl: public MatmulLayerImpl {
public:
    virtual ~MatmulLayerCPUImpl();
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input, size_t seq_len) override;
    virtual bool Init(Llama2Model* model, const std::string& weight_path) override;
    virtual void UnInit() override;
};
