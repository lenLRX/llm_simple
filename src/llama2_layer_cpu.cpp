#include <spdlog/spdlog.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

#include "llama2_layer_cpu.hpp"


Llama2EmbeddingLayerCPUImpl::~Llama2EmbeddingLayerCPUImpl() {

}


std::shared_ptr<Tensor> Llama2EmbeddingLayerCPUImpl::Forward(std::shared_ptr<Tensor> input, size_t seq_len) {
    auto output = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT16);
    if (input->data_type != DT_UINT32) {
        spdlog::critical("Llama2EmbeddingLayerCPUImpl::Forward invalid input type");
        return output;
    }

    uint32_t* input_ptr = static_cast<uint32_t*>(input->data_ptr);
    uint16_t* output_ptr = static_cast<uint16_t*>(output->data_ptr);
    for (size_t i = 0;i < seq_len; ++i) {
        auto tok_id = input_ptr[i];
        spdlog::info("Llama2EmbeddingLayerCPUImpl::Forward i {}/{} tok_id {}", i, seq_len, tok_id);
        memcpy(output_ptr, embedding_weight +  tok_id * sizeof(uint16_t) * hidden_dim, sizeof(uint16_t) * hidden_dim);
        output_ptr = output_ptr + hidden_dim;
    }

    return output;
}


bool Llama2EmbeddingLayerCPUImpl::Init(Llama2Model* model) {
    if (!Llama2EmbeddingLayerImpl::Init(model)) {
        return false;
    }
    return true;
}

void Llama2EmbeddingLayerCPUImpl::UnInit() {
    Llama2EmbeddingLayerImpl::UnInit();
}


std::shared_ptr<Tensor> Llamma2TransformerLayerCPUImpl::Forward(std::shared_ptr<Tensor> input, size_t seq_len) {
    return input;
}

Llamma2TransformerLayerCPUImpl::~Llamma2TransformerLayerCPUImpl() {

}


bool Llamma2TransformerLayerCPUImpl::Init(Llama2Model* model, int layer_no) {
    if (!Llamma2TransformerLayerImpl::Init(model, layer_no)) {
        return false;
    }
    return true;
}

void Llamma2TransformerLayerCPUImpl::UnInit() {
    Llamma2TransformerLayerImpl::UnInit();
}


RMSNormLayerCPUImpl::~RMSNormLayerCPUImpl() {

}

std::shared_ptr<Tensor> RMSNormLayerCPUImpl::Forward(std::shared_ptr<Tensor> input, size_t seq_len) {
    //Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign> source_tensor(seq_len, hidden_size);

    auto output = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT16);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_map(static_cast<Eigen::half*>(output->data_ptr), seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_map(static_cast<Eigen::half*>(input->data_ptr), seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    weight_map((float*)input->data_ptr, 1, hidden_dim);

    auto input_fp32 = input_map.cast<float>();
    std::array<long,1> mean_dims         = {1};
    auto mean = (input_fp32*input_fp32).mean(mean_dims);
    auto sqrt_mean_add_eps = (mean + eps).sqrt();

    output_map = (input_fp32 / sqrt_mean_add_eps * weight_map).cast<Eigen::half>() ;
    return output;
}


bool RMSNormLayerCPUImpl::Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm) {
    if (!RMSNormLayerImpl::Init(model, layer_no, pre_norm, last_norm)) {
        return false;
    }
    // cast weight from half to float for CPU
    weight_size = sizeof(float) * model->hidden_dim;
    uint8_t* norm_weight_new = (uint8_t*)malloc(weight_size);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_map((Eigen::half*)norm_weight, 1, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_map((float*)norm_weight_new, 1, hidden_dim);

    output_map = input_map.cast<float>();
    free(norm_weight);
    norm_weight = norm_weight_new;

    return true;
}

void RMSNormLayerCPUImpl::UnInit() {
    RMSNormLayerImpl::UnInit();
}
