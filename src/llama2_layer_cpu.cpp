#include <sstream>

#include <boost/filesystem.hpp>
#include <spdlog/spdlog.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

#include "util.h"
#include "llama2_layer_cpu.hpp"


Llama2EmbeddingLayerCPUImpl::~Llama2EmbeddingLayerCPUImpl() {

}


std::shared_ptr<Tensor> Llama2EmbeddingLayerCPUImpl::Forward(std::shared_ptr<Tensor> input, size_t seq_len) {
    auto output = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT16);
    if (input->data_type != DT_UINT32) {
        spdlog::critical("Llama2EmbeddingLayerCPUImpl::Forward invalid input type");
        return output;
    }

    int32_t* input_ptr = static_cast<int32_t*>(input->data_ptr);
    uint16_t* output_ptr = static_cast<uint16_t*>(output->data_ptr);
    for (size_t i = 0;i < seq_len; ++i) {
        auto tok_id = input_ptr[i];
        if (tok_id == pad_id) {
            break;
        }
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
    spdlog::info("pre_norm.Forward");
    auto pre_norm_out = pre_norm.Forward(input, seq_len);
    spdlog::info("q_proj.Forward");
    auto q = q_proj.Forward(pre_norm_out, seq_len);
    spdlog::info("k_proj.Forward");
    auto k = k_proj.Forward(pre_norm_out, seq_len);
    spdlog::info("v_proj.Forward");
    auto v = v_proj.Forward(pre_norm_out, seq_len);

    spdlog::info("rope_emb.Forward");
    auto q_k_emb = rope_emb.Forward(q, k, seq_len);

    spdlog::info("scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)");
    auto q_emb = std::get<0>(q_k_emb);
    auto k_emb = std::get<1>(q_k_emb);

    // (seq_length, n_heads, head_dim) -> (n_heads, seq_length, head_dim)
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    q_emb_map(static_cast<Eigen::half*>(q_emb->data_ptr), seq_len, n_heads, head_dim);

    // (seq_length, n_heads, head_dim)-> (n_heads, seq_length, head_dim) -> (n_heads, head_dim, seq_length)
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    k_emb_map(static_cast<Eigen::half*>(k_emb->data_ptr), seq_len, n_heads, head_dim);

    auto q_matmul_k = Tensor::MakeCPUTensor(n_heads * seq_len * seq_len, DT_FLOAT16);
    float qk_scale = 1/sqrtf(static_cast<float>(head_dim));

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    q_matmul_k_map(static_cast<Eigen::half*>(q_matmul_k->data_ptr), n_heads, seq_len, seq_len);

    Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign> q_emb_trans(n_heads, seq_len, head_dim);
    q_emb_trans = q_emb_map.cast<float>().shuffle(Eigen::array<int, 3>({1, 0, 2}));
    Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign> k_emb_trans(n_heads, head_dim, seq_len);
    k_emb_trans = k_emb_map.cast<float>().shuffle(Eigen::array<int, 3>({1, 2, 0}));

    auto q_matmul_k_f32 = Tensor::MakeCPUTensor(n_heads * seq_len * seq_len, DT_FLOAT32);
    Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    q_matmul_k_f32_map(static_cast<float*>(q_matmul_k_f32->data_ptr), n_heads, seq_len, seq_len);

    // tensor contraction does not support batch matmul
    // need a for loop to bmm
    // https://gitlab.com/libeigen/eigen/-/issues/2449
    Eigen::array<Eigen::IndexPair<int>, 1> qk_product_dims = { Eigen::IndexPair<int>(1, 0) };
    for (int i = 0; i < n_heads; ++i) {
        q_matmul_k_f32_map.chip<0>(i) = q_emb_trans.chip<0>(i).contract(k_emb_trans.chip<0>(i), qk_product_dims);
    }

    q_matmul_k_map = (q_matmul_k_f32_map * q_matmul_k_f32_map.constant(qk_scale)).cast<Eigen::half>();

    spdlog::info("scores = F.softmax(scores.float(), dim=-1).type_as(xq)");
    auto softmax_qk = softmax.Forward(q_matmul_k, seq_len);

    // (N, S, S)
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    softmax_qk_map(static_cast<Eigen::half*>(softmax_qk->data_ptr), n_heads, seq_len, seq_len);

    // (seq_length, n_heads, head_dim) -> (n_heads, seq_length, head_dim)
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    v_map(static_cast<Eigen::half*>(v->data_ptr), seq_len, n_heads, head_dim);

    auto vmap_trans = v_map.shuffle(Eigen::array<int, 3>({1, 0, 2}));

    spdlog::info("output = torch.matmul(scores, values)");

    auto tmp_output_tensor = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT16);
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    tmp_output_tensor_map(static_cast<Eigen::half*>(tmp_output_tensor->data_ptr), seq_len, hidden_dim);
    
    // tensor contraction does not support batch matmul
    // need a for loop to bmm
    // https://gitlab.com/libeigen/eigen/-/issues/2449
    Eigen::array<Eigen::IndexPair<int>, 1> output_product_dims = { Eigen::IndexPair<int>(1, 0) };
    // tmp_output: (n_heads, seq_length, head_dim)
    Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign> tmp_output(n_heads, seq_len, head_dim);
    for (int i = 0;i < n_heads; ++i) {
        tmp_output.chip<0>(i) = softmax_qk_map.chip<0>(i).cast<float>().contract(vmap_trans.chip<0>(i).cast<float>(), output_product_dims).cast<Eigen::half>();
    }

    // tmp_output: (n_heads, seq_length, head_dim) -> (seq_length, n_heads, head_dim)
    tmp_output_tensor_map = tmp_output.shuffle(Eigen::array<int, 3>({1, 0, 2})).reshape(std::array<long,2>{(long)seq_len, (long)hidden_dim});

    spdlog::info("o_proj.Forward");
    auto output = o_proj.Forward(tmp_output_tensor, seq_len);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_map(static_cast<Eigen::half*>(output->data_ptr), seq_len, hidden_dim);

    auto output_add_input = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT16);
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_add_input_map(static_cast<Eigen::half*>(output_add_input->data_ptr), seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_map(static_cast<Eigen::half*>(input->data_ptr), seq_len, hidden_dim);

    spdlog::info("attn output + input");
    output_add_input_map = (output_map.cast<float>() + input_map.cast<float>()).cast<Eigen::half>();

    spdlog::info("post_norm.Forward");
    auto post_norm_out = post_norm.Forward(output_add_input, seq_len);


    spdlog::info("gate_proj.Forward");
    auto w1_h = gate_proj.Forward(post_norm_out, seq_len);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    w1_h_map(static_cast<Eigen::half*>(w1_h->data_ptr), seq_len, ffn_hidden);

    spdlog::info("up_proj.Forward");
    auto w3_h = up_proj.Forward(post_norm_out, seq_len);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    w3_h_map(static_cast<Eigen::half*>(w3_h->data_ptr), seq_len, ffn_hidden);
    
    spdlog::info("silu(gate_proj.Forward) * up_proj.Forward");
    auto silu_out_mul_w3 = Tensor::MakeCPUTensor(ffn_hidden * seq_len, DT_FLOAT16);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    silu_out_mul_w3_map(static_cast<Eigen::half*>(silu_out_mul_w3->data_ptr), seq_len, ffn_hidden);

    auto w1_h_f32_map = w1_h_map.cast<float>();

    silu_out_mul_w3_map = ((w1_h_f32_map / ((-w1_h_f32_map).exp() + w1_h_f32_map.constant(1))) * w3_h_map.cast<float>()).cast<Eigen::half>();

    spdlog::info("down_proj.Forward");
    auto w2_h = down_proj.Forward(silu_out_mul_w3, seq_len);

    spdlog::info("w2_h + output");
    auto ffn_output = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT16);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    w2_h_map(static_cast<Eigen::half*>(w2_h->data_ptr), seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    ffn_output_map(static_cast<Eigen::half*>(ffn_output->data_ptr), seq_len, hidden_dim);

    ffn_output_map = (w2_h_map.cast<float>() + output_add_input_map.cast<float>()).cast<Eigen::half>();

    return ffn_output;
}

Llamma2TransformerLayerCPUImpl::~Llamma2TransformerLayerCPUImpl() {

}


bool Llamma2TransformerLayerCPUImpl::Init(Llama2Model* model, int layer_no) {
    if (!Llamma2TransformerLayerImpl::Init(model, layer_no)) {
        return false;
    }
    if (!pre_norm.Init(model, layer_no, true, false)) {
        return false;
    }

    if (!post_norm.Init(model, layer_no, false, false)) {
        return false;
    }

    auto q_name = std::string("model.layers.") + std::to_string(layer_no) + ".self_attn.q_proj.weight.bin";
    auto q_path = boost::filesystem::path(model->config.model_path) / q_name;
    if (!q_proj.Init(model, q_path.string(), hidden_dim, hidden_dim)) {
        return false;
    }
    auto k_name = std::string("model.layers.") + std::to_string(layer_no) + ".self_attn.k_proj.weight.bin";
    auto k_path = boost::filesystem::path(model->config.model_path) / k_name;
    if (!k_proj.Init(model, k_path.string(), hidden_dim, hidden_dim)) {
        return false;
    }

    auto v_name = std::string("model.layers.") + std::to_string(layer_no) + ".self_attn.v_proj.weight.bin";
    auto v_path = boost::filesystem::path(model->config.model_path) / v_name;
    if (!v_proj.Init(model, v_path.string(), hidden_dim, hidden_dim)) {
        return false;
    }

    auto o_name = std::string("model.layers.") + std::to_string(layer_no) + ".self_attn.o_proj.weight.bin";
    auto o_path = boost::filesystem::path(model->config.model_path) / o_name;
    if (!o_proj.Init(model, o_path.string(), hidden_dim, hidden_dim)) {
        return false;
    }

    auto inv_freq_name = std::string("model.layers.") + std::to_string(layer_no) + ".self_attn.rotary_emb.inv_freq.bin";
    auto inv_freq_path = boost::filesystem::path(model->config.model_path) / inv_freq_name;

    if (!rope_emb.Init(model, inv_freq_path.string())) {
        return false;
    }

    if (!softmax.Init(model)) {
        return false;
    }

    spdlog::info("ffn_hidden dim: {}", ffn_hidden);

    auto gate_proj_name = std::string("model.layers.") + std::to_string(layer_no) + ".mlp.gate_proj.weight.bin";
    auto gate_proj_path = boost::filesystem::path(model->config.model_path) / gate_proj_name;
    if (!gate_proj.Init(model, gate_proj_path.string(), ffn_hidden, hidden_dim)) {
        return false;
    }

    auto down_proj_name = std::string("model.layers.") + std::to_string(layer_no) + ".mlp.down_proj.weight.bin";
    auto down_proj_path = boost::filesystem::path(model->config.model_path) / down_proj_name;
    if (!down_proj.Init(model, down_proj_path.string(), hidden_dim, ffn_hidden)) {
        return false;
    }

    auto up_proj_name = std::string("model.layers.") + std::to_string(layer_no) + ".mlp.up_proj.weight.bin";
    auto up_proj_path = boost::filesystem::path(model->config.model_path) / up_proj_name;
    if (!up_proj.Init(model, up_proj_path.string(), ffn_hidden, hidden_dim)) {
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
    weight_map((float*)norm_weight, 1, hidden_dim);

    Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign> input_fp32 = input_map.cast<float>();
    std::array<long,1> mean_dims         = {1};
    Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign> mean = (input_fp32*input_fp32).mean(mean_dims).eval().reshape(std::array<long,2>{seq_len, 1});
    Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign> sqrt_mean_add_eps = (mean + mean.constant(eps)).sqrt().eval().reshape(std::array<long,2>{seq_len, 1});
    output_map = (input_fp32 / sqrt_mean_add_eps.broadcast(std::array<size_t, 2>{1, hidden_dim})
     * weight_map.broadcast(std::array<size_t, 2>{seq_len, 1})).cast<Eigen::half>();
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

ArgMaxLayerCPUImpl::~ArgMaxLayerCPUImpl() {

}

std::shared_ptr<Tensor>
ArgMaxLayerCPUImpl::Forward(std::shared_ptr<Tensor> input, size_t seq_len) {
    auto output = Tensor::MakeCPUTensor(1 * seq_len, DT_INT32);

    Eigen::TensorMap<Eigen::Tensor<int32_t, 1, Eigen::RowMajor|Eigen::DontAlign>> 
    output_map(static_cast<int32_t*>(output->data_ptr), seq_len);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_map(static_cast<Eigen::half*>(input->data_ptr), seq_len, hidden_dim);


    Eigen::Tensor<Eigen::DenseIndex, 1, Eigen::RowMajor|Eigen::DontAlign>
    tensor_argmax;

    tensor_argmax = input_map.argmax(1);

    spdlog::info("argmax result shape {}", TensorShapeToStr(tensor_argmax));

    output_map = tensor_argmax.cast<int32_t>();
    return output;
}

bool ArgMaxLayerCPUImpl::Init(Llama2Model* model) {
    return ArgMaxLayerImpl::Init(model);
}

void ArgMaxLayerCPUImpl::UnInit() {

}



SoftmaxLayerCPUImpl::~SoftmaxLayerCPUImpl() {

}

std::shared_ptr<Tensor>
SoftmaxLayerCPUImpl::Forward(std::shared_ptr<Tensor> input, size_t seq_len) {
    auto hs = n_heads * seq_len;
    auto output = Tensor::MakeCPUTensor(hs * seq_len, DT_FLOAT16);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_map(static_cast<Eigen::half*>(output->data_ptr), hs, seq_len);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_map(static_cast<Eigen::half*>(input->data_ptr), hs, seq_len);

    auto input_f32 = input_map.cast<float>();
    auto input_max = input_f32.maximum(Eigen::array<int, 1>{1}).eval()
        .reshape(Eigen::array<int, 2>{hs, 1}).broadcast(Eigen::array<int, 2>{1, seq_len});

    auto input_diff = (input_f32 - input_max).exp().eval();

    auto input_sum = input_diff.sum(Eigen::array<int, 1>{1}).eval()
        .reshape(Eigen::array<int, 2>{hs, 1}).broadcast(Eigen::array<int, 2>{1, seq_len});

    output_map = (input_diff/input_sum).cast<Eigen::half>();
    return output;
}

bool SoftmaxLayerCPUImpl::Init(Llama2Model* model) {
    return SoftmaxLayerImpl::Init(model);
}

void SoftmaxLayerCPUImpl::UnInit() {

}



MatmulLayerCPUImpl::~MatmulLayerCPUImpl() {

}

std::shared_ptr<Tensor> MatmulLayerCPUImpl::Forward(std::shared_ptr<Tensor> input, size_t seq_len) {
    auto output = Tensor::MakeCPUTensor(n * seq_len, DT_FLOAT16);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_map(static_cast<Eigen::half*>(output->data_ptr), seq_len, n);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_map(static_cast<Eigen::half*>(input->data_ptr), seq_len, k);

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    weight_map((float*)weight, k, n);

    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
    output_map = input_map.cast<float>().contract(weight_map, product_dims).cast<Eigen::half>();
    return output;
}

bool MatmulLayerCPUImpl::Init(Llama2Model* model, const std::string& weight_path, size_t n, size_t k) {
    if (!MatmulLayerImpl::Init(model, weight_path, n, k)) {
        return false;
    }
    // cast weight from half to float for CPU
    weight_size = sizeof(float) * n * k;
    uint8_t* mm_weight_new = (uint8_t*)malloc(weight_size);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_map((Eigen::half*)weight, 1, n*k);

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_map((float*)mm_weight_new, 1, n*k);

    output_map = input_map.cast<float>();
    free(weight);
    weight = mm_weight_new;
    return true;
}

void MatmulLayerCPUImpl::UnInit() {
    MatmulLayerImpl::UnInit();
}


RoPELayerCPUImpl::~RoPELayerCPUImpl() {

}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RoPELayerCPUImpl::Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k, size_t seq_len) {
    auto input_q_f32 = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT32);
    auto input_k_f32 = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT32);

    auto output_q_f32 = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT32);
    auto output_k_f32 = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT32);

    auto output_q = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT16);
    auto output_k = Tensor::MakeCPUTensor(hidden_dim * seq_len, DT_FLOAT16);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_q_map(static_cast<Eigen::half*>(input_q->data_ptr), seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_k_map(static_cast<Eigen::half*>(input_k->data_ptr), seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_q_f32_map(static_cast<float*>(input_q_f32->data_ptr), seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    input_k_f32_map(static_cast<float*>(input_k_f32->data_ptr), seq_len, hidden_dim);


    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_q_map(static_cast<Eigen::half*>(output_q->data_ptr), seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_k_map(static_cast<Eigen::half*>(output_k->data_ptr), seq_len, hidden_dim);


    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_q_f32_map(static_cast<float*>(output_q_f32->data_ptr), seq_len, hidden_dim);

    Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_k_f32_map(static_cast<float*>(output_k_f32->data_ptr), seq_len, hidden_dim);

    input_q_f32_map = input_q_map.cast<float>();
    input_k_f32_map = input_k_map.cast<float>();


    int freq_len = hidden_dim / 2;

    float* input_q_ptr = static_cast<float*>(input_q_f32->data_ptr);
    float* input_k_ptr = static_cast<float*>(input_k_f32->data_ptr);

    float* output_q_ptr = static_cast<float*>(output_q_f32->data_ptr);
    float* output_k_ptr = static_cast<float*>(output_k_f32->data_ptr);

    // https://mathworld.wolfram.com/ComplexMultiplication.html
    // (ac - bd), i(ad + bc)
    for (int s = 0; s < seq_len; ++s) {
        for (int f = 0;f < freq_len; ++f) {
            float fc = freqs_cis[2*f];
            float fd = freqs_cis[2*f+1];

            float qa = input_q_ptr[s*hidden_dim + 2*f];
            float qb = input_q_ptr[s*hidden_dim + 2*f+1];

            float ka = input_k_ptr[s*hidden_dim + 2*f];
            float kb = input_k_ptr[s*hidden_dim + 2*f+1];

            output_q_ptr[s*hidden_dim + 2*f] = qa * fc - qb * fd;
            output_q_ptr[s*hidden_dim + 2*f] = qa * fd + qb * fc;

            output_k_ptr[s*hidden_dim + 2*f] = ka * fc - kb * fd;
            output_k_ptr[s*hidden_dim + 2*f] = ka * fd + kb * fc;
        }
    }

    output_q_map = output_q_f32_map.cast<Eigen::half>();
    output_k_map = output_k_f32_map.cast<Eigen::half>();

    return std::make_tuple(output_q, output_k);
}

bool RoPELayerCPUImpl::Init(Llama2Model* model, const std::string& weight_path) {
    if (!RoPELayerImpl::Init(model, weight_path)) {
        return false;
    }
    return true;
}

void RoPELayerCPUImpl::UnInit() {

}

