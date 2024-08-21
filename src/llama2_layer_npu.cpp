#include <spdlog/spdlog.h>
#include <iostream>
#include <algorithm>
#include <boost/filesystem.hpp>

#include "llama2_layer_npu.hpp"
#include "acl_util.hpp"
#include "util.h"
#include "npu_ops/npu_ops.h"


Llama2EmbeddingLayerNPUImpl::~Llama2EmbeddingLayerNPUImpl() {

}


std::shared_ptr<Tensor> Llama2EmbeddingLayerNPUImpl::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    spdlog::info("Llama2EmbeddingLayerNPUImpl::Forward!");
    input = input->to(DEV_NPU);
    auto output = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);
    if (input->data_type != DT_UINT32) {
        spdlog::critical("Llama2EmbeddingLayerNPUImpl::Forward invalid input type");
        return output;
    }

    int32_t* input_ptr = static_cast<int32_t*>(input->data_ptr);
    uint16_t* output_ptr = static_cast<uint16_t*>(output->data_ptr);

    npu_embedding_layer((void*)output_ptr, (void*)embedding_weight, (void*)input_ptr, ctx.cur_size, hidden_dim, DT_FLOAT16, ctx.npu_stream);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

    return output->to(DEV_CPU);
}


bool Llama2EmbeddingLayerNPUImpl::Init(Llama2Model* model) {
    nwords_size = model->tokenizer.n_words;
    hidden_dim = model->hidden_dim;
    weight_size = sizeof(uint16_t) * nwords_size * model->hidden_dim;
    pad_id = model->tokenizer.pad_id;

    CHECK_ACL(aclrtMalloc((void **)&embedding_weight, weight_size, ACL_MEM_MALLOC_HUGE_FIRST));

    if (embedding_weight == nullptr) {
        spdlog::critical("oom!");
        return false;
    }

    auto weight_path = boost::filesystem::path(model->config.model_path) / "model.embed_tokens.weight.bin";

    void* embedding_weight_host = (uint8_t*)malloc(weight_size);

    if (!LoadBinaryFile(weight_path.c_str(), embedding_weight_host, weight_size)) {
        return false;
    }

    CHECK_ACL(aclrtMemcpy(embedding_weight, weight_size, embedding_weight_host, weight_size, ACL_MEMCPY_HOST_TO_DEVICE));

    free(embedding_weight_host);
    return true;
}

void Llama2EmbeddingLayerNPUImpl::UnInit() {
    CHECK_ACL(aclrtFree(embedding_weight));
    Llama2EmbeddingLayerImpl::UnInit();
}




RMSNormLayerNPUImpl::~RMSNormLayerNPUImpl() {

}

std::shared_ptr<Tensor> RMSNormLayerNPUImpl::Forward(std::shared_ptr<Tensor> input,  Llama2InferenceCtx& ctx) {
    spdlog::info("RMSNormLayerNPUImpl::Forward!");
    input = input->to(DEV_NPU);
    auto output = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

    uint16_t* input_ptr = static_cast<uint16_t*>(input->data_ptr);
    uint16_t* output_ptr = static_cast<uint16_t*>(output->data_ptr);

    npu_rmsnorm_layer((void*)output_ptr, (void*)norm_weight, (void*)input_ptr, ctx.cur_size, hidden_dim, eps, DT_FLOAT16, ctx.npu_stream);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

    return output->to(DEV_CPU);
}


bool RMSNormLayerNPUImpl::Init(Llama2Model* model, int layer_no, bool pre_norm, bool last_norm) {
    hidden_dim = model->hidden_dim;
    eps = model->norm_eps;

    weight_size = sizeof(uint16_t) * model->hidden_dim;
    CHECK_ACL(aclrtMalloc((void **)&norm_weight, weight_size, ACL_MEM_MALLOC_HUGE_FIRST));

    void* norm_weight_host = (uint8_t*)malloc(weight_size);

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

    if (!LoadBinaryFile(weight_path.c_str(), norm_weight_host, weight_size)) {
        return false;
    }

    CHECK_ACL(aclrtMemcpy(norm_weight, weight_size, norm_weight_host, weight_size, ACL_MEMCPY_HOST_TO_DEVICE));

    free(norm_weight_host);

    return true;
}

void RMSNormLayerNPUImpl::UnInit() {
    RMSNormLayerImpl::UnInit();
}



SoftmaxLayerNPUImpl::~SoftmaxLayerNPUImpl() {

}

std::shared_ptr<Tensor>
SoftmaxLayerNPUImpl::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    
    input = input->to(DEV_NPU);
    auto hs = n_heads * ctx.cur_size;
    auto output = Tensor::MakeCPUTensor(hs * ctx.cur_pos, DT_FLOAT16);

    spdlog::info("SoftmaxLayerNPUImpl::Forward! hs {} cur_pos {} cur_size {}", hs, ctx.cur_pos, ctx.cur_size);

    uint16_t* input_ptr = static_cast<uint16_t*>(input->data_ptr);
    uint16_t* output_ptr = static_cast<uint16_t*>(output->data_ptr);

    npu_softmax_layer((void*)output_ptr, (void*)input_ptr, hs, ctx.cur_pos, DT_FLOAT16, ctx.npu_stream);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

    return output->to(DEV_CPU);
}

bool SoftmaxLayerNPUImpl::Init(Llama2Model* model) {
    return SoftmaxLayerImpl::Init(model);
}

void SoftmaxLayerNPUImpl::UnInit() {

}



RoPELayerNPUImpl::~RoPELayerNPUImpl() {

}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RoPELayerNPUImpl::Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k,  Llama2InferenceCtx& ctx) {
    spdlog::info("RoPELayerNPUImpl::Forward seq len: {} n_head {} head_dim {} hidden_dim {}", ctx.cur_size, n_heads, hidden_dim/n_heads, hidden_dim);
    input_q = input_q->to(DEV_NPU);
    input_k = input_k->to(DEV_NPU);

    auto output_q = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);
    auto output_k = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

    uint16_t* input_q_ptr = static_cast<uint16_t*>(input_q->data_ptr);
    uint16_t* input_k_ptr = static_cast<uint16_t*>(input_k->data_ptr);

    uint16_t* output_q_ptr = static_cast<uint16_t*>(output_q->data_ptr);
    uint16_t* output_k_ptr = static_cast<uint16_t*>(output_k->data_ptr);

    npu_rope_layer(output_q_ptr, output_k_ptr, freqs_cis, input_q_ptr, input_k_ptr,
                    ctx.cur_size, n_heads, hidden_dim, DT_FLOAT16, ctx.npu_stream);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

    return std::make_tuple(output_q->to(DEV_CPU), output_k->to(DEV_CPU));
}

bool RoPELayerNPUImpl::Init(Llama2Model* model, const std::string& weight_path) {
    if (!RoPELayerImpl::Init(model, weight_path)) {
        return false;
    }
    return true;
}

void RoPELayerNPUImpl::UnInit() {

}



std::shared_ptr<Tensor> Llamma2TransformerLayerNPUImpl::Forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> mask, Llama2InferenceCtx& ctx) {
    spdlog::debug("pre_norm.Forward");
    auto pre_norm_out = pre_norm.Forward(input, ctx);
    spdlog::debug("q_proj.Forward");
    auto q = q_proj.Forward(pre_norm_out, ctx);
    spdlog::debug("k_proj.Forward");
    auto k = k_proj.Forward(pre_norm_out, ctx);
    spdlog::debug("v_proj.Forward");
    auto v = v_proj.Forward(pre_norm_out, ctx);

    {
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        pre_norm_out_map(static_cast<Eigen::half*>(pre_norm_out->data_ptr), ctx.cur_size, hidden_dim);

        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = pre_norm_out_map.slice(print_offsets, print_extents);
        std::cout << "pre_norm output \n" << print_slice << "\n";
    }


    {
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        q_map(static_cast<Eigen::half*>(q->data_ptr), ctx.cur_size, hidden_dim);

        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = q_map.slice(print_offsets, print_extents);
        std::cout << "q emb input \n" << print_slice << "\n";
    }

    spdlog::debug("rope_emb.Forward");
    auto q_k_emb = rope_emb.Forward(q, k, ctx);

    spdlog::debug("scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)");
    auto q_emb = std::get<0>(q_k_emb);
    auto k_emb = std::get<1>(q_k_emb);

    // (seq_length, n_heads, head_dim) -> (n_heads, seq_length, head_dim)
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    q_emb_map(static_cast<Eigen::half*>(q_emb->data_ptr), ctx.cur_size, n_heads, head_dim);

    {
        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {ctx.cur_size, 4, 4};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_slice = q_emb_map.slice(print_offsets, print_extents);
        std::cout << "q emb output \n" << print_slice << "\n";    
    }
    

    // (seq_length, n_heads, head_dim)-> (n_heads, seq_length, head_dim) -> (n_heads, head_dim, seq_length)
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    k_emb_map(static_cast<Eigen::half*>(k_emb->data_ptr), ctx.cur_size, n_heads, head_dim);

    {
        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {ctx.cur_size, 4, 4};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_slice = k_emb_map.slice(print_offsets, print_extents);
        std::cout << "k emb output \n" << print_slice << "\n";    
    }

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    v_map(static_cast<Eigen::half*>(v->data_ptr), ctx.cur_size, n_heads, head_dim);


    // update kv cache
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    k_cache_map(static_cast<Eigen::half*>(k_cache->data_ptr), ctx.cur_pos, n_heads, head_dim);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    v_cache_map(static_cast<Eigen::half*>(v_cache->data_ptr), ctx.cur_pos, n_heads, head_dim);

    {
        Eigen::array<Eigen::Index, 3> cache_offsets = {ctx.prev_pos, 0, 0};
        Eigen::array<Eigen::Index, 3> cache_extents = {ctx.cur_size, n_heads, head_dim};

        k_cache_map.slice(cache_offsets, cache_extents) = k_emb_map;
        v_cache_map.slice(cache_offsets, cache_extents) = v_map;
    }

    {
        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {std::min((Eigen::Index)4, (Eigen::Index)ctx.cur_pos), 4, 4};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_k_slice = k_cache_map.slice(print_offsets, print_extents);
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_v_slice = v_cache_map.slice(print_offsets, print_extents);
        std::cout << "k_cache_map output \n" << print_k_slice << "\n";
        std::cout << "v_cache_map output \n" << print_v_slice << "\n";
    }

    auto q_matmul_k = Tensor::MakeNPUTensor(n_heads * ctx.cur_pos * ctx.cur_size, DT_FLOAT16);
    float qk_scale = 1/sqrtf(static_cast<float>(head_dim));

    // (bs, nh, seqlen, hd) @ (bs, nh, hd, cache_len+seqlen) => bs, nh, seqlen, cache_len+seqlen
    

    std::cout << "n_heads: " << n_heads << " head_dim: " << head_dim
      << " ctx.cur_size :" << ctx.cur_size << " ctx.cur_pos: " << ctx.cur_pos << "\n";

    auto q_emb_npu = q_emb->to(DEV_NPU);
    auto k_emb_npu = k_cache->to(DEV_NPU);
    npu_batch_matmul_qk_trans_causual_layer(q_matmul_k->data_ptr, q_emb_npu->data_ptr, k_emb_npu->data_ptr,
                           n_heads, ctx.cur_size, ctx.cur_pos, head_dim, ctx.prev_pos, qk_scale, DT_FLOAT16, ctx.npu_stream);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));


    q_matmul_k = q_matmul_k->to(DEV_CPU);
    
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>>
    q_matmul_k_map(static_cast<Eigen::half*>(q_matmul_k->data_ptr), n_heads, ctx.cur_size, ctx.cur_pos);

    {
        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {n_heads, ctx.cur_size, ctx.cur_pos};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_slice = q_matmul_k_map.slice(print_offsets, print_extents);
        std::cout << "score output \n" << print_slice << "\n";    
    }
    

    spdlog::debug("scores = F.softmax(scores.float(), dim=-1).type_as(xq)");
    auto softmax_qk = softmax.Forward(q_matmul_k, ctx);

    // (N, S, S)
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    softmax_qk_map(static_cast<Eigen::half*>(softmax_qk->data_ptr), n_heads, ctx.cur_size, ctx.cur_pos);

    {
        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {n_heads, ctx.cur_size, ctx.cur_pos};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_slice = softmax_qk_map.slice(print_offsets, print_extents);
        std::cout << "score softmax output \n" << print_slice << "\n";    
    }

    // (seq_length, n_heads, head_dim) -> (n_heads, seq_length, head_dim)

    spdlog::debug("output = torch.matmul(scores, values)");

    auto tmp_output_tensor_npu = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);
    auto softmax_qk_npu = softmax_qk->to(DEV_NPU);

    std::cout << "bmm 2 n_heads: " << n_heads << " head_dim: " << head_dim
      << " ctx.cur_size :" << ctx.cur_size << " ctx.cur_pos: " << ctx.cur_pos << "\n";

    auto v_cache_npu = v_cache->to(DEV_NPU);

    npu_batch_matmul_trans_v_layer(tmp_output_tensor_npu->data_ptr, softmax_qk_npu->data_ptr, v_cache_npu->data_ptr,
                           n_heads, ctx.cur_size, head_dim, ctx.cur_pos, 1.0, DT_FLOAT16, ctx.npu_stream);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

    auto bmm2_cpu = tmp_output_tensor_npu->to(DEV_CPU);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>>
    bmm2_map(static_cast<Eigen::half*>(bmm2_cpu->data_ptr), n_heads, ctx.cur_size, head_dim);

    auto tmp_output_tensor = Tensor::MakeCPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);;
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    tmp_output_tensor_map(static_cast<Eigen::half*>(tmp_output_tensor->data_ptr), ctx.cur_size, hidden_dim);

    // tmp_output: (n_heads, seq_length, head_dim) -> (seq_length, n_heads, head_dim)
    tmp_output_tensor_map = bmm2_map.shuffle(Eigen::array<int, 3>({1, 0, 2})).reshape(std::array<long,2>{(long)ctx.cur_size, (long)hidden_dim});


    {
        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {n_heads, ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_slice = bmm2_map.slice(print_offsets, print_extents);
        std::cout << "xv output \n" << print_slice << "\n";    
    }

    {
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, hidden_dim};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = tmp_output_tensor_map.slice(print_offsets, print_extents);
        std::cout << "proj_o input \n" << print_slice << "\n";    
    }

    spdlog::debug("o_proj.Forward");
    auto output = o_proj.Forward(tmp_output_tensor, ctx);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_map(static_cast<Eigen::half*>(output->data_ptr), ctx.cur_size, hidden_dim);

    {
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = output_map.slice(print_offsets, print_extents);
        std::cout << "score output \n" << print_slice << "\n";    
    }

    auto output_add_input = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

    spdlog::debug("attn output + input");

    auto input_npu = input->to(DEV_NPU);
    auto output_npu = output->to(DEV_NPU);

    npu_add_layer(output_add_input->data_ptr, input_npu->data_ptr, output_npu->data_ptr,
                    ctx.cur_size * hidden_dim, DT_FLOAT16, ctx.npu_stream);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

    output_add_input = output_add_input->to(DEV_CPU);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    output_add_input_map(static_cast<Eigen::half*>(output_add_input->data_ptr), ctx.cur_size, hidden_dim);


    {
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = output_add_input_map.slice(print_offsets, print_extents);
        std::cout << "z output \n" << print_slice << "\n";    
    }

    spdlog::debug("post_norm.Forward");
    auto post_norm_out = post_norm.Forward(output_add_input, ctx);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    post_norm_out_map(static_cast<Eigen::half*>(post_norm_out->data_ptr), ctx.cur_size, hidden_dim);
    {
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = post_norm_out_map.slice(print_offsets, print_extents);
        std::cout << "post_norm output \n" << print_slice << "\n";    
    }


    spdlog::debug("gate_proj.Forward");
    auto w1_h = gate_proj.Forward(post_norm_out, ctx);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    w1_h_map(static_cast<Eigen::half*>(w1_h->data_ptr), ctx.cur_size, ffn_hidden);

    {
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = w1_h_map.slice(print_offsets, print_extents);
        std::cout << "gate_proj output \n" << print_slice << "\n";    
    }

    spdlog::debug("up_proj.Forward");
    auto w3_h = up_proj.Forward(post_norm_out, ctx);
    
    spdlog::debug("silu(gate_proj.Forward) * up_proj.Forward");

    auto silu_size = ffn_hidden * ctx.cur_size;
    auto silu_out_mul_w3 = Tensor::MakeNPUTensor(silu_size, DT_FLOAT16);

    auto w1_h_npu = w1_h->to(DEV_NPU);
    auto w3_h_npu = w3_h->to(DEV_NPU);

    npu_silu_mul_layer(silu_out_mul_w3->data_ptr, w1_h_npu->data_ptr, w3_h_npu->data_ptr,
                           silu_size, DT_FLOAT16, ctx.npu_stream);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

    silu_out_mul_w3 = silu_out_mul_w3->to(DEV_CPU);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    silu_out_mul_w3_map(static_cast<Eigen::half*>(silu_out_mul_w3->data_ptr), ctx.cur_size, ffn_hidden);
    
    {
        // bug here
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = silu_out_mul_w3_map.slice(print_offsets, print_extents);
        std::cout << "silu output \n" << print_slice << "\n";    
    }


    spdlog::debug("down_proj.Forward");
    auto w2_h = down_proj.Forward(silu_out_mul_w3, ctx);

    spdlog::debug("w2_h + output");
    auto ffn_output = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);
   
    auto w2_h_npu = w2_h->to(DEV_NPU);
    auto output_add_input_npu = output_add_input->to(DEV_NPU);

    npu_add_layer(ffn_output->data_ptr, w2_h_npu->data_ptr, output_add_input_npu->data_ptr,
                    ctx.cur_size * hidden_dim, DT_FLOAT16, ctx.npu_stream);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

    return ffn_output->to(DEV_CPU);
}

Llamma2TransformerLayerNPUImpl::~Llamma2TransformerLayerNPUImpl() {

}


bool Llamma2TransformerLayerNPUImpl::Init(Llama2Model* model, int layer_no) {
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

    spdlog::debug("ffn_hidden dim: {}", ffn_hidden);

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

    k_cache = Tensor::MakeCPUTensor(hidden_dim * max_seq_len, DT_FLOAT16);
    v_cache = Tensor::MakeCPUTensor(hidden_dim * max_seq_len, DT_FLOAT16);

    return true;
}

void Llamma2TransformerLayerNPUImpl::UnInit() {
    Llamma2TransformerLayerImpl::UnInit();
}


MatmulLayerNPUImpl::~MatmulLayerNPUImpl() {

}

std::shared_ptr<Tensor> MatmulLayerNPUImpl::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    auto output = Tensor::MakeNPUTensor(n * ctx.cur_size, DT_FLOAT16);
    input = input->to(DEV_NPU);

    
    uint16_t* input_ptr = static_cast<uint16_t*>(input->data_ptr);
    uint16_t* output_ptr = static_cast<uint16_t*>(output->data_ptr);

    spdlog::debug("MatmulLayerNPUImpl::Forward m {} n {} k {}", ctx.cur_size, n, k);

    npu_matmul_nz_layer((void*)output_ptr, (void*)input_ptr, (void*)weight, ctx.cur_size, n, k, DT_FLOAT16, ctx.npu_stream);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

    /*
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
    weight_map((Eigen::half*)(weight), k, n);

    {
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {k, std::min(n, size_t(128))};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = weight_map.slice(print_offsets, print_extents);
        std::cout << "weight_map output \n" << print_slice << "\n";    
    }
    */

    return output->to(DEV_CPU);
}

bool MatmulLayerNPUImpl::Init(Llama2Model* model, const std::string& weight_path, size_t n, size_t k) {
    if (!MatmulLayerImpl::Init(model, weight_path, n, k)) {
        return false;
    }
    if (k % 16 != 0) {
        spdlog::critical("k {} not aligned to 16", k);
        return false;
    }
    if (n % 16 != 0) {
        spdlog::critical("n {} not aligned to 16", n);
        return false;
    }
    size_t n1 = n / 16;
    weight_size = sizeof(uint16_t) * n * k;
    uint8_t* mm_weight_new = (uint8_t*)malloc(weight_size);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    input_map((Eigen::half*)weight, n1, 16, k);

    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    output_map((Eigen::half*)mm_weight_new, n1, k, 16);

    // transpose
    output_map = input_map.shuffle(Eigen::array<int, 3>({0,2,1}));
    
    free(weight);

    weight = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&weight, weight_size, ACL_MEM_MALLOC_HUGE_FIRST));

    if (weight == nullptr) {
        spdlog::critical("oom!");
        return false;
    }

    CHECK_ACL(aclrtMemcpy(weight, weight_size, mm_weight_new, weight_size, ACL_MEMCPY_HOST_TO_DEVICE));
    free(mm_weight_new);
    return true;
}

void MatmulLayerNPUImpl::UnInit() {
    MatmulLayerImpl::UnInit();
}
