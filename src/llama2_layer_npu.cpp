#include <spdlog/spdlog.h>
#include <fmt/core.h>
#include <iostream>
#include <algorithm>
#include <boost/filesystem.hpp>

#include "llama2_layer_npu.hpp"
#include "acl_util.hpp"
#include "util.h"
#include "npu_ops.h"
#include "profiling.hpp"


Llama2EmbeddingLayerNPUImpl::~Llama2EmbeddingLayerNPUImpl() {

}


std::shared_ptr<Tensor> Llama2EmbeddingLayerNPUImpl::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    spdlog::debug("Llama2EmbeddingLayerNPUImpl::Forward!");
    auto output = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);
    if (input->data_type != DT_UINT32) {
        spdlog::critical("Llama2EmbeddingLayerNPUImpl::Forward invalid input type");
        return output;
    }

    int32_t* input_ptr = static_cast<int32_t*>(input->data_ptr);
    uint16_t* output_ptr = static_cast<uint16_t*>(output->data_ptr);

    APP_PROFILE("Llama2EmbeddingLayer", fmt::format("hidden_dim: {} ctx.cur_size: {}", hidden_dim, ctx.cur_size).c_str(),
        ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);

    npu_embedding_layer((void*)output_ptr, (void*)embedding_weight, (void*)input_ptr, ctx.cur_size, hidden_dim, DT_FLOAT16, ctx.npu_stream);
    if (ctx.model->debug_print) {
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    }

    return output;
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
    spdlog::debug("RMSNormLayerNPUImpl::Forward!");
    auto output = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

    uint16_t* input_ptr = static_cast<uint16_t*>(input->data_ptr);
    uint16_t* output_ptr = static_cast<uint16_t*>(output->data_ptr);

    APP_PROFILE("RMSNormLayer", fmt::format("hidden_dim: {} ctx.cur_size: {}", hidden_dim, ctx.cur_size).c_str(),
        ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    npu_rmsnorm_layer((void*)output_ptr, (void*)norm_weight, (void*)input_ptr, ctx.cur_size, hidden_dim, eps, DT_FLOAT16, ctx.npu_stream);
    if (ctx.model->debug_print) {
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    }

    return output;
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
    auto hs = n_heads * ctx.cur_size;
    auto output = Tensor::MakeNPUTensor(hs * ctx.cur_pos, DT_FLOAT16);

    spdlog::debug("SoftmaxLayerNPUImpl::Forward! hs {} cur_pos {} cur_size {}", hs, ctx.cur_pos, ctx.cur_size);

    uint16_t* input_ptr = static_cast<uint16_t*>(input->data_ptr);
    uint16_t* output_ptr = static_cast<uint16_t*>(output->data_ptr);

    APP_PROFILE("SoftmaxLayer", fmt::format("hs {} cur_pos {} cur_size {}", hs, ctx.cur_pos, ctx.cur_size).c_str(),
        ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    npu_softmax_layer((void*)output_ptr, (void*)input_ptr, hs, ctx.cur_pos, DT_FLOAT16, ctx.npu_stream);
    if (ctx.model->debug_print) {
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    }

    return output;
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
    spdlog::debug("RoPELayerNPUImpl::Forward pos: {} seq len: {} n_head {} head_dim {} hidden_dim {}", ctx.cur_pos, ctx.cur_size, n_heads, hidden_dim/n_heads, hidden_dim);

    auto output_q = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);
    auto output_k = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

    uint16_t* input_q_ptr = static_cast<uint16_t*>(input_q->data_ptr);
    uint16_t* input_k_ptr = static_cast<uint16_t*>(input_k->data_ptr);

    uint16_t* output_q_ptr = static_cast<uint16_t*>(output_q->data_ptr);
    uint16_t* output_k_ptr = static_cast<uint16_t*>(output_k->data_ptr);

    APP_PROFILE("RoPELayer", fmt::format("pos: {} seq len: {} n_head {} head_dim {} hidden_dim {}", ctx.prev_pos, ctx.cur_size, n_heads, hidden_dim/n_heads, hidden_dim).c_str(),
        ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    npu_rope_layer(output_q_ptr, output_k_ptr, freqs_cis, input_q_ptr, input_k_ptr,
                    ctx.prev_pos, ctx.cur_size, n_heads, hidden_dim, DT_FLOAT16, ctx.npu_stream);
    if (ctx.model->debug_print) {
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    }

    return std::make_tuple(output_q, output_k);
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

    if (ctx.model->debug_print) {
        auto pre_norm_out_cpu = pre_norm_out->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        pre_norm_out_map(static_cast<Eigen::half*>(pre_norm_out_cpu->data_ptr), ctx.cur_size, hidden_dim);

        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = pre_norm_out_map.slice(print_offsets, print_extents);
        std::cout << "pre_norm output \n" << print_slice << "\n";
    }


    if (ctx.model->debug_print) {
        auto q_cpu = q->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        q_map(static_cast<Eigen::half*>(q_cpu->data_ptr), ctx.cur_size, hidden_dim);

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

    if (ctx.model->debug_print) {
        auto q_emb_cpu = q_emb->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        q_emb_map(static_cast<Eigen::half*>(q_emb->data_ptr), ctx.cur_size, n_heads, head_dim);

        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {ctx.cur_size, 4, 4};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_slice = q_emb_map.slice(print_offsets, print_extents);
        std::cout << "q emb output \n" << print_slice << "\n";    
    }
    

    // (seq_length, n_heads, head_dim)-> (n_heads, seq_length, head_dim) -> (n_heads, head_dim, seq_length)
    

    if (ctx.model->debug_print) {
        auto k_emb_cpu = k_emb->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        k_emb_map(static_cast<Eigen::half*>(k_emb->data_ptr), ctx.cur_size, n_heads, head_dim);
        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {ctx.cur_size, 4, 4};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_slice = k_emb_map.slice(print_offsets, print_extents);
        std::cout << "k emb output \n" << print_slice << "\n";    
    }
    // update kv cache
    size_t copy_size = ctx.cur_size * n_heads * head_dim * sizeof(uint16_t);
    size_t copy_offset = ctx.prev_pos * n_heads * head_dim * sizeof(uint16_t);
    
    {
        APP_PROFILE("UpdateKVCache", fmt::format("copy_size {} byte", copy_size).c_str(),
            ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
        CHECK_ACL(aclrtMemcpyAsync((void*)((uint8_t*)k_cache->data_ptr + copy_offset), copy_size, k_emb->data_ptr, copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.npu_stream));
        CHECK_ACL(aclrtMemcpyAsync((void*)((uint8_t*)v_cache->data_ptr + copy_offset), copy_size, v->data_ptr, copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.npu_stream));
    }

    if (ctx.model->debug_print) {
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
        auto k_cache_cpu = k_cache->to(DEV_CPU);
        auto v_cache_cpu = v_cache->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        k_cache_map(static_cast<Eigen::half*>(k_cache_cpu->data_ptr), ctx.cur_pos, n_heads, head_dim);

        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        v_cache_map(static_cast<Eigen::half*>(v_cache_cpu->data_ptr), ctx.cur_pos, n_heads, head_dim);
        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {std::min((Eigen::Index)4, (Eigen::Index)ctx.cur_pos), 4, 4};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_k_slice = k_cache_map.slice(print_offsets, print_extents);
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_v_slice = v_cache_map.slice(print_offsets, print_extents);
        std::cout << "k_cache_map output \n" << print_k_slice << "\n";
        std::cout << "v_cache_map output \n" << print_v_slice << "\n";
        v_cache_cpu->to_file("v_cache_cpu.data");
    }

    auto q_matmul_k = Tensor::MakeNPUTensor(n_heads * ctx.cur_pos * ctx.cur_size, DT_FLOAT16);
    float qk_scale = 1/sqrtf(static_cast<float>(head_dim));

    // (bs, nh, seqlen, hd) @ (bs, nh, hd, cache_len+seqlen) => bs, nh, seqlen, cache_len+seqlen
    

    spdlog::debug("n_heads: {} head_dim: {} ctx.cur_size: {} ctx.cur_pos: {}", 
                    n_heads, head_dim, ctx.cur_size, ctx.cur_pos);

    {
        APP_PROFILE("BMM_QK", fmt::format("n_heads: {} head_dim: {} ctx.cur_size: {} ctx.cur_pos: {}", n_heads, head_dim, ctx.cur_size, ctx.cur_pos).c_str(),
                ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
        npu_batch_matmul_qk_trans_causual_layer(q_matmul_k->data_ptr, q_emb->data_ptr, k_cache->data_ptr,
                            n_heads, ctx.cur_size, ctx.cur_pos, head_dim, ctx.prev_pos, qk_scale, DT_FLOAT16, ctx.npu_stream);
    }
    
    if (ctx.model->debug_print) {
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
        auto q_matmul_k_cpu = q_matmul_k->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>>
        q_matmul_k_map(static_cast<Eigen::half*>(q_matmul_k->data_ptr), n_heads, ctx.cur_size, ctx.cur_pos);
        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {n_heads, ctx.cur_size, ctx.cur_pos};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_slice = q_matmul_k_map.slice(print_offsets, print_extents);
        std::cout << "score output \n" << print_slice << "\n";
        //q_matmul_k_cpu->to_file("first_qk.data");
    }
    

    spdlog::debug("scores = F.softmax(scores.float(), dim=-1).type_as(xq)");
    auto softmax_qk = softmax.Forward(q_matmul_k, ctx);

    // (N, S, S)
    if (ctx.model->debug_print) {
        auto softmax_qk_cpu = softmax_qk->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
        softmax_qk_map(static_cast<Eigen::half*>(softmax_qk_cpu->data_ptr), n_heads, ctx.cur_size, ctx.cur_pos);
        Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
        Eigen::array<Eigen::Index, 3> print_extents = {n_heads, ctx.cur_size, ctx.cur_pos};
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>  print_slice = softmax_qk_map.slice(print_offsets, print_extents);
        std::cout << "score softmax output \n" << print_slice << "\n";
        //softmax_qk_cpu->to_file("first_softmax_qk.data");
    }

    // (seq_length, n_heads, head_dim) -> (n_heads, seq_length, head_dim)

    spdlog::debug("output = torch.matmul(scores, values)");

    auto tmp_output_tensor = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

    {
        APP_PROFILE("BMM_SCORE_V", fmt::format("n_heads: {} head_dim: {} ctx.cur_size: {} ctx.cur_pos: {}", n_heads, head_dim, ctx.cur_size, ctx.cur_pos).c_str(),
                    ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
        npu_batch_matmul_trans_v_layer(tmp_output_tensor->data_ptr, softmax_qk->data_ptr, v_cache->data_ptr,
                            n_heads, ctx.cur_size, head_dim, ctx.cur_pos, 1.0, DT_FLOAT16, ctx.npu_stream);
    }
    

    if (ctx.model->debug_print) {
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
        auto tmp_output_tensor_cpu = tmp_output_tensor->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        tmp_output_tensor_map(static_cast<Eigen::half*>(tmp_output_tensor_cpu->data_ptr), ctx.cur_size, hidden_dim);
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 16};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = tmp_output_tensor_map.slice(print_offsets, print_extents);
        std::cout << "proj_o input \n" << print_slice << "\n";
        //tmp_output_tensor_cpu->to_file("xv_output.data");
    }

    spdlog::debug("o_proj.Forward");
    auto output = o_proj.Forward(tmp_output_tensor, ctx);

    if (ctx.model->debug_print) {
        auto output_cpu = output->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        output_map(static_cast<Eigen::half*>(output_cpu->data_ptr), ctx.cur_size, hidden_dim);
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = output_map.slice(print_offsets, print_extents);
        std::cout << "score output \n" << print_slice << "\n";
        //output_cpu->to_file("first_output.data");
    }

    auto output_add_input = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

    spdlog::debug("attn output + input");
    {
        APP_PROFILE("Add", fmt::format("attn output + input").c_str(),
                        ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
        npu_add_layer(output_add_input->data_ptr, input->data_ptr, output->data_ptr,
                        ctx.cur_size * hidden_dim, DT_FLOAT16, ctx.npu_stream);
    }

    if (ctx.model->debug_print) {
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
        auto output_add_input_cpu = output_add_input->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        output_add_input_map(static_cast<Eigen::half*>(output_add_input_cpu->data_ptr), ctx.cur_size, hidden_dim);
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = output_add_input_map.slice(print_offsets, print_extents);
        std::cout << "z output \n" << print_slice << "\n";    
    }

    spdlog::debug("post_norm.Forward");
    auto post_norm_out = post_norm.Forward(output_add_input, ctx);

    if (ctx.model->debug_print) {
        auto post_norm_out_cpu = post_norm_out->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        post_norm_out_map(static_cast<Eigen::half*>(post_norm_out_cpu->data_ptr), ctx.cur_size, hidden_dim);
        Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
        Eigen::array<Eigen::Index, 2> print_extents = {ctx.cur_size, 4};
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>  print_slice = post_norm_out_map.slice(print_offsets, print_extents);
        std::cout << "post_norm output \n" << print_slice << "\n";    
    }

    spdlog::debug("gate_proj.Forward");
    auto w1_h = gate_proj.Forward(post_norm_out, ctx);

    if (ctx.model->debug_print) {
        auto w1_h_cpu = w1_h->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        w1_h_map(static_cast<Eigen::half*>(w1_h->data_ptr), ctx.cur_size, ffn_hidden);
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

    {
        APP_PROFILE("SILU_MUL", fmt::format("silu(gate_proj.Forward) * up_proj.Forward").c_str(),
                            ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
        npu_silu_mul_layer(silu_out_mul_w3->data_ptr, w1_h->data_ptr, w3_h->data_ptr,
                            silu_size, DT_FLOAT16, ctx.npu_stream);
    }
    if (ctx.model->debug_print) {
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
        auto silu_out_mul_w3_cpu = silu_out_mul_w3->to(DEV_CPU);
        Eigen::TensorMap<Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor|Eigen::DontAlign>> 
        silu_out_mul_w3_map(static_cast<Eigen::half*>(silu_out_mul_w3_cpu->data_ptr), ctx.cur_size, ffn_hidden);
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
   
    {
        APP_PROFILE("Add", fmt::format("w2_h + output").c_str(),
                            ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
        npu_add_layer(ffn_output->data_ptr, w2_h->data_ptr, output_add_input->data_ptr,
                        ctx.cur_size * hidden_dim, DT_FLOAT16, ctx.npu_stream);
    }

    //if (ctx.model->debug_print) {
    {
        // need to sync here, to make sure all temp tensor are read
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    }

    return ffn_output;
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

    k_cache = Tensor::MakeNPUTensor(hidden_dim * max_seq_len, DT_FLOAT16);
    v_cache = Tensor::MakeNPUTensor(hidden_dim * max_seq_len, DT_FLOAT16);

    return true;
}

void Llamma2TransformerLayerNPUImpl::UnInit() {
    Llamma2TransformerLayerImpl::UnInit();
}


MatmulLayerNPUImpl::~MatmulLayerNPUImpl() {

}

std::shared_ptr<Tensor> MatmulLayerNPUImpl::Forward(std::shared_ptr<Tensor> input, Llama2InferenceCtx& ctx) {
    auto output = Tensor::MakeNPUTensor(n * ctx.cur_size, DT_FLOAT16);

    
    uint16_t* input_ptr = static_cast<uint16_t*>(input->data_ptr);
    uint16_t* output_ptr = static_cast<uint16_t*>(output->data_ptr);

    spdlog::debug("MatmulLayerNPUImpl::Forward m {} n {} k {}", ctx.cur_size, n, k);

    {
        APP_PROFILE("MatmulLayer", fmt::format("m {} n {} k {}", ctx.cur_size, n, k).c_str(),
                            ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
        npu_matmul_nz_layer((void*)output_ptr, (void*)input_ptr, (void*)weight, ctx.cur_size, n, k, DT_FLOAT16, ctx.npu_stream);
    }
    if (ctx.model->debug_print) {
        CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    }

    return output;
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

#if 1
    void* tmp_dev_weight;
    CHECK_ACL(aclrtMalloc((void **)&tmp_dev_weight, weight_size, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMemcpy(tmp_dev_weight, weight_size, weight, weight_size, ACL_MEMCPY_HOST_TO_DEVICE));
    
    
    free(weight);
    weight = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&weight, weight_size, ACL_MEM_MALLOC_HUGE_FIRST));

    if (weight == nullptr) {
        spdlog::critical("oom!");
        return false;
    }

    npu_mamtul_weight_transpose_layer(weight, tmp_dev_weight, n, k, DT_FLOAT16, model->model_stream);
    CHECK_ACL(aclrtSynchronizeStream(model->model_stream));
    CHECK_ACL(aclrtFree(tmp_dev_weight));
#else
    auto n1 = n / 16;
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    curr_weight_map((Eigen::half*)(weight), k, n1, 16);
    Eigen::half* temp_output_ptr = new Eigen::half[n*k];
    Eigen::TensorMap<Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor|Eigen::DontAlign>> 
    tmp_output_map(temp_output_ptr, n1, k, 16);
    tmp_output_map = curr_weight_map.shuffle(Eigen::array<int, 3>({1, 0, 2}));
    
    void* tmp_dev_weight;
    CHECK_ACL(aclrtMalloc((void **)&tmp_dev_weight, weight_size, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tmp_dev_weight, weight_size, temp_output_ptr, weight_size, ACL_MEMCPY_HOST_TO_DEVICE));
    free(weight);
    weight = (uint8_t*)tmp_dev_weight;
    delete[] temp_output_ptr;
#endif
    return true;
}

void MatmulLayerNPUImpl::UnInit() {
    MatmulLayerImpl::UnInit();
}
