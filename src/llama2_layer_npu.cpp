#include <algorithm>
#include <boost/filesystem.hpp>
#include <exception>
#include <fmt/core.h>
#include <iostream>
#include <spdlog/spdlog.h>

#include "acl_util.hpp"
#include "defs.hpp"
#include "llama2_layer_npu.hpp"
#include "model_base.hpp"
#include "npu_ops.h"
#include "profiling.hpp"
#include "util.h"

EmbeddingLayerNPUImpl::~EmbeddingLayerNPUImpl() {}

std::shared_ptr<Tensor>
EmbeddingLayerNPUImpl::Forward(std::shared_ptr<Tensor> input,
                               InferenceCtx &ctx) {
  spdlog::debug("EmbeddingLayerNPUImpl::Forward! hidden_dim {} curr_size {} ",
                hidden_dim, ctx.cur_size);
  auto output = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);
  if (input->data_type != DT_UINT32) {
    spdlog::critical("EmbeddingLayerNPUImpl::Forward invalid input type");
    return output;
  }

  int32_t *input_ptr = static_cast<int32_t *>(input->data_ptr);
  uint16_t *output_ptr = static_cast<uint16_t *>(output->data_ptr);

  APP_PROFILE(
      "EmbeddingLayer",
      fmt::format("hidden_dim: {} ctx.cur_size: {}", hidden_dim, ctx.cur_size)
          .c_str(),
      ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);

  npu_embedding_layer((void *)output_ptr, (void *)embedding_weight,
                      (void *)input_ptr, ctx.cur_size, hidden_dim, DT_FLOAT16,
                      ctx.npu_stream);
  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
  }

  return output;
}

bool EmbeddingLayerNPUImpl::Init(ModelBase *model,
                                 const std::string &weight_path) {
  nwords_size = model->n_words;
  hidden_dim = model->hidden_dim;
  weight_size = sizeof(uint16_t) * nwords_size * model->hidden_dim;

  CHECK_ACL(aclrtMalloc((void **)&embedding_weight, weight_size,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  if (embedding_weight == nullptr) {
    spdlog::critical("oom!");
    return false;
  }

  void *embedding_weight_host = (uint8_t *)malloc(weight_size);

  if (!LoadBinaryFile(weight_path.c_str(), embedding_weight_host,
                      weight_size)) {
    return false;
  }

  CHECK_ACL(aclrtMemcpy(embedding_weight, weight_size, embedding_weight_host,
                        weight_size, ACL_MEMCPY_HOST_TO_DEVICE));

  free(embedding_weight_host);
  return true;
}

void EmbeddingLayerNPUImpl::UnInit() {
  CHECK_ACL(aclrtFree(embedding_weight));
  EmbeddingLayerImpl::UnInit();
}

RMSNormLayerNPUImpl::~RMSNormLayerNPUImpl() {}

std::shared_ptr<Tensor>
RMSNormLayerNPUImpl::Forward(std::shared_ptr<Tensor> input, InferenceCtx &ctx) {
  spdlog::debug("RMSNormLayerNPUImpl::Forward!");
  auto output = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

  uint16_t *input_ptr = static_cast<uint16_t *>(input->data_ptr);
  uint16_t *output_ptr = static_cast<uint16_t *>(output->data_ptr);

  APP_PROFILE(
      "RMSNormLayer",
      fmt::format("hidden_dim: {} ctx.cur_size: {}", hidden_dim, ctx.cur_size)
          .c_str(),
      ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
  npu_rmsnorm_layer((void *)output_ptr, (void *)norm_weight, (void *)input_ptr,
                    ctx.cur_size, hidden_dim, eps, ctx.model->config.data_type,
                    ctx.npu_stream);
  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
  }

  return output;
}

bool RMSNormLayerNPUImpl::Init(ModelBase *model, int layer_no, bool pre_norm,
                               bool last_norm) {
  hidden_dim = model->hidden_dim;
  eps = model->norm_eps;

  weight_size = sizeof(uint16_t) * model->hidden_dim;
  CHECK_ACL(aclrtMalloc((void **)&norm_weight, weight_size,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  void *norm_weight_host = (uint8_t *)malloc(weight_size);

  auto weight_path = boost::filesystem::path(model->config.model_path);
  if (last_norm) {
    // model.norm.weight.bin
    weight_path = weight_path / "model.norm.weight.bin";
  } else {
    std::stringstream ss;
    if (pre_norm) {
      ss << "model.layers." << layer_no << ".input_layernorm.weight.bin";
    } else {
      ss << "model.layers." << layer_no
         << ".post_attention_layernorm.weight.bin";
    }

    weight_path = weight_path / ss.str();
  }

  if (!LoadBinaryFile(weight_path.c_str(), norm_weight_host, weight_size)) {
    return false;
  }

  CHECK_ACL(aclrtMemcpy(norm_weight, weight_size, norm_weight_host, weight_size,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  free(norm_weight_host);

  return true;
}

void RMSNormLayerNPUImpl::UnInit() { RMSNormLayerImpl::UnInit(); }

SoftmaxLayerNPUImpl::~SoftmaxLayerNPUImpl() {}

std::shared_ptr<Tensor>
SoftmaxLayerNPUImpl::Forward(std::shared_ptr<Tensor> input, InferenceCtx &ctx) {
  auto hs = n_heads * ctx.cur_size;
  auto output = Tensor::MakeNPUTensor(hs * ctx.cur_pos, DT_FLOAT16);

  spdlog::debug("SoftmaxLayerNPUImpl::Forward! hs {} cur_pos {} cur_size {}",
                hs, ctx.cur_pos, ctx.cur_size);

  uint16_t *input_ptr = static_cast<uint16_t *>(input->data_ptr);
  uint16_t *output_ptr = static_cast<uint16_t *>(output->data_ptr);

  APP_PROFILE(
      "SoftmaxLayer",
      fmt::format("hs {} cur_pos {} cur_size {}", hs, ctx.cur_pos, ctx.cur_size)
          .c_str(),
      ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
  npu_softmax_layer((void *)output_ptr, (void *)input_ptr, hs, ctx.cur_pos,
                    DT_FLOAT16, ctx.npu_stream);
  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
  }

  return output;
}

bool SoftmaxLayerNPUImpl::Init(ModelBase *model) {
  return SoftmaxLayerImpl::Init(model);
}

void SoftmaxLayerNPUImpl::UnInit() {}

RoPELayerNPUImpl::~RoPELayerNPUImpl() {}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RoPELayerNPUImpl::Forward(std::shared_ptr<Tensor> input_q,
                          std::shared_ptr<Tensor> input_k, InferenceCtx &ctx) {
  spdlog::debug("RoPELayerNPUImpl::Forward pos: {} seq len: {} n_head {} "
                "head_dim {} hidden_dim {}",
                ctx.cur_pos, ctx.cur_size, n_heads, hidden_dim / n_heads,
                hidden_dim);

  auto output_q = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);
  auto output_k = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

  uint16_t *input_q_ptr = static_cast<uint16_t *>(input_q->data_ptr);
  uint16_t *input_k_ptr = static_cast<uint16_t *>(input_k->data_ptr);

  uint16_t *output_q_ptr = static_cast<uint16_t *>(output_q->data_ptr);
  uint16_t *output_k_ptr = static_cast<uint16_t *>(output_k->data_ptr);

  APP_PROFILE(
      "RoPELayer",
      fmt::format("pos: {} seq len: {} n_head {} head_dim {} hidden_dim {}",
                  ctx.prev_pos, ctx.cur_size, n_heads, hidden_dim / n_heads,
                  hidden_dim)
          .c_str(),
      ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
  npu_rope_layer(output_q_ptr, output_k_ptr, freqs_cis, input_q_ptr,
                 input_k_ptr, ctx.prev_pos, ctx.cur_size, n_heads, hidden_dim,
                 rope_is_neox_style, DT_FLOAT16, ctx.npu_stream);
  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
  }

  return std::make_tuple(output_q, output_k);
}

bool RoPELayerNPUImpl::Init(ModelBase *model, const std::string &weight_path) {
  if (!RoPELayerImpl::Init(model, weight_path)) {
    return false;
  }
  return true;
}

void RoPELayerNPUImpl::UnInit() {}

std::shared_ptr<Tensor>
Llamma2TransformerLayerNPUImpl::Forward(std::shared_ptr<Tensor> input,
                                        std::shared_ptr<Tensor> mask,
                                        InferenceCtx &ctx) {
  spdlog::debug("pre_norm.Forward");
  auto pre_norm_out = pre_norm.Forward(input, ctx);
  spdlog::debug("q_proj.Forward");
  auto q = q_proj.Forward(pre_norm_out, ctx);
  spdlog::debug("k_proj.Forward");
  auto k = k_proj.Forward(pre_norm_out, ctx);
  spdlog::debug("v_proj.Forward");
  auto v = v_proj.Forward(pre_norm_out, ctx);

  if (ctx.model->config.debug_print) {
    auto pre_norm_out_cpu = pre_norm_out->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
        pre_norm_out_map(static_cast<Eigen::half *>(pre_norm_out_cpu->data_ptr),
                         ctx.cur_size, hidden_dim);

    Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> print_extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4};
    Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = pre_norm_out_map.slice(print_offsets, print_extents);
    std::cout << "pre_norm output \n" << print_slice << "\n";
  }

  if (ctx.model->config.debug_print) {
    auto q_cpu = q->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
        q_map(static_cast<Eigen::half *>(q_cpu->data_ptr), ctx.cur_size,
              hidden_dim);

    Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> print_extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4};
    Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = q_map.slice(print_offsets, print_extents);
    std::cout << "q emb input \n" << print_slice << "\n";
  }

  if (ctx.model->config.debug_print) {
    auto k_cpu = k->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
        k_map(static_cast<Eigen::half *>(k_cpu->data_ptr), ctx.cur_size,
              hidden_dim);

    Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> print_extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4};
    Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = k_map.slice(print_offsets, print_extents);
    std::cout << "k emb input \n" << print_slice << "\n";
  }

  spdlog::debug("rope_emb.Forward");
  auto q_k_emb = rope_emb.Forward(q, k, ctx);

  spdlog::debug("scores = torch.matmul(xq, keys.transpose(2, 3)) / "
                "math.sqrt(self.head_dim)");
  auto q_emb = std::get<0>(q_k_emb);
  auto k_emb = std::get<1>(q_k_emb);

  // (seq_length, n_heads, head_dim) -> (n_heads, seq_length, head_dim)

  if (ctx.model->config.debug_print) {
    auto q_emb_cpu = q_emb->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
        q_emb_map(static_cast<Eigen::half *>(q_emb->data_ptr), ctx.cur_size,
                  n_heads, head_dim);

    Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> print_extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4, 4};
    Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = q_emb_map.slice(print_offsets, print_extents);
    std::cout << "q emb output \n" << print_slice << "\n";
  }

  // (seq_length, n_heads, head_dim)-> (n_heads, seq_length, head_dim) ->
  // (n_heads, head_dim, seq_length)

  if (ctx.model->config.debug_print) {
    auto k_emb_cpu = k_emb->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
        k_emb_map(static_cast<Eigen::half *>(k_emb->data_ptr), ctx.cur_size,
                  n_heads, head_dim);
    Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> print_extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4, 4};
    Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = k_emb_map.slice(print_offsets, print_extents);
    std::cout << "k emb output \n" << print_slice << "\n";
  }
  // update kv cache
  size_t copy_size = ctx.cur_size * n_heads * head_dim * sizeof(uint16_t);
  size_t copy_offset = ctx.prev_pos * n_heads * head_dim * sizeof(uint16_t);

  {
    APP_PROFILE("UpdateKVCache",
                fmt::format("copy_size {} byte", copy_size).c_str(),
                ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    CHECK_ACL(
        aclrtMemcpyAsync((void *)((uint8_t *)k_cache->data_ptr + copy_offset),
                         copy_size, k_emb->data_ptr, copy_size,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.npu_stream));
    CHECK_ACL(aclrtMemcpyAsync(
        (void *)((uint8_t *)v_cache->data_ptr + copy_offset), copy_size,
        v->data_ptr, copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.npu_stream));
  }

  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto k_cache_cpu = k_cache->to(DEV_CPU);
    auto v_cache_cpu = v_cache->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
        k_cache_map(static_cast<Eigen::half *>(k_cache_cpu->data_ptr),
                    ctx.cur_pos, n_heads, head_dim);

    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
        v_cache_map(static_cast<Eigen::half *>(v_cache_cpu->data_ptr),
                    ctx.cur_pos, n_heads, head_dim);
    Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
    Eigen::array<Eigen::Index, 3> print_extents = {
        std::min((Eigen::Index)4, (Eigen::Index)ctx.cur_pos), 4, 4};
    Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>
        print_k_slice = k_cache_map.slice(print_offsets, print_extents);
    Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>
        print_v_slice = v_cache_map.slice(print_offsets, print_extents);
    std::cout << "k_cache_map output \n" << print_k_slice << "\n";
    std::cout << "v_cache_map output \n" << print_v_slice << "\n";
    v_cache_cpu->to_file("v_cache_cpu.data");
  }

  float qk_scale = 1 / sqrtf(static_cast<float>(head_dim));
  auto tmp_output_tensor =
      Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

  if (true) {
    npu_flash_attn_opt_prefill_layer(
        tmp_output_tensor->data_ptr, q_emb->data_ptr, k_cache->data_ptr,
        v_cache->data_ptr, ctx.cur_size, ctx.cur_pos, ctx.prev_pos, n_heads,
        head_dim, DT_FLOAT16, ctx.npu_stream);
  } else {

    auto q_matmul_k =
        Tensor::MakeNPUTensor(n_heads * ctx.cur_pos * ctx.cur_size, DT_FLOAT16);

    // (bs, nh, seqlen, hd) @ (bs, nh, hd, cache_len+seqlen) => bs, nh, seqlen,
    // cache_len+seqlen

    spdlog::debug("n_heads: {} head_dim: {} ctx.cur_size: {} ctx.cur_pos: {}",
                  n_heads, head_dim, ctx.cur_size, ctx.cur_pos);

    {
      APP_PROFILE(
          "BMM_QK",
          fmt::format(
              "n_heads: {} head_dim: {} ctx.cur_size: {} ctx.cur_pos: {}",
              n_heads, head_dim, ctx.cur_size, ctx.cur_pos)
              .c_str(),
          ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
      npu_batch_matmul_qk_trans_causual_layer(
          q_matmul_k->data_ptr, q_emb->data_ptr, k_cache->data_ptr, n_heads,
          ctx.cur_size, ctx.cur_pos, head_dim, ctx.prev_pos, qk_scale,
          DT_FLOAT16, ctx.npu_stream);
    }

    if (ctx.model->config.debug_print) {
      CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
      auto q_matmul_k_cpu = q_matmul_k->to(DEV_CPU);
      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
          q_matmul_k_map(static_cast<Eigen::half *>(q_matmul_k->data_ptr),
                         n_heads, ctx.cur_size, ctx.cur_pos);
      Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
      Eigen::array<Eigen::Index, 3> print_extents = {
          static_cast<Eigen::Index>(n_heads),
          static_cast<Eigen::Index>(ctx.cur_size),
          static_cast<Eigen::Index>(ctx.cur_pos)};
      Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>
          print_slice = q_matmul_k_map.slice(print_offsets, print_extents);
      std::cout << "score output \n" << print_slice << "\n";
      // q_matmul_k_cpu->to_file("first_qk.data");
    }

    spdlog::debug("scores = F.softmax(scores.float(), dim=-1).type_as(xq)");
    auto softmax_qk = softmax.Forward(q_matmul_k, ctx);

    // (N, S, S)
    if (ctx.model->config.debug_print) {
      auto softmax_qk_cpu = softmax_qk->to(DEV_CPU);
      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>>
          softmax_qk_map(static_cast<Eigen::half *>(softmax_qk_cpu->data_ptr),
                         n_heads, ctx.cur_size, ctx.cur_pos);
      Eigen::array<Eigen::Index, 3> print_offsets = {0, 0, 0};
      Eigen::array<Eigen::Index, 3> print_extents = {
          static_cast<Eigen::Index>(n_heads),
          static_cast<Eigen::Index>(ctx.cur_size),
          static_cast<Eigen::Index>(ctx.cur_pos)};
      Eigen::Tensor<Eigen::half, 3, Eigen::RowMajor | Eigen::DontAlign>
          print_slice = softmax_qk_map.slice(print_offsets, print_extents);
      std::cout << "score softmax output \n" << print_slice << "\n";
      // softmax_qk_cpu->to_file("first_softmax_qk.data");
    }

    // (seq_length, n_heads, head_dim) -> (n_heads, seq_length, head_dim)

    spdlog::debug("output = torch.matmul(scores, values)");

    {
      APP_PROFILE(
          "BMM_SCORE_V",
          fmt::format(
              "n_heads: {} head_dim: {} ctx.cur_size: {} ctx.cur_pos: {}",
              n_heads, head_dim, ctx.cur_size, ctx.cur_pos)
              .c_str(),
          ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
      npu_batch_matmul_trans_v_layer(
          tmp_output_tensor->data_ptr, softmax_qk->data_ptr, v_cache->data_ptr,
          n_heads, ctx.cur_size, head_dim, ctx.cur_pos, 1.0, DT_FLOAT16,
          ctx.npu_stream);
    }

    if (ctx.model->config.debug_print) {
      CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
      auto tmp_output_tensor_cpu = tmp_output_tensor->to(DEV_CPU);
      Eigen::TensorMap<
          Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
          tmp_output_tensor_map(
              static_cast<Eigen::half *>(tmp_output_tensor_cpu->data_ptr),
              ctx.cur_size, hidden_dim);
      Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
      Eigen::array<Eigen::Index, 2> print_extents = {
          static_cast<Eigen::Index>(ctx.cur_size), 16};
      Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
          print_slice =
              tmp_output_tensor_map.slice(print_offsets, print_extents);
      std::cout << "proj_o input \n" << print_slice << "\n";
      // tmp_output_tensor_cpu->to_file("xv_output.data");
    }
  }

  spdlog::debug("o_proj.Forward");
  auto output = o_proj.Forward(tmp_output_tensor, ctx);

  if (ctx.model->config.debug_print) {
    auto output_cpu = output->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
        output_map(static_cast<Eigen::half *>(output_cpu->data_ptr),
                   ctx.cur_size, hidden_dim);
    Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> print_extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4};
    Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = output_map.slice(print_offsets, print_extents);
    std::cout << "score output \n" << print_slice << "\n";
    // output_cpu->to_file("first_output.data");
  }

  auto output_add_input =
      Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

  spdlog::debug("attn output + input");
  {
    APP_PROFILE("Add", fmt::format("attn output + input").c_str(),
                ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    npu_add_layer(output_add_input->data_ptr, input->data_ptr, output->data_ptr,
                  ctx.cur_size * hidden_dim, DT_FLOAT16, ctx.npu_stream);
  }

  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto output_add_input_cpu = output_add_input->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
        output_add_input_map(
            static_cast<Eigen::half *>(output_add_input_cpu->data_ptr),
            ctx.cur_size, hidden_dim);
    Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> print_extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4};
    Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = output_add_input_map.slice(print_offsets, print_extents);
    std::cout << "z output \n" << print_slice << "\n";
  }

  spdlog::debug("post_norm.Forward");
  auto post_norm_out = post_norm.Forward(output_add_input, ctx);

  if (ctx.model->config.debug_print) {
    auto post_norm_out_cpu = post_norm_out->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
        post_norm_out_map(
            static_cast<Eigen::half *>(post_norm_out_cpu->data_ptr),
            ctx.cur_size, hidden_dim);
    Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> print_extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4};
    Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = post_norm_out_map.slice(print_offsets, print_extents);
    std::cout << "post_norm output \n" << print_slice << "\n";
  }

  spdlog::debug("gate_proj.Forward");
  auto w1_h = gate_proj.Forward(post_norm_out, ctx);

  if (ctx.model->config.debug_print) {
    auto w1_h_cpu = w1_h->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
        w1_h_map(static_cast<Eigen::half *>(w1_h->data_ptr), ctx.cur_size,
                 ffn_hidden);
    Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> print_extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4};
    Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = w1_h_map.slice(print_offsets, print_extents);
    std::cout << "gate_proj output \n" << print_slice << "\n";
  }

  spdlog::debug("up_proj.Forward");
  auto w3_h = up_proj.Forward(post_norm_out, ctx);

  spdlog::debug("silu(gate_proj.Forward) * up_proj.Forward");

  auto silu_size = ffn_hidden * ctx.cur_size;
  auto silu_out_mul_w3 = Tensor::MakeNPUTensor(silu_size, DT_FLOAT16);

  {
    APP_PROFILE(
        "SILU_MUL",
        fmt::format("silu(gate_proj.Forward) * up_proj.Forward").c_str(),
        ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    npu_silu_mul_layer(silu_out_mul_w3->data_ptr, w1_h->data_ptr,
                       w3_h->data_ptr, silu_size, DT_FLOAT16, ctx.npu_stream);
  }
  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto silu_out_mul_w3_cpu = silu_out_mul_w3->to(DEV_CPU);
    Eigen::TensorMap<
        Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>>
        silu_out_mul_w3_map(
            static_cast<Eigen::half *>(silu_out_mul_w3_cpu->data_ptr),
            ctx.cur_size, ffn_hidden);
    // bug here
    Eigen::array<Eigen::Index, 2> print_offsets = {0, 0};
    Eigen::array<Eigen::Index, 2> print_extents = {
        static_cast<Eigen::Index>(ctx.cur_size), 4};
    Eigen::Tensor<Eigen::half, 2, Eigen::RowMajor | Eigen::DontAlign>
        print_slice = silu_out_mul_w3_map.slice(print_offsets, print_extents);
    std::cout << "silu output \n" << print_slice << "\n";
  }

  spdlog::debug("down_proj.Forward");
  auto w2_h = down_proj.Forward(silu_out_mul_w3, ctx);

  spdlog::debug("w2_h + output");
  auto ffn_output =
      Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, DT_FLOAT16);

  {
    APP_PROFILE("Add", fmt::format("w2_h + output").c_str(), ctx.npu_stream,
                &ctx.model->profiler, ctx.model->is_profiling);
    npu_add_layer(ffn_output->data_ptr, w2_h->data_ptr,
                  output_add_input->data_ptr, ctx.cur_size * hidden_dim,
                  DT_FLOAT16, ctx.npu_stream);
  }

  // if (ctx.model->config.debug_print) {
  {
    // need to sync here, to make sure all temp tensor are read
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
  }

  return ffn_output;
}

Llamma2TransformerLayerNPUImpl::~Llamma2TransformerLayerNPUImpl() {}

bool Llamma2TransformerLayerNPUImpl::Init(ModelBase *model, int layer_no) {
  if (!Llamma2TransformerLayerImpl::Init(model, layer_no)) {
    return false;
  }
  if (!pre_norm.Init(model, layer_no, true, false)) {
    return false;
  }

  if (!post_norm.Init(model, layer_no, false, false)) {
    return false;
  }

  if (model->config.q_type == QuantType::NoQuant) {
    auto q_name =
        fmt::format("model.layers.{}.self_attn.q_proj.weight.bin", layer_no);
    auto q_path = boost::filesystem::path(model->config.model_path) / q_name;
    if (!q_proj.Init(model, q_path.string(), hidden_dim, hidden_dim)) {
      return false;
    }
    auto k_name =
        fmt::format("model.layers.{}.self_attn.k_proj.weight.bin", layer_no);
    auto k_path = boost::filesystem::path(model->config.model_path) / k_name;
    if (!k_proj.Init(model, k_path.string(), hidden_dim, hidden_dim)) {
      return false;
    }

    auto v_name =
        fmt::format("model.layers.{}.self_attn.v_proj.weight.bin", layer_no);
    auto v_path = boost::filesystem::path(model->config.model_path) / v_name;
    if (!v_proj.Init(model, v_path.string(), hidden_dim, hidden_dim)) {
      return false;
    }

    auto o_name =
        fmt::format("model.layers.{}.self_attn.o_proj.weight.bin", layer_no);
    auto o_path = boost::filesystem::path(model->config.model_path) / o_name;
    if (!o_proj.Init(model, o_path.string(), hidden_dim, hidden_dim)) {
      return false;
    }

    auto gate_proj_name =
        fmt::format("model.layers.{}.mlp.gate_proj.weight.bin", layer_no);
    auto gate_proj_path =
        boost::filesystem::path(model->config.model_path) / gate_proj_name;
    if (!gate_proj.Init(model, gate_proj_path.string(), ffn_hidden,
                        hidden_dim)) {
      return false;
    }

    auto down_proj_name =
        fmt::format("model.layers.{}.mlp.down_proj.weight.bin", layer_no);
    auto down_proj_path =
        boost::filesystem::path(model->config.model_path) / down_proj_name;
    if (!down_proj.Init(model, down_proj_path.string(), hidden_dim,
                        ffn_hidden)) {
      return false;
    }

    auto up_proj_name =
        fmt::format("model.layers.{}.mlp.up_proj.weight.bin", layer_no);
    auto up_proj_path =
        boost::filesystem::path(model->config.model_path) / up_proj_name;
    if (!up_proj.Init(model, up_proj_path.string(), ffn_hidden, hidden_dim)) {
      return false;
    }
  } else if (model->config.q_type == QuantType::AWQ_4B) {
#define INIT_AWQ_MM(layer, x, n, k)                                            \
  auto x##_weight_name = fmt::format(                                          \
      "model.layers.{}." #layer "." #x "_proj.qweight.bin", layer_no);         \
  auto x##_zero_name = fmt::format(                                            \
      "model.layers.{}." #layer "." #x "_proj.qzeros.bin", layer_no);          \
  auto x##_scale_name = fmt::format(                                           \
      "model.layers.{}." #layer "." #x "_proj.scales.bin", layer_no);          \
  auto x##_weight_path =                                                       \
      boost::filesystem::path(model->config.model_path) / x##_weight_name;     \
  auto x##_zero_path =                                                         \
      boost::filesystem::path(model->config.model_path) / x##_zero_name;       \
  auto x##_scale_path =                                                        \
      boost::filesystem::path(model->config.model_path) / x##_scale_name;      \
  if (!x##_proj.InitAWQ(model, x##_weight_path.string(),                       \
                        x##_zero_path.string(), x##_scale_path.string(), n, k, \
                        model->config.q_type)) {                               \
    return false;                                                              \
  }

    INIT_AWQ_MM(self_attn, q, hidden_dim, hidden_dim);
    INIT_AWQ_MM(self_attn, k, hidden_dim, hidden_dim);
    INIT_AWQ_MM(self_attn, v, hidden_dim, hidden_dim);
    INIT_AWQ_MM(self_attn, o, hidden_dim, hidden_dim);

    INIT_AWQ_MM(mlp, gate, ffn_hidden, hidden_dim);
    INIT_AWQ_MM(mlp, down, hidden_dim, ffn_hidden);
    INIT_AWQ_MM(mlp, up, ffn_hidden, hidden_dim);
#undef INIT_AWQ_MM
  }

  auto inv_freq_name = std::string("model.layers.") + std::to_string(layer_no) +
                       ".self_attn.rotary_emb.inv_freq.bin";
  auto inv_freq_path =
      boost::filesystem::path(model->config.model_path) / inv_freq_name;

  if (!rope_emb.Init(model, inv_freq_path.string())) {
    return false;
  }

  if (!softmax.Init(model)) {
    return false;
  }

  spdlog::debug("ffn_hidden dim: {}", ffn_hidden);

  k_cache = Tensor::MakeNPUTensor(hidden_dim * max_seq_len, DT_FLOAT16);
  v_cache = Tensor::MakeNPUTensor(hidden_dim * max_seq_len, DT_FLOAT16);

  return true;
}

void Llamma2TransformerLayerNPUImpl::UnInit() {
  Llamma2TransformerLayerImpl::UnInit();
}

// Qwen2

std::shared_ptr<Tensor>
Qwen2TransformerLayerNPUImpl::Forward(std::shared_ptr<Tensor> input,
                                      std::shared_ptr<Tensor> mask,
                                      InferenceCtx &ctx) {
  spdlog::debug("pre_norm.Forward");
  auto pre_norm_out = pre_norm.Forward(input, ctx);
  spdlog::debug("q_proj.Forward");
  auto q = q_proj.Forward(pre_norm_out, ctx);
  spdlog::debug("k_proj.Forward");
  auto k = k_proj.Forward(pre_norm_out, ctx);
  spdlog::debug("v_proj.Forward");
  auto v = v_proj.Forward(pre_norm_out, ctx);

  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));

    auto pre_norm_out_cpu = pre_norm_out->to(DEV_CPU);
    std::cout << "pre_norm output \n"
              << print_tensor<2>(pre_norm_out_cpu->data_ptr,
                                 ctx.model->config.data_type,
                                 {ctx.cur_size, hidden_dim}, {ctx.cur_size, 4})
              << "\n";
  }

  if (ctx.model->config.debug_print) {
    auto q_cpu = q->to(DEV_CPU);
    std::cout << "q emb input \n"
              << print_tensor<2>(q_cpu->data_ptr, ctx.model->config.data_type,
                                 {ctx.cur_size, hidden_dim}, {ctx.cur_size, 4})
              << "\n";
  }

  if (ctx.model->config.debug_print) {
    auto k_cpu = k->to(DEV_CPU);
    std::cout << "k emb input \n"
              << print_tensor<2>(k_cpu->data_ptr, ctx.model->config.data_type,
                                 {ctx.cur_size, n_kv_heads * head_dim}, {ctx.cur_size, 4})
              << "\n";
  }

  spdlog::debug("q size {} k size {}", q->data_size, k->data_size);

  spdlog::debug("rope_emb.Forward");
  // auto q_k_emb = rope_emb.Forward(q, k, ctx);
  auto q_emb = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, dtype);
  auto k_emb =
      Tensor::MakeNPUTensor(n_kv_heads * head_dim * ctx.cur_size, dtype);

  spdlog::debug("n_heads {} n_kv_heads {} head_dim {}", n_heads, n_kv_heads,
                head_dim);
  spdlog::debug("prev_pos {} cur_size {}", ctx.prev_pos, ctx.cur_size);

  npu_rope_single_layer(q_emb->data_ptr, ctx.model->freq_cis, q->data_ptr,
                        ctx.prev_pos, ctx.cur_size, n_heads, hidden_dim,
                        ctx.model->config.rope_is_neox_style, dtype,
                        ctx.npu_stream);

  npu_rope_single_layer(
      k_emb->data_ptr, ctx.model->freq_cis, k->data_ptr, ctx.prev_pos,
      ctx.cur_size, n_kv_heads, n_kv_heads * head_dim,
      ctx.model->config.rope_is_neox_style, dtype, ctx.npu_stream);

  spdlog::debug("scores = torch.matmul(xq, keys.transpose(2, 3)) / "
                "math.sqrt(self.head_dim)");
  // auto q_emb = std::get<0>(q_k_emb);
  // auto k_emb = std::get<1>(q_k_emb);

  // (seq_length, n_heads, head_dim) -> (n_heads, seq_length, head_dim)

  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto q_emb_cpu = q_emb->to(DEV_CPU);
    std::cout << "q emb output \n"
              << print_tensor<3>(
                     q_emb_cpu->data_ptr, ctx.model->config.data_type,
                     {ctx.cur_size, n_heads, head_dim}, {ctx.cur_size, 2, 4})
              << std::endl;
    q_emb_cpu->to_file("q_emb_cpu.data");
  }

  // (seq_length, n_heads, head_dim)-> (n_heads, seq_length, head_dim) ->
  // (n_heads, head_dim, seq_length)

  if (ctx.model->config.debug_print) {
    auto k_emb_cpu = k_emb->to(DEV_CPU);
    std::cout << "k emb output \n"
              << print_tensor<3>(
                     k_emb_cpu->data_ptr, ctx.model->config.data_type,
                     {ctx.cur_size, n_kv_heads, head_dim}, {ctx.cur_size, 2, 4})
              << "\n";
  }
  // update kv cache
  size_t copy_size = ctx.cur_size * n_kv_heads * head_dim * sizeof(uint16_t);
  size_t copy_offset = ctx.prev_pos * n_kv_heads * head_dim * sizeof(uint16_t);

  {
    APP_PROFILE("UpdateKVCache",
                fmt::format("copy_size {} byte", copy_size).c_str(),
                ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    CHECK_ACL(
        aclrtMemcpyAsync((void *)((uint8_t *)k_cache->data_ptr + copy_offset),
                         copy_size, k_emb->data_ptr, copy_size,
                         ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.npu_stream));
    CHECK_ACL(aclrtMemcpyAsync(
        (void *)((uint8_t *)v_cache->data_ptr + copy_offset), copy_size,
        v->data_ptr, copy_size, ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.npu_stream));
  }

  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto k_cache_cpu = k_cache->to(DEV_CPU);
    auto v_cache_cpu = v_cache->to(DEV_CPU);
    std::cout << "k_cache_map output \n"
              << print_tensor<3>(
                     k_cache_cpu->data_ptr, ctx.model->config.data_type,
                     {ctx.cur_size, n_kv_heads, head_dim}, {ctx.cur_size, 2, 4})
              << "\n";
    std::cout << "v_cache_map output \n"
              << print_tensor<3>(
                     v_cache_cpu->data_ptr, ctx.model->config.data_type,
                     {ctx.cur_size, n_kv_heads, head_dim}, {ctx.cur_size, 2, 4})
              << "\n";
  }

  float qk_scale = 1 / sqrtf(static_cast<float>(head_dim));
  auto tmp_output_tensor =
      Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, dtype);

  npu_flash_attn_gqa_layer(tmp_output_tensor->data_ptr, q_emb->data_ptr,
                           k_cache->data_ptr, v_cache->data_ptr, ctx.cur_size,
                           ctx.cur_pos, ctx.prev_pos, n_heads / n_kv_heads,
                           n_kv_heads, head_dim, dtype, ctx.npu_stream);
  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto tmp_output_cpu = tmp_output_tensor->to(DEV_CPU);
    std::cout << "attn gqa output \n"
              << print_tensor<2>(tmp_output_cpu->data_ptr,
                                 ctx.model->config.data_type,
                                 {ctx.cur_size, hidden_dim}, {ctx.cur_size, 4})
              << "\n";
  }

  spdlog::debug("o_proj.Forward");
  auto output = o_proj.Forward(tmp_output_tensor, ctx);

  if (ctx.model->config.debug_print) {
    auto output_cpu = output->to(DEV_CPU);
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    std::cout << "score output \n"
              << print_tensor<2>(output_cpu->data_ptr,
                                 ctx.model->config.data_type,
                                 {ctx.cur_size, hidden_dim}, {ctx.cur_size, 4})
              << "\n";
  }

  auto output_add_input =
      Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, dtype);

  spdlog::debug("attn output + input");
  {
    APP_PROFILE("Add", fmt::format("attn output + input").c_str(),
                ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    npu_add_layer(output_add_input->data_ptr, input->data_ptr, output->data_ptr,
                  ctx.cur_size * hidden_dim, dtype, ctx.npu_stream);
  }

  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto output_add_input_cpu = output_add_input->to(DEV_CPU);
    std::cout << "z output \n"
              << print_tensor<2>(output_add_input_cpu->data_ptr,
                                 ctx.model->config.data_type,
                                 {ctx.cur_size, hidden_dim}, {ctx.cur_size, 4})
              << "\n";
  }

  spdlog::debug("post_norm.Forward");
  auto post_norm_out = post_norm.Forward(output_add_input, ctx);

  if (ctx.model->config.debug_print) {
    auto post_norm_out_cpu = post_norm_out->to(DEV_CPU);
    std::cout << "post_norm output \n"
              << print_tensor<2>(post_norm_out_cpu->data_ptr,
                                 ctx.model->config.data_type,
                                 {ctx.cur_size, hidden_dim}, {ctx.cur_size, 4})
              << "\n";
  }

  spdlog::debug("gate_proj.Forward");
  auto w1_h = gate_proj.Forward(post_norm_out, ctx);

  if (ctx.model->config.debug_print) {
    auto w1_h_cpu = w1_h->to(DEV_CPU);
    std::cout << "gate_proj output \n"
              << print_tensor<2>(w1_h_cpu->data_ptr,
                                 ctx.model->config.data_type,
                                 {ctx.cur_size, hidden_dim}, {ctx.cur_size, 4})
              << "\n";
  }

  spdlog::debug("up_proj.Forward");
  auto w3_h = up_proj.Forward(post_norm_out, ctx);

  spdlog::debug("silu(gate_proj.Forward) * up_proj.Forward");

  auto silu_size = ffn_hidden * ctx.cur_size;
  auto silu_out_mul_w3 = Tensor::MakeNPUTensor(silu_size, dtype);

  {
    APP_PROFILE(
        "SILU_MUL",
        fmt::format("silu(gate_proj.Forward) * up_proj.Forward").c_str(),
        ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    npu_silu_mul_layer(silu_out_mul_w3->data_ptr, w1_h->data_ptr,
                       w3_h->data_ptr, silu_size, dtype, ctx.npu_stream);
  }
  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto silu_out_mul_w3_cpu = silu_out_mul_w3->to(DEV_CPU);
    std::cout << "silu output \n"
              << print_tensor<2>(silu_out_mul_w3_cpu->data_ptr,
                                 ctx.model->config.data_type,
                                 {ctx.cur_size, ffn_hidden}, {ctx.cur_size, 4})
              << std::endl;
    silu_out_mul_w3_cpu->to_file("silu_output.bin");
  }

  spdlog::debug("down_proj.Forward");
  auto w2_h = down_proj.Forward(silu_out_mul_w3, ctx);

  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
    auto w2_h_cpu = w2_h->to(DEV_CPU);
    std::cout << "w2 output \n"
              << print_tensor<2>(w2_h_cpu->data_ptr,
                                 ctx.model->config.data_type,
                                 {ctx.cur_size, hidden_dim}, {ctx.cur_size, 4})
              << std::endl;
  }

  spdlog::debug("w2_h + output");
  auto ffn_output = Tensor::MakeNPUTensor(hidden_dim * ctx.cur_size, dtype);

  {
    APP_PROFILE("Add", fmt::format("w2_h + output").c_str(), ctx.npu_stream,
                &ctx.model->profiler, ctx.model->is_profiling);
    npu_add_layer(ffn_output->data_ptr, w2_h->data_ptr,
                  output_add_input->data_ptr, ctx.cur_size * hidden_dim, dtype,
                  ctx.npu_stream);
  }

  // if (ctx.model->config.debug_print) {
  {
    // need to sync here, to make sure all temp tensor are read
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
  }

  return ffn_output;
}

Qwen2TransformerLayerNPUImpl::~Qwen2TransformerLayerNPUImpl() {}

bool Qwen2TransformerLayerNPUImpl::Init(ModelBase *model, int layer_no) {
  if (!Qwen2TransformerLayerImpl::Init(model, layer_no)) {
    return false;
  }
  if (!pre_norm.Init(model,
                     fmt::format("model.layers.{}.input_layernorm.weight.bin",
                                 layer_no))) {
    return false;
  }

  if (!post_norm.Init(
          model,
          fmt::format("model.layers.{}.post_attention_layernorm.weight.bin",
                      layer_no))) {
    return false;
  }

  size_t kv_head_dim = model->n_kv_heads * model->head_dim;
  spdlog::debug("Init Qwen2 Transformer layer kv_head_dim: {}", kv_head_dim);

  if (model->config.q_type == QuantType::NoQuant) {
    auto q_name =
        fmt::format("model.layers.{}.self_attn.q_proj.weight.bin", layer_no);
    auto q_path = boost::filesystem::path(model->config.model_path) / q_name;
    auto q_bias_name =
        fmt::format("model.layers.{}.self_attn.q_proj.bias.bin", layer_no);
    auto q_bias_path =
        boost::filesystem::path(model->config.model_path) / q_bias_name;

    if (!q_proj.InitWithBias(model, q_path.string(), q_bias_path.string(),
                             hidden_dim, hidden_dim)) {
      return false;
    }
    auto k_name =
        fmt::format("model.layers.{}.self_attn.k_proj.weight.bin", layer_no);
    auto k_path = boost::filesystem::path(model->config.model_path) / k_name;
    auto k_bias_name =
        fmt::format("model.layers.{}.self_attn.k_proj.bias.bin", layer_no);
    auto k_bias_path =
        boost::filesystem::path(model->config.model_path) / k_bias_name;
    if (!k_proj.InitWithBias(model, k_path.string(), k_bias_path.string(),
                             kv_head_dim, hidden_dim)) {
      return false;
    }

    auto v_name =
        fmt::format("model.layers.{}.self_attn.v_proj.weight.bin", layer_no);
    auto v_path = boost::filesystem::path(model->config.model_path) / v_name;
    auto v_bias_name =
        fmt::format("model.layers.{}.self_attn.v_proj.bias.bin", layer_no);
    auto v_bias_path =
        boost::filesystem::path(model->config.model_path) / v_bias_name;
    if (!v_proj.InitWithBias(model, v_path.string(), v_bias_path.string(),
                             kv_head_dim, hidden_dim)) {
      return false;
    }

    auto o_name =
        fmt::format("model.layers.{}.self_attn.o_proj.weight.bin", layer_no);
    auto o_path = boost::filesystem::path(model->config.model_path) / o_name;
    if (!o_proj.Init(model, o_path.string(), hidden_dim, hidden_dim)) {
      return false;
    }

    auto gate_proj_name =
        fmt::format("model.layers.{}.mlp.gate_proj.weight.bin", layer_no);
    auto gate_proj_path =
        boost::filesystem::path(model->config.model_path) / gate_proj_name;
    if (!gate_proj.Init(model, gate_proj_path.string(), ffn_hidden,
                        hidden_dim)) {
      return false;
    }

    auto down_proj_name =
        fmt::format("model.layers.{}.mlp.down_proj.weight.bin", layer_no);
    auto down_proj_path =
        boost::filesystem::path(model->config.model_path) / down_proj_name;
    if (!down_proj.Init(model, down_proj_path.string(), hidden_dim,
                        ffn_hidden)) {
      return false;
    }

    auto up_proj_name =
        fmt::format("model.layers.{}.mlp.up_proj.weight.bin", layer_no);
    auto up_proj_path =
        boost::filesystem::path(model->config.model_path) / up_proj_name;
    if (!up_proj.Init(model, up_proj_path.string(), ffn_hidden, hidden_dim)) {
      return false;
    }
  } else if (model->config.q_type == QuantType::AWQ_4B) {
#define INIT_AWQ_MM(layer, x, n, k)                                            \
  auto x##_weight_name = fmt::format(                                          \
      "model.layers.{}." #layer "." #x "_proj.qweight.bin", layer_no);         \
  auto x##_zero_name = fmt::format(                                            \
      "model.layers.{}." #layer "." #x "_proj.qzeros.bin", layer_no);          \
  auto x##_scale_name = fmt::format(                                           \
      "model.layers.{}." #layer "." #x "_proj.scales.bin", layer_no);          \
  auto x##_weight_path =                                                       \
      boost::filesystem::path(model->config.model_path) / x##_weight_name;     \
  auto x##_zero_path =                                                         \
      boost::filesystem::path(model->config.model_path) / x##_zero_name;       \
  auto x##_scale_path =                                                        \
      boost::filesystem::path(model->config.model_path) / x##_scale_name;      \
  if (!x##_proj.InitAWQ(model, x##_weight_path.string(),                       \
                        x##_zero_path.string(), x##_scale_path.string(), n, k, \
                        model->config.q_type)) {                               \
    return false;                                                              \
  }

    INIT_AWQ_MM(self_attn, q, hidden_dim, hidden_dim);
    q_proj.AddBias(
        (boost::filesystem::path(model->config.model_path) /
         fmt::format("model.layers.{}.self_attn.q_proj.bias.bin", layer_no))
            .string());
    INIT_AWQ_MM(self_attn, k, kv_head_dim, hidden_dim);
    k_proj.AddBias(
        (boost::filesystem::path(model->config.model_path) /
         fmt::format("model.layers.{}.self_attn.k_proj.bias.bin", layer_no))
            .string());
    INIT_AWQ_MM(self_attn, v, kv_head_dim, hidden_dim);
    v_proj.AddBias(
        (boost::filesystem::path(model->config.model_path) /
         fmt::format("model.layers.{}.self_attn.v_proj.bias.bin", layer_no))
            .string());
    INIT_AWQ_MM(self_attn, o, hidden_dim, hidden_dim);

    INIT_AWQ_MM(mlp, gate, ffn_hidden, hidden_dim);
    INIT_AWQ_MM(mlp, down, hidden_dim, ffn_hidden);
    INIT_AWQ_MM(mlp, up, ffn_hidden, hidden_dim);

#undef INIT_AWQ_MM
  }

  auto inv_freq_name = std::string("model.layers.") + std::to_string(layer_no) +
                       ".self_attn.rotary_emb.inv_freq.bin";
  auto inv_freq_path =
      boost::filesystem::path(model->config.model_path) / inv_freq_name;

  if (!rope_emb.Init(model, inv_freq_path.string())) {
    return false;
  }

  if (!softmax.Init(model)) {
    return false;
  }

  spdlog::debug("ffn_hidden dim: {}", ffn_hidden);

  k_cache = Tensor::MakeNPUTensor(kv_head_dim * max_seq_len, DT_FLOAT16);
  v_cache = Tensor::MakeNPUTensor(kv_head_dim * max_seq_len, DT_FLOAT16);

  return true;
}

void Qwen2TransformerLayerNPUImpl::UnInit() {
  Qwen2TransformerLayerImpl::UnInit();
}

MatmulLayerNPUImpl::~MatmulLayerNPUImpl() {}

std::shared_ptr<Tensor>
MatmulLayerNPUImpl::Forward(std::shared_ptr<Tensor> input, InferenceCtx &ctx) {
  auto output = Tensor::MakeNPUTensor(n * ctx.cur_size, DT_FLOAT16);

  uint16_t *input_ptr = static_cast<uint16_t *>(input->data_ptr);
  uint16_t *output_ptr = static_cast<uint16_t *>(output->data_ptr);

  spdlog::debug("MatmulLayerNPUImpl::Forward m {} n {} k {} dtype {}",
                ctx.cur_size, n, k, dtype);

  if (qtype == QuantType::NoQuant) {
    APP_PROFILE("MatmulLayer",
                fmt::format("m {} n {} k {}", ctx.cur_size, n, k).c_str(),
                ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    if (bias != nullptr) {
      npu_matmul_bias_nz_layer((void *)output_ptr, (void *)input_ptr,
                               (void *)weight, (void *)bias, ctx.cur_size, n, k,
                               dtype, ctx.npu_stream);
    } else {
      npu_matmul_nz_layer((void *)output_ptr, (void *)input_ptr, (void *)weight,
                          ctx.cur_size, n, k, dtype, ctx.npu_stream);
    }

  } else if (qtype == QuantType::AWQ_4B) {
    APP_PROFILE("MatmulLayerAWQ4Bit",
                fmt::format("m {} n {} k {}", ctx.cur_size, n, k).c_str(),
                ctx.npu_stream, &ctx.model->profiler, ctx.model->is_profiling);
    if (bias != nullptr) {
      npu_matmul_nz_awq_4bit_bias_layer(
          (void *)output_ptr, (void *)input_ptr, (void *)weight, (void *)qzeros,
          (void *)qscales, (void *)bias, ctx.cur_size, n, k, dtype,
          ctx.npu_stream);

    } else {
      npu_matmul_nz_awq_4bit_layer(
          (void *)output_ptr, (void *)input_ptr, (void *)weight, (void *)qzeros,
          (void *)qscales, ctx.cur_size, n, k, dtype, ctx.npu_stream);
    }
  }
  if (ctx.model->config.debug_print) {
    CHECK_ACL(aclrtSynchronizeStream(ctx.npu_stream));
  }

  return output;
}

bool MatmulLayerNPUImpl::Init(ModelBase *model, const std::string &weight_path,
                              size_t n, size_t k) {
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

  dtype = model->config.data_type;

  void *tmp_dev_weight;
  CHECK_ACL(aclrtMalloc((void **)&tmp_dev_weight, weight_size,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK_ACL(aclrtMemcpy(tmp_dev_weight, weight_size, weight, weight_size,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  free(weight);
  weight = nullptr;
  CHECK_ACL(
      aclrtMalloc((void **)&weight, weight_size, ACL_MEM_MALLOC_HUGE_FIRST));

  if (weight == nullptr) {
    spdlog::critical("oom!");
    return false;
  }

  npu_mamtul_weight_transpose_layer(weight, tmp_dev_weight, n, k, DT_FLOAT16,
                                    model->model_stream);
  CHECK_ACL(aclrtSynchronizeStream(model->model_stream));
  CHECK_ACL(aclrtFree(tmp_dev_weight));
#if 0
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

bool MatmulLayerNPUImpl::InitWithBias(ModelBase *model,
                                      const std::string &weight_path,
                                      const std::string &bias_path, size_t n,
                                      size_t k) {
  if (!MatmulLayerNPUImpl::Init(model, weight_path, n, k)) {
    return false;
  }

  if (!MatmulLayerImpl::InitWithBias(model, weight_path, bias_path, n, k)) {
    return false;
  }

  dtype = model->config.data_type;
  uint8_t *dev_bias = nullptr;
  CHECK_ACL(
      aclrtMalloc((void **)&dev_bias, bias_size, ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK_ACL(aclrtMemcpy(dev_bias, bias_size, bias, bias_size,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  free(bias);
  bias = dev_bias;

  return true;
}

bool MatmulLayerNPUImpl::InitAWQ(ModelBase *model,
                                 const std::string &weight_path,
                                 const std::string &zero_path,
                                 const std::string &scale_path, size_t n,
                                 size_t k, QuantType quant_type) {
  if (!MatmulLayerImpl::InitAWQ(model, weight_path, zero_path, scale_path, n, k,
                                quant_type)) {
    return false;
  }
  dtype = model->config.data_type;

  void *tmp_dev_weight;
  void *tmp_dev_zero;
  void *tmp_dev_scale;
  CHECK_ACL(aclrtMalloc((void **)&tmp_dev_weight, weight_size,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&tmp_dev_zero, zero_size,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc((void **)&tmp_dev_scale, scale_size,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(tmp_dev_weight, weight_size, weight, weight_size,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(tmp_dev_zero, zero_size, qzeros, zero_size,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK_ACL(aclrtMemcpy(tmp_dev_scale, scale_size, qscales, scale_size,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  free(weight);
  free(qzeros);
  free(qscales);
  weight = (uint8_t *)tmp_dev_weight;
  qzeros = (uint8_t *)tmp_dev_zero;
  qscales = (uint8_t *)tmp_dev_scale;
  return true;
}

bool MatmulLayerNPUImpl::AddBias(const std::string &bias_path) {
  bias_size = n * sizeof(float);

  float *temp_host_bias = new float[n];

  if (!LoadBinaryFile(bias_path.c_str(), temp_host_bias, bias_size)) {
    delete[] temp_host_bias;
    return false;
  }

  CHECK_ACL(aclrtMalloc((void **)&bias, bias_size, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemcpy(bias, bias_size, temp_host_bias, bias_size,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  delete[] temp_host_bias;
  return true;
}

void MatmulLayerNPUImpl::UnInit() { MatmulLayerImpl::UnInit(); }
