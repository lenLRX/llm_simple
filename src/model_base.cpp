#include <algorithm>
#include <boost/filesystem.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>

#include "device.hpp"
#include "llama2_layer_cpu.hpp"
#include "llama2_layer_npu.hpp"
#include "llama2_model.hpp"
#include "model_base.hpp"
#include "qwen2_model.hpp"
#include "util.h"

using namespace std::literals;

InferenceCtx::InferenceCtx(ModelBase *model, size_t cur_pos, size_t prev_pos)
    : model(model), cur_pos(cur_pos), prev_pos(prev_pos) {
  cur_size = cur_pos - prev_pos;
}

bool ModelBase::InitFreqCIS(const float theta, const DataType dt) {
  int head_dim = hidden_dim / n_heads;
  int freq_len = head_dim / 2;
  float *freq = new float[freq_len];

  for (int i = 0; i < freq_len; ++i) {
    freq[i] =
        1.0f /
        (pow(theta, static_cast<double>(i * 2) / static_cast<double>(head_dim)));
  }

  float *t = new float[config.max_seq_len];
  for (int i = 0; i < config.max_seq_len; ++i) {
    t[i] = static_cast<float>(i);
  }

  float *freq_outer = new float[freq_len * config.max_seq_len];

  // max_seq_len row, freq_len column
  for (int i = 0; i < config.max_seq_len; ++i) {
    for (int j = 0; j < freq_len; ++j) {
      freq_outer[i * freq_len + j] = t[i] * freq[j];
    }
  }

  freq_cis = new float[freq_len * config.max_seq_len * 2];

  for (int i = 0; i < config.max_seq_len * freq_len; ++i) {
    freq_cis[i * 2] = std::cos(freq_outer[i]);
    freq_cis[i * 2 + 1] = std::sin(freq_outer[i]);
  }

  if (dt == DT_BFLOAT16) {
    spdlog::debug("freq_cis convert fp32 to bf16");
    uint32_t* freq_cis_binary = reinterpret_cast<uint32_t*>(freq_cis);
    for (int i = 0; i < freq_len * config.max_seq_len * 2; ++i) {
        freq_cis_binary[i] = fp32_to_bfloat16(freq_cis_binary[i]);
        freq_cis_binary[i] = freq_cis_binary[i] << 16; 
    }
  }
  // TODO: float16

  if (config.device_type == DEV_NPU) {
    float *old_freq_cis = freq_cis;
    size_t freq_cis_size = freq_len * config.max_seq_len * 2 * sizeof(float);
    CHECK_ACL(aclrtMalloc((void **)&freq_cis, freq_cis_size,
                          ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(freq_cis, freq_cis_size, old_freq_cis, freq_cis_size,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    delete[] old_freq_cis;
  }

  delete[] freq;
  delete[] t;
  delete[] freq_outer;
  return true;
}

std::shared_ptr<Tensor> EmbeddingLayer::Forward(std::shared_ptr<Tensor> input,
                                                InferenceCtx &ctx) {
  return impl->Forward(input, ctx);
}

bool EmbeddingLayer::Init(ModelBase *model, const std::string &weight_path) {
  switch (model->config.device_type) {
  case DEV_NPU:
    impl = new EmbeddingLayerNPUImpl();
    break;
  case DEV_CPU:
    impl = new EmbeddingLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->Init(model, weight_path);
}

void EmbeddingLayer::UnInit() {}

EmbeddingLayerImpl::~EmbeddingLayerImpl() {}

bool EmbeddingLayerImpl::Init(ModelBase *model,
                              const std::string &weight_path) {
  nwords_size = model->n_words;
  hidden_dim = model->hidden_dim;
  weight_size = sizeof(uint16_t) * nwords_size * model->hidden_dim;

  embedding_weight = (uint8_t *)malloc(weight_size);
  if (embedding_weight == nullptr) {
    spdlog::critical("oom!");
    return false;
  }

  if (!LoadBinaryFile(weight_path.c_str(), embedding_weight, weight_size)) {
    return false;
  }
  return true;
}

void EmbeddingLayerImpl::UnInit() { free(embedding_weight); }

EmbeddingLayer::~EmbeddingLayer() { delete impl; }

RMSNormLayerImpl::~RMSNormLayerImpl() {}

bool RMSNormLayerImpl::Init(ModelBase *model, int layer_no, bool pre_norm,
                            bool last_norm) {
  std::string weight_path;
  if (last_norm) {
    // model.norm.weight.bin
    weight_path = "model.norm.weight.bin";
  } else {
    std::stringstream ss;
    if (pre_norm) {
      ss << "model.layers." << layer_no << ".input_layernorm.weight.bin";
    } else {
      ss << "model.layers." << layer_no
         << ".post_attention_layernorm.weight.bin";
    }

    weight_path = ss.str();
  }

  return Init(model, weight_path);
}

bool RMSNormLayerImpl::Init(ModelBase *model, const std::string &weight_path) {
  hidden_dim = model->hidden_dim;
  eps = model->norm_eps;

  weight_size = sizeof(uint16_t) * model->hidden_dim;
  norm_weight = (uint8_t *)malloc(weight_size);
  if (norm_weight == nullptr) {
    spdlog::critical("oom!");
    return false;
  }

  auto full_weight_path =
      boost::filesystem::path(model->config.model_path) / weight_path;

  if (!LoadBinaryFile(full_weight_path.c_str(), norm_weight, weight_size)) {
    return false;
  }
  return true;
}

void RMSNormLayerImpl::UnInit() { free(norm_weight); }

RMSNormLayer::~RMSNormLayer() { delete impl; }

std::shared_ptr<Tensor> RMSNormLayer::Forward(std::shared_ptr<Tensor> input,
                                              InferenceCtx &ctx) {
  return impl->Forward(input, ctx);
}

bool RMSNormLayer::Init(ModelBase *model, int layer_no, bool pre_norm,
                        bool last_norm) {
  switch (model->config.device_type) {
  case DEV_NPU:
    impl = new RMSNormLayerNPUImpl();
    break;
  case DEV_CPU:
    impl = new RMSNormLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->Init(model, layer_no, pre_norm, last_norm);
}

bool RMSNormLayer::Init(ModelBase *model, const std::string &weight_path) {
  switch (model->config.device_type) {
  case DEV_NPU:
    impl = new RMSNormLayerNPUImpl();
    break;
  case DEV_CPU:
    impl = new RMSNormLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->Init(model, weight_path);
}

void RMSNormLayer::UnInit() {}

bool Llamma2TransformerLayerImpl::Init(ModelBase *model, int layer_no) {
  Llama2Model *llama2_model = dynamic_cast<Llama2Model *>(model);
  if (llama2_model == nullptr) {
    return false;
  }
  hidden_dim = model->hidden_dim;
  head_dim = model->hidden_dim / model->n_heads;
  n_heads = model->n_heads;
  ffn_hidden = 4 * ((hidden_dim * 2) / 3);
  ffn_hidden = (ffn_hidden + llama2_model->multiple_of - 1) /
               llama2_model->multiple_of * llama2_model->multiple_of;
  max_seq_len = model->config.max_seq_len;
  return true;
}

void Llamma2TransformerLayerImpl::UnInit() {}

Llamma2TransformerLayer::~Llamma2TransformerLayer() { delete impl; }

std::shared_ptr<Tensor>
Llamma2TransformerLayer::Forward(std::shared_ptr<Tensor> input,
                                 std::shared_ptr<Tensor> mask,
                                 InferenceCtx &ctx) {
  return impl->Forward(input, mask, ctx);
}

bool Llamma2TransformerLayer::Init(ModelBase *model, int layer_no) {
  switch (model->config.device_type) {
  case DEV_NPU:
    impl = new Llamma2TransformerLayerNPUImpl();
    break;
  case DEV_CPU:
    impl = new Llamma2TransformerLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }

  return impl->Init(model, layer_no);
}

void Llamma2TransformerLayer::UnInit() {}

bool Qwen2TransformerLayerImpl::Init(ModelBase *model, int layer_no) {
  Qwen2Model *qwen2_model = dynamic_cast<Qwen2Model *>(model);
  if (qwen2_model == nullptr) {
    return false;
  }
  hidden_dim = model->hidden_dim;
  head_dim = model->hidden_dim / model->n_heads;
  n_heads = model->n_heads;
  n_kv_heads = model->n_kv_heads;
  ffn_hidden = qwen2_model->intermediate_size;
  max_seq_len = model->config.max_seq_len;
  dtype = model->config.data_type;
  return true;
}

void Qwen2TransformerLayerImpl::UnInit() {}

Qwen2TransformerLayer::~Qwen2TransformerLayer() { delete impl; }

std::shared_ptr<Tensor>
Qwen2TransformerLayer::Forward(std::shared_ptr<Tensor> input,
                               std::shared_ptr<Tensor> mask,
                               InferenceCtx &ctx) {
  return impl->Forward(input, mask, ctx);
}

bool Qwen2TransformerLayer::Init(ModelBase *model, int layer_no) {
  switch (model->config.device_type) {
  case DEV_NPU:
    impl = new Qwen2TransformerLayerNPUImpl();
    break;
  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }

  return impl->Init(model, layer_no);
}

void Qwen2TransformerLayer::UnInit() {}

ArgMaxLayerImpl::~ArgMaxLayerImpl() {}

bool ArgMaxLayerImpl::Init(ModelBase *model) {
  // TODO: change name
  hidden_dim = model->n_words;
  dt = model->config.data_type;
  return true;
}

void ArgMaxLayerImpl::UnInit() {}

ArgMaxLayer::~ArgMaxLayer() {}

std::shared_ptr<Tensor> ArgMaxLayer::Forward(std::shared_ptr<Tensor> input,
                                             InferenceCtx &ctx) {
  return impl->Forward(input, ctx);
}

bool ArgMaxLayer::Init(ModelBase *model) {
  switch (model->config.device_type) {
  case DEV_NPU:
  case DEV_CPU:
    impl = new ArgMaxLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->Init(model);
}

void ArgMaxLayer::UnInit() {}

SampleTopPLayerImpl::~SampleTopPLayerImpl() {}

bool SampleTopPLayerImpl::Init(ModelBase *model) {
  temperature = model->config.temperature;
  top_p = model->config.top_p;
  vocab_size = model->n_words;
  return true;
}

void SampleTopPLayerImpl::UnInit() {}

SampleTopPLayer::~SampleTopPLayer() {}

int SampleTopPLayer::Forward(std::shared_ptr<Tensor> input, InferenceCtx &ctx) {
  return impl->Forward(input, ctx);
}

bool SampleTopPLayer::Init(ModelBase *model) {
  switch (model->config.device_type) {
  case DEV_NPU:
  case DEV_CPU:
    impl = new SampleTopPLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->Init(model);
}

void SampleTopPLayer::UnInit() {}

SoftmaxLayerImpl::~SoftmaxLayerImpl() {}

bool SoftmaxLayerImpl::Init(ModelBase *model) {
  hidden_dim = model->hidden_dim;
  n_heads = model->n_heads;
  eps = 1E-5;
  return true;
}

void SoftmaxLayerImpl::UnInit() {}

SoftmaxLayer::~SoftmaxLayer() {}

std::shared_ptr<Tensor> SoftmaxLayer::Forward(std::shared_ptr<Tensor> input,
                                              InferenceCtx &ctx) {
  return impl->Forward(input, ctx);
}

bool SoftmaxLayer::Init(ModelBase *model) {
  switch (model->config.device_type) {
  case DEV_NPU:
    impl = new SoftmaxLayerNPUImpl();
    break;
  case DEV_CPU:
    impl = new SoftmaxLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->Init(model);
}

void SoftmaxLayer::UnInit() {}

CausualMaskLayerImpl::~CausualMaskLayerImpl() {}

bool CausualMaskLayerImpl::Init(ModelBase *model) { return true; }

void CausualMaskLayerImpl::UnInit() {}

CausualMaskLayer::~CausualMaskLayer() {}

std::shared_ptr<Tensor> CausualMaskLayer::Forward(InferenceCtx &ctx) {
  return impl->Forward(ctx);
}

bool CausualMaskLayer::Init(ModelBase *model) {
  switch (model->config.device_type) {
  case DEV_NPU:
  case DEV_CPU:
    impl = new CausualMaskLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->Init(model);
}

void CausualMaskLayer::UnInit() {}

MatmulLayerImpl::~MatmulLayerImpl() {}

bool MatmulLayerImpl::Init(ModelBase *model, const std::string &weight_path,
                           size_t n, size_t k) {
  this->n = n;
  this->k = k;
  weight_size = sizeof(uint16_t) * n * k;

  weight = (uint8_t *)malloc(weight_size);
  if (weight == nullptr) {
    spdlog::critical("oom!");
    return false;
  }

  if (!LoadBinaryFile(weight_path.c_str(), weight, weight_size)) {
    return false;
  }
  return true;
}

bool MatmulLayerImpl::InitWithBias(ModelBase *model,
                                   const std::string &weight_path,
                                   const std::string &bias_path, size_t n,
                                   size_t k) {
  this->n = n;
  this->k = k;
  bias_size = sizeof(float) * n;

  bias = (uint8_t *)malloc(bias_size);
  if (weight == nullptr) {
    spdlog::critical("oom!");
    return false;
  }

  if (!LoadBinaryFile(bias_path.c_str(), bias, bias_size)) {
    return false;
  }

  return true;
}

bool MatmulLayerImpl::InitAWQ(ModelBase *model, const std::string &weight_path,
                              const std::string &zero_path,
                              const std::string &scale_path, size_t n, size_t k,
                              QuantType quant_type) {
  qtype = quant_type;
  constexpr int group_size = 128;
  this->n = n;
  this->k = k;
  weight_size = n * k / 2;
  zero_size = n * k / group_size * sizeof(uint16_t);
  scale_size = n * k / group_size * sizeof(uint16_t);

  weight = (uint8_t *)malloc(weight_size);
  if (weight == nullptr) {
    spdlog::critical("oom!");
    return false;
  }

  if (!LoadBinaryFile(weight_path.c_str(), weight, weight_size)) {
    return false;
  }

  qzeros = (uint8_t *)malloc(zero_size);
  if (!LoadBinaryFile(zero_path.c_str(), qzeros, zero_size)) {
    return false;
  }

  qscales = (uint8_t *)malloc(scale_size);
  if (!LoadBinaryFile(scale_path.c_str(), qscales, scale_size)) {
    return false;
  }
  return true;
}

bool MatmulLayerImpl::AddBias(const std::string &bias_path) {
  return false;
}

void MatmulLayerImpl::UnInit() {}

MatmulLayer::~MatmulLayer() {}

std::shared_ptr<Tensor> MatmulLayer::Forward(std::shared_ptr<Tensor> input,
                                             InferenceCtx &ctx) {
  return impl->Forward(input, ctx);
}

bool MatmulLayer::Init(ModelBase *model, const std::string &weight_path,
                       size_t n, size_t k) {
  switch (model->config.device_type) {
  case DEV_NPU:
    impl = new MatmulLayerNPUImpl();
    break;
  case DEV_CPU:
    impl = new MatmulLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->Init(model, weight_path, n, k);
}

bool MatmulLayer::InitWithBias(ModelBase *model, const std::string &weight_path,
                               const std::string &bias, size_t n, size_t k) {
  switch (model->config.device_type) {
  case DEV_NPU:
    impl = new MatmulLayerNPUImpl();
    break;
  case DEV_CPU:
    impl = new MatmulLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->InitWithBias(model, weight_path, bias, n, k);
}

bool MatmulLayer::InitAWQ(ModelBase *model, const std::string &weight_path,
                          const std::string &zero_path,
                          const std::string &scale_path, size_t n, size_t k,
                          QuantType quant_type) {
  switch (model->config.device_type) {
  case DEV_NPU:
    impl = new MatmulLayerNPUImpl();
    break;
  case DEV_CPU:
    impl = new MatmulLayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->InitAWQ(model, weight_path, zero_path, scale_path, n, k,
                       quant_type);
}

bool MatmulLayer::AddBias(const std::string &bias_path) {
  return impl->AddBias(bias_path);
}

void MatmulLayer::UnInit() {}

RoPELayerImpl::~RoPELayerImpl() {}

bool RoPELayerImpl::Init(ModelBase *model, const std::string &weight_path) {
  hidden_dim = model->hidden_dim;
  head_dim = model->hidden_dim / model->n_heads;
  n_heads = model->n_heads;
  rope_dim = model->config.max_seq_len * model->hidden_dim;
  weight_size = sizeof(float) * rope_dim;
  freqs_cis = model->freq_cis;
  rope_is_neox_style = model->config.rope_is_neox_style;

  return true;
}

void RoPELayerImpl::UnInit() {}

RoPELayer::~RoPELayer() {}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RoPELayer::Forward(std::shared_ptr<Tensor> input_q,
                   std::shared_ptr<Tensor> input_k, InferenceCtx &ctx) {
  return impl->Forward(input_q, input_k, ctx);
}

bool RoPELayer::Init(ModelBase *model, const std::string &weight_path) {
  switch (model->config.device_type) {
  case DEV_NPU:
    impl = new RoPELayerNPUImpl();
    break;
  case DEV_CPU:
    impl = new RoPELayerCPUImpl();
    break;

  default:
    spdlog::critical("invalid device type");
    return false;
    break;
  }
  return impl->Init(model, weight_path);
}

void RoPELayer::UnInit() {}
