#pragma once

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

#include "acl_util.hpp"

#include "defs.hpp"
#include "device.hpp"
#include "profiling.hpp"
#include "tokenizer.hpp"

enum class QuantType { NoQuant, AWQ_4B };

class ModelConfig {
public:
  std::string tok_path;
  std::string model_path;
  std::string config_path;
  nlohmann::json config;
  std::string model_type;
  DeviceType device_type;
  DataType data_type;
  int max_seq_len;
  int max_gen_len;
  float norm_eps;
  float temperature{0.0f};
  float top_p{0.0f};
  QuantType q_type{QuantType::NoQuant};
  int quant_group_size{-1};
  bool rope_is_neox_style;
  bool debug_print{false};
};

class ModelBase;

class InferenceCtx {
public:
  InferenceCtx(ModelBase *model, size_t cur_pos, size_t prev_pos);
  ModelBase *model;
  size_t cur_pos;
  size_t prev_pos;
  size_t cur_size;
  aclrtStream npu_stream{nullptr};
};

class EmbeddingLayerImpl {
public:
  virtual ~EmbeddingLayerImpl();
  // N -> [N, hidden_dim]
  virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                          InferenceCtx &ctx) = 0;
  virtual bool Init(ModelBase *model, const std::string& weight_path);

  virtual void UnInit();

  size_t nwords_size;
  size_t weight_size;
  size_t hidden_dim;
  uint8_t *embedding_weight{nullptr};
};

class EmbeddingLayer {
public:
  ~EmbeddingLayer();
  // N -> [N, hidden_dim]
  std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                  InferenceCtx &ctx);
  bool Init(ModelBase *model, const std::string& weight_path);

  void UnInit();

  EmbeddingLayerImpl *impl{nullptr};
};

class RMSNormLayerImpl {
public:
  virtual ~RMSNormLayerImpl();
  virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                          InferenceCtx &ctx) = 0;
  virtual bool Init(ModelBase *model, int layer_no, bool pre_norm,
                    bool last_norm);
  virtual bool Init(ModelBase *model, const std::string& weight_path);
  virtual void UnInit();

  size_t hidden_dim;
  size_t weight_size;
  float eps;
  uint8_t *norm_weight{nullptr};
};

class RMSNormLayer {
public:
  ~RMSNormLayer();
  virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                          InferenceCtx &ctx);
  bool Init(ModelBase *model, int layer_no, bool pre_norm, bool last_norm);
  bool Init(ModelBase *model, const std::string& weight_path);
  void UnInit();
  RMSNormLayerImpl *impl{nullptr};
};

class RoPELayerImpl {
public:
  virtual ~RoPELayerImpl();
  virtual std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
  Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k,
          InferenceCtx &ctx) = 0;
  virtual bool Init(ModelBase *model, const std::string &weight_path);
  virtual void UnInit();

  size_t hidden_dim;
  size_t n_heads;
  size_t head_dim;
  size_t rope_dim;
  size_t weight_size;
  bool rope_is_neox_style;
  float *freqs_cis{nullptr};
};

class RoPELayer {
public:
  ~RoPELayer();
  virtual std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
  Forward(std::shared_ptr<Tensor> input_q, std::shared_ptr<Tensor> input_k,
          InferenceCtx &ctx);
  bool Init(ModelBase *model, const std::string &weight_path);
  void UnInit();
  RoPELayerImpl *impl{nullptr};
};

class ArgMaxLayerImpl {
public:
  virtual ~ArgMaxLayerImpl();
  virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                          InferenceCtx &ctx) = 0;
  virtual bool Init(ModelBase *model);
  virtual void UnInit();

  size_t hidden_dim;
  DataType dt;
};

class ArgMaxLayer {
public:
  ~ArgMaxLayer();
  virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                          InferenceCtx &ctx);
  bool Init(ModelBase *model);
  void UnInit();
  ArgMaxLayerImpl *impl{nullptr};
};

class SampleTopPLayerImpl {
public:
  virtual ~SampleTopPLayerImpl();
  virtual int Forward(std::shared_ptr<Tensor> input, InferenceCtx &ctx) = 0;
  virtual bool Init(ModelBase *model);
  virtual void UnInit();

  float temperature;
  float top_p;
  size_t vocab_size;
};

class SampleTopPLayer {
public:
  ~SampleTopPLayer();
  virtual int Forward(std::shared_ptr<Tensor> input, InferenceCtx &ctx);
  bool Init(ModelBase *model);
  void UnInit();
  SampleTopPLayerImpl *impl{nullptr};
};

class SoftmaxLayerImpl {
public:
  virtual ~SoftmaxLayerImpl();
  virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                          InferenceCtx &ctx) = 0;
  virtual bool Init(ModelBase *model);
  virtual void UnInit();

  size_t hidden_dim;
  size_t n_heads;
  float eps;
};

class SoftmaxLayer {
public:
  ~SoftmaxLayer();
  virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                          InferenceCtx &ctx);
  bool Init(ModelBase *model);
  void UnInit();
  SoftmaxLayerImpl *impl{nullptr};
};

class CausualMaskLayerImpl {
public:
  virtual ~CausualMaskLayerImpl();
  virtual std::shared_ptr<Tensor> Forward(InferenceCtx &ctx) = 0;
  virtual bool Init(ModelBase *model);
  virtual void UnInit();
};

class CausualMaskLayer {
public:
  ~CausualMaskLayer();
  virtual std::shared_ptr<Tensor> Forward(InferenceCtx &ctx);
  bool Init(ModelBase *model);
  void UnInit();
  CausualMaskLayerImpl *impl{nullptr};
};

class MatmulLayerImpl {
public:
  virtual ~MatmulLayerImpl();
  virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                          InferenceCtx &ctx) = 0;
  virtual bool Init(ModelBase *model, const std::string &weight_path, size_t n,
                    size_t k);
  virtual bool InitWithBias(ModelBase *model, const std::string &weight_path,
                            const std::string &bias, size_t n, size_t k);
  virtual bool InitAWQ(ModelBase *model, const std::string &weight_path,
                       const std::string &zero_path,
                       const std::string &scale_path, size_t n, size_t k,
                       QuantType quant_type);
  virtual void UnInit();

  size_t n;
  size_t k;
  size_t weight_size;
  size_t bias_size;
  size_t zero_size;
  size_t scale_size;
  uint8_t *weight{nullptr};
  uint8_t *qzeros{nullptr};
  uint8_t *qscales{nullptr};
  uint8_t *bias{nullptr};

  DataType dtype;
  QuantType qtype{QuantType::NoQuant};
};

class MatmulLayer {
public:
  ~MatmulLayer();
  std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                  InferenceCtx &ctx);
  bool Init(ModelBase *model, const std::string &weight_path, size_t n,
            size_t k);
  bool InitWithBias(ModelBase *model, const std::string &weight_path,
                    const std::string &bias, size_t n, size_t k);
  bool InitAWQ(ModelBase *model, const std::string &weight_path,
               const std::string &zero_path, const std::string &scale_path,
               size_t n, size_t k, QuantType quant_type);
  void UnInit();
  MatmulLayerImpl *impl{nullptr};
};

class Llamma2TransformerLayerImpl {
public:
  virtual ~Llamma2TransformerLayerImpl() = default;
  virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                          std::shared_ptr<Tensor> mask,
                                          InferenceCtx &ctx) = 0;
  virtual bool Init(ModelBase *model, int layer_no);
  virtual void UnInit();

  size_t ffn_hidden;
  size_t hidden_dim;
  size_t head_dim;
  size_t n_heads;
  size_t max_seq_len;
};

class Llamma2TransformerLayer {
public:
  ~Llamma2TransformerLayer();
  std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                  std::shared_ptr<Tensor> mask,
                                  InferenceCtx &ctx);
  bool Init(ModelBase *model, int layer_no);
  void UnInit();

  Llamma2TransformerLayerImpl *impl{nullptr};
};

class Qwen2TransformerLayerImpl {
public:
  virtual ~Qwen2TransformerLayerImpl() = default;
  virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                          std::shared_ptr<Tensor> mask,
                                          InferenceCtx &ctx) = 0;
  virtual bool Init(ModelBase *model, int layer_no);
  virtual void UnInit();

  DataType dtype;
  size_t ffn_hidden;
  size_t hidden_dim;
  size_t head_dim;
  size_t n_heads;
  size_t n_kv_heads;
  size_t max_seq_len;
};

class Qwen2TransformerLayer {
public:
  ~Qwen2TransformerLayer();
  std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input,
                                  std::shared_ptr<Tensor> mask,
                                  InferenceCtx &ctx);
  bool Init(ModelBase *model, int layer_no);
  void UnInit();

  Qwen2TransformerLayerImpl *impl{nullptr};
};

class ModelBase {
public:
  ModelBase() = default;
  virtual ~ModelBase() = default;

  virtual bool Init() = 0;

  bool InitFreqCIS(const float theta=10000.0f, const DataType dt=DT_FLOAT32);

  virtual void Chat(const std::string &input_seq, const std::string &reverse_prompt) = 0;
  virtual void TextCompletion(const std::string &input_seq) = 0;
  virtual void Benchmark(int input_seq_len, int output_seq_len) = 0;


  ModelConfig config;
  int hidden_dim;
  int head_dim;
  int n_heads;
  int n_kv_heads;
  int n_layers;
  float norm_eps;

  int n_words;
  int pad_id;

  aclrtStream model_stream;

  float *freq_cis{nullptr};
  bool is_profiling{false};
  AppProfiler profiler;
};
