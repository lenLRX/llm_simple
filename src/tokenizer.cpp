#include <boost/filesystem.hpp>
#include <fstream>
#include <spdlog/spdlog.h>

#include "base64.h"
#include "tokenizer.hpp"

bool Tokenizer::Init(const std::string &token_model_path) {
  const auto status = processor.Load(token_model_path);
  if (!status.ok()) {
    spdlog::critical("failed to init tokenizer from path {}", token_model_path);
    return false;
  }

  n_words = processor.GetPieceSize();
  bos_id = processor.bos_id();
  eos_id = processor.eos_id();
  pad_id = processor.pad_id();

  spdlog::info("initialized tokenizer from {}, nwords: {}, bos_id: {}, eos_id: "
               "{}, pad_id: {}",
               token_model_path, n_words, bos_id, eos_id, pad_id);

  return true;
}

std::vector<int32_t> Tokenizer::Encode(const std::string &text, bool bos,
                                       bool eos) {
  std::vector<int32_t> result;
  processor.Encode(text, &result);
  if (bos) {
    result.insert(result.begin(), bos_id);
  }
  if (eos) {
    result.push_back(eos_id);
  }
  return result;
}

std::string Tokenizer::Decode(const std::vector<int32_t> &ids) {
  std::string result;
  auto status = processor.Decode(ids, &result);
  if (!status.ok()) {
    spdlog::critical("failed to Decode {}", status.error_message());
  }
  return result;
}

static std::pair<std::string, int> _parse(const std::string &line) {
  auto pos = line.find(" ");
  if (pos == std::string::npos) {
    throw std::runtime_error("invalid encoder line: " + line);
  }

  auto token = base64::decode({line.data(), pos});
  int rank = 0;
  try {
    rank = std::stoul(line.substr(pos + 1));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid encoder rank: " + line);
  }

  return {std::move(token), rank};
}

void QwenTokenizer::Init(const std::string &tiktoken_path) {
  std::ifstream file(tiktoken_path);
  if (!file) {
    throw std::runtime_error("failed to open encoder file: " + tiktoken_path);
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  std::string line;
  while (std::getline(file, line)) {
    auto [token, rank] = _parse(line);

    if (!encoder.emplace(std::move(token), rank).second) {
      throw std::runtime_error("duplicate item: " + line);
    }
  }

  std::vector<std::string> special_tokens_s{"<|endoftext|>", "<|im_start|>",
                                            "<|im_end|>"};
  char buffer[14];
  for (size_t i = 0; i < 205; i++) {
    snprintf(buffer, 14, "<|extra_%zu|>", i);
    special_tokens_s.push_back(buffer);
  }
  size_t encoder_size = encoder.size();
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  special_tokens.reserve(special_tokens_s.size());
  for (size_t i = 0; i < special_tokens_s.size(); i++) {
    special_tokens[special_tokens_s[i]] = encoder_size + i;
  }

  tokenizer = tiktoken::tiktoken(std::move(encoder), special_tokens, PAT_STR);
  eos_token_id = encoder_size + 0;
  im_start_id = encoder_size + 1;
  im_end_id = encoder_size + 2;
}

auto QwenTokenizer::build_prompt(const std::vector<std::string> &history) const
    -> std::string {
  if (!(history.size() % 2 == 1)) {
    spdlog::critical("invalid history size {}", history.size());
  }

  std::ostringstream oss_prompt;
  oss_prompt << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
  for (size_t i = 0; i < history.size() - 1; i += 2) {
    oss_prompt << "\n<|im_start|>user\n"
               << history[i] << "<|im_end|>\n<|im_start|>" << history[i + 1]
               << "<|im_end|>";
  }
  oss_prompt << "\n<|im_start|>user\n"
             << history.back() << "<|im_end|>\n<|im_start|>assistant\n";

  return oss_prompt.str();
}

auto QwenTokenizer::encode(const std::string &text, int max_length) const
    -> std::vector<int> {
  auto ids = tokenizer.encode(text);
  if ((int)ids.size() > max_length) {
    ids.erase(ids.begin(), ids.end() - max_length);
  }
  return ids;
}

auto QwenTokenizer::decode(const std::vector<int> &ids) const -> std::string {
  std::vector<int> normal_ids(ids);
  normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(),
                                  [this](int id) { return is_special_id(id); }),
                   normal_ids.end());
  auto text = tokenizer.decode(normal_ids);
  return text;
}

auto QwenTokenizer::encode_history(const std::vector<std::string> &history,
                                   int max_length) const -> std::vector<int> {
  std::string prompt = build_prompt(history);
  std::vector<int> input_ids = encode(prompt, max_length);
  return input_ids;
}

auto QwenTokenizer::is_special_id(int id) const -> bool {
  return id == eos_token_id || id == im_start_id || id == im_end_id;
}

void Qwen2HFTokenizer::from_pretrained(const std::string &tokenizer_dir) {
  py_transformers_module =
      PyImport_ImportModule("transformers");
  if (py_transformers_module == nullptr) {
    PyErr_Print();
    throw std::exception();
  }

  

  nlohmann::json tokenizer_config;

  auto cfg_json = boost::filesystem::path(tokenizer_dir) / "tokenizer_config.json";


  std::ifstream config_fs(cfg_json.c_str());
  if (!config_fs) {
    spdlog::error("failed to open tokenizer conifg {}", cfg_json.c_str());
    throw std::exception();
  }

  config_fs >> tokenizer_config;

  auto tokenizer_class = tokenizer_config["tokenizer_class"].get<std::string>();
  spdlog::info("using tokenizer_class {}", tokenizer_class);

  py_tokenizer_clz =
      PyObject_GetAttrString(py_transformers_module, tokenizer_class.c_str());
  if (py_transformers_module == nullptr) {
    PyErr_Print();
    throw std::exception();
  }

  PyObject *init_args = PyTuple_New(1);
  PyTuple_SetItem(init_args, 0, PyUnicode_FromString(tokenizer_dir.c_str()));

  PyObject *kwargs = PyDict_New();
  PyDict_SetItemString(kwargs, "trust_remote_code", Py_True);
  PyDict_SetItemString(kwargs, "local_files_only", Py_True);

  PyObject *py_tokenizer =
      PyObject_Call(PyObject_GetAttrString(py_tokenizer_clz, "from_pretrained"),
                    init_args, kwargs);

  Py_DECREF(init_args);
  Py_DECREF(kwargs);

  if (py_tokenizer == nullptr) {
    PyErr_Print();
    throw std::exception();
  }

  py_encode_func = PyObject_GetAttrString(py_tokenizer, "encode");
  if (py_encode_func == nullptr) {
    PyErr_Print();
    throw std::exception();
  }

  py_decode_func = PyObject_GetAttrString(py_tokenizer, "decode");
  if (py_decode_func == nullptr) {
    PyErr_Print();
    throw std::exception();
  }

  PyObject* py_eos = PyObject_GetAttrString(py_tokenizer, "eos_token");
  if (py_eos == nullptr) {
    PyErr_Print();
    throw std::exception();
  }
  std::string eos_str = PyUnicode_AsUTF8(py_eos);

  nlohmann::json js_tok;
  std::ifstream tok_fs(
      (boost::filesystem::path(tokenizer_dir) / "tokenizer.json").c_str());
  tok_fs >> js_tok;
  auto add_tokens = js_tok["added_tokens"];

  for (const auto &d : add_tokens) {
    if (d["content"].get<std::string>() == eos_str) {
      eos_token_id = d["id"].get<int>();
    }
    if (d["content"].get<std::string>() == "<|im_start|>") {
      im_start_id = d["id"].get<int>();
    }
    if (d["content"].get<std::string>() == "<|im_end|>") {
      im_end_id = d["id"].get<int>();
    }
  }
}

std::vector<int> Qwen2HFTokenizer::encode(const std::string &text,
                                          int max_length) const {
  PyObject *text_args = PyTuple_New(1);
  PyTuple_SetItem(text_args, 0, PyUnicode_FromString(text.c_str()));
  PyObject *result = PyObject_CallObject(py_encode_func, text_args);
  Py_DECREF(text_args);

  if (result == nullptr) {
    PyErr_Print();
    throw std::exception();
  }

  std::vector<int> ids;
  if (PyList_Check(result)) {
    Py_ssize_t size = PyList_Size(result);
    ids.reserve(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *item = PyList_GetItem(result, i);
      ids.push_back(PyLong_AsLong(item));
    }
  }

  return ids;
}

std::string Qwen2HFTokenizer::decode(const std::vector<int> &ids) {
  PyObject *id_list = PyList_New(ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    PyList_SetItem(id_list, i, PyLong_FromLong(ids[i]));
  }

  PyObject *id_list_args = PyTuple_New(1);
  PyTuple_SetItem(id_list_args, 0, id_list);

  PyObject *result = PyObject_CallObject(py_decode_func, id_list_args);
  Py_DECREF(id_list_args);

  if (result == nullptr) {
    PyErr_Print();
    throw std::exception();
  }

  if (!PyUnicode_Check(result)) {
    Py_DECREF(result);
    throw std::exception();
  }
  const char *str_result = PyUnicode_AsUTF8(result);
  std::string decoded_str(str_result);
  Py_DECREF(result);
  return decoded_str;
}

auto Qwen2HFTokenizer::build_prompt(
    const std::vector<std::string> &history) const -> std::string {
  if (!(history.size() % 2 == 1)) {
    spdlog::critical("invalid history size {}", history.size());
  }

  std::ostringstream oss_prompt;
  oss_prompt << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
  for (size_t i = 0; i < history.size() - 1; i += 2) {
    oss_prompt << "\n<|im_start|>user\n"
               << history[i] << "<|im_end|>\n<|im_start|>" << history[i + 1]
               << "<|im_end|>";
  }
  oss_prompt << "\n<|im_start|>user\n"
             << history.back() << "<|im_end|>\n<|im_start|>assistant\n";

  return oss_prompt.str();
}

auto Qwen2HFTokenizer::encode_history(const std::vector<std::string> &history,
                                      int max_length) const
    -> std::vector<int> {
  std::string prompt = build_prompt(history);
  std::vector<int> input_ids = encode(prompt, max_length);
  return input_ids;
}

auto Qwen2HFTokenizer::is_special_id(int id) const -> bool {
  return id == eos_token_id || id == im_start_id || id == im_end_id;
}
