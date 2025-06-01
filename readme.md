# Orange Pi LLM推理
支持OrangePi LLM推理，当前测试硬件版本: Orange Pi 20T 24GB
## 安装
[安装文档](orangepi_install.md)

## QWen2
支持*Qwen2ForCausalLM*模型
### 模型下载
建议通过git直接从modelscope下载(需要安装git lfs)，比如DeepSeek-R1-Distill-Qwen-1.5B:
```git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B.git```
### 权重转换
#### BF16模型 （以Qwen2.5-3B-Instruct为例，请将路径替换为自己的路径）
```python3 /data/llm_simple/scripts/convert_qwen2_weight.py --input_model_path /ssd/models/Qwen2.5-3B-Instruct --output_dir /ssd/models/Qwen2.5-3B-Instruct_converted ```
#### AWQ模型 （以Qwen2.5-14B-Instruct-AWQ为例，请将路径替换为自己的路径）
```python3 /data/llm_simple/scripts/convert_qwen2_awq_weight.py --input_model_path /ssd/models/Qwen2.5-14B-Instruct-AWQ --output_dir /ssd/models/Qwen2.5-14B-Instruct-AWQ_converted```
### 运行
*请将脚本中的路径改为自己的路径*

```bash scripts/example_text_completion_deepseek_r1_qwen2.5_1.5B_bf16_orangepi.sh```

### 性能（输入256token/输出256token）
|模型大小|ttft(ms)|decode(ms/token)|
|---|---|---|
|1.5B|461|142|
|3B|776|284|
|7B|3215|881|
|3B-AWQ|3215|113|
|7B-AWQ|2358|206
|14B-AWQ|8181|653|


## LLAMA2
### 权重转换
#### LLAMA2-7B FP16 (支持llama官方发布的格式, 包含tokenizer.model,params.json,consolidated.00.pth文件)
```python3 scripts/convert_llama2_weight.py --input_dir <llama_path> --model_size 7B --output_dir <output dir>```
#### LLAMA2-7B-AWQ 4bit
权重下载链接:[model.safetensors](https://huggingface.co/TheBloke/Llama-2-7B-AWQ/blob/main/model.safetensors)
```python3 scripts/convert_llama_awq_4bit.py --input_safetensor <model.safetensors path> --output_dir <weight output path>```
#### LLAMA2-13B-AWQ 4bit
权重下载链接:[model.safetensors](https://huggingface.co/TheBloke/Llama-2-13B-AWQ/resolve/main/model.safetensors)
```python3 scripts/convert_llama_awq_4bit.py --input_safetensor <model.safetensors path> --output_dir <weight output path>```

### 运行
*请将转化后的权重文件夹，配置文件, tokenizer文件拷贝到设备上并修改bash文件中对应的路径*

1. ```bash scripts/example_chat_llama2_7B_fp16_orangepi.sh```
2. ```bash scripts/example_text_completion_llama2_7B_fp16_orangepi.sh```
3. ```bash scripts/example_chat_llama2_7B_awq_4bit_orangepi.sh```
4. ```bash scripts/example_text_completion_llama2_7B_awq_4bit_orangepi.sh```
5. ```bash scripts/example_chat_llama2_13B_awq_4bit_orangepi.sh```
6. ```bash scripts/example_text_completion_llama2_13B_awq_4bit_orangepi.sh```

### 性能
|场景|ttft(ms)|decode(ms/token)|
|---|---|---|
|llama2-7B-AWQ-4bit|886|176.7|
|llama2-7B-FP16|4498|568.4|
|llama2-13B-AWQ-4bit|1819|320.1|
