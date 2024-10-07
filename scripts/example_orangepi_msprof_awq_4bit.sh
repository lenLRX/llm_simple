#gdb --args
msprof --aic-mode=sample-based --output=./profiling_output --application="./build/src/llama2_main \
--config=/data/llama2/7B/params.json \
--tokenizer=/data/llama2/tokenizer.model \
--weight=/data/llama2_7b_awq/llama2_7b_awq_4bit/ \
--device_type=npu \
--quant_method=awq_4bit \
--quant_group_size=128 \
--max_seq_len=128 \z
--prompt=\"Once upon\""