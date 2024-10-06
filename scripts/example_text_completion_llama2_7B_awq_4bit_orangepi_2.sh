#gdb --args \
./build/src/llama2_main \
--config=/data/llama2/7B/params.json \
--tokenizer=/data/llama2_7b_awq/tokenizer.model \
--weight=/data/llama2_7b_awq/llama2_7b_awq_4bit/ \
--device_type=npu \
--max_seq_len=2048 \
--log_level=info \
--quant_method=awq_4bit \
--quant_group_size=128 \
--prompt="Please write a story of a poor little girl."