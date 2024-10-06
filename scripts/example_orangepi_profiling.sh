#gdb --args
./build/src/llama2_main \
--log_level=info \
--profiling_output=llama_full_prof.json \
--config=/data/llama2/7B/params.json \
--tokenizer=/data/llama2/tokenizer.model \
--weight=/data/llama2/7B/model_output \
--device_type=npu \
--max_seq_len=128 \
--prompt="Once upon"