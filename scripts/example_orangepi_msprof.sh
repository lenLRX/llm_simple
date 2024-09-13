#gdb --args
msprof --aic-mode=sample-based --output=./profiling_output --application="./build/src/llama2_main \
--config=/data/llama2/7B/params.json \
--tokenizer=/data/llama2/tokenizer.model \
--weight=/data/llama2/7B/model_output \
--device_type=npu \
--prompt=\"Once upon\""