#gdb --args
./build/src/llama2_main \
--config=/data/llama2/7B/params.json \
--tokenizer=/data/llama2/tokenizer.model \
--weight=/data/llama2/7B/model_output \
--device_type=npu \
--max_seq_len=128 \
--log_level=debug \
--debug_print=true \
--prompt="Translate English to French:
    sea otter => loutre de mer
    peppermint => menthe poivrée
    plush girafe => girafe peluche
    cheese =>"