#gdb --args \
./build/src/llama2_main \
--config=llama/llama2/13B/params.json \
--tokenizer=llama/llama2/tokenizer.model \
--weight=/data/Llama-2-13B-AWQ/llama2_13B_awq_4bit \
--device_type=npu \
--max_seq_len=2048 \
--max_gen_token=256 \
--log_level=info \
--debug_print=false \
--quant_method=awq_4bit \
--quant_group_size=128 \
--rope_is_neox_style=true \
--prompt="You are a virtual tour guide from 1901. You have tourists visiting Eiffel Tower. Describe Eiffel Tower to your audience. Begin with
1. Why it was built
2. Then by how long it took them to build
3. Where were the materials sourced to build
4. Number of people it took to build
5. End it with the number of people visiting the Eiffel tour annually in the 1900's, the amount of time it completes a full tour and why so many people visit this place each year.
Make your tour funny by including 1 or 2 funny jokes at the end of the tour."
