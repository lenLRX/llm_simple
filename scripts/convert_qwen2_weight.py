from ml_dtypes import bfloat16
from safetensors.numpy import load_file
import numpy as np
import argparse
import os
import json
import re


def change_weight_layout(t):
    return t
    n, k = t.shape
    t = t.reshape(n//16,16,k)
    t = np.transpose(t, (0, 2, 1))
    t = np.ascontiguousarray(t)
    return t



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model_path",
        required=True,
        help="Location of Qwen Git repo",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write converted model",
    )

    args = parser.parse_args()

    input_path = args.input_model_path
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    weight_config_f = open(os.path.join(input_path, "model.safetensors.index.json"))
    weight_config = json.load(weight_config_f)

    weight_map = weight_config["weight_map"]

    safetensor_map = {}

    for k, wst in weight_map.items():
        tensor_name = k
        if wst not in safetensor_map:
            safetensor_map[wst] = load_file(os.path.join(input_path, wst))
        t = safetensor_map[wst][k]
        print(k, t.shape, t.dtype)
        if k == "lm_head.weight":
            t = change_weight_layout(t)
            t.tofile(os.path.join(output_path, k + ".bin"))
        elif k == "model.norm.weight":
            t.tofile(os.path.join(output_path, k + ".bin"))
        elif k == "model.embed_tokens.weight":
            t.tofile(os.path.join(output_path, k + ".bin"))
        else:
            m = re.match("model.*layernorm.*", k)
            if m:
                t.tofile(os.path.join(output_path, k + ".bin"))
                continue
            m = re.match(".*bias", k)
            if m:
                t = t.astype("float32")
                t.tofile(os.path.join(output_path, k + ".bin"))
                continue

            t = change_weight_layout(t)
            t.tofile(os.path.join(output_path, k + ".bin"))


if __name__ == '__main__':
    main()
