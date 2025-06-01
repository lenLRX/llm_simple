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

    index_json_path = os.path.join(input_path, "model.safetensors.index.json")

    safetensor_map = {}

    if os.path.exists(index_json_path):
        weight_config_f = open(index_json_path)
        weight_config = json.load(weight_config_f)

        weight_map = weight_config["weight_map"]
        st_set = set()

        for k, wst in weight_map.items():
            st_set.add(wst)

        for st_name in st_set:
            st = load_file(os.path.join(input_path, st_name))
            for k, t in st.items():
                safetensor_map[k] = t
    else:
        # only one file
        safetensor_map = load_file(os.path.join(input_path, "model.safetensors"))


    for k, t in safetensor_map.items():
        print(k, t.shape, t.dtype)
        if "qweight" in k:
            # (k, n//8) -> (k, n//2)
            t = t.view("uint8")
            k_dim, n_dim = t.shape
            print(f"qweight k:{k_dim}, n:{n_dim}")
            t = t.reshape(k_dim, n_dim, 1)
            t = np.repeat(t, 2, axis=-1)
            t[..., 0] = t[..., 0] & 0xf
            t[..., 1] = (t[..., 1] >> 4) & 0xf
            n_dim = n_dim * 2
            t = t.reshape(k_dim, n_dim//8, 2, 4)
            t = np.transpose(t, (0, 1, 3, 2))
            # transpose to (k, n)
            #v = np.transpose(v, (1, 0))
            t = t.reshape(k_dim//16, 16, n_dim)
            t = np.transpose(t, (0, 2, 1))
            print(f"new shape: {t.shape}")
            d1 = t.size // 512
            t = t.reshape(d1, 4, 64, 2)
            t = np.transpose(t, (0, 2, 1, 3))
            t = (t + 8)&0xf
            t[..., 0] = t[..., 0] | (t[...,1] << 4)
            t = np.ascontiguousarray(t[..., 0])
            print(f"qweight output shape {t.shape}, dtype: {t.dtype}")
        if "qzeros" in k:
            # (k//128,n//8) -> (k//128, n//2)
            t = t.view("uint8")
            k_dim, n_dim = t.shape
            t = t.reshape(k_dim, n_dim, 1)
            t = np.repeat(t, 2, axis=-1)
            t[..., 0] = t[..., 0] & 0xf
            t[..., 1] = (t[..., 1] >> 4) & 0xf
            t = t.astype("float16")
            t = t - 8.0
            n_dim = n_dim * 2
            t = t.reshape(k_dim, n_dim//8, 2, 4)
            t = np.transpose(t, (0, 1, 3, 2))
            t = t.reshape(k_dim, n_dim)
            #v = np.transpose(v, (1, 0))
            print(f"qzeros new shape: {t.shape}")
            t = np.ascontiguousarray(t)
        if "scales" in k:
            # (k//128,n)
            #v = np.transpose(v, (1, 0))
            print(f"new qscale shape: {t.shape}")
            t = np.ascontiguousarray(t)


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
