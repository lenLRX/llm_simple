from safetensors.numpy import load_file
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_safetensor",
        required=True,
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write converted model",
    )

    args = parser.parse_args()

    input_path = args.input_safetensor
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)

    loaded = load_file(input_path)
    for k, v in loaded.items():
        tname = f"{k}.bin"
        print(f"writting {tname} shape: {v.shape} dtype: {v.dtype}")
        if "qweight" in k:
            # (k, n//8) -> (k, n//2)
            v = v.view("uint8")
            k_dim, n_dim = v.shape
            print(f"weight k:{k_dim}, n:{n_dim}")
            v = v.reshape(k_dim, n_dim, 1)
            v = np.repeat(v, 2, axis=-1)
            v[..., 0] = v[..., 0] & 0xf
            v[..., 1] = (v[..., 1] >> 4) & 0xf
            n_dim = n_dim * 2
            v = v.reshape(k_dim, n_dim//8, 2, 4)
            v = np.transpose(v, (0, 1, 3, 2))
            # transpose to (k, n)
            #v = np.transpose(v, (1, 0))
            v = v.reshape(k_dim//16, 16, n_dim)
            v = np.transpose(v, (0, 2, 1))
            print(f"new shape: {v.shape}")
            d1 = v.size // 512
            v = v.reshape(d1, 4, 64, 2)
            v = np.transpose(v, (0, 2, 1, 3))
            v[..., 0] = v[..., 0] | (v[...,1] << 4)
            v = np.ascontiguousarray(v[..., 0])
            print(f"weight output shape {v.shape}, dtype: {v.dtype}")
        if "qzeros" in k:
            # (k//128,n//8) -> (k//128, n//2)
            v = v.view("uint8")
            k_dim, n_dim = v.shape
            v = v.reshape(k_dim, n_dim, 1)
            v = np.repeat(v, 2, axis=-1)
            v[..., 0] = v[..., 0] & 0xf
            v[..., 1] = (v[..., 1] >> 4) & 0xf
            v = v.astype("float16")
            n_dim = n_dim * 2
            v = v.reshape(k_dim, n_dim//8, 2, 4)
            v = np.transpose(v, (0, 1, 3, 2))
            v = v.reshape(k_dim, n_dim)
            #v = np.transpose(v, (1, 0))
            print(f"new shape: {v.shape}")
            v = np.ascontiguousarray(v)
        if "scales" in k:
            # (k//128,n)
            #v = np.transpose(v, (1, 0))
            print(f"new shape: {v.shape}")
            v = np.ascontiguousarray(v)

        v.tofile(os.path.join(output_path, tname))


if __name__ == '__main__':
    main()
