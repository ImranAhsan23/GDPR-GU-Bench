#!/usr/bin/env python
from __future__ import annotations

import platform
import sys


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    try:
        import torch
    except Exception as e:
        print("torch: NOT INSTALLED")
        print("error:", repr(e))
        print("Install PyTorch first. Example for CUDA 12.1:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return 1

    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda_version_built_with:", torch.version.cuda)
        print("gpu_count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024 ** 3)
            print(f"gpu_{i}: {torch.cuda.get_device_name(i)} | {total_gb:.2f} GB")
    else:
        print("No CUDA GPU detected by PyTorch. Experiments can still run with --device cpu or --device auto, but they will be slower.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
