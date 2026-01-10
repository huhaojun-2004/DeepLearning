import os
import numpy as np

def load_mnist_from_npy(root_dir, split):
    """
    从 .npy 缓存中加载 MNIST 数据（mmap）

    参数:
        root_dir: 项目中 mnist-pngs 的路径
        split:    "train" 或 "test"

    返回:
        X: (N, 28, 28) float32  (np.memmap)
        y: (N,) int64          (np.memmap)
    """
    cache_dir = os.path.join(root_dir, "cache")

    X_path = os.path.join(cache_dir, f"X_{split}.npy")
    y_path = os.path.join(cache_dir, f"y_{split}.npy")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"MNIST cache not found: {X_path}, {y_path}\n"
            f"请先运行 tools/build_mnist_npy.py"
        )

    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")

    return X, y
