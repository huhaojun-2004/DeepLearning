import os
import numpy as np
from utils.mnist_dataset_load import load_png_mnist_from_csv

def build_cache():
    # 当前脚本所在目录：DeepLearning/tools
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # 项目根目录：DeepLearning
    project_root = os.path.dirname(this_dir)

    # mnist-pngs 和 tools 同级
    root_dir = os.path.join(project_root, "mnist-pngs")

    cache_dir = os.path.join(root_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    for split, csv_file in [("train", "train.csv"), ("test", "test.csv")]:
        print(f"building {split}...")

        X, y = load_png_mnist_from_csv(
            root_dir=root_dir,
            csv_file=csv_file,
            normalize=True
        )

        np.save(os.path.join(cache_dir, f"X_{split}.npy"), X.astype(np.float32))
        np.save(os.path.join(cache_dir, f"y_{split}.npy"), y.astype(np.int64))

        print(f"saved {split}: {X.shape}, {y.shape}")

if __name__ == "__main__":
    build_cache()
