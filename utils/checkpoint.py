import os
import numpy as np

def save_model_npz(model, path: str):
    """
    保存模型参数到 .npz
    默认约定：有参数的层具备 W, b 属性（例如 Linear）
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    params = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "W") and hasattr(layer, "b"):
            params[f"layer{i}.W"] = layer.W
            params[f"layer{i}.b"] = layer.b

    np.savez(path, **params)
    print(f"[save] {path} (saved {len(params)} arrays)")

def load_model_npz(model, path: str):
    """
    从 .npz 加载参数到 model
    要求：模型结构一致（layers 顺序一致，Linear 位置一致）
    """
    data = np.load(path)

    for i, layer in enumerate(model.layers):
        if hasattr(layer, "W") and hasattr(layer, "b"):
            W_key = f"layer{i}.W"
            b_key = f"layer{i}.b"
            if W_key not in data or b_key not in data:
                raise KeyError(f"missing keys: {W_key}, {b_key} in {path}")

            if layer.W.shape != data[W_key].shape or layer.b.shape != data[b_key].shape:
                raise ValueError(
                    f"shape mismatch at layer {i}: "
                    f"W {layer.W.shape} vs {data[W_key].shape}, "
                    f"b {layer.b.shape} vs {data[b_key].shape}"
                )

            layer.W[...] = data[W_key]
            layer.b[...] = data[b_key]

    print(f"[load] {path}")
