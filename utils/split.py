import numpy as np

def train_val_split(X,y,val_ratio=0.2, shuffle=True, seed=0):
    N = X.shape[0]
    idx = np.arange(N)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    val_size = int(N * val_ratio)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    return X[train_idx],y[train_idx],X[val_idx],y[val_idx]