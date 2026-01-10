import numpy as np

def he(in_dim,out_dim):
    return np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)