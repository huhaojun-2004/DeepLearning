import numpy as np

def xavier(in_dim,out_dim):
    limit = np.sqrt(2.0/(in_dim + out_dim))
    return np.random.randn(in_dim,out_dim )* limit