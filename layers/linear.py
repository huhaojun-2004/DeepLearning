from typing import Dict, Any

import numpy as np
from layers import Layer


class Linear(Layer):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.W = self.init(in_dim,out_dim).astype(self.dtype)
        self.b = np.zeros(out_dim).astype(self.dtype)
        self.x = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self,x):
        # x:(batch,in_dim)
        self.x = x #cache
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW[...] = self.x.T @ dout
        self.db[...] = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx

    def params(self) -> Dict[str, Any]:
        return {"W":self.W , "b":self.b}

    def grads(self) -> Dict[str, Any]:
        return {"W":self.dW , "b":self.db}