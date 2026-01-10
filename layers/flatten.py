import numpy as np
from layers import Layer

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.orig_shape = None

    def forward(self,x):
        # x: (N, ...) -> (N, D)
        self.orig_shape = x.shape
        return x.reshape(x.shape[0],-1)

    def backward(self,dout):
        # dout: (N, D) -> (N, ...)
        return dout.reshape(self.orig_shape)