import numpy as np
from layers import Layer

class Dropout(Layer):
    def __init__(self,p:float =0.5):
        super().__init__()
        assert 0.0<=p<1,"Dropout p must be in [0, 1)"
        self.p = p
        self.mask=None

    def forward(self,x):
        if not self.training or self.p == 0.0:
            return x

        self.mask = (np.random.rand(*x.shape)>self.p).astype(self.dtype)

        return x* self.mask / (1.0-self.p)

    def backward(self, dout):
        if not self.training or self.p == 0.0:
            return dout

        return dout * self.mask / (1.0 - self.p)