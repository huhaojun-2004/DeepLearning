import numpy as np
from layers import Layer
class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.x = None

    def forward(self,x):
        self.x = x
        return np.maximum(0,x)

    def backward(self, dout):
        return dout * (self.x > 0)