import numpy as np
from layers import Layer

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.y = None

    def forward(self,x):
        self.y = 1 / (1+np.exp(-x))
        return self.y

    def backward(self,dout):
        return dout * self.y * (1-self.y)