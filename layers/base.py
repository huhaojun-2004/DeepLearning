from __future__ import annotations
from abc import ABC,abstractmethod
from typing import Dict,Any
from initializers import he,xavier
import numpy as np

class Layer(ABC):

    def __init__(self, init=he,dtype=np.float32):
        self.training = True
        self.init =init
        self.dtype = dtype

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self,x):
        return self.forward(x)

    @abstractmethod
    def forward(self,x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, dout):
        raise NotImplementedError

    def params(self) -> Dict[str, Any]:
        return {}

    def grads(self) -> Dict[str, Any]:
        return {}

    def zero_grad(self):
        # Default: clear grads if layer has any
        for k, g in self.grads().items():
            if g is None:
                continue
            g[...] = 0
