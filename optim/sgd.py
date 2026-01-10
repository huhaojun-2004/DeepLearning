from sympy.physics.vector.printing import params


class SGD:
    def __init__(self,layers,lr=1e-2,weight_decay = 0.0):
        self.layers = layers
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for layer in self.layers:
            params = layer.params()
            grads = layer.grads()
            if not params:
                continue

            for k,p in params.items():
                g = grads[k]
                if self.weight_decay !=0.0:
                    g = g + self.weight_decay * p #L2正则化技术??
                p[...] = p[...] -self.lr * g

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()



