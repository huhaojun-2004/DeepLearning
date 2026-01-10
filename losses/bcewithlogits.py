import numpy as np

class BCEWithLogitsLoss:
    def __init__(self):
        self.logits = None
        self.targets = None
        self.probs = None

    def forward(self,logits,targets):
        logits = np.asarray(logits,dtype = np.float32)
        targets = np.asarray(targets , dtype=np.float32)

        self.logits = logits
        self.targets = targets

        #sigmoid
        self.probs = 1.0 / (1.0 + np.exp(-logits))

        x=logits
        loss = np.maximum(x, 0) - x * targets + np.log1p(np.exp(-np.abs(x)))
        return loss.mean()

    def backward(self):
        """
        dL/dlogits = sigmoid(logits) - y
        """
        N = self.logits.size
        return (self.probs - self.targets) / N