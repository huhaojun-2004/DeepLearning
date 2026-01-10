import numpy as np

class SoftmaxCrossEntropyLoss:
    def __init__(self):
        self.logits = None
        self.targets = None
        self.probs = None

    def forward(self,logits,targets):
        # logits:(B,C) targets:(B,)
        logits = np.asarray(logits, dtype=np.float32)
        targets = np.asarray(targets,dtype=np.int64)

        self.logits = logits
        self.targets = targets

        z = logits - np.max(logits , axis=1, keepdims=True)  #平移减少exp值 广播机制keep缺失
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z,axis=1,keepdims=True)
        self.probs =probs

        B = logits.shape[0] #B个样本
        p_true = probs[np.arange(B),targets]
        loss = -np.log(p_true + 1e-12)
        return loss.mean()

    def backward(self):
        """
               dL/dlogits = (softmax(logits) - one_hot(targets)) / N
        """
        N, C = self.probs.shape
        dlogits = self.probs.copy()
        dlogits[np.arange(N), self.targets] -= 1.0
        dlogits /= N
        return dlogits

