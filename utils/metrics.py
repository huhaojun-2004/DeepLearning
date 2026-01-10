import numpy as np

def accuracy_from_logits(logits,targets):
    # logits(N,C) targets(N,) int 64

    pred = np.argmax(logits,axis=1)
    return (pred == targets).mean()