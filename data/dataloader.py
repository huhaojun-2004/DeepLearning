import numpy as np

class DataLoadwer:
    def __init__(self,X,y,batch_size=16,shuffle=True,drop_last=False):
        """
        :param X: (B,····)
        :param y: (B,) OR (B,1)
        :param batch_size:
        :param shuffle:
        :param drop_last:
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.N = X.shape[0]

    def __iter__(self):
        idx = np.arange(self.N)
        if self.shuffle:
            np.random.shuffle(idx)

        bs = self.batch_size
        for start in range(0,self.N,bs):
            end = start+bs
            if end > self.N and self.drop_last:
                break
            batch_idx = idx[start:end]
            yield self.X[batch_idx],self.y[batch_idx]

    def __len__(self):
        if self.drop_last:
            return self.N // self.batch_size
        return (self.N + self.batch_size -1) // self.batch_size




