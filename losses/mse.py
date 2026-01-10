class MSELoss:
    def forward(self,pred,target):
        self.pred=pred
        self.target=target
        loss = ((pred-target)**2).mean()
        return loss

    def backward(self):
        return 2 * (self.pred - self.target) / self.pred.size