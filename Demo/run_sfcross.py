import numpy as np
from layers.linear import Linear
from layers.relu import ReLU
from layers.sigmoid import Sigmoid
from model import Model
from losses.mse import MSELoss
from optim.sgd import SGD
from losses.bcewithlogits import BCEWithLogitsLoss
from losses.softmaxcrossentropy import SoftmaxCrossEntropyLoss
import time

x = np.array([[0,0],[1,1],[0,1],[1,0]]) #(batch,2) (4,2)
y = np.array([[0],[0],[1],[1]])   #(4,1)
y = y.reshape(-1)

model = Model([Linear(2, 30),ReLU(),Linear(30,2)])
loss_fn = SoftmaxCrossEntropyLoss()
opt = SGD(model.layers, lr=1)

for step in range(50000):
    opt.zero_grad()
    pred = model.forward(x)
    loss = loss_fn.forward(pred, y)
    dout = loss_fn.backward()
    model.backward(dout)
    opt.step()

    print(f"step:{step},loss:{loss}\n")

pred_label = np.argmax(model.forward(x), axis=1).tolist()
true_y = y.flatten().tolist()

print(f"pre_value \n {pred_label}")
print(f"true_value \n {true_y}")