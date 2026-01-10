import numpy as np
from utils import ProgressBar,load_mnist_from_npy,accuracy_from_logits
from data.dataloader import DataLoadwer
from model import Model
from layers import Linear,ReLU,Flatten,Dropout
from losses.softmaxcrossentropy import SoftmaxCrossEntropyLoss
from optim.sgd import SGD
from utils.checkpoint import save_model_npz

def main():


    batch_size = 512
    epochs = 80
    lr = 0.1

    # -------- 1 读数据（返回 NumPy）--------
    X_train,y_train = load_mnist_from_npy("mnist-pngs",split="train")
    X_test,y_test = load_mnist_from_npy("mnist-pngs",split="test")

    print("Train:", X_train.shape, y_train.shape, X_train.dtype, y_train.dtype)
    print("Test: ", X_test.shape, y_test.shape, X_test.dtype, y_test.dtype)

    train_loader = DataLoadwer(X_train,y_train,batch_size=batch_size,shuffle=True,drop_last=True)
    test_loader = DataLoadwer(X_test, y_test, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Model([Flatten(),Linear(784,1024),ReLU(),Dropout(0.2),Linear(1024,512),ReLU(),Dropout(0.2),Linear(512,256),ReLU(),Linear(256,10)])

    loss_fn = SoftmaxCrossEntropyLoss()
    opt = SGD(model.layers,lr=lr,weight_decay=1e-4)

    num_batches_train = len(train_loader)
    num_batches_test = len(test_loader)

    # 开始训练
    for epoch in range(1,epochs+1):
        train_losses = []
        train_accs = []

        test_losses = []
        test_accs = []

        pbar_train = ProgressBar(num_batches_train, prefix=f"Epoch of Train {epoch}/{epochs}",leave=True)
        pbar_test = ProgressBar(num_batches_test, prefix=f"Epoch of Test {epoch}/{epochs}")

        step_train = 1
        step_test = 1
        for xb,yb in train_loader:
            opt.zero_grad()

            logits = model.forward(xb)
            loss = loss_fn.forward(logits,yb)
            dout = loss_fn.backward()

            model.backward(dout)
            opt.step()

            train_losses.append(loss)
            acc = accuracy_from_logits(logits,yb)
            train_accs.append(acc)

            pbar_train.update(step_train,postfix=f"loss={loss:.4f} acc={acc:.4f}")
            step_train += 1

        pbar_train.close()

        #开始评估
        for xb,yb in test_loader:
            logits = model.forward(xb)

            loss = loss_fn.forward(logits,yb)
            test_losses.append(loss)

            acc = accuracy_from_logits(logits, yb)
            test_accs.append(acc)

            pbar_test.update(step_test, postfix=f"loss={loss:.4f} acc={acc:.4f}")
            step_test += 1


        pbar_test.close()

        print(
            f"\n Epoch {epoch}/{epochs} | "
            f"train_loss={np.mean(train_losses):.4f}, train_acc={np.mean(train_accs):.4f} | "
            f"test_loss={np.mean(test_losses):.4f}, test_acc={np.mean(test_accs):.4f}\n"
        )

    save_model_npz(model, "checkpoints/mnist_mlp.npz")


if __name__ == "__main__":
    main()