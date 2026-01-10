import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =====================
# Config
# =====================
ROOT_DIR = "mnist-pngs"          # 你的本地目录
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
BATCH_SIZE = 512
EPOCHS = 80
LR = 1e-3

# Windows 上 tqdm “卡 0%”通常和 worker 启动有关：
# 想要最顺滑：NUM_WORKERS = 0
# 想更快：2 或 4，但第一次会卡一下（正常）
NUM_WORKERS = 0
PIN_MEMORY = True


# =====================
# Local npy loader (mmap)
# =====================
def load_mnist_from_npy(cache_dir: str, split: str):
    X_path = os.path.join(cache_dir, f"X_{split}.npy")
    y_path = os.path.join(cache_dir, f"y_{split}.npy")
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"missing: {X_path} / {y_path}")
    X = np.load(X_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    return X, y


class LocalNpyMNIST(Dataset):
    def __init__(self, X, y):
        self.X = X  # memmap/ndarray, (N,28,28) float32
        self.y = y  # memmap/ndarray, (N,) int64

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        # 注意：这里不把整个数据拷进内存，只取一张
        x = np.array(self.X[idx], copy=False)        # (28,28) float32
        y = int(self.y[idx])                         # python int
        x = torch.from_numpy(x).float()              # CPU tensor
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# =====================
# Model (MLP)
# =====================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ---- load local data ----
    X_train, y_train = load_mnist_from_npy(CACHE_DIR, "train")
    X_test, y_test = load_mnist_from_npy(CACHE_DIR, "test")
    print("Train:", X_train.shape, y_train.shape, X_train.dtype, y_train.dtype)
    print("Test: ", X_test.shape, y_test.shape, X_test.dtype, y_test.dtype)

    train_ds = LocalNpyMNIST(X_train, y_train)
    test_ds = LocalNpyMNIST(X_test, y_test)

    # persistent_workers 只有 num_workers>0 才能用
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # ---- model / loss / opt ----
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ===== Train =====
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]", ncols=100)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)   # (B,28,28)
            yb = yb.to(device, non_blocking=True)   # (B,)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            train_correct += (pred == yb).sum().item()
            train_total += yb.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{train_correct/train_total:.4f}")

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ===== Test =====
        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            pbar_t = tqdm(test_loader, desc=f"Epoch {epoch}/{EPOCHS} [Test ]", ncols=100)
            for xb, yb in pbar_t:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(xb)
                loss = criterion(logits, yb)

                test_loss_sum += loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                test_correct += (pred == yb).sum().item()
                test_total += yb.size(0)

                pbar_t.set_postfix(acc=f"{test_correct/test_total:.4f}")

        test_loss = test_loss_sum / test_total
        test_acc = test_correct / test_total

        print(
            f"\nEpoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}\n"
        )

        # save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "test_acc": best_acc},
                "checkpoints/mnist_mlp_best.pt",
            )

    torch.save({"model_state": model.state_dict()}, "checkpoints/mnist_mlp_latest.pt")
    print("saved checkpoints/mnist_mlp_best.pt and mnist_mlp_latest.pt")


if __name__ == "__main__":
    main()
