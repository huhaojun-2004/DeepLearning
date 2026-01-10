import numpy as np
from PIL import Image

from model import Model
from layers import Linear, ReLU, Faltten
from utils.checkpoint import load_model_npz

def build_model():
    # 和你训练时保持一致
    return Model([Faltten(),Linear(784,1024),ReLU(),Linear(1024,512),ReLU(),Linear(512,256),ReLU(),Linear(256,10)])
def preprocess_mnist_image(img_path: str, invert: bool = False):
    """
    返回 X: (1, 28, 28) float32, 范围 [0,1]
    """
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))

    x = np.array(img, dtype=np.float32) / 255.0

    if invert:
        x = 1.0 - x  # 如果你的图是白底黑字，通常需要反色

    x = x[None, :, :]  # (1,28,28)
    return x

def predict_one(model, img_path: str, invert: bool = False):
    x = preprocess_mnist_image(img_path, invert=invert)
    logits = model.forward(x)                  # (1,10)
    pred = int(np.argmax(logits, axis=1)[0])   # 类别
    return pred, logits

def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)

if __name__ == "__main__":
    ckpt_path = "checkpoints/mnist_mlp.npz"
    img_path = "img3.png"   # 改成你的图片路径

    model = build_model()
    load_model_npz(model, ckpt_path)

    pred, logits = predict_one(model, img_path, invert=True)
    print("pred =", pred)
    print("logits =",logits)
