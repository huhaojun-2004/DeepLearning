import cv2
import numpy as np

# 1. 准备输入数据 (N=1, C=3, H=3, W=3)
# 为了方便观察，我们让三个通道的数据相同
single_channel = np.array([[1, 1, 1],
                           [2, 2, 2],
                           [3, 3, 3]],dtype=np.uint8)
x = np.array([[single_channel, single_channel, single_channel]])
img = cv2.imread('cat.png').transpose(2,0,1)
img = np.array([img])


# 2. 准备卷积核 W (C_out=2, C_in=3, K_h=2, K_w=2)
# 这里我们定义 2 个卷积核，每个核都要处理 3 个通道
W = np.ones((1, 3, 2, 2))
print(W)
# 3. 准备偏置 b (C_out=2)
b = np.zeros(1)

def conv2d_forward_naive(x, W, b, stride=1, padding=0):
    N, C_in, H, W_in = x.shape
    C_out, _, K_h, K_w = W.shape

    # padding 逻辑
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")

    H_out = (H + 2*padding - K_h) // stride + 1
    W_out = (W_in + 2*padding - K_w) // stride + 1

    out = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    for n in range(N):           # 遍历每个样本
        for co in range(C_out):  # 遍历每个卷积核 (输出通道)
            for i in range(H_out):
                hs = i * stride
                for j in range(W_out):
                    ws = j * stride
                    # 取出窗口：包含了输入的所有通道 (C_in, K_h, K_w)
                    window = x_pad[n, :, hs:hs+K_h, ws:ws+K_w]
                    # 核心操作：窗口与对应的核点乘求和，再加上偏置
                    out[n, co, i, j] = np.sum(window * W[co]) + b[co]
    return out

output = conv2d_forward_naive(img, W, b).transpose(0,2,3,1)
cv2.imshow("1",output[0])
cv2.waitKey(0)


print("输出的维度是:", output.shape)
print("输出的内容是:\n", len(output[0][0][0]))