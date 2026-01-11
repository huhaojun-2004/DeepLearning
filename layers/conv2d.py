import numpy as np
from layers.base import Layer


def _pair(v):
    return (v, v) if isinstance(v, int) else v


def im2col(x, K_h, K_w, stride=1, padding=0):
    """
    x: (N, C, H, W)
    return:
      cols: (N * H_out * W_out, C * K_h * K_w)
      (H_out, W_out): output spatial size
      x_pad_shape: padded x shape (for col2im)
    """
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)

    N, C, H, W = x.shape

    H_out = (H + 2 * pad_h - K_h) // stride_h + 1
    W_out = (W + 2 * pad_w - K_w) // stride_w + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Invalid output size: H_out={H_out}, W_out={W_out}. "
                         f"Check padding/stride/kernel.")

    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant"
    )

    # cols shape: (N, C, K_h, K_w, H_out, W_out)
    cols = np.empty((N, C, K_h, K_w, H_out, W_out), dtype=x.dtype)

    # gather windows
    for i in range(K_h):
        i_end = i + stride_h * H_out
        for j in range(K_w):
            j_end = j + stride_w * W_out
            cols[:, :, i, j, :, :] = x_pad[:, :, i:i_end:stride_h, j:j_end:stride_w]

    # -> (N, H_out, W_out, C, K_h, K_w) -> (N*H_out*W_out, C*K_h*K_w)
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)

    return cols, (H_out, W_out), x_pad.shape


def col2im(cols, x_shape, K_h, K_w, out_hw, stride=1, padding=0):
    """
    cols: (N * H_out * W_out, C * K_h * K_w)
    x_shape: original x shape (N, C, H, W)
    out_hw: (H_out, W_out)
    return:
      dx: (N, C, H, W)
    """
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)

    N, C, H, W = x_shape
    H_out, W_out = out_hw

    H_pad = H + 2 * pad_h
    W_pad = W + 2 * pad_w

    cols_reshaped = cols.reshape(N, H_out, W_out, C, K_h, K_w).transpose(0, 3, 4, 5, 1, 2)
    dx_pad = np.zeros((N, C, H_pad, W_pad), dtype=cols.dtype)

    for i in range(K_h):
        i_end = i + stride_h * H_out
        for j in range(K_w):
            j_end = j + stride_w * W_out
            dx_pad[:, :, i:i_end:stride_h, j:j_end:stride_w] += cols_reshaped[:, :, i, j, :, :]

    if pad_h == 0 and pad_w == 0:
        return dx_pad
    return dx_pad[:, :, pad_h:pad_h + H, pad_w:pad_w + W]


class Conv2D(Layer):
    """
    2D Convolution (NCHW) using im2col/col2im.

    Weight:
      W: (C_out, C_in, K_h, K_w)
      b: (C_out,)
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True,
                 weight_init="he", weight_scale=0.01, dtype=np.float32):
        super().__init__()

        K_h, K_w = _pair(kernel_size)
        self.K_h, self.K_w = K_h, K_w
        self.stride = stride
        self.padding = padding
        self.use_bias = bias
        self.dtype = dtype

        # init weights
        if weight_init == "he":
            fan_in = in_channels * K_h * K_w
            std = np.sqrt(2.0 / fan_in)
            W = np.random.randn(out_channels, in_channels, K_h, K_w) * std
        elif weight_init == "xavier":
            fan_in = in_channels * K_h * K_w
            fan_out = out_channels * K_h * K_w
            std = np.sqrt(2.0 / (fan_in + fan_out))
            W = np.random.randn(out_channels, in_channels, K_h, K_w) * std
        else:
            # simple small Gaussian
            W = np.random.randn(out_channels, in_channels, K_h, K_w) * weight_scale

        self.W = W.astype(dtype)
        self.b = (np.zeros((out_channels,), dtype=dtype) if bias else None)

        self.dW = np.zeros_like(self.W)
        self.db = (np.zeros_like(self.b) if bias else None)

        # cache for backward
        self.x_shape = None
        self.X_col = None
        self.out_hw = None

    def params(self):
        if self.use_bias:
            return {"W": self.W, "b": self.b}
        return {"W": self.W}

    def grads(self):
        if self.use_bias:
            return {"W": self.dW, "b": self.db}
        return {"W": self.dW}

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(self.dtype, copy=False)
        self.x_shape = x.shape

        C_out, C_in, K_h, K_w = self.W.shape
        if x.shape[1] != C_in:
            raise ValueError(f"Conv2D expected in_channels={C_in}, got x.shape[1]={x.shape[1]}")

        X_col, (H_out, W_out), _ = im2col(
            x, K_h, K_w, stride=self.stride, padding=self.padding
        )
        self.X_col = X_col
        self.out_hw = (H_out, W_out)

        W_col = self.W.reshape(C_out, -1)  # (C_out, C_in*K_h*K_w)

        out = X_col @ W_col.T  # (N*H_out*W_out, C_out)
        if self.use_bias:
            out += self.b  # broadcast over rows

        N = x.shape[0]
        out = out.reshape(N, H_out, W_out, C_out).transpose(0, 3, 1, 2)  # (N, C_out, H_out, W_out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        dout: (N, C_out, H_out, W_out)
        returns dx: (N, C_in, H, W)
        """
        if self.X_col is None or self.x_shape is None or self.out_hw is None:
            raise RuntimeError("Must call forward() before backward().")

        dout = dout.astype(self.dtype, copy=False)
        N, C_out, H_out, W_out = dout.shape
        if (H_out, W_out) != self.out_hw:
            raise ValueError(f"dout spatial {(H_out, W_out)} mismatch cached {self.out_hw}")

        dout_col = dout.transpose(0, 2, 3, 1).reshape(N * H_out * W_out, C_out)  # (N*H_out*W_out, C_out)

        # db
        if self.use_bias:
            self.db[...] = dout_col.sum(axis=0)

        # dW
        # dW_col: (C_out, C_in*K_h*K_w) = dout_col.T @ X_col
        dW_col = dout_col.T @ self.X_col
        self.dW[...] = dW_col.reshape(self.W.shape)

        # dx
        W_col = self.W.reshape(C_out, -1)               # (C_out, C_in*K_h*K_w)
        dX_col = dout_col @ W_col                       # (N*H_out*W_out, C_in*K_h*K_w)
        dx = col2im(
            dX_col, self.x_shape, self.K_h, self.K_w, self.out_hw,
            stride=self.stride, padding=self.padding
        )
        return dx
