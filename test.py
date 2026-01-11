import cv2
import numpy as np
p=0.8
x=np.array([[[[1,1,1],[2,2,2]],[[1,1,1],[2,2,2]]]])
mask = np.random.rand(*x.shape)>p
print(mask)

x=x* mask / (1.0-p)
print(x)