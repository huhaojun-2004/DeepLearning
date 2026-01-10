import os
import csv
import numpy as np
from PIL import Image

def load_png_mnist_from_csv(root_dir,csv_file,normalize = True):
    image_paths = []
    labels = []

    #读取csv 并且跳过表头
    csv_file = os.path.join(root_dir,csv_file)
    with open(csv_file , "r",encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            rel_path,label = row
            # check_img_path = os.path.join(root_dir,rel_path)
            # check_img = Image.open(check_img_path).convert("L")
            # w,h = check_img.size
            # if(w,h) !=(28,28):
            #     print(f"Expected 28x28 images, got {w}x{h} at {check_img_path}")
            #     continue

            image_paths.append(rel_path)
            labels.append(int(label))

    N = len(image_paths)

    #分配numpy数组
    X = np.zeros((N,28,28),dtype=np.float32)
    y = np.array(labels,dtype=np.int64)

    for i,rel_path in enumerate(image_paths):
        img_path = os.path.join(root_dir,rel_path)
        img = Image.open(img_path).convert("L")
        arr = np.array(img ,dtype=np.float32)

        if normalize:
            arr/= 255.0

        X[i] = arr

    return X,y
