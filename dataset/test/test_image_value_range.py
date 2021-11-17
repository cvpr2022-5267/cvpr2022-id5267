#!/usr/bin/python3

from PIL import Image
import torchvision.transforms.functional as F
import numpy as np

img_path = "./test.png"

if __name__ == "__main__":
    img = Image.open(img_path).convert('RGB')
    img = F.to_tensor(img)
    if list(img.shape) != [3, 128, 128]:
        raise ValueError("channel is not in the first dimension!")
    if img.max() > 1 or img.min() < 0:
        raise ValueError("image value is not between [0,1 ]!")
    img = F.normalize(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
