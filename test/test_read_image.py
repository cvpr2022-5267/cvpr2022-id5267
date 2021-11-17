from PIL import Image
import torch
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    image = Image.open("../data/train/images/00001.png")
    image = image.convert('RGB')
    trans_to_tensor = transforms.ToTensor()
    trans_to_PIL = transforms.ToPILImage()
    img_tensor = trans_to_tensor(image)
    norm = transforms.Normalize((0.65, 0.65, 0.65), (0.27, 0.27, 0.27))
    img_tensor = norm(img_tensor)
    tensor_r = img_tensor[0].numpy()
    tensor_g = img_tensor[1].numpy()
    tensor_b = img_tensor[2].numpy()

    img_np = np.array(image)
    np_r = img_np[:, :, 0]
    np_g = img_np[:, :, 1]
    np_b = img_np[:, :, 2]
    tensor_t_numpy = trans_to_PIL(img_tensor)
    plt.imshow(tensor_t_numpy)
    plt.show()
    plt.clf()
