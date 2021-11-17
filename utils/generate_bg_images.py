import numpy as np
from PIL import Image


if __name__ == "__main__":
    bg_imgs = np.load('../data/bg_imgs.npy').astype(np.str)

    bg_img_list = np.random.choice(bg_imgs, 1000)
    bg_img = [Image.open(i).convert('RGB').resize((64, 64), resample=Image.LANCZOS) for i in bg_img_list]
    bg_img = [np.array(i).astype('float32').reshape((64, 64, 3)) / 255. for i in bg_img]
    bg_img = np.stack(bg_img)

    with open('bg_images.npy', 'wb') as f:
        np.save(f, bg_img)

