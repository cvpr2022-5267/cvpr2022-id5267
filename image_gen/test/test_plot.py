#!/usr/bin/python3

from image_gen.image_generation import plot_bar, make_dirs, gen_dataset
from dataset.normalize_label import cal_label_mu_sigma

savepath = "/home/ning/toy_dexdexnet/data"

kwargs = {
    "figsize" : (16, 16),
    "train_num" : 10,
    "test_num" : 10,
    "val_num" : 10,
    "colors" : ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "purple", "pink", "brown"],
    "color_test": ["olive"],
    "verbose" : True,
    "exist_ok" : False,
}


if __name__ == '__main__':
    # generate training images
    gen_dataset(kwargs["train_num"], kwargs["colors"], flag="train_10", kwargs=kwargs)
    cal_label_mu_sigma(flag="train_10")
    # # generate testing images
    # gen_dataset(kwargs["test_num"], kwargs["colors"], flag="test", kwargs=kwargs)
    #
    # # generate val images
    # gen_dataset(kwargs["val_num"], kwargs["color_test"], flag="val", kwargs=kwargs)
