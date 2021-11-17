#!/usr/bin/python3

import os
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np
# import json
import yaml
from PIL import Image
import csv
import random


class BarImage(Dataset):
    """
    Bar images dataset
    """
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.data_path = os.path.join(self.root_dir, self.mode)  # ../train or ../test
        # label_mu_sigma_path = os.path.join(self.data_path, "label_mu_sigma.txt")
        # with open(label_mu_sigma_path, "r") as f:
        #     r = json.load(f)
        #     self.label_mu, self.label_sigma = np.float32(r["mu"]), np.float32(r["sigma"])

    def __getitem__(self, ind):
        self.image_dir_path = os.path.join(self.data_path, "images")
        self.label_dir_path = os.path.join(self.data_path, "labels")
        self.image_name = f"{ind:06d}.png"
        self.label_name = f"{ind:06d}.yaml"
        self.image_path = os.path.join(self.image_dir_path, self.image_name)
        self.label_path = os.path.join(self.label_dir_path, self.label_name)
        sample = Image.open(self.image_path)
        label = {}
        length = []
        with open(self.label_path) as f:
            file = yaml.load(f, Loader=yaml.FullLoader)

            # for c in file:
            #     l_c = np.float32(file[c])
            #     # normalize label
            #     # length = (l_c - self.label_mu) / self.label_sigma
            #     length = np.float32(l_c / 23)
            #     # length = np.float32(l_c / 1)
            #     label[c] = length

            for c in file:
                l_c = np.float32(file[c])
                # normalize label
                # length = (l_c - self.label_mu) / self.label_sigma
                # l_c = np.float32(l_c / 23)
                l_c = np.float32(l_c / 23) - 0.5
                # length = np.float32(l_c / 1)
                length.append(l_c)

        # label = (l - self.label_mu) / self.label_sigma
        if self.transform:
            sample = self.transform(sample)
            sample -= 0.5
        label_c = np.array(eval(c), dtype=np.float32)
        length = np.array(length, dtype=np.float32)
        return sample, length, label_c

    def __len__(self):
        return len(glob.glob(self.data_path + '/images/' + '*.png'))


# def custom_collate(batch):
#     """Puts each data field into a tensor with outer dimension batch size"""
#
#     elem = batch[0]
#     elem_type = type(elem)
#     if isinstance(elem, torch.Tensor):
#         out = None
#         if torch.utils.data.get_worker_info() is not None:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = elem.storage()._new_shared(numel)
#             out = elem.new(storage)
#         return torch.stack(batch, 0, out=out)
#     elif isinstance(elem, dict):
#         return batch
#     elif isinstance(elem, tuple):
#         transposed = zip(*batch)
#         return [custom_collate(samples) for samples in transposed]
#
#     raise TypeError("{}Error".format(elem_type))


def bar(batch_size, path_to_data, mode, img_size, shuffle=True, test=False):
    """
    batch_size:
    mode: str "train" or "test"
    img_size: image size, default: 128
    path_to_data: path to data directory
    shuffle: default: True
    """

    # TODO: add mode ["train", "test"] for different transform
    transform = transforms.Compose([lambda x: x.convert('RGB'),
                                    transforms.Resize(img_size),
                                    # transforms.CenterCrop(224),
                                    # transforms.Resize((img_size, img_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    # transforms.RandomRotation(30, fill=256),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    # transforms.Normalize((0.65, 0.65, 0.65), (0.27, 0.27, 0.27))
                                    # transforms.Normalize((0.54, 0.559, 0.469), (0.293, 0.273, 0.282))
                                    ])
    if test:
        transform = transforms.Compose([lambda x: x.convert('RGB'),
                                        transforms.Resize(img_size),
                                        # transforms.CenterCrop(224),
                                        # transforms.Resize((img_size, img_size)),
                                        # transforms.RandomHorizontalFlip(),
                                        # transforms.RandomVerticalFlip(),
                                        # transforms.RandomRotation(30, fill=256),
                                        transforms.ToTensor(),
                                        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        # transforms.Normalize((0.65, 0.65, 0.65), (0.27, 0.27, 0.27))
                                        # transforms.Normalize((0.54, 0.559, 0.469), (0.293, 0.273, 0.282))
                                        ])
    data = BarImage(path_to_data, mode, transform=transform)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return data_loader


class BACODataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]
        assert x.shape[1] == y.shape[1]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y
