import numpy as np
import pickle
import torch
import pickle
import sys
import os
from PIL import Image
from utils import Augmenter, task_augment, convert_channel_last_np_to_tensor
from dataset import BaseData

"""
    used for refinement and eval_one_task only
"""


class ShapeNet1DRefinement(BaseData):
    """
        ShapeNet1D dataset for pose estimation

    """
    def __init__(self, path, img_size, seed, data_size="large", aug=None):
        super().__init__(img_size)
        self.num_classes = 1
        self.alpha = 0.3

        assert set(aug).issubset(set(["MR", "data_aug", "task_aug"]))
        self.aug_list = aug
        if "data_aug" in self.aug_list:
            self.Augmentor = Augmenter()
            self.data_aug = True
        if "task_aug" in self.aug_list:
            self.task_aug = True
            self.num_noise = 15
        self.data_size = data_size

        self.x, self.y = pickle.load(open(os.path.join(path, "test_data.pkl"), 'rb'))
        self.x, self.y = np.array(self.x[0]), np.array(self.y[0])
        self.y = self.y[:, -1, None]
        self.x_train, self.x_test = self.x[:-15], self.x[-15:]
        self.y_train, self.y_test = self.y[:-15], self.y[-15:]
        self.max_shot_train = self.x_train.shape[0]
        self.test_shot = self.x_test.shape[0]
        print(f"Test Samples = {self.x_test.shape[0]}, Train samples : {self.x_train.shape[0]}")
        self.test_rng = np.random.RandomState(seed)
        self.val_rng = np.random.RandomState(seed)
        self.test_counter = 0
        np.random.seed(seed)

    def get_batch(self, source, tasks_per_batch, shot):
        """Get data batch."""
        xs, ys, xq, yq = [], [], [], []

        tasks_per_batch = 1  # refinement only have one task

        for _ in range(tasks_per_batch):
            # sample WAY classes
            classes = [0]  # only use the first task

            support_set = []
            query_set = []
            support_sety = []
            query_sety = []
            for k in list(classes):
                # sample SHOT and QUERY instances
                # idx = np.random.choice(range(np.shape(self.x_train)[0]), size=shot, replace=False)
                # if source == "refine_train":
                idx = np.random.permutation(shot)
                x_k = self.x_train[idx]
                y_k = self.y_train[idx]

                support_set.append(x_k)
                support_sety.append(y_k)
                query_set.append(self.x_test)
                query_sety.append(self.y_test)

            xs_k = np.concatenate(support_set, 0)
            xq_k = np.concatenate(query_set, 0)
            ys_k = np.concatenate(support_sety, 0)
            yq_k = np.concatenate(query_sety, 0)

            xs.append(xs_k)
            xq.append(xq_k)
            ys.append(ys_k)
            yq.append(yq_k)

        xs, ys = np.stack(xs, 0), np.stack(ys, 0)
        xq, yq = np.stack(xq, 0), np.stack(yq, 0)

        xs = np.reshape(
            xs,
            [tasks_per_batch, shot * self.num_classes, *self.img_size])
        xq = np.reshape(
            xq,
            [tasks_per_batch, self.test_shot * self.num_classes, *self.img_size])

        ys = ys.astype(np.float32) * 2 * np.pi
        yq = yq.astype(np.float32) * 2 * np.pi

        if self.data_aug and (source == 'train' or source == 'refine_train'):
            xs = self.Augmentor.generate(xs)
            xq = self.Augmentor.generate(xq)
            # plot image for debug
            # tmp_img = Image.fromarray(xs[0, 0, :, :, 0])
            # tmp_img.show()
        # if self.task_aug and source == 'train':
        #     noise = np.linspace(0, 2, self.num_noise+1)[:-1]
        #     y_noise = np.random.choice(noise, (tasks_per_batch, 1))[:, None, :]
        #     # y_noise = np.random.uniform(-1, 1, size=(tasks_per_batch, 1))[:, None, :] * 10.0 * self.alpha
        #     ys += y_noise
        #     yq += y_noise
        #     ys %= 2 * np.pi
        #     yq %= 2 * np.pi

        xs = xs.astype(np.float32) / 255.0
        xq = xq.astype(np.float32) / 255.0

        ys = np.concatenate([np.cos(ys), np.sin(ys), ys], axis=-1)
        yq = np.concatenate([np.cos(yq), np.sin(yq), yq], axis=-1)
        xs = convert_channel_last_np_to_tensor(xs)
        xq = convert_channel_last_np_to_tensor(xq)
        return xs, xq, torch.from_numpy(ys).type(torch.FloatTensor), torch.from_numpy(yq).type(torch.FloatTensor)

    def gen_bg(self, config, data="all"):
        pass
