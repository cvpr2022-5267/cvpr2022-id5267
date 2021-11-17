#!/usr/bin/python3

# generate bar images for toy_dexdexnet
# create three dataset, train, test, and val
# first two used for meta-learn, val used for testing on new task

import os
from pathlib import Path
import random
import json
import math
import numpy as np
import yaml
from itertools import permutations, product
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process, Pool

from dataset.normalize_label import cal_label_mu_sigma
from utils import save_config
from utils.algebra import calc_function
from dataset.dataset import BarImage, bar


class BarGenerator:
    """Bar generator"""
    def __init__(self, kwargs):
        self.save_path       = kwargs['save_path']
        self.task_num  = kwargs["task_num"]
        # self.val_task_num   = kwargs["val_task_num"]
        # self.test_task_num    = kwargs["test_task_num"]
        self.img_num_per_task = kwargs['img_num_per_task']
        # self.img_num_per_task_others = kwargs['img_num_per_task_others']
        self.is_parallel = kwargs["parallel"]
        self.exist_ok = kwargs['exist_ok']
        self.c_num = kwargs['c_choose_num']
        self.verbose = kwargs["verbose"]
        self.figsize = kwargs["figsize"]
        self.dpi = kwargs["figure_dpi"]
        self.linewidth = kwargs["line_width"]
        self.img_w, self.img_h = kwargs["img_w"], kwargs["img_h"]
        self.len_range = kwargs["length_range"]
        self.length_type = kwargs['length_type']
        self.seed = kwargs['seed']
        np.random.seed(self.seed)
        random.seed(self.seed)
        # self.seed_val = kwargs['seed'] * 2
        # self.seed_test = kwargs['seed'] * 3
        # generate color dict
        self._color_dict = self.generate_color(self.task_num, self.seed)
        # self._val_dict = self.generate_color(self.val_task_num, self.seed_val)
        # self._val_dict = {i: self._train_dict[i] for i in range(self.val_task_num)}
        # self._val_dict = self._train_dict
        # self._test_dict = self.generate_color(self.test_task_num, self.seed_test)
        save_config(self.save_path, "config.cfg", kwargs)


    @property
    def color_dict(self):
        return self._color_dict

    # @property
    # def test_dict(self):
    #     return self._test_dict
    #
    # @property
    # def val_dict(self):
    #     return self._val_dict

    def make_dirs(self, flag, exist_ok):
        color_path = os.path.join(self.save_path, flag)
        os.makedirs(os.path.join(color_path, "images"), exist_ok=exist_ok)
        os.makedirs(os.path.join(color_path, "labels"), exist_ok=exist_ok)

    def generate_color_dict(self, colors):
        color_dict = {}
        for i, c in enumerate(colors):
            color_dict[i] = c
        return color_dict

    # def generate_colors(self):
    #     """return train_colors(list, len: task_num), test_color(list, len: 1) and val_color(list, len: 1)"""
    #     a = np.linspace(0.3, 0.9, 6).round(decimals=1)
    #     # colors = list(product(a, repeat=3))
    #     colors = list(permutations(a, 3))
    #     random.shuffle(colors)
    #     train_colors = colors[:self.train_task_num]
    #     val_colors = train_colors
    #     test_colors = colors[self.train_task_num:self.train_task_num + self.test_task_num]
    #     self._train_dict  = self.generate_color_dict(train_colors)
    #     self._val_dict    = self.generate_color_dict(val_colors)
    #     self._test_dict   = self.generate_color_dict(test_colors)

    def generate_color(self, task_num, seed):
        """return train_colors(list, len: task_num), test_color(list, len: 1) and val_color(list, len: 1)"""
        # color = np.random.rand(task_num, 3)
        color = np.random.uniform(0.3, 0.8, (task_num, 3))
        # color = np.random.randint(10, 240, (task_num, 3))
        # a = np.linspace(0.3, 0.9, 30).round(decimals=4)
        # a = np.linspace(0.3, 0.9, 30)
        # color = list(product(a, repeat=3))
        # color = product(a, repeat=3)
        # color = np.array(color)
        # np.random.shuffle(color)
        # color = color[:task_num]

        color_check = np.unique(color, axis=0)
        assert color.shape[0] == color_check.shape[0]
        color_dict = {}
        for i, c in enumerate(color):
            color_dict[i] = tuple(c)
        return color_dict

    def generate_bar(self,):
        x1, y1, y2 = random.randint(3, self.img_w - 3), random.randint(3, self.img_h - 3), random.randint(3, self.img_h - 3)

        if self.is_parallel:
            x2 = x1
        else:
            x2 = random.randint(3, self.img_w - 3)
        length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        while (x1 == x2 and y1 == y2) or (length > self.len_range[1]) or (length < self.len_range[0]):
            x1, y1, y2 = random.randint(3, self.img_w - 3), random.randint(3, self.img_h - 3), random.randint(3, self.img_h - 3)
            if self.is_parallel:
                x2 = x1
            else:
                x2 = random.randint(3, self.img_w - 3)
            length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return length, x1, x2, y1, y2

    def draw_bar(self, c, x_select):
        """

        c: color
        linewidth:
        img_w:
        img_h:
        len_range: list() of bar length range

        """
        length, x1, x2, y1, y2 = self.generate_bar()
        # while x1 in x_select:
        #     length, x1, x2, y1, y2 = generate_bar(img_w, img_h, len_range)
        if self.is_parallel:
            if x_select:
                x = x_select[-1]
                a = random.randint(5, 10)
                a = a + x
                if a > self.img_w:
                    a = a - self.img_w
                x1 = x2 = a

        plt.xlim(0, self.img_w)
        plt.ylim(0, self.img_h)
        plt.axis('off')
        plt.plot((x1, x2), (y1, y2), color=c, linewidth=self.linewidth, antialiased=True)

        return length, x1, x2, y1, y2

    def get_line_x_y(self, center_x, center_y, length, angle):
        x1 = -length / 2 * math.cos(angle) + center_x
        x2 = length / 2 * math.cos(angle) + center_x
        y1 = -length / 2 * math.sin(angle) + center_y
        y2 = length / 2 * math.sin(angle) + center_y
        return x1, x2, y1, y2

    def draw_bar_same_range(self, c, x_select, length, angle):
        """

        c: color
        linewidth:
        img_w:
        img_h:
        len_range: list() of bar length range

        """
        if self.length_type == 'random':
            length = np.random.randint(*self.len_range)
        if not self.is_parallel:
            x_min, x_max = length / 2, self.img_w - length / 2
            y_min, y_max = length / 2, self.img_h - length / 2
            line_center_x = random.uniform(x_min, x_max)
            line_center_y = random.uniform(y_min, y_max)
            angle = np.random.uniform(0, math.pi / 2)
            b = calc_function(line_center_x, line_center_y, angle)
            x1, x2, y1, y2 = self.get_line_x_y(line_center_x, line_center_y, length, angle)
            # while x1 in x_select:
            #     length, x1, x2, y1, y2 = generate_bar(img_w, img_h, len_range)
        elif self.is_parallel:
            x_min, x_max = 2, self.img_w - 2
            # x_min, x_max = length / 2 + 2, self.img_w - length / 2 - 1
            y_min, y_max = length / 2, self.img_h - length / 2
            line_center_x = random.uniform(x_min, x_max)
            line_center_y = random.uniform(y_min, y_max)
            # b = calc_function(line_center_x, line_center_y, angle)
            angle = math.pi / 2
            th = 3
            ### generate rotate parallel bars with some distance th
            # if x_select:
            #     dist = []
            #     for b_old in x_select:
            #         d = abs(b - b_old)
            #         d = abs(d * math.cos(angle))
            #         if d < th:
            #             dist.append(d)
            #     while len(dist):
            #         dist = []
            #         line_center_x = np.random.randint(x_min, x_max)
            #         line_center_y = np.random.randint(y_min, y_max)
            #         b = calc_function(line_center_x, line_center_y, angle)
            #         for b_old in x_select:
            #             d = abs(b - b_old)
            #             d = abs(d * math.cos(angle))
            #             if d < th:
            #                 dist.append(d)

            x1, x2, y1, y2 = self.get_line_x_y(line_center_x, line_center_y, length, angle)
            # if x_select:
            #     x = x_select[-1]
            #     # a = random.randint(5, 10)
            #     a = random.randint(5, 8)
            #     a = a + x
            #     if a > self.img_w - 2:
            #         a = a - self.img_w
            #         if a < 2:
            #             a += random.randint(5, 8)
            #     x1 = x2 = a

            # x1 = x2 = 7
            # if x_select:
            #     a = x_select[-1] + 8
            #     x1 = x2 = a

            # x1 = x2 = 7
            if x_select:
                # a = x_select[-1] + np.random.randint(4, 50)
                a = x_select[-1] + np.random.randint(4, 20)
                if a > self.img_w - 2:
                    x1 = x2 = a - (self.img_w - 2) + 2
                else:
                    x1 = x2 = a
            b = 0

        plt.xlim(0, self.img_w)
        plt.ylim(0, self.img_h)
        plt.axis('off')
        plt.tight_layout()
        # plt.plot((x1, x2), (y1, y2), color=c, linewidth=self.linewidth, antialiased=False)
        plt.plot((x1, x2), (y1, y2), color=c, linewidth=self.linewidth)

        return length, x1, x2, y1, y2, b

    def draw_all_bars(self, color_test, color_ctx):
        """
        draw all bars for target and context colors
        len_candidate: list of length range [[s1, e1], [s2, e2] ...]
        data: dict(), used to save bar length for each color
        color_test: target color
        color_ctx: list of context color
        """
        data_length = {}
        x_select = []
        # save target label
        color_ctx.append(color_test)
        # random the order between color_test and color_ctx
        random.shuffle(color_ctx)
        len_list = [12, 20, 28]
        random.shuffle(len_list)
        angle = random.uniform(0, math.pi)
        for i, ctx_c in enumerate(color_ctx):
            length = len_list[i]
            length, x1, x2, y1, y2, b = self.draw_bar_same_range(ctx_c, x_select, length, angle)
            # length, x1, x2, y1, y2 = self.draw_bar(ctx_c, x_select)
            x_select.append(x1)
            # x_select.append(b)
            data_length[str(ctx_c)] = length

        # data[color_test] = length_t
        return data_length

    # def plot_and_save_bars(self, i, k, counter, color_dict, color_ctx_ind, flag, img_num_per_task, fixed=False):
    #     """
    #     plot bar for each color and save line length
    #     i: ith image for single task
    #     k: kth test color
    #     counter: counter images number for each test color
    #     color_dict: color dict
    #     color_ctx_ind: ind for context color
    #     fixed: used in the end when there is no context color to choose
    #     """
    #     plt.rcParams["figure.figsize"] = self.figsize
    #     plt.rcParams['savefig.dpi'] = self.dpi
    #
    #     color_test = color_dict[k]
    #     ind_test = i + k * img_num_per_task
    #     save_img_path = os.path.join(os.path.join(self.save_path, flag), "images")
    #     save_label_path = os.path.join(os.path.join(self.save_path, flag), "labels")
    #
    #     data = {}
    #     color_ctx = [color_dict[i] for i in color_ctx_ind]
    #
    #     # draw bars and save target label
    #     data = self.draw_all_bars(data, color_test, color_ctx)
    #
    #     # save before show
    #     plt.savefig(os.path.join(save_img_path, f"{ind_test:05d}.png"))
    #     if fixed:
    #         print(f"fixed for {ind_test:05d}.png")
    #     with open(os.path.join(save_label_path, f"{ind_test:05d}.yaml"), "w") as f:
    #         yaml.dump({str(color_test): data[color_test]}, f)
    #     if not fixed:
    #         for i in color_ctx_ind:
    #             ind_ctx = counter[i] + img_num_per_task * i
    #             plt.savefig(os.path.join(save_img_path, f"{ind_ctx:05d}.png"))
    #             with open(os.path.join(save_label_path, f"{ind_ctx:05d}.yaml"), "w") as f:
    #                 yaml.dump({str(color_dict[i]): data[color_dict[i]]}, f)
    #     if not self.verbose:
    #         plt.show()
    #     plt.clf()
    #
    #     if not fixed:
    #         for i in color_ctx_ind:
    #             counter[i] = counter[i] + 1
    #             if counter[i] >= img_num_per_task:
    #                 del color_dict[i]
    #     counter[k] = counter[k] + 1
    #     if counter[k] >= img_num_per_task:
    #         del color_dict[k]
    #     return color_dict, counter

    def plot_and_save_bars(self, data, i, k, color_dict, color_ctx_ind, folder_path):
        """
        plot bar for each color and save line length
        i: ith image for single task
        k: kth test color
        counter: counter images number for each test color
        color_dict: color dict
        color_ctx_ind: ind for context color
        fixed: used in the end when there is no context color to choose
        """
        plt.rcParams["figure.figsize"] = self.figsize
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['image.interpolation'] = 'bilinear'
        plt.rcParams['image.resample'] = False
        color_test = color_dict[k]
        ind_test = i
        # save_img_path = os.path.join(os.path.join(self.save_path, flag), "images")
        # save_label_path = os.path.join(os.path.join(self.save_path, flag), "labels")

        color_ctx = [color_dict[i] for i in color_ctx_ind]
        # draw bars and save target label
        data_length = self.draw_all_bars(color_test, color_ctx)
        data['gt'].append(data_length[str(color_test)])
        # save before show
        plt.savefig(os.path.join(folder_path, f"{ind_test:06d}.png"))
        result = {}
        # result[str(color_test)] = data[color_test]
        print(folder_path, f"{ind_test:06d}.png", "finished")
        if not self.verbose:
            plt.show()
        plt.clf()

    # def gen_dataset(self, color_dict, img_num_per_task, seed, flag):
    #     """
    #     generate images for color lists
    #     num: images number for each color
    #     color_dict: dict(), {0:[0.1, 0.5, 0.7], ...}
    #     flag: str(), "train", "test", "val"
    #     """
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     color_fix = color_dict.copy()
    #     # make dir for each flag
    #     self.make_dirs(flag, exist_ok=self.exist_ok)
    #     counter = {k: 0 for k in range(len(color_dict))}
    #     ind = 0
    #     for k in range(len(color_dict)):
    #         i = counter[k]
    #         while i < img_num_per_task:
    #             choose_from = list(color_dict.keys())
    #             choose_from.remove(k)
    #             try:
    #                 color_ctx_ind = random.sample(choose_from, self.c_num - 1)
    #                 color_dict, counter = self.plot_and_save_bars(i, k, counter, color_dict, color_ctx_ind, flag,
    #                                                          img_num_per_task)
    #             except ValueError as e:
    #                 print("choose colors are less than 2", "k = ", k)
    #                 choose_from = list(color_fix.keys())
    #                 choose_from.remove(k)
    #                 color_ctx_ind = random.sample(choose_from, self.c_num - 1)
    #                 color_fix, counter = self.plot_and_save_bars(i, k, counter, color_fix, color_ctx_ind, flag,
    #                                                         img_num_per_task, fixed=True)
    #             i = i + 1
    #     return color_dict, counter

    def generate_for_each_task(self, k, color_dict, img_num_per_task, flag):
        i = 0

        while i < img_num_per_task:
            choose_from = list(color_dict.keys())
            choose_from.remove(k)
            color_ctx_ind = random.sample(choose_from, self.c_num - 1)
            self.plot_and_save_bars(i, k, color_dict, color_ctx_ind, flag, img_num_per_task)
            i = i + 1

    def generate_for_each_task_same_color(self, k, color_dict, img_num_per_task, start_id):
        folder_id = k + start_id
        folder_path = os.path.join(self.save_path, str(folder_id))
        os.makedirs(folder_path, exist_ok=False)

        choose_from = list(color_dict.keys())
        choose_from.remove(k)
        color_ctx_ind = random.sample(choose_from, self.c_num - 1)
        color_ctx = [color_dict[i] for i in color_ctx_ind]
        data = {}
        data['color_test'] = str(color_dict[k])
        data['color_ctx'] = str(color_ctx)
        data['gt'] = []

        i = 0
        while i < img_num_per_task:
            self.plot_and_save_bars(data, i, k, color_dict, color_ctx_ind, folder_path)
            i = i + 1
        with open(os.path.join(folder_path, "label.yaml"), "w") as f:
            yaml.dump(data, f)

    def gen_dataset_mp(self, color_dict, img_num_per_task, seed, flag, round=1):
        """
        generate images for color lists
        num: images number for each color
        color_dict: dict(), {0:[0.1, 0.5, 0.7], ...}
        flag: str(), "train", "test", "val"
        """
        random.seed(seed)
        np.random.seed(seed)
        processes = []
        # make dir for each flag
        self.make_dirs(flag, exist_ok=self.exist_ok)
        counter = {k: 0 for k in range(len(color_dict))}
        p = Pool(multiprocessing.cpu_count() - 1)
        k = list(range(len(color_dict)))
        # args = [(i, color_dict, img_num_per_task, flag) for i in k]

        # for k in range(len(color_dict)):
        # p.starmap(self.generate_for_each_task, [(i, color_dict, img_num_per_task, flag) for i in k])
        for r in range(round):
            start_id = r * img_num_per_task * len(color_dict)
            p.starmap(self.generate_for_each_task_same_color, [(i, color_dict, img_num_per_task, flag, start_id) for i in k])
        p.close()
        p.join()
        # processes.append(p)
        # [x.start() for x in processes]
        # [x.join() for x in processes]
        return color_dict, counter

    def gen_dataset(self, color_dict, img_num_per_task, round=1):
        """
        generate images for color lists
        num: images number for each color
        color_dict: dict(), {0:[0.1, 0.5, 0.7], ...}
        flag: str(), "train", "test", "val"
        """
        processes = []
        # make dir for each flag
        # self.make_dirs(flag, exist_ok=self.exist_ok)
        counter = {k: 0 for k in range(len(color_dict))}
        k = list(range(len(color_dict)))
        # args = [(i, color_dict, img_num_per_task, flag) for i in k]

        # for k in range(len(color_dict)):
        # p.starmap(self.generate_for_each_task, [(i, color_dict, img_num_per_task, flag) for i in k])
        for r in range(round):
            start_id = r * len(color_dict)
            for i in k:
                self.generate_for_each_task_same_color(i, color_dict, img_num_per_task, start_id)

        return color_dict, counter

    def save_colors(self):
        with open(os.path.join(self.save_path, "./colors.json"), "w") as f:
            json.dump(self._color_dict, f, sort_keys=True, indent=4)
        # with open(os.path.join(self.save_path, "test/colors.json"), "w") as f:
        #     json.dump(self._test_dict, f, sort_keys=True, indent=4)
        # with open(os.path.join(self.save_path, "val/colors.json"), "w") as f:
        #     json.dump(self._val_dict, f, sort_keys=True, indent=4)

    def cal_label_mu_sigma(self):
        # generate label mu and sigma
        for flag in ["train", "test", "val"]:
            cal_label_mu_sigma(flag)

    def extract_length_range(self, len_range, c_num):
        """
        len_range: bar length range [start, end]
        c_num: number of bars + 1
        return: length range list for each bar
        """
        len_candidate = []
        len_choose_list = np.linspace(len_range[0], len_range[1], c_num + 1)
        for i in range(len(len_choose_list) - 1):
            l = [len_choose_list[i], len_choose_list[i + 1] - 2]  # [10, 15, 20, 25] -> [[10, 13], [15, 18], [20, 23]]
            len_candidate.append(l)
        return len_candidate


if __name__ == "__main__":

    file_path = Path(__file__)
    root_path = file_path.parent.parent

    img_kwargs = {
        "figsize": (8, 8),
        "figure_dpi": 4,
        "parallel": True,
        "img_h": 32,
        "img_w": 32,
        "line_width": 30,
        "length_range": [6, 23],
        "length_type": "random",
        "task_num": 40,
        "img_num_per_task": 20,
        "c_choose_num": 2,
        "save_path": str(root_path) + "/data/bars",
        "verbose": True,
        "exist_ok": True,
        "round": 10,
        "seed": 1234,
    }

    # image generator
    generator = BarGenerator(img_kwargs)
    color_dict = generator.color_dict
    # test_dict = generator.test_dict
    # val_dict = generator.val_dict

    generator.gen_dataset(color_dict, img_kwargs['img_num_per_task'], round=img_kwargs['round'])
    # generate val images
    # generator.gen_dataset(val_dict, img_kwargs['img_num_per_task_others'], img_kwargs['seed'] * 2, flag="val")
    # # generate test images
    # generator.gen_dataset(test_dict, img_kwargs['img_num_per_task_others'], img_kwargs['seed'] * 3, flag="test")

    generator.save_colors()

    # train_bs = kwargs["bs"] * kwargs['img_num_per_task_train']
    # val_bs = kwargs["val_task_num"] * kwargs['img_num_per_task_others']
    # test_bs = kwargs["test_task_num"] * kwargs['img_num_per_task_others']

    # train_loader = bar(batch_size=train_bs, path_to_data=kwargs["root_dir"],
    #                    mode="train", img_size=kwargs['img_size'][0], shuffle=False)
    #
    # test_loader = bar(batch_size=test_bs, path_to_data=kwargs["root_dir"],
    #                   mode="test", img_size=kwargs['img_size'][0], shuffle=False)
    #
    # val_loader = bar(batch_size=val_bs, path_to_data=kwargs["root_dir"],
    #                  mode="val", img_size=kwargs['img_size'][0], shuffle=False)


