from abc import abstractmethod


class BaseData:
    def __init__(self, img_size):
        self.img_size = img_size
        self.image_height = img_size[0]
        self.image_width = img_size[1]
        self.image_channels = img_size[2]
        self.data_aug = False
        self.task_aug = False

    def get_image_height(self):
        return self.image_height

    def get_image_width(self):
        return self.image_width

    def get_image_channels(self):
        return self.image_channels

    @abstractmethod
    def get_batch(self, source, tasks_per_batch, shot):
        raise NotImplementedError

    @abstractmethod
    def gen_bg(self, config, data="all"):
        raise NotImplementedError

